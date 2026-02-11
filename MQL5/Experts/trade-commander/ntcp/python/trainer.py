"""
NTCP Trainer — training loop with factor-looping, early stopping, and AMP.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig, TrainConfig
from .data_manager import NTCPDataManager
from .model import NTCPModel

logger = logging.getLogger(__name__)

LogCallback = Callable[[str], None]


class TrainingResult:
    """Stores results from a single training run."""

    def __init__(self, stf_factor: int) -> None:
        self.stf_factor = stf_factor
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.model_state: Optional[dict] = None
        self.feature_cols: list[str] = []
        self.target_cols: list[str] = []
        self.cls_target_cols: list[str] = []
        self.scaling_params: dict = {}
        self.num_features: int = 0
        self.dataset_size: int = 0


class NTCPTrainer:
    """Training engine with time-series split, early stopping, and CUDA AMP."""

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        log_callback: Optional[LogCallback] = None,
    ) -> None:
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.log_callback = log_callback or (lambda msg: None)
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def _log(self, msg: str) -> None:
        logger.info(msg)
        self.log_callback(msg)

    def _get_device(self) -> torch.device:
        if self.train_cfg.use_cuda and torch.cuda.is_available():
            dev = torch.device("cuda")
            # TF32 optimizations for Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            self._log(f"Using CUDA: {torch.cuda.get_device_name(0)} (TF32 enabled)")
            return dev
        self._log("Using CPU")
        return torch.device("cpu")

    def train_single_factor(
        self,
        stf_factor: int,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
        sample_callback: Optional[Callable] = None,
    ) -> TrainingResult:
        """
        Run full pipeline for a single STF factor:
        DataManager -> Dataset -> Model -> Train -> Checkpoint.
        """
        self._stop_requested = False
        result = TrainingResult(stf_factor)

        # Data pipeline
        cfg = DataConfig(
            m5_path=self.data_cfg.m5_path,
            base_tf_minutes=self.data_cfg.base_tf_minutes,
            stf_factor=stf_factor,
            lookback=self.data_cfg.lookback,
            clip_percentile=self.data_cfg.clip_percentile,
            rolling_window_size=self.data_cfg.rolling_window_size,
            hurst_window=self.data_cfg.hurst_window,
            regime_n_clusters=self.data_cfg.regime_n_clusters,
            ma_min=self.data_cfg.ma_min,
            ma_max=self.data_cfg.ma_max,
            ma_count=self.data_cfg.ma_count,
            signal_ma_count=self.data_cfg.signal_ma_count,
            strategy=self.data_cfg.strategy,
            mafilter_nh=self.data_cfg.mafilter_nh,
            mafilter_fac_slope=self.data_cfg.mafilter_fac_slope,
            mafilter_fslope_thresh=self.data_cfg.mafilter_fslope_thresh,
            mafilter_max_bw_factor=self.data_cfg.mafilter_max_bw_factor,
            mafilter_stddev_factor=self.data_cfg.mafilter_stddev_factor,
            mafilter_exit_option=self.data_cfg.mafilter_exit_option,
            active_feature_groups=self.data_cfg.active_feature_groups,
        )
        dm = NTCPDataManager(cfg)
        dataset = dm.run_pipeline(log_callback=self.log_callback)
        result.feature_cols = dm.feature_cols
        result.target_cols = dm.target_cols
        result.cls_target_cols = dm.cls_target_cols
        result.scaling_params = dm.scaling_params
        result.num_features = len(dm.feature_cols)
        result.dataset_size = len(dataset)

        self._log(f"Factor {stf_factor}: {len(dataset)} samples, "
                   f"{result.num_features} features")

        # Time-series split (no shuffle for val)
        n = len(dataset)
        val_size = int(n * self.train_cfg.val_split)
        train_size = n - val_size

        train_ds = torch.utils.data.Subset(dataset, range(train_size))
        val_ds = torch.utils.data.Subset(dataset, range(train_size, n))

        # DataLoader with optional multiprocessing
        use_cuda = self.train_cfg.use_cuda and torch.cuda.is_available()
        num_workers = min(4, os.cpu_count() or 1) if use_cuda else 0
        dl_kwargs = {}
        if num_workers > 0:
            dl_kwargs = dict(
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
            )

        train_loader = DataLoader(
            train_ds, batch_size=self.train_cfg.batch_size, shuffle=True,
            **dl_kwargs,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.train_cfg.batch_size, shuffle=False,
            **dl_kwargs,
        )

        # Batch stats for debugging
        first_seq, first_tgt, first_cls = next(iter(train_loader))
        self._log(
            f"Batch stats — Input:  mean={first_seq.mean():.4f} "
            f"std={first_seq.std():.4f} min={first_seq.min():.4f} "
            f"max={first_seq.max():.4f}"
        )
        self._log(
            f"Batch stats — Target: mean={first_tgt.mean():.4f} "
            f"std={first_tgt.std():.4f} min={first_tgt.min():.4f} "
            f"max={first_tgt.max():.4f}"
        )
        self._log(
            f"Batch stats — Cls:    pos_long={first_cls[:,0].sum():.0f}/{len(first_cls)} "
            f"pos_short={first_cls[:,1].sum():.0f}/{len(first_cls)}"
        )

        # Model
        self.model_cfg.input_size = result.num_features
        device = self._get_device()
        model = NTCPModel(self.model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.train_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=max(1, self.train_cfg.patience // 2),
            factor=0.5,
        )
        criterion_reg = nn.MSELoss()
        criterion_cls = nn.BCEWithLogitsLoss()
        cls_weight = self.train_cfg.cls_loss_weight

        # AMP scaler for CUDA
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_val = float("inf")
        patience_counter = 0

        for epoch in range(1, self.train_cfg.epochs + 1):
            if self._stop_requested:
                self._log("Training stopped by user.")
                break

            # --- Train ---
            model.train()
            train_loss_sum = 0.0
            train_mse_sum = 0.0
            train_bce_sum = 0.0
            train_batches = 0

            for seq, tgt, cls_tgt in train_loader:
                seq, tgt, cls_tgt = seq.to(device), tgt.to(device), cls_tgt.to(device)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        reg_pred, cls_pred = model(seq)
                        mse_loss = criterion_reg(reg_pred, tgt)
                        bce_loss = criterion_cls(cls_pred, cls_tgt)
                        loss = mse_loss + cls_weight * bce_loss
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    reg_pred, cls_pred = model(seq)
                    mse_loss = criterion_reg(reg_pred, tgt)
                    bce_loss = criterion_cls(cls_pred, cls_tgt)
                    loss = mse_loss + cls_weight * bce_loss
                    loss.backward()
                    optimizer.step()

                train_loss_sum += loss.item()
                train_mse_sum += mse_loss.item()
                train_bce_sum += bce_loss.item()
                train_batches += 1

            avg_train = train_loss_sum / max(train_batches, 1)
            avg_train_mse = train_mse_sum / max(train_batches, 1)
            avg_train_bce = train_bce_sum / max(train_batches, 1)

            # --- Validate ---
            model.eval()
            val_loss_sum = 0.0
            val_mse_sum = 0.0
            val_bce_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for seq, tgt, cls_tgt in val_loader:
                    seq, tgt, cls_tgt = seq.to(device), tgt.to(device), cls_tgt.to(device)
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            reg_pred, cls_pred = model(seq)
                            mse_loss = criterion_reg(reg_pred, tgt)
                            bce_loss = criterion_cls(cls_pred, cls_tgt)
                            loss = mse_loss + cls_weight * bce_loss
                    else:
                        reg_pred, cls_pred = model(seq)
                        mse_loss = criterion_reg(reg_pred, tgt)
                        bce_loss = criterion_cls(cls_pred, cls_tgt)
                        loss = mse_loss + cls_weight * bce_loss
                    val_loss_sum += loss.item()
                    val_mse_sum += mse_loss.item()
                    val_bce_sum += bce_loss.item()
                    val_batches += 1

            avg_val = val_loss_sum / max(val_batches, 1)
            avg_val_mse = val_mse_sum / max(val_batches, 1)
            avg_val_bce = val_bce_sum / max(val_batches, 1)
            scheduler.step(avg_val)

            result.train_losses.append(avg_train)
            result.val_losses.append(avg_val)

            # Sample predictions for live monitor
            if sample_callback:
                with torch.no_grad():
                    s_seq, s_tgt, s_cls = next(iter(val_loader))
                    s_seq = s_seq.to(device)
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            s_reg_pred, s_cls_pred = model(s_seq)
                    else:
                        s_reg_pred, s_cls_pred = model(s_seq)
                    # Cls confidence: sigmoid of logits, column 0 = long
                    cls_probs = torch.sigmoid(s_cls_pred).cpu().numpy()
                    cls_actuals = s_cls.numpy()
                    # Regression: MFE 10-bar horizon (index 6 in 18-target layout)
                    tgt_idx = min(6, s_tgt.shape[1] - 1)
                    sample_callback(
                        epoch,
                        s_tgt[:, tgt_idx].numpy(),
                        s_reg_pred[:, tgt_idx].cpu().numpy(),
                        cls_actuals,
                        cls_probs,
                    )

            # Early stopping checkpoint
            if avg_val < best_val:
                best_val = avg_val
                patience_counter = 0
                result.best_val_loss = avg_val
                result.best_epoch = epoch
                result.model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
            else:
                patience_counter += 1

            lr = optimizer.param_groups[0]["lr"]
            self._log(
                f"Epoch {epoch}/{self.train_cfg.epochs} | "
                f"Train: {avg_train:.6f} (MSE={avg_train_mse:.6f} BCE={avg_train_bce:.4f}) | "
                f"Val: {avg_val:.6f} (MSE={avg_val_mse:.6f} BCE={avg_val_bce:.4f}) | "
                f"LR: {lr:.2e} | Best: {best_val:.6f} (ep {result.best_epoch})"
            )
            if progress_callback:
                progress_callback(epoch, self.train_cfg.epochs, avg_train, avg_val)

            if patience_counter >= self.train_cfg.patience:
                self._log(f"Early stopping at epoch {epoch} "
                           f"(no improvement for {self.train_cfg.patience} epochs)")
                break

        return result

    def train_factor_range(
        self,
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
        sample_callback: Optional[Callable] = None,
    ) -> list[TrainingResult]:
        """Iterate STF factors [min..max] and run full pipeline per factor."""
        results: list[TrainingResult] = []
        f_min = self.train_cfg.stf_factor_min
        f_max = self.train_cfg.stf_factor_max

        for factor in range(f_min, f_max + 1):
            if self._stop_requested:
                break
            self._log(f"\n{'='*60}")
            self._log(f"STF Factor {factor} of [{f_min}..{f_max}]")
            self._log(f"{'='*60}")
            result = self.train_single_factor(factor, progress_callback, sample_callback)
            results.append(result)
            self._log(f"Factor {factor} done — best val loss: "
                       f"{result.best_val_loss:.6f} at epoch {result.best_epoch}")

        # Log summary
        if results:
            best = min(results, key=lambda r: r.best_val_loss)
            self._log(f"\nBest factor: {best.stf_factor} "
                       f"(val loss {best.best_val_loss:.6f})")

        return results
