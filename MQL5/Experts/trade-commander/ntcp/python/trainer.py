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
            stf_factor=stf_factor,
            lookback=self.data_cfg.lookback,
            clip_percentile=self.data_cfg.clip_percentile,
            rolling_window_size=self.data_cfg.rolling_window_size,
            hurst_window=self.data_cfg.hurst_window,
            regime_n_clusters=self.data_cfg.regime_n_clusters,
            macd_fast=self.data_cfg.macd_fast,
            macd_slow=self.data_cfg.macd_slow,
            macd_signal=self.data_cfg.macd_signal,
            active_feature_groups=self.data_cfg.active_feature_groups,
        )
        dm = NTCPDataManager(cfg)
        dataset = dm.run_pipeline(log_callback=self.log_callback)
        result.feature_cols = dm.feature_cols
        result.target_cols = dm.target_cols
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

        # Model
        self.model_cfg.input_size = result.num_features
        device = self._get_device()
        model = NTCPModel(self.model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.train_cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=max(1, self.train_cfg.patience // 2),
            factor=0.5,
        )
        criterion = nn.MSELoss()

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
            train_batches = 0

            for seq, tgt in train_loader:
                seq, tgt = seq.to(device), tgt.to(device)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred = model(seq)
                        loss = criterion(pred, tgt)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(seq)
                    loss = criterion(pred, tgt)
                    loss.backward()
                    optimizer.step()

                train_loss_sum += loss.item()
                train_batches += 1

            avg_train = train_loss_sum / max(train_batches, 1)

            # --- Validate ---
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for seq, tgt in val_loader:
                    seq, tgt = seq.to(device), tgt.to(device)
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            pred = model(seq)
                            loss = criterion(pred, tgt)
                    else:
                        pred = model(seq)
                        loss = criterion(pred, tgt)
                    val_loss_sum += loss.item()
                    val_batches += 1

            avg_val = val_loss_sum / max(val_batches, 1)
            scheduler.step(avg_val)

            result.train_losses.append(avg_train)
            result.val_losses.append(avg_val)

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
                f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
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
            result = self.train_single_factor(factor, progress_callback)
            results.append(result)
            self._log(f"Factor {factor} done — best val loss: "
                       f"{result.best_val_loss:.6f} at epoch {result.best_epoch}")

        # Log summary
        if results:
            best = min(results, key=lambda r: r.best_val_loss)
            self._log(f"\nBest factor: {best.stf_factor} "
                       f"(val loss {best.best_val_loss:.6f})")

        return results
