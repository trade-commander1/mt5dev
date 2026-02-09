"""
NTCP Backtest — equity simulation and validation metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TARGET_HORIZONS, ModelConfig
from .data_manager import NTCPDataset
from .model import NTCPModel

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Container for backtest results."""
    equity_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    per_target_r2: dict[str, float] = field(default_factory=dict)
    per_target_mse: dict[str, float] = field(default_factory=dict)
    target_check: dict[str, dict[str, float]] = field(default_factory=dict)


class NTCPBacktester:
    """Run model inference on validation set and compute trading metrics."""

    def __init__(
        self,
        model_state: dict,
        model_cfg: ModelConfig,
        crv_threshold: float = 1.5,
    ) -> None:
        self.model_state = model_state
        self.model_cfg = model_cfg
        self.crv_threshold = crv_threshold

    def run(
        self,
        dataset: NTCPDataset,
        target_cols: list[str],
        batch_size: int = 512,
        use_cuda: bool = True,
    ) -> BacktestMetrics:
        """
        Run backtest on the given dataset.

        Strategy: If predicted MFE / |MAE| >= crv_threshold, take the trade.
        PnL = actual momentum target.
        """
        device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        model = NTCPModel(self.model_cfg).to(device)
        model.load_state_dict(self.model_state)
        model.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for seq, tgt in loader:
                seq = seq.to(device)
                pred = model(seq)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(tgt.numpy())

        preds = np.concatenate(all_preds, axis=0)
        actuals = np.concatenate(all_targets, axis=0)

        metrics = BacktestMetrics()

        # Per-target R^2, MSE, and comprehensive target check
        for j, name in enumerate(target_cols):
            y_true = actuals[:, j]
            y_pred = preds[:, j]
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-9)
            mse = float(np.mean((y_true - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_true - y_pred)))
            corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0
            if np.isnan(corr):
                corr = 0.0
            pred_mean = float(np.mean(y_pred))
            actual_mean = float(np.mean(y_true))
            pred_std = float(np.std(y_pred))
            actual_std = float(np.std(y_true))

            metrics.per_target_r2[name] = float(r2)
            metrics.per_target_mse[name] = mse
            metrics.target_check[name] = {
                "r2": float(r2),
                "mse": mse,
                "mae": mae,
                "correlation": corr,
                "pred_mean": pred_mean,
                "actual_mean": actual_mean,
                "pred_std": pred_std,
                "actual_std": actual_std,
                "ratio": pred_mean / (actual_mean + 1e-9),
            }

        # Equity simulation using the first horizon (3 bars)
        # Target layout per horizon: MFE, MAE, Momentum
        # Use horizon index 0 (3 bars) for trading
        mfe_idx = 0  # tgt_mfe_3
        mae_idx = 1  # tgt_mae_3
        mom_idx = 2  # tgt_mom_3

        pred_mfe = preds[:, mfe_idx]
        pred_mae = np.abs(preds[:, mae_idx])
        actual_mom = actuals[:, mom_idx]

        equity = [0.0]
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0

        for i in range(len(pred_mfe)):
            # CRV filter: predicted MFE / |MAE| must exceed threshold
            if pred_mae[i] < 1e-9:
                continue
            crv = pred_mfe[i] / pred_mae[i]
            if crv < self.crv_threshold:
                continue

            pnl = actual_mom[i]
            equity.append(equity[-1] + pnl)

            if pnl > 0:
                wins += 1
                gross_profit += pnl
            else:
                losses += 1
                gross_loss += abs(pnl)

        metrics.equity_curve = np.array(equity)
        metrics.total_trades = wins + losses
        metrics.win_rate = wins / max(wins + losses, 1)
        metrics.profit_factor = gross_profit / max(gross_loss, 1e-9)

        # Max drawdown
        peak = equity[0]
        max_dd = 0.0
        for val in equity:
            if val > peak:
                peak = val
            dd = peak - val
            if dd > max_dd:
                max_dd = dd
        metrics.max_drawdown = max_dd

        logger.info(
            "Backtest: %d trades, WR=%.1f%%, PF=%.2f, MaxDD=%.6f",
            metrics.total_trades,
            metrics.win_rate * 100,
            metrics.profit_factor,
            metrics.max_drawdown,
        )
        return metrics
