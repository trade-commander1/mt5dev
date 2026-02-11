"""
NTCP Backtest — bidirectional equity simulation with dollar PnL.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TARGET_HORIZONS, TARGET_SCALE_FACTOR, ModelConfig
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
    long_trades: int = 0
    short_trades: int = 0
    num_winners: int = 0
    num_losers: int = 0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    total_pnl: float = 0.0
    show_in_dollars: bool = False
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
        spread: float = 0.0,
        slippage: float = 0.0,
        point: float = 0.01,
        tick_value: float = 1.0,
        lot_size: float = 0.10,
        show_in_dollars: bool = False,
        strategy: str = "trendcatcher",
    ) -> None:
        self.model_state = model_state
        self.model_cfg = model_cfg
        self.crv_threshold = crv_threshold
        self.spread = spread
        self.slippage = slippage
        self.point = point
        self.tick_value = tick_value
        self.lot_size = lot_size
        self.show_in_dollars = show_in_dollars
        self.strategy = strategy

    def run(
        self,
        dataset: NTCPDataset,
        target_cols: list[str],
        batch_size: int = 512,
        use_cuda: bool = True,
        entry_prices: np.ndarray | None = None,
    ) -> BacktestMetrics:
        """
        Run backtest on the given dataset.

        Strategy: Bidirectional — uses classification head confidence + CRV filter.
        Long PnL = +actual_mom, Short PnL = -actual_mom.
        """
        device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        model = NTCPModel(self.model_cfg).to(device)
        model.load_state_dict(self.model_state)
        model.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_reg_preds = []
        all_cls_preds = []
        all_targets = []
        all_cls_targets = []

        with torch.no_grad():
            for seq, tgt, cls_tgt in loader:
                seq = seq.to(device)
                reg_pred, cls_pred = model(seq)
                all_reg_preds.append(reg_pred.cpu().numpy())
                all_cls_preds.append(cls_pred.cpu().numpy())
                all_targets.append(tgt.numpy())
                all_cls_targets.append(cls_tgt.numpy())

        preds = np.concatenate(all_reg_preds, axis=0)
        cls_preds = np.concatenate(all_cls_preds, axis=0)
        actuals = np.concatenate(all_targets, axis=0)

        # Apply sigmoid to classification logits
        cls_probs = 1.0 / (1.0 + np.exp(-cls_preds))

        metrics = BacktestMetrics(show_in_dollars=self.show_in_dollars)

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
        mfe_idx = 0  # tgt_mfe_3
        mae_idx = 1  # tgt_mae_3
        mom_idx = 2  # tgt_mom_3

        pred_mfe = preds[:, mfe_idx]
        pred_mae = np.abs(preds[:, mae_idx])
        actual_mom = actuals[:, mom_idx]

        cls_long_conf = cls_probs[:, 0]
        cls_short_conf = cls_probs[:, 1]

        # Diagnostic: log prediction distributions
        logger.info(
            "Pred stats — MFE: mean=%.6f std=%.6f | MAE: mean=%.6f std=%.6f | "
            "cls_long: mean=%.4f | cls_short: mean=%.4f",
            pred_mfe.mean(), pred_mfe.std(),
            pred_mae.mean(), pred_mae.std(),
            cls_long_conf.mean(), cls_short_conf.mean(),
        )

        equity = [0.0]
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        long_count = 0
        short_count = 0

        for i in range(len(pred_mfe)):
            is_long = False
            is_short = False

            if self.strategy == "mafilter":
                # MA Filter: cls head is a hard gate — only trade when
                # confidence exceeds CRV threshold (used as cls threshold)
                if cls_long_conf[i] >= self.crv_threshold:
                    is_long = True
                elif cls_short_conf[i] >= self.crv_threshold:
                    is_short = True
            else:
                # Trendcatcher: regression CRV filter + cls direction selector
                if pred_mae[i] < 1e-9 or pred_mfe[i] < 1e-9:
                    continue
                crv_long = pred_mfe[i] / pred_mae[i]
                crv_short = pred_mae[i] / pred_mfe[i]
                is_long = cls_long_conf[i] >= cls_short_conf[i]
                crv = crv_long if is_long else crv_short
                if crv < self.crv_threshold:
                    continue
                is_short = not is_long

            if not is_long and not is_short:
                continue

            # Compute PnL
            raw_mom = actual_mom[i]
            if is_long:
                pnl_scaled = raw_mom
                long_count += 1
            else:
                pnl_scaled = -raw_mom
                short_count += 1

            # Trading cost: spread + slippage (both in points)
            cost_points = self.spread + self.slippage * self.point

            if self.show_in_dollars and entry_prices is not None and i < len(entry_prices):
                ep = entry_prices[i]
                mom_unscaled = raw_mom / TARGET_SCALE_FACTOR
                if is_long:
                    pnl = (mom_unscaled * ep / self.point) * self.tick_value * self.lot_size
                else:
                    pnl = (-mom_unscaled * ep / self.point) * self.tick_value * self.lot_size
                trade_cost = (cost_points / self.point) * self.tick_value * self.lot_size
                pnl -= trade_cost
            else:
                pnl = pnl_scaled
                if entry_prices is not None and i < len(entry_prices):
                    ep = entry_prices[i]
                    spread_cost = (cost_points / (ep + 1e-9)) * TARGET_SCALE_FACTOR
                else:
                    spread_cost = 0.0
                pnl -= spread_cost

            equity.append(equity[-1] + pnl)

            if pnl > 0:
                wins += 1
                gross_profit += pnl
            else:
                losses += 1
                gross_loss += abs(pnl)

        metrics.equity_curve = np.array(equity)
        metrics.total_trades = wins + losses
        metrics.long_trades = long_count
        metrics.short_trades = short_count
        metrics.num_winners = wins
        metrics.num_losers = losses
        metrics.avg_winner = gross_profit / max(wins, 1)
        metrics.avg_loser = gross_loss / max(losses, 1)
        metrics.win_rate = wins / max(wins + losses, 1)
        metrics.profit_factor = gross_profit / max(gross_loss, 1e-9)
        metrics.total_pnl = equity[-1] if equity else 0.0

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
            "Backtest: %d trades (%d long, %d short), "
            "Winners=%d (avg=%.4f), Losers=%d (avg=%.4f), "
            "WR=%.1f%%, PF=%.2f, MaxDD=%.6f, PnL=%.4f",
            metrics.total_trades, metrics.long_trades, metrics.short_trades,
            wins, metrics.avg_winner, losses, metrics.avg_loser,
            metrics.win_rate * 100,
            metrics.profit_factor,
            metrics.max_drawdown,
            metrics.total_pnl,
        )
        return metrics
