"""
TrendcatcherStrategy — exit-condition-aware target labeling.

For each bar T and horizon H, compute:
  - eff_h = min(H, exit_distance[T])  where exit_distance is the number
    of bars until the fastest MA slope reverses sign.
  - MFE, MAE, Momentum over [T+1, T+1+eff_h)
  - exit_reached = 1.0 if exit_distance <= H else 0.0
"""

from __future__ import annotations

import numpy as np

from ..config import TARGET_HORIZONS, CLS_CRV_THRESHOLD, EPSILON, DataConfig


class TrendcatcherStrategy:
    """Trendcatcher target generator with exit-condition awareness."""

    def __init__(self, cfg: DataConfig, ma_spectrum: list[int]) -> None:
        self.cfg = cfg
        self.ma_spectrum = ma_spectrum

    def compute_exit_bars(
        self,
        closes: np.ndarray,
        ma_fastest_values: np.ndarray,
    ) -> np.ndarray:
        """
        For each bar T, compute the distance to the first bar where the
        slope of the fastest MA reverses sign relative to bar T's slope.

        Returns array of shape (n,) with exit distances. Bars where no
        reversal is found within 2*max(TARGET_HORIZONS) get that cap value.
        """
        n = len(closes)
        max_look = 2 * max(TARGET_HORIZONS)
        exit_dist = np.full(n, max_look, dtype=np.int32)

        # Compute slope of fastest MA: diff between consecutive values
        slopes = np.diff(ma_fastest_values, prepend=ma_fastest_values[0])

        for t in range(n - 1):
            ref_sign = 1.0 if slopes[t] >= 0 else -1.0
            limit = min(t + max_look, n)
            for j in range(t + 1, limit):
                current_sign = 1.0 if slopes[j] >= 0 else -1.0
                if current_sign != ref_sign:
                    exit_dist[t] = j - t
                    break

        return exit_dist

    def generate_targets(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        ma_fastest: np.ndarray,
        **kwargs,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Generate exit-aware targets for all horizons.

        Returns:
            targets: array of shape (n, len(TARGET_HORIZONS) * 4)
            names: list of target column names
        """
        n = len(closes)
        exit_dist = self.compute_exit_bars(closes, ma_fastest)

        target_names: list[str] = []
        for h in TARGET_HORIZONS:
            target_names.extend([
                f"tgt_mfe_{h}",
                f"tgt_mae_{h}",
                f"tgt_mom_{h}",
                f"tgt_exit_{h}",
            ])

        targets = np.full((n, len(target_names)), np.nan, dtype=np.float64)
        col_idx = 0

        for horizon in TARGET_HORIZONS:
            mfe = np.full(n, np.nan)
            mae = np.full(n, np.nan)
            mom = np.full(n, np.nan)
            exit_flag = np.full(n, np.nan)

            for i in range(n - 1):
                eff_h = min(horizon, exit_dist[i])
                if eff_h < 1:
                    eff_h = 1
                end_idx = i + 1 + eff_h
                if end_idx > n:
                    continue

                entry = closes[i]
                fs = slice(i + 1, end_idx)
                mfe[i] = (highs[fs].max() / entry) - 1.0
                mae[i] = (lows[fs].min() / entry) - 1.0
                # Momentum uses the bar at the end of effective horizon
                mom_bar = min(i + eff_h, n - 1)
                mom[i] = (closes[mom_bar] / entry) - 1.0
                exit_flag[i] = 1.0 if exit_dist[i] <= horizon else 0.0

            targets[:, col_idx] = mfe
            targets[:, col_idx + 1] = mae
            targets[:, col_idx + 2] = mom
            targets[:, col_idx + 3] = exit_flag
            col_idx += 4

        # Classification labels from 3-bar horizon
        mfe_3 = targets[:, 0]  # tgt_mfe_3
        mae_3 = targets[:, 1]  # tgt_mae_3
        mom_3 = targets[:, 2]  # tgt_mom_3

        abs_mae_3 = np.abs(mae_3)
        cls_long = np.zeros(n, dtype=np.float64)
        cls_short = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if np.isnan(mfe_3[i]) or np.isnan(mae_3[i]) or np.isnan(mom_3[i]):
                cls_long[i] = np.nan
                cls_short[i] = np.nan
                continue
            # Long: upside reward ratio good AND positive momentum
            if abs_mae_3[i] > EPSILON and mfe_3[i] / abs_mae_3[i] >= CLS_CRV_THRESHOLD and mom_3[i] > 0:
                cls_long[i] = 1.0
            # Short: downside reward ratio good AND negative momentum
            if mfe_3[i] > EPSILON and abs_mae_3[i] / mfe_3[i] >= CLS_CRV_THRESHOLD and mom_3[i] < 0:
                cls_short[i] = 1.0

        # Append cls columns
        cls_cols = np.column_stack([cls_long, cls_short])
        targets = np.concatenate([targets, cls_cols], axis=1)
        target_names.extend(["tgt_cls_long", "tgt_cls_short"])

        return targets, target_names
