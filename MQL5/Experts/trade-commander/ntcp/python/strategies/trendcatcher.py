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

from ..config import TARGET_HORIZONS, DataConfig


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

        return targets, target_names
