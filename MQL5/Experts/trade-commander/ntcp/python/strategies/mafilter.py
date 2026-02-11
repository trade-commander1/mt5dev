"""
MAFilterStrategy — MA Filter entry/exit detection + forward-simulation labeling.

Entry conditions (all must hold simultaneously):
  1. Close vs K shortest MAs (all above for long, all below for short)
  2. All K slopes share the same sign (positive for long, negative for short)
  3. Each slope magnitude >= fac_slope * |AverageSlope|
  4. Fslope >= fslope_thresh (long) or Fslope <= -fslope_thresh (short)
  5. BWFactor <= max_bw_factor

Exit options:
  A (exit_option=0): Close crosses fastest MA ± stddev_factor * StdDev
  B (exit_option=1): Fslope crosses zero
"""

from __future__ import annotations

import numpy as np

from ..config import TARGET_HORIZONS, EPSILON, DataConfig


class MAFilterStrategy:
    """MA Filter target generator with forward-simulated exit labeling."""

    def __init__(self, cfg: DataConfig, ma_spectrum: list[int]) -> None:
        self.cfg = cfg
        self.ma_spectrum = ma_spectrum
        self.K = cfg.signal_ma_count  # number of shortest MAs for entry
        self.nh = cfg.mafilter_nh
        self.fac_slope = cfg.mafilter_fac_slope
        self.fslope_thresh = cfg.mafilter_fslope_thresh
        self.max_bw_factor = cfg.mafilter_max_bw_factor
        self.stddev_factor = cfg.mafilter_stddev_factor
        self.exit_option = cfg.mafilter_exit_option

    # ------------------------------------------------------------------
    # Internal computations
    # ------------------------------------------------------------------

    def _compute_slopes(self, raw_ma_values: np.ndarray) -> np.ndarray:
        """
        Raw diffs of each MA: slope[i] = ma[i] - ma[i-1].
        Returns (n, num_mas) array; bar 0 = 0.
        """
        slopes = np.diff(raw_ma_values, axis=0, prepend=raw_ma_values[:1, :])
        slopes[0, :] = 0.0
        return slopes

    def _compute_slope_hist_avg(self, slopes: np.ndarray) -> np.ndarray:
        """
        Per-MA rolling average of |slope| over NH bars.
        Returns (n, num_mas) array of historical average absolute slopes.
        """
        n, num_mas = slopes.shape
        abs_slopes = np.abs(slopes)
        hist_avg = np.zeros_like(abs_slopes)
        cumsum = np.cumsum(abs_slopes, axis=0)
        for i in range(n):
            window_start = max(0, i - self.nh + 1)
            window_len = i - window_start + 1
            if window_start == 0:
                hist_avg[i] = cumsum[i] / window_len
            else:
                hist_avg[i] = (cumsum[i] - cumsum[window_start - 1]) / window_len
        return hist_avg

    def _compute_bandwidth(
        self, raw_ma_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        BW = max(K MAs) - min(K MAs) per bar.
        BWFactor = BW / SMA(BW, NH).

        Returns (bw, bw_factor), each shape (n,).
        """
        n = raw_ma_values.shape[0]
        k_mas = raw_ma_values[:, :self.K]
        bw = k_mas.max(axis=1) - k_mas.min(axis=1)

        # Cumsum-based running SMA of BW over NH window
        bw_sma = np.zeros(n, dtype=np.float64)
        cumsum = np.cumsum(bw)
        for i in range(n):
            window_start = max(0, i - self.nh + 1)
            window_len = i - window_start + 1
            if window_start == 0:
                bw_sma[i] = cumsum[i] / window_len
            else:
                bw_sma[i] = (cumsum[i] - cumsum[window_start - 1]) / window_len

        bw_factor = np.where(bw_sma > EPSILON, bw / bw_sma, 1.0)
        return bw, bw_factor

    def _compute_fslope(self, k_slopes: np.ndarray) -> np.ndarray:
        """
        Pairwise slope ordering metric for the K shortest MAs.
        Fslope = (concordant - discordant) / total_pairs.
        Range: [-1, 1]. +1 means all slopes in ascending period order,
        -1 means all in descending order.

        k_slopes: (n, K)
        Returns: (n,)
        """
        n, K = k_slopes.shape
        fslope = np.zeros(n, dtype=np.float64)
        total_pairs = K * (K - 1) // 2
        if total_pairs == 0:
            return fslope

        for i in range(n):
            concordant = 0
            for a in range(K):
                for b in range(a + 1, K):
                    # Concordant if shorter-period MA (index a) has larger slope
                    # than longer-period MA (index b) — proper trend ordering
                    if k_slopes[i, a] > k_slopes[i, b]:
                        concordant += 1
                    elif k_slopes[i, a] < k_slopes[i, b]:
                        concordant -= 1
            fslope[i] = concordant / total_pairs

        return fslope

    def _detect_entry_signals(
        self,
        closes: np.ndarray,
        raw_ma_values: np.ndarray,
        k_slopes: np.ndarray,
        k_slope_hist_avg: np.ndarray,
        fslope: np.ndarray,
        bw_factor: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect long/short entry signals.

        Returns (long_entry, short_entry) bool arrays of shape (n,).
        """
        n = len(closes)
        k_mas = raw_ma_values[:, :self.K]

        # Condition 1: Close vs K MAs
        close_above_all = np.all(closes[:, None] > k_mas, axis=1)
        close_below_all = np.all(closes[:, None] < k_mas, axis=1)

        # Condition 2: All K slopes same sign
        all_slopes_pos = np.all(k_slopes > 0, axis=1)
        all_slopes_neg = np.all(k_slopes < 0, axis=1)

        # Condition 3: Each slope vs its own NH-bar historical average
        # |slope[k]| >= fac_slope * SMA(|slope[k]|, NH)
        slope_threshold = self.fac_slope * k_slope_hist_avg
        slopes_strong = np.all(np.abs(k_slopes) >= slope_threshold, axis=1)

        # Condition 4: Fslope (LaminarLevel) threshold
        fslope_long = fslope >= self.fslope_thresh
        fslope_short = fslope <= -self.fslope_thresh

        # Condition 5: BW filter
        bw_ok = bw_factor <= self.max_bw_factor

        long_entry = close_above_all & all_slopes_pos & slopes_strong & fslope_long & bw_ok
        short_entry = close_below_all & all_slopes_neg & slopes_strong & fslope_short & bw_ok

        return long_entry, short_entry

    def _simulate_exit(
        self,
        bar: int,
        direction: int,
        closes: np.ndarray,
        raw_ma_values: np.ndarray,
        raw_stddev_values: np.ndarray,
        k_slopes: np.ndarray,
        fslope: np.ndarray,
        max_bars: int,
    ) -> tuple[int, float]:
        """
        Forward-simulate from entry bar to exit.

        direction: +1 for long, -1 for short.
        Returns (exit_bar, pnl_fraction).
        """
        n = len(closes)
        entry_price = closes[bar]
        slowest_idx = raw_ma_values.shape[1] - 1  # index of slowest MA

        limit = min(bar + max_bars, n)

        for j in range(bar + 1, limit):
            exited = False

            if self.exit_option == 0:
                # Exit A: Close breaches slowest MA ± StdDevFactor × StdDev
                slowest_ma = raw_ma_values[j, slowest_idx]
                stddev = raw_stddev_values[j, slowest_idx]
                if direction == 1:
                    # Long exit: close drops below slowest_ma - factor * stddev
                    if closes[j] < slowest_ma - self.stddev_factor * stddev:
                        exited = True
                else:
                    # Short exit: close rises above slowest_ma + factor * stddev
                    if closes[j] > slowest_ma + self.stddev_factor * stddev:
                        exited = True
            else:
                # Exit B: Fslope crosses zero
                if direction == 1 and fslope[j] <= 0:
                    exited = True
                elif direction == -1 and fslope[j] >= 0:
                    exited = True

            if exited:
                pnl = direction * (closes[j] / entry_price - 1.0)
                return j, pnl

        # No exit found within max_bars — exit at limit
        end = limit - 1
        pnl = direction * (closes[end] / entry_price - 1.0)
        return end, pnl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_targets(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        ma_fastest: np.ndarray,
        *,
        raw_ma_values: np.ndarray,
        raw_stddev_values: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Generate targets for MA Filter strategy.

        Phase 1: Universal regression targets (MFE/MAE/Mom x 6 horizons = 18 cols).
        Phase 2: Compute slopes, Fslope, bandwidth, detect entry signals.
        Phase 3: Forward-simulate exits → cls labels.

        Returns (targets [n, 20], target_names).
        """
        n = len(closes)
        num_mas = raw_ma_values.shape[1]
        K = min(self.K, num_mas)
        max_look = 2 * max(TARGET_HORIZONS)

        # ---- Phase 1: Universal regression targets (no exit truncation) ----
        reg_names: list[str] = []
        for h in TARGET_HORIZONS:
            reg_names.extend([f"tgt_mfe_{h}", f"tgt_mae_{h}", f"tgt_mom_{h}"])

        reg_targets = np.full((n, len(reg_names)), np.nan, dtype=np.float64)
        col = 0
        for horizon in TARGET_HORIZONS:
            for i in range(n - horizon):
                end_idx = i + 1 + horizon
                if end_idx > n:
                    continue
                entry = closes[i]
                fs = slice(i + 1, end_idx)
                reg_targets[i, col] = (highs[fs].max() / entry) - 1.0     # MFE
                reg_targets[i, col + 1] = (lows[fs].min() / entry) - 1.0  # MAE
                mom_bar = min(i + horizon, n - 1)
                reg_targets[i, col + 2] = (closes[mom_bar] / entry) - 1.0  # Mom
            col += 3

        # ---- Phase 2: Strategy signal detection ----
        slopes = self._compute_slopes(raw_ma_values)
        k_slopes = slopes[:, :K]
        slope_hist_avg = self._compute_slope_hist_avg(slopes)
        k_slope_hist_avg = slope_hist_avg[:, :K]
        fslope = self._compute_fslope(k_slopes)
        _, bw_factor = self._compute_bandwidth(raw_ma_values)
        long_entry, short_entry = self._detect_entry_signals(
            closes, raw_ma_values, k_slopes, k_slope_hist_avg, fslope, bw_factor,
        )

        # ---- Diagnostic: per-condition counts ----
        import logging
        _log = logging.getLogger(__name__)
        k_mas = raw_ma_values[:, :K]
        c1_long = np.sum(np.all(closes[:, None] > k_mas, axis=1))
        c1_short = np.sum(np.all(closes[:, None] < k_mas, axis=1))
        c2_pos = np.sum(np.all(k_slopes > 0, axis=1))
        c2_neg = np.sum(np.all(k_slopes < 0, axis=1))
        c3 = np.sum(np.all(np.abs(k_slopes) >= self.fac_slope * k_slope_hist_avg, axis=1))
        c4_long = np.sum(fslope >= self.fslope_thresh)
        c4_short = np.sum(fslope <= -self.fslope_thresh)
        c5 = np.sum(bw_factor <= self.max_bw_factor)
        _log.info(
            "MAFilter conditions (n=%d, K=%d): "
            "C1 close>MAs=%d close<MAs=%d | "
            "C2 allPos=%d allNeg=%d | "
            "C3 slopesStrong=%d | "
            "C4 laminar+=%d laminar-=%d | "
            "C5 bwOk=%d | "
            "LONG=%d SHORT=%d",
            n, K, c1_long, c1_short, c2_pos, c2_neg, c3,
            c4_long, c4_short, c5,
            int(long_entry.sum()), int(short_entry.sum()),
        )

        # ---- Phase 3: Forward-simulate exits for each signal bar ----
        cls_long = np.full(n, np.nan, dtype=np.float64)
        cls_short = np.full(n, np.nan, dtype=np.float64)

        # Only label bars where regression targets are valid
        has_reg = ~np.isnan(reg_targets[:, 0])

        pos_long = 0
        pos_short = 0

        for i in range(n):
            if not has_reg[i]:
                continue

            # Default: 0 (no signal or losing trade)
            cls_long[i] = 0.0
            cls_short[i] = 0.0

            if long_entry[i]:
                _, pnl = self._simulate_exit(
                    i, +1, closes, raw_ma_values, raw_stddev_values,
                    k_slopes, fslope, max_look,
                )
                if pnl > 0:
                    cls_long[i] = 1.0
                    pos_long += 1

            if short_entry[i]:
                _, pnl = self._simulate_exit(
                    i, -1, closes, raw_ma_values, raw_stddev_values,
                    k_slopes, fslope, max_look,
                )
                if pnl > 0:
                    cls_short[i] = 1.0
                    pos_short += 1

        import logging
        logger = logging.getLogger(__name__)
        n_long_signals = int(long_entry.sum())
        n_short_signals = int(short_entry.sum())
        logger.info(
            "MAFilter signals: %d long (%d profitable), %d short (%d profitable)",
            n_long_signals, pos_long, n_short_signals, pos_short,
        )

        # ---- Assemble final target array ----
        target_names = reg_names + ["tgt_cls_long", "tgt_cls_short"]
        targets = np.column_stack([reg_targets, cls_long, cls_short])

        return targets, target_names
