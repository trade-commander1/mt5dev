"""
NTCP DataManager — single M5 CSV + dynamic STF aggregation pipeline.

Replaces the dual-CSV (M5+H1) approach with a configurable STF_FACTOR
that aggregates M5 bars into a super-timeframe on-the-fly, including
a developing (partial) bar that mirrors the MQL5 tcMA snapshot/rollback
pattern from moving_average.mqh.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import (
    MA_SPECTRUM,
    TARGET_HORIZONS,
    TARGET_SCALE_FACTOR,
    TICK_SMA_PERIOD,
    EPSILON,
    RSI_PERIOD,
    ATR_PERIOD,
    DataConfig,
    generate_ma_spectrum,
)
from .strategies import get_strategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature group mapping — dynamic factory
# ---------------------------------------------------------------------------

def build_feature_group_map(
    ma_spectrum: list[int],
    n_clusters: int,
) -> dict[str, list[str]]:
    """Build the feature-group-to-column-name mapping for a given MA spectrum."""
    return {
        "m5_ma": [f"m5_ma_{p}" for p in ma_spectrum],
        "m5_sd": [f"m5_sd_{p}" for p in ma_spectrum],
        "m5_slope": [f"m5_slope_{p}" for p in ma_spectrum],
        "m5_misc": ["m5_rsi", "m5_atr", "m5_vol_zscore"],
        "m5_candle": ["m5_body", "m5_wick_upper", "m5_wick_lower"],
        "m5_range": ["m5_squeeze", "m5_dist_day_high", "m5_dist_day_low"],
        "m5_time": ["m5_hour_sin", "m5_hour_cos", "m5_dow_sin", "m5_dow_cos"],
        "m5_hurst": ["m5_hurst"],
        "m5_regime": [f"m5_regime_{k}" for k in range(n_clusters)],
        "stf_ma": [f"stf_ma_{p}" for p in ma_spectrum],
        "stf_sd": [f"stf_sd_{p}" for p in ma_spectrum],
        "stf_slope": [f"stf_slope_{p}" for p in ma_spectrum],
        "stf_misc": ["stf_rsi", "stf_atr", "stf_vol_zscore", "stf_close", "stf_high", "stf_low"],
        "stf_squeeze": ["stf_squeeze"],
        "stf_hurst": ["stf_hurst"],
    }


# Backward-compatible module-level mapping using default spectrum
FEATURE_GROUP_MAP: dict[str, list[str]] = build_feature_group_map(MA_SPECTRUM, 4)

ALL_FEATURE_GROUPS = list(FEATURE_GROUP_MAP.keys())


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class NTCPDataset(Dataset):
    """Sliding-window dataset returning (feature_sequence, reg_target, cls_target) triples."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lookback: int,
        cls_targets: np.ndarray | None = None,
    ) -> None:
        assert features.shape[0] == targets.shape[0], "Feature/target length mismatch"
        assert lookback < features.shape[0], "Lookback exceeds available data"

        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.lookback = lookback
        self._length = features.shape[0] - lookback

        if cls_targets is not None:
            assert cls_targets.shape[0] == features.shape[0], "Cls target length mismatch"
            self.cls_targets = torch.as_tensor(cls_targets, dtype=torch.float32)
        else:
            self.cls_targets = torch.zeros(features.shape[0], 2, dtype=torch.float32)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = self.features[idx : idx + self.lookback]
        tgt = self.targets[idx + self.lookback - 1]
        cls_tgt = self.cls_targets[idx + self.lookback - 1]
        return seq, tgt, cls_tgt


# ---------------------------------------------------------------------------
# Online Indicator — Python port of tcMA snapshot/rollback (moving_average.mqh)
# ---------------------------------------------------------------------------

class OnlineIndicator:
    """
    Single windowed-SMA + Welford stddev indicator with snapshot/rollback.
    Mirrors the tcMA class from moving_average.mqh lines 96-216.
    """

    __slots__ = (
        "length", "count", "window_sum", "circ_buf", "circ_idx",
        "ma", "m2", "running_sum", "std_dev",
        # snapshot state
        "_snap_count", "_snap_window_sum", "_snap_circ_idx",
        "_snap_ma", "_snap_m2", "_snap_running_sum", "_snap_std_dev",
        "_snap_old_buf_val",
    )

    def __init__(self, length: int) -> None:
        self.length = length
        self.count = 0
        self.window_sum = 0.0
        self.circ_buf = np.zeros(length, dtype=np.float64)
        self.circ_idx = 0
        self.ma = 0.0
        self.m2 = 0.0
        self.running_sum = 0.0
        self.std_dev = 0.0
        # snapshot
        self._snap_count = 0
        self._snap_window_sum = 0.0
        self._snap_circ_idx = 0
        self._snap_ma = 0.0
        self._snap_m2 = 0.0
        self._snap_running_sum = 0.0
        self._snap_std_dev = 0.0
        self._snap_old_buf_val = 0.0

    def update(self, value: float, is_new_bar: bool = True) -> float:
        """
        Feed a new value. If is_new_bar=True, save snapshot first.
        If is_new_bar=False, rollback to snapshot then re-apply.
        Returns current MA value.
        """
        if is_new_bar:
            self._snap_count = self.count
            self._snap_window_sum = self.window_sum
            self._snap_circ_idx = self.circ_idx
            self._snap_ma = self.ma
            self._snap_m2 = self.m2
            self._snap_running_sum = self.running_sum
            self._snap_std_dev = self.std_dev
            self._snap_old_buf_val = self.circ_buf[self.circ_idx]
        else:
            self.count = self._snap_count
            self.window_sum = self._snap_window_sum
            self.circ_idx = self._snap_circ_idx
            self.ma = self._snap_ma
            self.m2 = self._snap_m2
            self.running_sum = self._snap_running_sum
            self.std_dev = self._snap_std_dev
            self.circ_buf[self.circ_idx] = self._snap_old_buf_val

        if self.count == 0:
            self.running_sum = value
            self.ma = value
            self.m2 = 0.0
            self.std_dev = 0.0
            self.circ_buf[0] = value
            self.window_sum = value
            self.circ_idx = 1 % self.length
        else:
            # Welford's online stddev
            old_mean = self.running_sum / self.count
            self.running_sum += value
            n = self.count + 1
            new_mean = self.running_sum / n
            delta = value - old_mean
            delta2 = value - new_mean
            self.m2 += delta * delta2
            self.std_dev = math.sqrt(self.m2 / n)

            # Windowed SMA via circular buffer
            if self.count >= self.length:
                self.window_sum -= self.circ_buf[self.circ_idx]
            self.window_sum += value
            self.circ_buf[self.circ_idx] = value
            self.circ_idx = (self.circ_idx + 1) % self.length
            self.ma = self.window_sum / min(self.count + 1, self.length)

        self.count += 1
        return self.ma

    def reset(self) -> None:
        self.count = 0
        self.window_sum = 0.0
        self.circ_buf[:] = 0.0
        self.circ_idx = 0
        self.ma = 0.0
        self.m2 = 0.0
        self.running_sum = 0.0
        self.std_dev = 0.0


class OnlineIndicatorBank:
    """
    Bank of OnlineIndicator instances for a configurable MA spectrum.
    Also tracks RSI (Wilder-style), ATR online.
    Supports snapshot/rollback for developing-bar simulation.
    """

    def __init__(self, ma_spectrum: list[int] | None = None) -> None:
        if ma_spectrum is None:
            ma_spectrum = MA_SPECTRUM
        self._ma_spectrum = ma_spectrum
        # One indicator per MA period
        self.ma_indicators: dict[int, OnlineIndicator] = {
            p: OnlineIndicator(p) for p in ma_spectrum
        }
        # RSI state (Wilder's smoothing)
        self._rsi_period = RSI_PERIOD
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._rsi_count = 0
        self._rsi_prev_value = 0.0
        self._rsi = 50.0
        # ATR state
        self._atr_period = ATR_PERIOD
        self._atr = 0.0
        self._atr_count = 0
        self._atr_prev_close = 0.0
        # Volume
        self._vol_indicator = OnlineIndicator(TICK_SMA_PERIOD)
        # Snapshots for RSI/ATR
        self._snap_rsi_state: tuple = ()
        self._snap_atr_state: tuple = ()

    def _snapshot_extras(self) -> None:
        self._snap_rsi_state = (
            self._avg_gain, self._avg_loss, self._rsi_count,
            self._rsi_prev_value, self._rsi,
        )
        self._snap_atr_state = (
            self._atr, self._atr_count, self._atr_prev_close,
        )

    def _rollback_extras(self) -> None:
        if self._snap_rsi_state:
            (self._avg_gain, self._avg_loss, self._rsi_count,
             self._rsi_prev_value, self._rsi) = self._snap_rsi_state
        if self._snap_atr_state:
            (self._atr, self._atr_count, self._atr_prev_close) = self._snap_atr_state

    def update(
        self,
        close: float,
        high: float,
        low: float,
        tick_volume: float,
        is_new_bar: bool = True,
    ) -> None:
        """Update all indicators with a new bar's data."""
        if is_new_bar:
            self._snapshot_extras()
        else:
            self._rollback_extras()

        # MA spectrum
        for ind in self.ma_indicators.values():
            ind.update(close, is_new_bar)

        # Volume
        self._vol_indicator.update(tick_volume, is_new_bar)

        # RSI (Wilder's smoothing)
        if self._rsi_count == 0:
            self._rsi_prev_value = close
            self._rsi_count = 1
        else:
            change = close - self._rsi_prev_value
            gain = max(change, 0.0)
            loss = max(-change, 0.0)
            alpha = 1.0 / self._rsi_period
            if self._rsi_count < self._rsi_period:
                self._avg_gain += gain
                self._avg_loss += loss
                if self._rsi_count == self._rsi_period - 1:
                    self._avg_gain /= self._rsi_period
                    self._avg_loss /= self._rsi_period
            else:
                self._avg_gain = alpha * gain + (1 - alpha) * self._avg_gain
                self._avg_loss = alpha * loss + (1 - alpha) * self._avg_loss

            rs = self._avg_gain / (self._avg_loss + EPSILON)
            self._rsi = 100.0 - (100.0 / (1.0 + rs))
            self._rsi_prev_value = close
            self._rsi_count += 1

        # ATR
        if self._atr_count == 0:
            self._atr_prev_close = close
            self._atr_count = 1
        else:
            tr = max(
                high - low,
                abs(high - self._atr_prev_close),
                abs(low - self._atr_prev_close),
            )
            if self._atr_count < self._atr_period:
                self._atr += tr
                if self._atr_count == self._atr_period - 1:
                    self._atr /= self._atr_period
            else:
                self._atr = (self._atr * (self._atr_period - 1) + tr) / self._atr_period
            self._atr_prev_close = close
            self._atr_count += 1

    def get_ma_features(self, ref_close: float) -> dict[str, float]:
        """Return normalized MA and SD features relative to ref_close."""
        feats: dict[str, float] = {}
        for p, ind in self.ma_indicators.items():
            feats[f"ma_{p}"] = (ind.ma / ref_close) - 1.0 if ref_close != 0 else 0.0
            feats[f"sd_{p}"] = ind.std_dev / ref_close if ref_close != 0 else 0.0
        return feats

    def get_rsi(self) -> float:
        return self._rsi / 100.0

    def get_atr(self, ref_close: float) -> float:
        return self._atr / ref_close if ref_close != 0 else 0.0

    def get_volume_zscore(self) -> float:
        """(last_vol - vol_SMA) / vol_stddev — replaces get_rel_volume()."""
        sma = self._vol_indicator.ma
        std = self._vol_indicator.std_dev
        if std < EPSILON:
            return 0.0
        last_vol = self._vol_indicator.circ_buf[
            (self._vol_indicator.circ_idx - 1) % self._vol_indicator.length
        ]
        return (last_vol - sma) / (std + EPSILON)

    def get_slopes(self, prev_ma_values: dict[int, float], ref_close: float) -> dict[str, float]:
        """Per-MA slope = (current_MA - prev_MA) / close."""
        slopes: dict[str, float] = {}
        for p, ind in self.ma_indicators.items():
            prev = prev_ma_values.get(p, ind.ma)
            slopes[f"slope_{p}"] = (ind.ma - prev) / (ref_close + EPSILON) if ref_close != 0 else 0.0
        return slopes

    def get_current_ma_values(self) -> dict[int, float]:
        """Return raw MA values for slope computation on next bar."""
        return {p: ind.ma for p, ind in self.ma_indicators.items()}

    def reset(self) -> None:
        for ind in self.ma_indicators.values():
            ind.reset()
        self._vol_indicator.reset()
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._rsi_count = 0
        self._rsi_prev_value = 0.0
        self._rsi = 50.0
        self._atr = 0.0
        self._atr_count = 0
        self._atr_prev_close = 0.0


# ---------------------------------------------------------------------------
# Standalone functions: Hurst & Market Regime
# ---------------------------------------------------------------------------

def compute_hurst_exponent(series: np.ndarray, window: int) -> np.ndarray:
    """
    Compute rolling Hurst exponent via R/S analysis.
    Returns array of same length as series; values are NaN where
    insufficient data exists.
    """
    n = len(series)
    hurst = np.full(n, np.nan, dtype=np.float64)
    if window < 10:
        return hurst

    for i in range(window, n):
        seg = series[i - window : i]
        mean_seg = np.mean(seg)
        deviations = seg - mean_seg
        cumdev = np.cumsum(deviations)
        r = np.max(cumdev) - np.min(cumdev)
        s = np.std(seg, ddof=1)
        if s < EPSILON:
            hurst[i] = 0.5
        else:
            rs = r / s
            if rs > 0:
                hurst[i] = math.log(rs) / math.log(window)
            else:
                hurst[i] = 0.5
    return hurst


def compute_market_regimes(
    feature_subset: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    GMM clustering on a subset of features.
    Returns (one_hot [n_bars, n_clusters], centroids [n_clusters, n_features]).
    """
    from sklearn.mixture import GaussianMixture

    n = feature_subset.shape[0]
    one_hot = np.zeros((n, n_clusters), dtype=np.float64)

    # Only fit on valid (non-NaN) rows
    valid_mask = ~np.isnan(feature_subset).any(axis=1)
    valid_data = feature_subset[valid_mask]

    if len(valid_data) < n_clusters * 10:
        # Not enough data, return zeros + dummy centroids
        centroids = np.zeros((n_clusters, feature_subset.shape[1]), dtype=np.float64)
        return one_hot, centroids

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="diag",
        max_iter=100,
        random_state=42,
        n_init=3,
    )
    gmm.fit(valid_data)

    labels = gmm.predict(valid_data)
    # Fill one-hot for valid rows
    valid_indices = np.where(valid_mask)[0]
    for j, idx in enumerate(valid_indices):
        one_hot[idx, labels[j]] = 1.0

    return one_hot, gmm.means_


# ---------------------------------------------------------------------------
# Main DataManager
# ---------------------------------------------------------------------------

class NTCPDataManager:
    """
    Pipeline: M5 CSV -> STF aggregation -> online indicators -> features
    & targets -> PyTorch Dataset.
    """

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.m5_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.target_matrix: Optional[np.ndarray] = None
        self.feature_cols: list[str] = []
        self.target_cols: list[str] = []
        self.cls_target_cols: list[str] = []
        self.scaling_params: dict = {}
        self.ma_spectrum: list[int] = []
        self._m5_fastest_ma: Optional[np.ndarray] = None
        self._m5_raw_ma_values: Optional[np.ndarray] = None
        self._m5_raw_stddev_values: Optional[np.ndarray] = None
        self.raw_closes: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # CSV Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        """Load an MT5 CSV export (comma or tab-separated) and parse timestamps."""
        # Auto-detect separator
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","

        df = pd.read_csv(path, sep=sep)
        # Normalize column names: strip angle brackets, whitespace, lowercase
        df.columns = [
            c.replace("<", "").replace(">", "").strip().lower()
            for c in df.columns
        ]

        # Parse datetime — handle single 'datetime' or separate 'date'+'time'
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
        elif "date" in df.columns and "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])
            df = df.drop(columns=["date", "time"])
        elif "date" in df.columns:
            df["datetime"] = pd.to_datetime(df["date"])
            df = df.drop(columns=["date"])

        df = df.rename(columns={
            "tickvol": "tick_volume",
            "tickvolume": "tick_volume",
            "vol": "volume",
            "spread": "spread",
        })

        # Derive OHLC from Bid/Ask when OHLC columns are absent
        if "open" not in df.columns and "bid" in df.columns:
            ask = df["ask"] if "ask" in df.columns else df["bid"]
            mid = (df["bid"] + ask) / 2.0
            df["open"] = mid
            df["high"] = df[["bid", "ask"]].max(axis=1) if "ask" in df.columns else mid
            df["low"] = df[["bid", "ask"]].min(axis=1) if "ask" in df.columns else mid
            df["close"] = mid
            df = df.drop(columns=[c for c in ["bid", "ask"] if c in df.columns])

        # Ensure tick_volume exists
        if "tick_volume" not in df.columns and "volume" in df.columns:
            df = df.rename(columns={"volume": "tick_volume"})
        if "tick_volume" not in df.columns:
            df["tick_volume"] = 0.0

        df = df.sort_values("datetime").reset_index(drop=True)
        return df

    def load(self) -> pd.DataFrame:
        """Load the M5 CSV."""
        self.m5_df = self._load_csv(Path(self.cfg.m5_path))
        logger.info("Loaded %d M5 bars from %s", len(self.m5_df), self.cfg.m5_path)
        return self.m5_df

    # ------------------------------------------------------------------
    # STF Developing Bar Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_ohlcv(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        tick_vols: np.ndarray,
        start: int,
        end: int,
    ) -> tuple[float, float, float, float, float]:
        """Aggregate M5 bars [start..end) into a single OHLCV bar."""
        o = opens[start]
        h = highs[start:end].max()
        l = lows[start:end].min()
        c = closes[end - 1]
        v = tick_vols[start:end].sum()
        return o, h, l, c, float(v)

    def build_features(
        self,
        log_callback=None,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Build M5 + STF features using OnlineIndicatorBank with
        snapshot/rollback for the developing STF bar.

        Returns (feature_matrix [T, F], feature_col_names).
        """
        if self.m5_df is None:
            raise RuntimeError("Call load() first.")

        df = self.m5_df
        n = len(df)
        stf_factor = self.cfg.stf_factor

        # Derive dynamic MA spectrum
        ma_spectrum = self.cfg.get_ma_spectrum()
        self.ma_spectrum = ma_spectrum
        self.scaling_params["ma_spectrum"] = ma_spectrum
        self.scaling_params["signal_ma_count"] = self.cfg.signal_ma_count
        fastest_period = ma_spectrum[0]

        opens = df["open"].values.astype(np.float64)
        highs = df["high"].values.astype(np.float64)
        lows = df["low"].values.astype(np.float64)
        closes = df["close"].values.astype(np.float64)
        tick_vols = df["tick_volume"].values.astype(np.float64)
        datetimes = df["datetime"].values

        # Online indicator banks
        m5_bank = OnlineIndicatorBank(ma_spectrum=ma_spectrum)
        stf_bank = OnlineIndicatorBank(ma_spectrum=ma_spectrum)

        # Feature names — bar-loop features (before hurst/regime post-loop)
        m5_loop_names = (
            [f"m5_ma_{p}" for p in ma_spectrum]
            + [f"m5_sd_{p}" for p in ma_spectrum]
            + [f"m5_slope_{p}" for p in ma_spectrum]
            + ["m5_rsi", "m5_atr", "m5_vol_zscore"]
            + ["m5_body", "m5_wick_upper", "m5_wick_lower"]
            + ["m5_squeeze"]
            + ["m5_dist_day_high", "m5_dist_day_low"]
            + ["m5_hour_sin", "m5_hour_cos", "m5_dow_sin", "m5_dow_cos"]
        )

        stf_loop_names = (
            [f"stf_ma_{p}" for p in ma_spectrum]
            + [f"stf_sd_{p}" for p in ma_spectrum]
            + [f"stf_slope_{p}" for p in ma_spectrum]
            + ["stf_rsi", "stf_atr", "stf_vol_zscore"]
            + ["stf_squeeze"]
            + ["stf_close", "stf_high", "stf_low"]
        )

        loop_feat_names = m5_loop_names + stf_loop_names
        num_loop_features = len(loop_feat_names)

        # Post-loop features: hurst + regime
        n_clusters = self.cfg.regime_n_clusters
        post_feat_names = (
            ["m5_hurst"]
            + [f"m5_regime_{k}" for k in range(n_clusters)]
            + ["stf_hurst"]
        )
        all_feat_names = loop_feat_names + post_feat_names
        num_total_features = len(all_feat_names)

        features = np.full((n, num_total_features), np.nan, dtype=np.float64)

        # Array to collect fastest MA values for strategy exit computation
        m5_fastest_ma = np.zeros(n, dtype=np.float64)

        # Raw MA and StdDev arrays for strategy use (n, num_mas)
        num_mas = len(ma_spectrum)
        m5_raw_ma_values = np.zeros((n, num_mas), dtype=np.float64)
        m5_raw_stddev_values = np.zeros((n, num_mas), dtype=np.float64)

        # Rolling daily high/low tracking
        day_bars = 1440 // self.cfg.base_tf_minutes
        day_highs = np.full(day_bars, np.nan)
        day_lows = np.full(day_bars, np.nan)
        day_idx = 0

        # Track previous MA values for slope computation (empty → first slope = 0)
        prev_m5_ma_values: dict[int, float] = {}
        prev_stf_ma_values: dict[int, float] = {}

        log_step = max(1, n // 20)

        for i in range(n):
            c = closes[i]
            h = highs[i]
            l = lows[i]
            o = opens[i]
            tv = tick_vols[i]

            # --- M5 bank update (always is_new_bar=True for M5) ---
            m5_bank.update(c, h, l, tv, is_new_bar=True)

            # Collect fastest MA value
            m5_fastest_ma[i] = m5_bank.ma_indicators[fastest_period].ma

            # Collect raw MA and StdDev values for all periods
            for k_idx, p in enumerate(ma_spectrum):
                m5_raw_ma_values[i, k_idx] = m5_bank.ma_indicators[p].ma
                m5_raw_stddev_values[i, k_idx] = m5_bank.ma_indicators[p].std_dev

            # --- Build developing STF bar ---
            group_idx = i // stf_factor
            pos_in_group = i % stf_factor
            group_start = group_idx * stf_factor

            stf_o, stf_h, stf_l, stf_c, stf_v = self._aggregate_ohlcv(
                opens, highs, lows, closes, tick_vols,
                group_start, i + 1,
            )

            # STF bank: snapshot on first bar of group, rollback on subsequent
            stf_is_new = (pos_in_group == 0)
            stf_bank.update(stf_c, stf_h, stf_l, stf_v, is_new_bar=stf_is_new)

            # --- Collect M5 features ---
            row = []
            m5_ma_feats = m5_bank.get_ma_features(c)
            for p in ma_spectrum:
                row.append(m5_ma_feats[f"ma_{p}"])
            for p in ma_spectrum:
                row.append(m5_ma_feats[f"sd_{p}"])

            # M5 slopes
            m5_slopes = m5_bank.get_slopes(prev_m5_ma_values, c)
            for p in ma_spectrum:
                row.append(m5_slopes[f"slope_{p}"])
            prev_m5_ma_values = m5_bank.get_current_ma_values()

            row.append(m5_bank.get_rsi())
            row.append(m5_bank.get_atr(c))
            row.append(m5_bank.get_volume_zscore())

            # Candle geometry
            hl_range = h - l + EPSILON
            row.append((c - o) / hl_range)            # body
            row.append((h - max(o, c)) / hl_range)    # wick_upper
            row.append((min(o, c) - l) / hl_range)    # wick_lower

            # Squeeze: (MA_fastest - MA_slowest) / close
            ma_fast_val = m5_bank.ma_indicators[ma_spectrum[0]].ma
            ma_slow_val = m5_bank.ma_indicators[ma_spectrum[-1]].ma
            row.append((ma_fast_val - ma_slow_val) / (c + EPSILON))

            # Daily high/low distance
            day_highs[day_idx % day_bars] = h
            day_lows[day_idx % day_bars] = l
            day_idx += 1
            valid_count = min(day_idx, day_bars)
            rolling_high = np.nanmax(day_highs[:valid_count])
            rolling_low = np.nanmin(day_lows[:valid_count])
            row.append((rolling_high / c) - 1.0)
            row.append((rolling_low / c) - 1.0)

            # Cyclic time
            dt = pd.Timestamp(datetimes[i])
            hour = dt.hour + dt.minute / 60.0
            dow = dt.dayofweek
            row.append(math.sin(2.0 * math.pi * hour / 24.0))
            row.append(math.cos(2.0 * math.pi * hour / 24.0))
            row.append(math.sin(2.0 * math.pi * dow / 7.0))
            row.append(math.cos(2.0 * math.pi * dow / 7.0))

            # --- Collect STF features ---
            stf_ma_feats = stf_bank.get_ma_features(c)
            for p in ma_spectrum:
                row.append(stf_ma_feats[f"ma_{p}"])
            for p in ma_spectrum:
                row.append(stf_ma_feats[f"sd_{p}"])

            # STF slopes
            stf_slopes = stf_bank.get_slopes(prev_stf_ma_values, c)
            for p in ma_spectrum:
                row.append(stf_slopes[f"slope_{p}"])
            if stf_is_new:
                prev_stf_ma_values = stf_bank.get_current_ma_values()

            row.append(stf_bank.get_rsi())
            row.append(stf_bank.get_atr(c))
            row.append(stf_bank.get_volume_zscore())

            # STF squeeze
            stf_fast_val = stf_bank.ma_indicators[ma_spectrum[0]].ma
            stf_slow_val = stf_bank.ma_indicators[ma_spectrum[-1]].ma
            row.append((stf_fast_val - stf_slow_val) / (c + EPSILON))

            # STF OHLC relative to M5 close
            row.append((stf_c / c) - 1.0 if c != 0 else 0.0)
            row.append((stf_h / c) - 1.0 if c != 0 else 0.0)
            row.append((stf_l / c) - 1.0 if c != 0 else 0.0)

            features[i, :num_loop_features] = row

            if log_callback and i % log_step == 0:
                pct = (i + 1) / n * 100
                log_callback(f"Building features: {pct:.0f}% ({i+1}/{n})")

        # Store fastest MA and raw arrays for strategy use
        self._m5_fastest_ma = m5_fastest_ma
        self._m5_raw_ma_values = m5_raw_ma_values
        self._m5_raw_stddev_values = m5_raw_stddev_values

        # Determine which post-loop groups are active (empty = all active)
        active_groups = self.cfg.active_feature_groups
        _grp_active = lambda g: not active_groups or g in active_groups

        # Post-loop column layout: [m5_hurst, m5_regime_0..K, stf_hurst]
        col_offset = num_loop_features

        # --- m5_hurst ---
        if _grp_active("m5_hurst"):
            if log_callback:
                log_callback("Computing M5 Hurst exponent...")
            features[:, col_offset] = compute_hurst_exponent(
                closes, self.cfg.hurst_window,
            )
        else:
            features[:, col_offset] = 0.0
        col_offset += 1

        # --- m5_regime ---
        if _grp_active("m5_regime"):
            if log_callback:
                log_callback("Computing market regimes (GMM)...")
            ma_col_indices = []
            for j, name in enumerate(loop_feat_names):
                if name.startswith("m5_ma_") or name.startswith("m5_sd_"):
                    ma_col_indices.append(j)
            regime_input = features[:, ma_col_indices]
            regime_one_hot, regime_centroids = compute_market_regimes(
                regime_input, n_clusters,
            )
            features[:, col_offset:col_offset + n_clusters] = regime_one_hot
            self.scaling_params["regime_centroids"] = regime_centroids.tolist()
        else:
            if log_callback:
                log_callback("Skipping market regimes (group unchecked).")
            features[:, col_offset:col_offset + n_clusters] = 0.0
            self.scaling_params["regime_centroids"] = np.zeros(
                (n_clusters, 2 * len(ma_spectrum)), dtype=np.float64,
            ).tolist()
        col_offset += n_clusters

        # --- stf_hurst ---
        if _grp_active("stf_hurst"):
            if log_callback:
                log_callback("Computing STF Hurst exponent...")
            stf_close_series = closes.copy()
            features[:, col_offset] = compute_hurst_exponent(
                stf_close_series, self.cfg.hurst_window,
            )
        else:
            features[:, col_offset] = 0.0

        # --- Feature group filtering ---
        feature_group_map = build_feature_group_map(ma_spectrum, n_clusters)
        active_groups = self.cfg.active_feature_groups
        if active_groups:
            # Build set of allowed feature names
            allowed_names: set[str] = set()
            for grp in active_groups:
                if grp in feature_group_map:
                    allowed_names.update(feature_group_map[grp])

            # Filter columns
            keep_indices = [j for j, name in enumerate(all_feat_names) if name in allowed_names]
            if keep_indices:
                features = features[:, keep_indices]
                all_feat_names = [all_feat_names[j] for j in keep_indices]
            else:
                logger.warning("No features matched active groups; keeping all.")

        # --- Per-feature Mean/StdDev storage ---
        feat_means: dict[str, float] = {}
        feat_stds: dict[str, float] = {}
        for j, name in enumerate(all_feat_names):
            col = features[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                feat_means[name] = float(np.mean(valid))
                feat_stds[name] = float(np.std(valid))
            else:
                feat_means[name] = 0.0
                feat_stds[name] = 1.0
        self.scaling_params["feature_means"] = feat_means
        self.scaling_params["feature_stds"] = feat_stds

        self.feature_cols = all_feat_names
        self.feature_matrix = features
        num_features = len(all_feat_names)
        logger.info("Built %d features for %d bars (STF factor=%d)",
                     num_features, n, stf_factor)
        return features, all_feat_names

    # ------------------------------------------------------------------
    # Target Generation
    # ------------------------------------------------------------------

    def generate_targets(self) -> tuple[np.ndarray, list[str]]:
        """
        Compute targets by delegating to the configured strategy.
        Clips at configured percentile (skipping binary exit columns).
        """
        if self.m5_df is None:
            raise RuntimeError("Call load() first.")
        if self._m5_fastest_ma is None:
            raise RuntimeError("Call build_features() first.")

        df = self.m5_df
        close = df["close"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)

        # Delegate to strategy
        strategy = get_strategy(self.cfg.strategy, self.cfg, self.ma_spectrum)
        targets, target_names = strategy.generate_targets(
            close, high, low, self._m5_fastest_ma,
            raw_ma_values=self._m5_raw_ma_values,
            raw_stddev_values=self._m5_raw_stddev_values,
        )

        # Percentile clipping (skip binary tgt_exit_* columns)
        lo_pct = self.cfg.clip_percentile
        hi_pct = 100.0 - self.cfg.clip_percentile
        clip_bounds: dict[str, dict[str, float]] = {}

        for j, name in enumerate(target_names):
            if name.startswith("tgt_exit_") or name.startswith("tgt_cls_"):
                continue
            col = targets[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            lb = float(np.percentile(valid, lo_pct))
            ub = float(np.percentile(valid, hi_pct))
            targets[:, j] = np.clip(col, lb, ub)
            clip_bounds[name] = {"lower": lb, "upper": ub}

        self.scaling_params["target_clip_bounds"] = clip_bounds
        self.target_cols = target_names
        self.target_matrix = targets
        logger.info("Generated %d target columns for %d bars.", len(target_names), len(df))
        return targets, target_names

    # ------------------------------------------------------------------
    # Dataset Creation
    # ------------------------------------------------------------------

    def create_dataset(self) -> NTCPDataset:
        """Build the final NTCPDataset from computed features and targets."""
        if self.feature_matrix is None or self.target_matrix is None:
            raise RuntimeError("Call build_features() and generate_targets() first.")

        # Find valid rows (no NaN in either features or targets)
        feat_valid = ~np.isnan(self.feature_matrix).any(axis=1)
        tgt_valid = ~np.isnan(self.target_matrix).any(axis=1)
        valid = feat_valid & tgt_valid

        # Find first and last valid index for contiguous slice
        valid_indices = np.where(valid)[0]
        if len(valid_indices) == 0:
            raise ValueError("No valid rows after NaN filtering.")

        start = valid_indices[0]
        end = valid_indices[-1] + 1

        features = self.feature_matrix[start:end].astype(np.float32)
        targets = self.target_matrix[start:end].astype(np.float32)

        # Store raw M5 close prices for dollar conversion in backtest
        if self.m5_df is not None:
            self.raw_closes = self.m5_df["close"].values[start:end].astype(np.float64)

        # Strip binary exit-flag columns — model does regression only
        keep = [i for i, name in enumerate(self.target_cols)
                if not name.startswith("tgt_exit_")]
        targets = targets[:, keep]
        self.target_cols = [self.target_cols[i] for i in keep]

        # Separate classification targets (binary, not scaled)
        cls_indices = [i for i, name in enumerate(self.target_cols)
                       if name.startswith("tgt_cls_")]
        reg_indices = [i for i, name in enumerate(self.target_cols)
                       if not name.startswith("tgt_cls_")]

        cls_targets = targets[:, cls_indices] if cls_indices else None
        self.cls_target_cols = [self.target_cols[i] for i in cls_indices]

        targets = targets[:, reg_indices]
        self.target_cols = [self.target_cols[i] for i in reg_indices]

        # Scale regression targets for better gradient flow
        targets *= TARGET_SCALE_FACTOR
        self.scaling_params["target_scale_factor"] = TARGET_SCALE_FACTOR

        # Replace any remaining interior NaN with 0
        features = np.nan_to_num(features, nan=0.0)
        targets = np.nan_to_num(targets, nan=0.0)
        if cls_targets is not None:
            cls_targets = np.nan_to_num(cls_targets, nan=0.0)

        dataset = NTCPDataset(features, targets, lookback=self.cfg.lookback,
                              cls_targets=cls_targets)
        logger.info(
            "Dataset ready: %d samples, lookback=%d, %d features, %d reg targets, %d cls targets.",
            len(dataset), self.cfg.lookback,
            features.shape[1], targets.shape[1],
            cls_targets.shape[1] if cls_targets is not None else 0,
        )
        return dataset

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run_pipeline(self, log_callback=None) -> NTCPDataset:
        """Execute load -> rolling trim -> features -> targets -> dataset."""
        self.load()

        # Rolling window trimming
        if self.cfg.rolling_window_size > 0 and len(self.m5_df) > self.cfg.rolling_window_size:
            trim_count = len(self.m5_df) - self.cfg.rolling_window_size
            self.m5_df = self.m5_df.iloc[trim_count:].reset_index(drop=True)
            logger.info("Rolling window: trimmed %d bars, keeping %d",
                         trim_count, len(self.m5_df))
            if log_callback:
                log_callback(f"Rolling window: trimmed {trim_count} bars, keeping {len(self.m5_df)}")

        self.build_features(log_callback=log_callback)
        self.generate_targets()
        return self.create_dataset()

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_info(self) -> dict:
        """Return summary info about the loaded data."""
        if self.m5_df is None:
            return {"loaded": False}
        df = self.m5_df
        return {
            "loaded": True,
            "bars": len(df),
            "date_start": str(df["datetime"].iloc[0]),
            "date_end": str(df["datetime"].iloc[-1]),
            "stf_factor": self.cfg.stf_factor,
            "num_features": len(self.feature_cols),
            "num_targets": len(self.target_cols),
        }
