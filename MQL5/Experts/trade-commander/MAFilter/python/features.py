"""
Build NN feature vector and labels FROM backtest results.

Flow: (1) Run strategy backtest → list of TradeRecord (entry state + profit).
      (2) For each trade: features = entry state at open; label = good/bad from profit.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .mafilter_engine import TradeRecord


def trade_to_features(t: TradeRecord) -> np.ndarray:
    """
    One trade → one feature vector (entry state only: laminar, bw, std_dev_ratio, slope ratios).
    Built from backtest result; no future data.
    """
    feats = [
        t.laminar_level,
        t.bw_factor,
        t.std_dev_current / (t.close_bar1 + 1e-12),
    ]
    for s in t.slopes:
        feats.append(s / (t.close_bar1 + 1e-12))
    return np.array(feats, dtype=np.float64)


def trades_to_xy(
    trades: list[TradeRecord],
    good_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix X and label vector y from backtest results.
    X: one row per trade (entry-state features). y: 0/1 per trade (1 = profit > threshold).
    """
    if not trades:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    X = np.stack([trade_to_features(t) for t in trades])
    y = (np.array([t.profit for t in trades]) > good_threshold).astype(np.int64)
    return X, y


def get_feature_names(n_slopes: int) -> list[str]:
    out = ["laminar_level", "bw_factor", "std_dev_ratio"]
    for i in range(n_slopes):
        out.append(f"slope_{i}_ratio")
    return out


def trades_to_xy_with_lookback(
    trades: list[TradeRecord],
    good_threshold: float = 0.0,
    lookback_enabled: bool = False,
    lookback_length: int = 50,
    lookback_ohlc: bool = True,
    lookback_laminar: bool = False,
    lookback_bw: bool = False,
    lookback_stddev: bool = False,
    lookback_slopes: bool = False,
    bar_ohlcv: Optional[tuple] = None,
    bar_indicators: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build X, y with optional lookback window. When lookback disabled, same as trades_to_xy.
    When enabled, requires bar_ohlcv and bar_indicators (per-bar OHLCV and strategy indicators)
    for each bar in the backtest range. For now, if lookback enabled but bar data missing,
    falls back to entry-only features.
    """
    if not lookback_enabled or bar_ohlcv is None or bar_indicators is None:
        return trades_to_xy(trades, good_threshold=good_threshold)
    # TODO: implement full lookback: for each trade, slice [entry_bar - N : entry_bar],
    # build (lookback_length, n_features_per_bar) + entry features, normalize per window or global
    return trades_to_xy(trades, good_threshold=good_threshold)
