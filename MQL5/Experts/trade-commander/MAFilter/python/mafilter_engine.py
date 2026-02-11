"""
MAFilter backtest engine — mirrors MAFilter.mq5 logic on OHLCV arrays.
Produces list of trades with entry bar, direction, exit bar, PnL and entry-state features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


def generate_ma_periods(min_len: int, max_len: int, count: int) -> list[int]:
    """Log-spaced unique MA periods (match EA)."""
    if count < 2:
        return [min_len]
    log_ratio = math.log(max_len / min_len)
    seen: set[int] = set()
    out: list[int] = []
    for i in range(count):
        raw = min_len * math.exp(i / (count - 1) * log_ratio)
        r = max(2, round(raw))
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def sma_series(close: np.ndarray, period: int) -> np.ndarray:
    """SMA of close, length = len(close). First (period-1) values are NaN, then valid."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if period < 1 or n < period:
        return out
    cum = np.cumsum(close)
    out[period - 1] = cum[period - 1] / period
    for i in range(period, n):
        out[i] = (cum[i] - cum[i - period]) / period
    return out


def stddev_series(close: np.ndarray, period: int) -> np.ndarray:
    """StdDev of close over period (MODE_SMA style)."""
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if period < 2 or n < period:
        return out
    sma = sma_series(close, period)
    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        out[i] = np.std(window)
    return out


def compute_laminar_level(slopes: np.ndarray) -> float:
    """Pairwise slope concordance in [-1, 1]. slopes: 1D array per MA."""
    count = len(slopes)
    total_pairs = count * (count - 1) // 2
    if total_pairs == 0:
        return 0.0
    concordant = 0
    for a in range(count):
        for b in range(a + 1, count):
            if slopes[a] > slopes[b]:
                concordant += 1
            elif slopes[a] < slopes[b]:
                concordant -= 1
    return concordant / total_pairs


def compute_bandwidth(ma_values: np.ndarray) -> float:
    """max(MAs) - min(MAs)."""
    return float(np.max(ma_values) - np.min(ma_values))


@dataclass
class EAParams:
    """Mirror of MAFilter.mq5 input parameters."""
    lot_size: float = 0.1
    min_len: int = 5
    max_len: int = 21
    nbr_ma: int = 4
    min_slope_factor: float = 1.0
    min_laminar_level: float = 0.8
    max_bandwidth_factor: float = 1.0
    nh: int = 1000
    std_dev_factor: float = 1.0
    exit_option: int = 0  # 0=StdDev, 1=Slope
    magic_number: int = 20260210


@dataclass
class TradeRecord:
    """Single trade from backtest with features at entry."""
    entry_bar: int
    exit_bar: int
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    profit: float  # dollar or points; same as label
    # Features at entry (bar 1 = last completed bar when we open)
    laminar_level: float = 0.0
    bw_factor: float = 0.0
    std_dev_current: float = 0.0
    slopes: tuple[float, ...] = field(default_factory=tuple)
    ma_bar1: tuple[float, ...] = field(default_factory=tuple)
    close_bar1: float = 0.0


def run_backtest(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    params: EAParams,
    tick_value: float = 1.0,
    point: float = 0.01,
    lot_size: Optional[float] = None,
    equity_callback: Optional[Callable[[int, float], None]] = None,
    stop_callback: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[TradeRecord]:
    """
    Run MAFilter logic bar-by-bar. Evaluates on bar 1 (last completed).
    Returns list of TradeRecord with entry features and profit.
    """
    n = len(close)
    lot = lot_size if lot_size is not None else params.lot_size
    periods = generate_ma_periods(params.min_len, params.max_len, params.nbr_ma)
    nbr_ma = len(periods)
    if nbr_ma < 2:
        return []

    # Precompute all MAs and StdDev (slowest period)
    ma_buffers: list[np.ndarray] = []
    for p in periods:
        ma_buffers.append(sma_series(close, p))
    std_dev_buf = stddev_series(close, periods[-1])
    bars_needed = params.nh + 2
    if n < bars_needed:
        return []

    trades: list[TradeRecord] = []
    pos = 0
    pos_entry_bar = -1
    pos_entry_price = 0.0
    pos_direction = 0
    entry_laminar = 0.0
    entry_bw_factor = 0.0
    entry_std_dev = 0.0
    entry_slopes: tuple[float, ...] = ()
    entry_ma_bar1: tuple[float, ...] = ()
    entry_close = 0.0

    total_bars = n - 1 - (bars_needed - 2)
    progress_interval = max(1, total_bars // 80)  # emit ~80 times during backtest
    for bar1_idx in range(bars_needed - 2, n - 1):
        if stop_callback and stop_callback():
            break
        if progress_callback and total_bars > 0:
            current = bar1_idx - (bars_needed - 2)
            if current % progress_interval == 0 or current == total_bars - 1:
                progress_callback(current, total_bars)
        bar2_idx = bar1_idx - 1
        close_price = close[bar1_idx]
        ma_bar1 = np.array([ma_buffers[k][bar1_idx] for k in range(nbr_ma)])
        ma_bar2 = np.array([ma_buffers[k][bar2_idx] for k in range(nbr_ma)])
        slopes = ma_bar1 - ma_bar2
        if np.any(np.isnan(ma_bar1)) or np.any(np.isnan(slopes)):
            continue
        std_dev_current = std_dev_buf[bar1_idx]
        if np.isnan(std_dev_current):
            std_dev_current = 0.0

        close_above_all = np.all(close_price > ma_bar1)
        close_below_all = np.all(close_price < ma_bar1)
        all_slopes_pos = np.all(slopes > 0)
        all_slopes_neg = np.all(slopes < 0)

        slopes_strong = True
        for k in range(nbr_ma):
            hist_sum = 0.0
            hist_count = 0
            for shift in range(params.nh):
                bc = bar1_idx - shift
                bp = bc - 1
                if bp < 0:
                    break
                s = abs(ma_buffers[k][bc] - ma_buffers[k][bp])
                hist_sum += s
                hist_count += 1
            avg_abs = hist_sum / hist_count if hist_count > 0 else 0.0
            thresh = params.min_slope_factor * avg_abs
            if abs(slopes[k]) < thresh:
                slopes_strong = False
                break

        laminar_level = compute_laminar_level(slopes)
        laminar_long = laminar_level >= params.min_laminar_level
        laminar_short = laminar_level <= -params.min_laminar_level

        bw_bar1 = compute_bandwidth(ma_bar1)
        bw_sum = 0.0
        bw_count = 0
        for shift in range(params.nh):
            bi = bar1_idx - shift
            if bi < 0:
                break
            mas = np.array([ma_buffers[k][bi] for k in range(nbr_ma)])
            bw_sum += compute_bandwidth(mas)
            bw_count += 1
        bw_sma = bw_sum / bw_count if bw_count > 0 else 1e-10
        bw_factor = bw_bar1 / bw_sma
        bw_ok = bw_factor <= params.max_bandwidth_factor

        if pos != 0:
            exit_signal = False
            if params.exit_option == 0:
                slowest_ma = ma_bar1[-1]
                band = params.std_dev_factor * std_dev_current
                if pos > 0 and close_price < slowest_ma - band:
                    exit_signal = True
                if pos < 0 and close_price > slowest_ma + band:
                    exit_signal = True
            else:
                if pos > 0 and slopes[0] < 0:
                    exit_signal = True
                if pos < 0 and slopes[0] > 0:
                    exit_signal = True
            if exit_signal:
                exit_price = close[bar1_idx]
                profit_pts = pos_direction * (exit_price - pos_entry_price)
                profit_dollars = profit_pts / point * tick_value * lot if point > 0 else 0.0
                tr = TradeRecord(
                    entry_bar=pos_entry_bar,
                    exit_bar=bar1_idx,
                    direction=pos_direction,
                    entry_price=pos_entry_price,
                    exit_price=exit_price,
                    profit=profit_dollars,
                    laminar_level=entry_laminar,
                    bw_factor=entry_bw_factor,
                    std_dev_current=entry_std_dev,
                    slopes=entry_slopes,
                    ma_bar1=entry_ma_bar1,
                    close_bar1=entry_close,
                )
                trades.append(tr)
                if equity_callback is not None:
                    cum = sum(t.profit for t in trades)
                    equity_callback(bar1_idx, cum)
                pos = 0
                continue
            continue

        long_entry = close_above_all and all_slopes_pos and slopes_strong and laminar_long and bw_ok
        short_entry = close_below_all and all_slopes_neg and slopes_strong and laminar_short and bw_ok

        if long_entry:
            pos = 1
            pos_direction = 1
            pos_entry_bar = bar1_idx
            pos_entry_price = close_price
            entry_laminar = laminar_level
            entry_bw_factor = bw_factor
            entry_std_dev = std_dev_current
            entry_slopes = tuple(float(s) for s in slopes)
            entry_ma_bar1 = tuple(float(m) for m in ma_bar1)
            entry_close = close_price
        elif short_entry:
            pos = -1
            pos_direction = -1
            pos_entry_bar = bar1_idx
            pos_entry_price = close_price
            entry_laminar = laminar_level
            entry_bw_factor = bw_factor
            entry_std_dev = std_dev_current
            entry_slopes = tuple(float(s) for s in slopes)
            entry_ma_bar1 = tuple(float(m) for m in ma_bar1)
            entry_close = close_price

    if pos != 0:
        exit_price = close[-1]
        profit_pts = pos_direction * (exit_price - pos_entry_price)
        profit_dollars = profit_pts / point * tick_value * lot if point > 0 else 0.0
        tr = TradeRecord(
            entry_bar=pos_entry_bar,
            exit_bar=n - 1,
            direction=pos_direction,
            entry_price=pos_entry_price,
            exit_price=exit_price,
            profit=profit_dollars,
            laminar_level=entry_laminar,
            bw_factor=entry_bw_factor,
            std_dev_current=entry_std_dev,
            slopes=entry_slopes,
            ma_bar1=entry_ma_bar1,
            close_bar1=entry_close,
        )
        trades.append(tr)
        if equity_callback is not None:
            cum = sum(t.profit for t in trades)
            equity_callback(n - 1, cum)

    return trades
