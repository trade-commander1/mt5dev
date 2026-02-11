"""
Load OHLCV CSV: DateTime,Open,High,Low,Close,Volume,Spread
Example line: 20250811 01:00:00,3397.298,3404.448,3397.298,3403.648,27360,0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Timeframe string -> bar duration in minutes
TF_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# Minutes -> pandas resample rule (for upsampling to higher TF)
MINUTES_TO_RESAMPLE_RULE = {
    1: "1min",
    5: "5min",
    15: "15min",
    30: "30min",
    60: "1h",
    240: "4h",
    1440: "1D",
}


def parse_timeframe(tf_str: str) -> int:
    """Parse e.g. '1M', '15M', '1H' to minutes. Default 1 if invalid."""
    key = (tf_str or "1M").strip().lower()
    return TF_MINUTES.get(key, 1)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """
    Load CSV with header: DateTime,Open,High,Low,Close,Volume,Spread
    Returns DataFrame with datetime index and columns Open,High,Low,Close,Volume,Spread.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    df = pd.read_csv(path)
    df = df.rename(columns=lambda c: c.strip())
    if "Date" in df.columns and "DateTime" not in df.columns:
        df = df.rename(columns={"Date": "DateTime"})
    if "TickVolume" in df.columns and "Volume" not in df.columns:
        df = df.rename(columns={"TickVolume": "Volume"})
    required = {"DateTime", "Open", "High", "Low", "Close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Parse datetime: "20250811 01:00:00" or "2025.11.30 08:52:00"
    dt = pd.to_datetime(df["DateTime"], format="%Y%m%d %H:%M:%S", errors="coerce")
    if dt.isna().any():
        dt = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.drop(columns=["DateTime"])
    df.insert(0, "DateTime", dt)
    df = df.dropna(subset=["DateTime"])
    df = df.sort_values("DateTime").reset_index(drop=True)

    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    else:
        df["Volume"] = 0
    if "Spread" not in df.columns:
        df["Spread"] = 0.0
    else:
        df["Spread"] = pd.to_numeric(df["Spread"], errors="coerce").fillna(0)

    return df


def resample_ohlcv(df: pd.DataFrame, from_minutes: int, to_minutes: int) -> pd.DataFrame:
    """
    Resample OHLCV DataFrame to a higher timeframe.
    from_minutes: timeframe of current bars (e.g. 1 for M1).
    to_minutes: target timeframe (e.g. 5 for M5); must be >= from_minutes and integer multiple.
    Returns DataFrame with same columns, bars aggregated: O=first, H=max, L=min, C=last, V=sum, Spread=last.
    """
    if to_minutes < from_minutes:
        raise ValueError(f"Target period ({to_minutes} min) must be >= data file period ({from_minutes} min).")
    if to_minutes % from_minutes != 0:
        raise ValueError(
            f"Target period ({to_minutes} min) must be an integer multiple of data file period ({from_minutes} min)."
        )
    if to_minutes == from_minutes:
        return df.copy()
    rule = MINUTES_TO_RESAMPLE_RULE.get(to_minutes)
    if not rule:
        raise ValueError(f"Unsupported target timeframe: {to_minutes} minutes. Use 1,5,15,30,60,240,1440.")
    idx = pd.DatetimeIndex(df["DateTime"])
    df = df.set_index(idx)
    resampled = df.resample(rule).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
        Spread=("Spread", "last"),
    ).dropna(how="all")
    # Drop rows with any NaN in OHLC so strategy gets valid numbers
    resampled = resampled.dropna(subset=["Open", "High", "Low", "Close"])
    resampled = resampled.reset_index().rename(columns={"index": "DateTime"})
    return resampled


def split_by_day_offset(
    df: pd.DataFrame,
    offset_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    Split DataFrame into training (all data up to offset_days before end) and application (last offset_days).
    Returns (df_training, df_application, split_date, last_date).
    split_date is the first date of the application part; training is df[DateTime < split_date].
    """
    if "DateTime" not in df.columns or len(df) == 0:
        return df.copy(), pd.DataFrame(), pd.NaT, pd.NaT
    last_ts = df["DateTime"].iloc[-1]
    if not pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
        last_ts = pd.to_datetime(last_ts)
    last_date = last_ts.normalize() if hasattr(last_ts, "normalize") else last_ts
    split_ts = last_date - pd.Timedelta(days=offset_days)
    df_train = df[df["DateTime"] < split_ts].copy()
    df_app = df[df["DateTime"] >= split_ts].copy()
    first_date = df["DateTime"].iloc[0]
    if not pd.api.types.is_datetime64_any_dtype(df["DateTime"]):
        first_date = pd.to_datetime(first_date)
    return df_train, df_app, split_ts, last_date


def df_to_arrays(df: pd.DataFrame):
    """Return (open, high, low, close, volume, spread) as 1D numpy arrays."""
    return (
        df["Open"].values.astype(np.float64),
        df["High"].values.astype(np.float64),
        df["Low"].values.astype(np.float64),
        df["Close"].values.astype(np.float64),
        df["Volume"].values.astype(np.float64),
        df["Spread"].values.astype(np.float64),
    )
