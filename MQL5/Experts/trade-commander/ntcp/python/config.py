"""
NTCP Configuration — shared constants and dataclasses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def generate_ma_spectrum(ma_min: int, ma_max: int, ma_count: int) -> list[int]:
    """Generate log-spaced MA periods, rounded to unique ints."""
    if ma_count < 2:
        return [ma_min]
    raw = [
        ma_min * math.exp(i / (ma_count - 1) * math.log(ma_max / ma_min))
        for i in range(ma_count)
    ]
    # Round and deduplicate while preserving order
    seen: set[int] = set()
    result: list[int] = []
    for v in raw:
        iv = max(2, round(v))
        if iv not in seen:
            seen.add(iv)
            result.append(iv)
    return result


# Default MA spectrum (backward-compatible with old Fibonacci list)
MA_SPECTRUM: list[int] = generate_ma_spectrum(5, 500, 11)

# Horizons for MFE / MAE / Momentum targets
TARGET_HORIZONS: list[int] = [3, 5, 10, 20, 40, 80]

# Volume smoothing period
TICK_SMA_PERIOD: int = 20

# Numerical stability
EPSILON: float = 1e-9

# RSI default period
RSI_PERIOD: int = 14

# ATR default period
ATR_PERIOD: int = 14

# Rolling window for daily high/low approximation (288 M5 bars = 24 h)
DAY_BARS: int = 288

# Hurst exponent window
HURST_WINDOW: int = 100

# Market regime clustering
REGIME_N_CLUSTERS: int = 4


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    m5_path: str | Path = ""
    stf_factor: int = 12  # number of M5 bars per STF bar (12 = H1)
    lookback: int = 20
    clip_percentile: float = 1.0
    rolling_window_size: int = 0  # 0 = unlimited (use all data)
    hurst_window: int = HURST_WINDOW
    regime_n_clusters: int = REGIME_N_CLUSTERS
    ma_min: int = 5
    ma_max: int = 500
    ma_count: int = 11
    signal_ma_count: int = 4
    strategy: str = "trendcatcher"
    active_feature_groups: list[str] = field(default_factory=list)  # empty = all

    def get_ma_spectrum(self) -> list[int]:
        """Generate the MA spectrum from current config."""
        return generate_ma_spectrum(self.ma_min, self.ma_max, self.ma_count)


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    input_size: int = 0  # set dynamically from feature count
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    cell_type: Literal["GRU", "LSTM"] = "GRU"


@dataclass
class TrainConfig:
    """Training loop configuration."""
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    val_split: float = 0.15
    patience: int = 10
    use_cuda: bool = True
    stf_factor_min: int = 6
    stf_factor_max: int = 24


@dataclass
class ExportConfig:
    """ONNX / MQH export configuration."""
    version: str = "1.0.0"
    onnx_path: str | Path = "model.onnx"
    mqh_path: str | Path = "../scaling_params.mqh"
