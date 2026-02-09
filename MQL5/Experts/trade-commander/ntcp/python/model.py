"""
NTCP Neural Network — configurable GRU/LSTM for multi-target regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, TARGET_HORIZONS


# 3 targets per horizon: MFE, MAE, Momentum
NUM_TARGETS = len(TARGET_HORIZONS) * 3


class NTCPModel(nn.Module):
    """
    RNN-based model for trading signal prediction.
    Architecture: RNN (GRU or LSTM) -> last hidden -> Dropout -> Linear -> targets.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        rnn_cls = nn.GRU if cfg.cell_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, NUM_TARGETS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, lookback, features]
        Returns:
            predictions: [batch, NUM_TARGETS]
        """
        output, _ = self.rnn(x)
        # Take last time step
        last = output[:, -1, :]
        last = self.dropout(last)
        return self.head(last)
