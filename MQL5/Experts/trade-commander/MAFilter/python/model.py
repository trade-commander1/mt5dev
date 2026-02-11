"""NN models for trade filter: Dense (MLP), LSTM, GRU."""

from __future__ import annotations

import torch
import torch.nn as nn


# Predefined architectures: (hidden_sizes tuple for Dense, or (rnn_units, dense_units) for RNN)
ARCH_DENSE = {
    "small": (128, 64),
    "medium": (256, 128, 64),
    "large": (512, 256, 128, 64),
}
ARCH_LSTM = {
    "small": (64, 32),
    "medium": (128, 64, 32),
    "large": (256, 128, 64, 32),
}
ARCH_GRU = {
    "small": (64, 32),
    "medium": (128, 64, 32),
    "large": (256, 128, 64, 32),
}


class TradeFilterNet(nn.Module):
    """Dense (MLP) binary classifier for good/bad trade."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TradeFilterLSTM(nn.Module):
    """LSTM for sequence (batch, seq_len, features) then dense to 1. Handles seq_len=1 (entry-only)."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        rnn_units = hidden_sizes[0]
        self.lstm = nn.LSTM(input_size, rnn_units, batch_first=True)
        dense_layers = []
        prev = rnn_units
        for h in hidden_sizes[1:]:
            dense_layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        dense_layers.append(nn.Linear(prev, 1))
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, (h_n, _) = self.lstm(x)
        out = self.dense(h_n.squeeze(0))
        return out


class TradeFilterGRU(nn.Module):
    """GRU for sequence (batch, seq_len, features) then dense to 1. Handles seq_len=1 (entry-only)."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self._input_size = input_size
        rnn_units = hidden_sizes[0]
        self.gru = nn.GRU(input_size, rnn_units, batch_first=True)
        dense_layers = []
        prev = rnn_units
        for h in hidden_sizes[1:]:
            dense_layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev = h
        dense_layers.append(nn.Linear(prev, 1))
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        _, h_n = self.gru(x)
        out = self.dense(h_n.squeeze(0))
        return out


def build_filter_model(
    nn_type: str,
    architecture: str,
    input_size: int,
    custom_hidden_sizes: tuple[int, ...] | None = None,
    dropout: float = 0.2,
) -> tuple[nn.Module, tuple[int, ...]]:
    """Build Dense, LSTM, or GRU model. Returns (model, hidden_sizes_used)."""
    arch = architecture.lower()
    if custom_hidden_sizes is not None and len(custom_hidden_sizes) > 0:
        hidden = custom_hidden_sizes
    elif "dense" in nn_type.lower():
        hidden = ARCH_DENSE.get(arch, ARCH_DENSE["medium"])
    elif "lstm" in nn_type.lower():
        hidden = ARCH_LSTM.get(arch, ARCH_LSTM["medium"])
    elif "gru" in nn_type.lower():
        hidden = ARCH_GRU.get(arch, ARCH_GRU["medium"])
    else:
        hidden = (64, 32)
    dropout = max(0.0, min(0.5, dropout))
    if "lstm" in nn_type.lower():
        return TradeFilterLSTM(input_size, hidden_sizes=hidden, dropout=dropout), hidden
    if "gru" in nn_type.lower():
        return TradeFilterGRU(input_size, hidden_sizes=hidden, dropout=dropout), hidden
    return TradeFilterNet(input_size, hidden_sizes=hidden, dropout=dropout), hidden
