"""
NTCP Neural Network — configurable GRU/LSTM for multi-target regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, TARGET_HORIZONS, NUM_CLS_TARGETS


# 3 targets per horizon: MFE, MAE, Momentum
NUM_TARGETS = len(TARGET_HORIZONS) * 3


class NTCPModel(nn.Module):
    """
    RNN-based model for trading signal prediction.
    Architecture: RNN -> last hidden -> Linear -> LeakyReLU -> Dropout -> Linear -> targets.
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
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, NUM_TARGETS),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, NUM_CLS_TARGETS),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform for Linear layers, orthogonal for RNN recurrent weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for name, param in self.rnn.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, lookback, features]
        Returns:
            reg_pred: [batch, NUM_TARGETS]  — regression head
            cls_pred: [batch, NUM_CLS_TARGETS]  — classification logits
        """
        output, _ = self.rnn(x)
        last = output[:, -1, :]
        return self.head(last), self.cls_head(last)
