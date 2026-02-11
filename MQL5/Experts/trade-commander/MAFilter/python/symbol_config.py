"""Symbol settings: name, mintick, tick value, default lot size."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SymbolConfig:
    name: str = ""
    mintick: float = 0.01
    tick_value: float = 1.0
    default_lots: float = 0.1

    def point_value(self, point: Optional[float] = None) -> float:
        """Dollar value per point move (per lot). point defaults to mintick."""
        p = point if point is not None else self.mintick
        return self.tick_value / p if p > 0 else 0.0
