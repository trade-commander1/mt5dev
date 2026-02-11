"""NTCP Strategies — pluggable target-generation strategies."""

from ..config import DataConfig
from .mafilter import MAFilterStrategy
from .trendcatcher import TrendcatcherStrategy

__all__ = ["TrendcatcherStrategy", "MAFilterStrategy", "get_strategy"]

_REGISTRY: dict[str, type] = {
    "trendcatcher": TrendcatcherStrategy,
    "mafilter": MAFilterStrategy,
}


def get_strategy(name: str, cfg: DataConfig, ma_spectrum: list[int]):
    """Factory: return an instantiated strategy by name."""
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY)}")
    return cls(cfg, ma_spectrum)
