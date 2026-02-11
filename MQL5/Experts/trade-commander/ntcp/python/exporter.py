"""
NTCP Exporter — ONNX model export and scaling_params.mqh generation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import (
    MA_SPECTRUM,
    NUM_CLS_TARGETS,
    TARGET_HORIZONS,
    TICK_SMA_PERIOD,
    ExportConfig,
    ModelConfig,
)
from .model import NTCPModel

logger = logging.getLogger(__name__)


class NTCPExporter:
    """Export trained model to ONNX and generate MQL5 header file."""

    def __init__(self, cfg: ExportConfig) -> None:
        self.cfg = cfg

    def export_onnx(
        self,
        model_state: dict,
        model_cfg: ModelConfig,
        lookback: int,
    ) -> Path:
        """
        Export PyTorch model to ONNX format with dynamic batch axis.
        Returns the path to the written file.
        """
        model = NTCPModel(model_cfg)
        model.load_state_dict(model_state)
        model.eval()

        dummy = torch.randn(1, lookback, model_cfg.input_size)
        onnx_path = Path(self.cfg.onnx_path)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["regression", "classification"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "regression": {0: "batch_size"},
                "classification": {0: "batch_size"},
            },
            opset_version=17,
        )

        logger.info("ONNX exported to %s", onnx_path)
        return onnx_path

    def export_mqh(
        self,
        feature_cols: list[str],
        target_cols: list[str],
        scaling_params: dict,
        stf_factor: int,
        lookback: int,
    ) -> Path:
        """
        Generate scaling_params.mqh — a MQL5 header file containing all
        preprocessing constants needed by the EA to replicate the Python
        feature pipeline.
        """
        mqh_path = Path(self.cfg.mqh_path)
        mqh_path.parent.mkdir(parents=True, exist_ok=True)

        clip_bounds = scaling_params.get("target_clip_bounds", {})

        lines: list[str] = []
        lines.append("//+------------------------------------------------------------------+")
        lines.append(f"//| NTCP Scaling Parameters v{self.cfg.version}")
        lines.append(f"//| Auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("//| DO NOT EDIT — regenerate from Python exporter.")
        lines.append("//+------------------------------------------------------------------+")
        lines.append(f"#property strict")
        lines.append("")
        lines.append(f"#define NTCP_VERSION \"{self.cfg.version}\"")
        lines.append(f"#define NTCP_STF_FACTOR {stf_factor}")
        lines.append(f"#define NTCP_LOOKBACK {lookback}")
        lines.append(f"#define NTCP_NUM_FEATURES {len(feature_cols)}")
        lines.append(f"#define NTCP_NUM_TARGETS {len(target_cols)}")
        lines.append(f"#define NTCP_NUM_CLS_TARGETS {NUM_CLS_TARGETS}")
        lines.append(f"#define NTCP_TICK_SMA_PERIOD {TICK_SMA_PERIOD}")
        scale = scaling_params.get("target_scale_factor", 1.0)
        lines.append(f"#define NTCP_TARGET_SCALE_FACTOR {scale:.1f}")
        lines.append("")

        # MA spectrum (dynamic from training)
        ma_spectrum = scaling_params.get("ma_spectrum", MA_SPECTRUM)
        signal_n = scaling_params.get("signal_ma_count", 4)
        ma_str = ", ".join(str(p) for p in ma_spectrum)
        lines.append(f"const int NTCP_MA_SPECTRUM[] = {{{ma_str}}};")
        lines.append(f"#define NTCP_MA_COUNT {len(ma_spectrum)}")
        lines.append(f"#define NTCP_SIGNAL_MA_COUNT {signal_n}")
        lines.append("")

        # Target horizons
        th_str = ", ".join(str(h) for h in TARGET_HORIZONS)
        lines.append(f"const int NTCP_TARGET_HORIZONS[] = {{{th_str}}};")
        lines.append(f"#define NTCP_HORIZON_COUNT {len(TARGET_HORIZONS)}")
        lines.append("")

        # Feature order array
        lines.append("// Feature order (must match Python pipeline)")
        lines.append(f"const string NTCP_FEATURE_NAMES[] = {{")
        for i, name in enumerate(feature_cols):
            comma = "," if i < len(feature_cols) - 1 else ""
            lines.append(f'    "{name}"{comma}')
        lines.append("};")
        lines.append("")

        # Target names
        lines.append(f"const string NTCP_TARGET_NAMES[] = {{")
        for i, name in enumerate(target_cols):
            comma = "," if i < len(target_cols) - 1 else ""
            lines.append(f'    "{name}"{comma}')
        lines.append("};")
        lines.append("")

        # Target clip bounds
        lines.append("// Target clipping bounds (1st/99th percentile)")
        for name, bounds in clip_bounds.items():
            safe_name = name.upper()
            lines.append(
                f"const double NTCP_CLIP_{safe_name}_LO = {bounds['lower']:.10f};"
            )
            lines.append(
                f"const double NTCP_CLIP_{safe_name}_HI = {bounds['upper']:.10f};"
            )
        lines.append("")

        # Per-feature Mean/StdDev arrays
        feat_means = scaling_params.get("feature_means", {})
        feat_stds = scaling_params.get("feature_stds", {})
        if feat_means:
            lines.append("// Per-feature normalization: mean values")
            lines.append("const double NTCP_FEATURE_MEAN[] = {")
            for i, name in enumerate(feature_cols):
                val = feat_means.get(name, 0.0)
                comma = "," if i < len(feature_cols) - 1 else ""
                lines.append(f"    {val:.10f}{comma}  // {name}")
            lines.append("};")
            lines.append("")

            lines.append("// Per-feature normalization: standard deviation")
            lines.append("const double NTCP_FEATURE_STD[] = {")
            for i, name in enumerate(feature_cols):
                val = feat_stds.get(name, 1.0)
                comma = "," if i < len(feature_cols) - 1 else ""
                lines.append(f"    {val:.10f}{comma}  // {name}")
            lines.append("};")
            lines.append("")

        # Regime centroids
        regime_centroids = scaling_params.get("regime_centroids")
        if regime_centroids:
            n_clusters = len(regime_centroids)
            n_regime_feats = len(regime_centroids[0]) if n_clusters > 0 else 0
            lines.append(f"#define NTCP_REGIME_CLUSTERS {n_clusters}")
            lines.append(f"#define NTCP_REGIME_FEATURES {n_regime_feats}")
            lines.append("// GMM regime centroids [cluster][feature]")
            lines.append("const double NTCP_REGIME_CENTROIDS[][" + str(n_regime_feats) + "] = {")
            for ci, centroid in enumerate(regime_centroids):
                vals = ", ".join(f"{v:.10f}" for v in centroid)
                comma = "," if ci < n_clusters - 1 else ""
                lines.append(f"    {{{vals}}}{comma}")
            lines.append("};")
            lines.append("")

        # Active feature groups metadata
        active_groups = scaling_params.get("active_feature_groups")
        if active_groups:
            groups_str = ", ".join(f'"{g}"' for g in active_groups)
            lines.append(f"// Active feature groups used during training")
            lines.append(f"const string NTCP_ACTIVE_FEATURE_GROUPS[] = {{{groups_str}}};")
            lines.append("")

        lines.append("//+------------------------------------------------------------------+")

        mqh_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("MQH header exported to %s", mqh_path)
        return mqh_path
