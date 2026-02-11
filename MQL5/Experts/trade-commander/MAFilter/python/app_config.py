"""
Load/save application configuration to config.json (same directory as app package).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Config file next to this package (MAFilter/python/config.json)
CONFIG_FILENAME = "config.json"


def get_config_path() -> Path:
    return Path(__file__).resolve().parent / CONFIG_FILENAME


def get_default_config() -> dict[str, Any]:
    return {
        "version": "1.0",
        "window": {"width": 860, "height": 1002, "position_x": 100, "position_y": 100},
        "data_manager": {
            "last_data_file": "",
            "native_bar_period": "1M",
            "target_bar_period": "1M",
            "day_offset": 20,
            "symbol_name": "",
            "mintick": 0.01,
            "tick_value": 1.0,
            "default_lots": 0.1,
        },
        "backtest_train": {
            "ea_parameters": {
                "lot_size": 0.1,
                "min_len": 5,
                "max_len": 21,
                "nbr_ma": 4,
                "min_slope_factor": 1.0,
                "min_laminar_level": 0.8,
                "max_bandwidth_factor": 1.0,
                "nh": 1000,
                "std_dev_factor": 1.0,
                "exit_option": 0,
                "magic_number": 20260210,
            },
            "training": {
                "good_threshold": 0.0,
                "epochs": 100,
                "batch_size": 64,
                "patience": 15,
            },
            "nn_config": {
                "nn_type": "Dense (Feedforward)",
                "architecture": "Medium",
                "custom_arch": False,
                "custom_layers": 3,
                "custom_sizes": "256,128,64",
                "dropout": 0.2,
                "use_dropout": True,
            },
            "lookback_config": {
                "enabled": True,
                "length": 50,
                "include_ohlcv": True,
                "include_laminar": False,
                "include_bw": False,
                "include_stddev": False,
                "include_slopes": False,
                "normalization": "Dynamic (per window)",
            },
            "output_dir": "",
            "device": "CPU",
        },
    }


def load_config() -> tuple[dict[str, Any], str | None]:
    """
    Load config from config.json. Returns (config_dict, error_message).
    error_message is None on success.
    """
    path = get_config_path()
    if not path.exists():
        return get_default_config(), None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return get_default_config(), "Config file invalid (not a dict)"
        # Merge with defaults so new keys exist
        default = get_default_config()
        for key in default:
            if key not in data:
                data[key] = default[key]
            elif isinstance(default[key], dict) and isinstance(data[key], dict):
                for k, v in default[key].items():
                    if k not in data[key]:
                        data[key][k] = v
        return data, None
    except json.JSONDecodeError as e:
        return get_default_config(), f"Config file corrupted: {e}"
    except Exception as e:
        return get_default_config(), str(e)


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to config.json."""
    config = dict(config)
    config["last_saved"] = datetime.now().isoformat()
    if "version" not in config:
        config["version"] = "1.0"
    path = get_config_path()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
