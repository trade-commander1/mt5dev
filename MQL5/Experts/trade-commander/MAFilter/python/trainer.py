"""
Train trade filter NN from backtest results.

Input: list of TradeRecord (from strategy backtest). We build feature matrix X and
labels y from these (entry state → features, profit → good/bad label), then train.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .features import get_feature_names, trade_to_features, trades_to_xy
from .mafilter_engine import TradeRecord
from .model import TradeFilterNet, build_filter_model


def train_filter(
    trades: list[TradeRecord],
    good_threshold: float = 0.0,
    hidden_sizes: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    patience: int = 15,
    use_cuda: bool = True,
    device: Optional[str] = None,
    seed: int = 42,
    out_dir: Optional[Path] = None,
    stop_callback: Optional[Callable[[], bool]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    nn_type: str = "Dense (Feedforward)",
    architecture: str = "Medium",
    custom_hidden_sizes: Optional[tuple[int, ...]] = None,
    use_dropout: bool = True,
) -> tuple[nn.Module, dict, list[str]]:
    """
    Returns (model, scaling_dict, feature_names).
    scaling_dict includes "nn_type", "architecture", "hidden_sizes", "dropout".
    device: e.g. "CPU", "CUDA:0 (NVIDIA ...)" -> use that device; else use use_cuda.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is not None and str(device).strip().upper().startswith("CPU"):
        dev = torch.device("cpu")
    elif device is not None and str(device).strip().upper().startswith("CUDA"):
        part = str(device).strip().split("(")[0].strip().rstrip(":")
        if ":" in part:
            idx = part.split(":")[-1]
            dev = torch.device(f"cuda:{idx}" if idx.isdigit() else "cuda")
        else:
            dev = torch.device("cuda")
    else:
        dev = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    device = dev
    drop = dropout if use_dropout else 0.0

    # Build feature matrix X and label vector y from backtest results (one row per trade)
    X, y = trades_to_xy(trades, good_threshold=good_threshold)
    if len(X) < 2:
        n_feat = X.shape[1] if X.size else 3 + len(trades[0].slopes) if trades else 3
        model, hidden = build_filter_model(nn_type, architecture.lower(), n_feat, custom_hidden_sizes, drop)
        mean = np.zeros(n_feat)
        std = np.ones(n_feat)
        names = get_feature_names(len(trades[0].slopes)) if trades else get_feature_names(0)
        scaling = {"mean": mean.tolist(), "std": std.tolist(), "n_features": n_feat, "feature_names": names,
                   "nn_type": nn_type, "architecture": architecture, "hidden_sizes": list(hidden), "dropout": drop}
        return model, scaling, names

    n_features = X.shape[1]
    feature_names = get_feature_names(X.shape[1] - 3)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_norm = (X - mean) / std

    n = len(X)
    idx = np.random.permutation(n)
    n_val = max(1, int(n * val_ratio))
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    X_tr = torch.from_numpy(X_norm[train_idx]).float()
    y_tr = torch.from_numpy(y[train_idx]).float().unsqueeze(1)
    X_va = torch.from_numpy(X_norm[val_idx]).float()
    y_va = torch.from_numpy(y[val_idx]).float().unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size)

    model, hidden_used = build_filter_model(nn_type, architecture.lower(), n_features, custom_hidden_sizes, drop)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val = float("inf")
    best_state = None
    wait = 0
    for epoch in range(epochs):
        if stop_callback and stop_callback():
            break
        if progress_callback:
            progress_callback(epoch + 1, epochs)
        model.train()
        for xb, yb in train_loader:
            if stop_callback and stop_callback():
                break
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = bce(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += bce(model(xb.to(device)), yb.to(device)).item()
        val_loss /= len(val_loader)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
    if best_state:
        model.load_state_dict(best_state)

    scaling = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_features": n_features,
        "feature_names": feature_names,
        "nn_type": nn_type,
        "architecture": architecture,
        "hidden_sizes": list(hidden_used),
        "dropout": drop,
    }
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            **scaling,
        }, out_dir / "best.pt")
        with open(out_dir / "scaling.json", "w") as f:
            json.dump({k: v for k, v in scaling.items() if k != "dropout"}, f, indent=2)
    return model, scaling, feature_names


def load_filter_model(out_dir: Path) -> tuple[nn.Module, dict]:
    """Load model and scaling from out_dir (best.pt). Returns (model, scaling). Supports Dense, LSTM, GRU."""
    out_dir = Path(out_dir)
    data = torch.load(out_dir / "best.pt", map_location="cpu")
    n_features = data["n_features"]
    hidden_sizes = tuple(data["hidden_sizes"])
    dropout = data.get("dropout", 0.2)
    nn_type = data.get("nn_type", "Dense (Feedforward)")
    architecture = data.get("architecture", "medium")
    model, _ = build_filter_model(nn_type, architecture, n_features, custom_hidden_sizes=hidden_sizes, dropout=dropout)
    model.load_state_dict(data["model_state"])
    model.eval()
    scaling = {k: v for k, v in data.items() if k != "model_state"}
    return model, scaling


def filter_trades_by_nn(
    trades: list[TradeRecord],
    model: nn.Module,
    scaling: dict,
    threshold: float = 0.5,
) -> list[TradeRecord]:
    """Return only trades where NN predicts good (probability >= threshold)."""
    if not trades:
        return []
    X = np.stack([trade_to_features(t) for t in trades])
    mean = np.array(scaling["mean"], dtype=np.float64)
    std = np.array(scaling["std"], dtype=np.float64)
    std[std < 1e-8] = 1.0
    X_norm = (X - mean) / std
    with torch.no_grad():
        logits = model(torch.from_numpy(X_norm).float())
        probs = torch.sigmoid(logits).squeeze(1).numpy()
    return [t for t, p in zip(trades, probs) if p >= threshold]
