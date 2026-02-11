"""
Trade Filter Network — train a neural network to filter bad trades of a given strategy.

The strategy is typically an MT5 Expert Advisor. You provide a CSV of historical trades
with features at entry and profit/outcome; the script trains a binary classifier
(good vs bad trade) and saves the model for use as a filter (e.g. only take trades
the model approves).

Data format (CSV):
  - Required: one column for outcome (e.g. 'profit' or 'pnl') to label good/bad.
  - Required: one or more feature columns (indicators/context at entry).
  - Optional: entry_time, type, entry_price, exit_price (for analysis only).

Example columns (align with what your EA can export at bar of entry):
  profit, type, entry_price, slope_1, slope_2, laminar, bandwidth_factor, rsi, atr_pips

Usage:
  python -m trade_filter_net train --csv data/trades.csv --out_dir models/filter
  python -m trade_filter_net predict --csv data/new_trades.csv --model models/filter/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Try pandas for CSV; fallback to csv for minimal deps
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TradeFilterNet(nn.Module):
    """
    MLP that takes a feature vector at trade entry and outputs P(good trade).
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = (64, 32),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
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
        """x: [batch, features] -> [batch, 1] logits."""
        return self.mlp(x)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# Default column names (override with --label_col / --feature_cols)
DEFAULT_LABEL_COL = "profit"
DEFAULT_FEATURE_COLS: list[str] = []  # empty = auto: all numeric except label/time/type


def load_trades_csv(
    path: str | Path,
    label_col: str = DEFAULT_LABEL_COL,
    feature_cols: Optional[list[str]] = None,
    good_threshold: float = 0.0,
    drop_na: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load trade data from CSV.

    Returns:
        X: (n_trades, n_features) float array
        y: (n_trades,) 0/1 (0=bad, 1=good)
        feature_names: list of column names used as features
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    if not HAS_PANDAS:
        raise RuntimeError("pandas is required for CSV loading. Install with: pip install pandas")

    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    if drop_na:
        df = df.dropna(axis=0, how="any")

    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not in CSV. Columns: {list(df.columns)}"
        )

    y_raw = df[label_col].values.astype(np.float64)
    y = (y_raw > good_threshold).astype(np.int64)

    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        use_cols = list(feature_cols)
    else:
        exclude = {
            label_col,
            "entry_time", "exit_time", "time", "date",
            "type", "direction", "magic", "ticket", "symbol",
        }
        use_cols = [
            c for c in df.columns
            if c != label_col and c not in exclude
            and np.issubdtype(df[c].dtype, np.number)
        ]
        if not use_cols:
            raise ValueError(
                "No feature columns found. Specify --feature_cols or add numeric columns."
            )

    X = df[use_cols].values.astype(np.float64)
    return X, y, use_cols


def get_class_weights(y: np.ndarray) -> torch.Tensor:
    """Weights for BCE to handle imbalance: more weight on minority class."""
    n_pos = int(np.sum(y))
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return torch.tensor([1.0, 1.0])
    w_neg = 1.0
    w_pos = n_neg / n_pos
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    csv_path: str | Path,
    out_dir: str | Path,
    label_col: str = DEFAULT_LABEL_COL,
    feature_cols: Optional[list[str]] = None,
    good_threshold: float = 0.0,
    hidden_sizes: tuple[int, ...] = (64, 32),
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    patience: int = 15,
    use_cuda: bool = True,
    seed: int = 42,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    X, y, feature_names = load_trades_csv(
        csv_path,
        label_col=label_col,
        feature_cols=feature_cols,
        good_threshold=good_threshold,
    )
    n_samples, n_features = X.shape
    n_good = int(np.sum(y))
    logger.info(
        "Loaded %d trades, %d features. Good: %d, Bad: %d",
        n_samples, n_features, n_good, n_samples - n_good,
    )

    # Normalize features (save mean/std for inference)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-8] = 1.0
    X_norm = (X - mean) / std

    # Shuffle and split
    idx = np.random.permutation(n_samples)
    n_val = max(1, int(n_samples * val_ratio))
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    X_tr = torch.from_numpy(X_norm[train_idx]).float()
    y_tr = torch.from_numpy(y[train_idx]).float().unsqueeze(1)
    X_va = torch.from_numpy(X_norm[val_idx]).float()
    y_va = torch.from_numpy(y[val_idx]).float().unsqueeze(1)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ds = TensorDataset(X_va, y_va)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TradeFilterNet(
        input_size=n_features,
        hidden_sizes=list(hidden_sizes),
        dropout=dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = get_class_weights(y).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=weights[1:2])

    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += bce(logits, yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            wait = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_names": feature_names,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "n_features": n_features,
                    "hidden_sizes": list(hidden_sizes),
                    "dropout": dropout,
                },
                out_dir / "best.pt",
            )
        else:
            wait += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "Epoch %3d  train_loss=%.4f  val_loss=%.4f  best=%d",
                epoch + 1, train_loss, val_loss, best_epoch + 1,
            )
        if wait >= patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Save scaling params for inference / ONNX
    scaling = {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
    }
    with open(out_dir / "scaling.json", "w") as f:
        json.dump(scaling, f, indent=2)

    logger.info("Best model saved to %s (epoch %d)", out_dir / "best.pt", best_epoch + 1)
    return {
        "best_epoch": best_epoch + 1,
        "best_val_loss": best_val_loss,
        "n_features": n_features,
        "feature_names": feature_names,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str | Path, device: Optional[torch.device] = None) -> tuple[TradeFilterNet, dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    mean = np.array(ckpt["mean"])
    std = np.array(ckpt["std"])
    n_features = ckpt["n_features"]
    hidden_sizes = ckpt.get("hidden_sizes", [64, 32])
    dropout = ckpt.get("dropout", 0.2)

    model = TradeFilterNet(
        input_size=n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
    )
    model.load_state_dict(ckpt["model_state"])
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    scaling = {
        "feature_names": ckpt["feature_names"],
        "mean": mean,
        "std": std,
    }
    return model, scaling


def load_trades_csv_for_predict(
    path: str | Path,
    feature_cols: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Load CSV with only feature columns and normalize. Returns (n, n_features)."""
    if not HAS_PANDAS:
        raise RuntimeError("pandas required")
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing feature columns: {missing}")
    X = df[feature_cols].values.astype(np.float64)
    std_safe = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std_safe


def predict(
    csv_path: str | Path,
    model_path: str | Path,
    out_path: Optional[str | Path] = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Run trained filter on a CSV of trades (same feature columns as training).
    Returns probabilities of good trade, and optionally writes CSV with column 'filter_prob'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaling = load_model(model_path, device)
    feature_names = scaling["feature_names"]
    mean = np.asarray(scaling["mean"])
    std = np.asarray(scaling["std"])

    X_norm = load_trades_csv_for_predict(Path(csv_path), feature_names, mean, std)
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(X_norm).float().to(device)
        logits = model(x_t)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

    if out_path:
        out_path = Path(out_path)
        if HAS_PANDAS:
            df = pd.read_csv(csv_path)
            df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
            df_out = df.copy()
            df_out["filter_prob"] = probs
            df_out["filter_approve"] = (probs >= threshold).astype(int)
            df_out.to_csv(out_path, index=False)
            logger.info("Predictions written to %s", out_path)

    return probs


def predict_from_features(
    model_path: str | Path,
    feature_vector: np.ndarray,
    feature_names: Optional[list[str]] = None,
) -> float:
    """
    Single-trade inference: feature_vector shape (n_features,) or (1, n_features).
    Returns P(good trade) in [0, 1].
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaling = load_model(model_path, device)
    mean = np.array(scaling["mean"])
    std = np.array(scaling["std"])
    X = np.atleast_2d(feature_vector).astype(np.float64)
    X_norm = (X - mean) / np.where(std < 1e-8, 1.0, std)
    with torch.no_grad():
        x_t = torch.from_numpy(X_norm).float().to(device)
        logits = model(x_t)
        prob = torch.sigmoid(logits).cpu().item()
    return prob


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train or run a neural network trade filter for MT5 strategy trades.",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # train
    train_p = sub.add_parser("train", help="Train filter from trades CSV")
    train_p.add_argument("--csv", required=True, help="Path to trades CSV (profit + features)")
    train_p.add_argument("--out_dir", default="models/trade_filter", help="Output directory for model and scaling")
    train_p.add_argument("--label_col", default=DEFAULT_LABEL_COL, help="Column name for outcome (e.g. profit)")
    train_p.add_argument("--feature_cols", nargs="*", default=None, help="Feature columns (default: all numeric except label/time/type)")
    train_p.add_argument("--good_threshold", type=float, default=0.0, help="Label as good if label_col > this")
    train_p.add_argument("--hidden", type=int, nargs="+", default=[64, 32], help="Hidden layer sizes")
    train_p.add_argument("--dropout", type=float, default=0.2)
    train_p.add_argument("--epochs", type=int, default=100)
    train_p.add_argument("--batch_size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--val_ratio", type=float, default=0.2)
    train_p.add_argument("--patience", type=int, default=15)
    train_p.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    train_p.add_argument("--seed", type=int, default=42)

    # predict
    pred_p = sub.add_parser("predict", help="Run trained filter on a CSV")
    pred_p.add_argument("--csv", required=True, help="Path to trades CSV (same feature columns as training)")
    pred_p.add_argument("--model", required=True, help="Path to best.pt")
    pred_p.add_argument("--out", default=None, help="Output CSV path with filter_prob and filter_approve")
    pred_p.add_argument("--threshold", type=float, default=0.5, help="Approve if P(good) >= this")

    args = ap.parse_args()

    if args.command == "train":
        train(
            csv_path=args.csv,
            out_dir=args.out_dir,
            label_col=args.label_col,
            feature_cols=args.feature_cols,
            good_threshold=args.good_threshold,
            hidden_sizes=tuple(args.hidden),
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_ratio=args.val_ratio,
            patience=args.patience,
            use_cuda=not args.no_cuda,
            seed=args.seed,
        )
        return 0

    if args.command == "predict":
        predict(
            csv_path=args.csv,
            model_path=args.model,
            out_path=args.out,
            threshold=args.threshold,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
