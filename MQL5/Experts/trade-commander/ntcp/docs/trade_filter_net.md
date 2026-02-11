# Trade Filter Network

A small Python script that **trains a neural network to classify trades as good or bad** for a given strategy (e.g. your MT5 Expert Advisor). You can then use the model as a filter: only open trades the model approves.

## Quick start

From the `ntcp/python` directory:

```bash
# Train on a CSV of historical trades (columns: profit + features at entry)
python trade_filter_net.py train --csv data/sample_trades_filter.csv --out_dir models/trade_filter

# Predict on new trades (same feature columns)
python trade_filter_net.py predict --csv data/new_trades.csv --model models/trade_filter/best.pt --out data/filtered.csv
```

## Data format (CSV)

- **Required**
  - One **label** column (e.g. `profit` or `pnl`) used to define good vs bad (default: good = profit > 0).
  - One or more **feature** columns (numeric) describing the market/strategy state **at trade entry**.

- **Optional** (ignored as features if not listed in `--feature_cols`)
  - `entry_time`, `exit_time`, `type`, `direction`, `entry_price`, `exit_price`, `magic`, `ticket`, `symbol`.

If you do not pass `--feature_cols`, the script uses all numeric columns except the label and common non-feature names above.

Example columns (e.g. for an MA-based EA):

```text
profit, type, entry_price, slope_1, slope_2, laminar, bandwidth_factor, rsi
```

## Getting trade data from MetaTrader 5

1. **Strategy Tester**
   - Run your EA in the Strategy Tester.
   - Use the **Report** tab and export to HTML, or use an EA/script that writes each closed trade to a file.

2. **Custom export from EA**
   - In your EA, when a position is closed (in `OnTradeTransaction` or on tick when you detect close), write one line to a CSV file:
     - Outcome: `profit` (or `pnl`) for the closed deal.
     - Features: indicator values and any context **at the bar when the trade was opened** (e.g. MA slopes, laminar level, bandwidth factor, RSI, ATR). You must store these at open time if you want them at close time for the CSV.

3. **Python backtester**
   - If you backtest the same logic in Python (e.g. in this repo’s pipeline), you can export a CSV of synthetic trades: one row per trade with `profit` and the same entry-time features.

The filter is trained on **historical trades only**. At live/demo, you compute the same features at entry and call the model (or an exported ONNX version) to get P(good trade); then only open when that probability is above your threshold.

## Training options

| Option | Default | Description |
|--------|--------|-------------|
| `--label_col` | `profit` | Column name for outcome |
| `--good_threshold` | `0.0` | Label as good if `label_col > good_threshold` |
| `--feature_cols` | (auto) | Explicit list of feature columns |
| `--hidden` | `64 32` | Hidden layer sizes |
| `--dropout` | `0.2` | Dropout |
| `--epochs` | `100` | Max epochs |
| `--patience` | `15` | Early stopping (epochs without val improvement) |
| `--val_ratio` | `0.2` | Fraction of data for validation |
| `--no_cuda` | — | Use CPU only |

## Outputs

- **Training**
  - `out_dir/best.pt`: PyTorch checkpoint (model state + feature names + mean/std for normalization).
  - `out_dir/scaling.json`: Feature names and scaling parameters (mean, std) for use in MT5 or other runtimes.

- **Prediction**
  - With `--out path`: CSV with original columns plus `filter_prob` and `filter_approve` (1 if `filter_prob >= threshold`).

## Using the filter in MT5

- **Option A**: Export the model to ONNX from Python, then load it in MQL5 and run inference on the same feature vector (using the same scaling as in `scaling.json`).
- **Option B**: Run the Python script periodically or via a bridge: MT5 (or a log/file) provides the feature row; Python returns `filter_prob`; your EA opens only when above threshold.

Feature order and scaling must match exactly between training and MT5.
