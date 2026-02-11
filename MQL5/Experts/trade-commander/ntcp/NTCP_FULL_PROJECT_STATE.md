# NTCP — Neural Trading Consultant Platform
## Full Project State Document (February 2026)

---

## 1. Project Overview & Hardware

### 1.1 Context
NTCP is a **commercial high-end MLOps platform** for MetaTrader 5, designed for distribution via Digistore24. It trains multi-target GRU/LSTM neural networks on normalized M5 price data and exports both ONNX models and MQL5 header files (`scaling_params.mqh`) that enable the MT5 Expert Advisor to replicate the exact Python feature pipeline at inference time.

### 1.2 Hardware Stack
| Component | Specification |
|---|---|
| **CPU** | AMD Ryzen 9 7950X3D — 16C/32T, 3D V-Cache (preprocessing, multiprocessing) |
| **RAM** | 64 GB DDR5 |
| **GPU** | NVIDIA RTX 5080 — CUDA 12.8, used for PyTorch training with TF32 + AMP |
| **OS** | Windows 11 Pro (primary) / Ubuntu 22.04 VM (alternative) |

### 1.3 Software Environment
| Package | Version |
|---|---|
| Python | 3.13 |
| PyTorch | 2.10.0+cu128 |
| PyQt6 | 6.10.2 |
| pyqtgraph | 0.14.0 |
| scikit-learn | 1.8.0 |
| ONNX opset | 17 |

### 1.4 Windows DLL Fix (Critical)
On Windows, `import torch` **must** occur before `import PyQt6` to prevent a `c10.dll` crash. The entry point (`main.py`) enforces this by importing torch first and registering its lib directory via `os.add_dll_directory()`.

---

## 2. Directory Structure & File Manifest

```
ntcp/                                         # Project root
├── ntcp.mq5                                  # EA stub (empty OnInit/OnTick/OnDeinit)
├── ntcp.ex5                                  # Compiled EA binary
├── ntcp.mqproj                               # MetaEditor project file
│
├── tools/
│   ├── ntcp_exporter.mq5                     # MT5 data export script (WinAPI)
│   └── ntcp_exporter.ex5                     # Compiled exporter binary
│
├── docs/
│   ├── Technical_Core_and_ML_Strategy.md     # ML pipeline & strategy spec
│   ├── datamanager.md                        # DataManager implementation spec
│   ├── gui.md                                # GUI specification (German)
│   └── porject_roadmap.md                    # Commercial roadmap (Digistore24)
│
├── python/                                   # Python package root
│   ├── __init__.py                           # Package marker
│   ├── main.py                               # Entry point — launches PyQt6 GUI
│   ├── config.py                             # Constants, dataclasses, MA generator
│   ├── data_manager.py                       # Feature engineering & dataset pipeline
│   ├── model.py                              # GRU/LSTM neural network
│   ├── trainer.py                            # Training loop with AMP & early stopping
│   ├── backtest.py                           # Equity simulation & validation metrics
│   ├── exporter.py                           # ONNX + scaling_params.mqh generation
│   │
│   ├── strategies/
│   │   ├── __init__.py                       # Exports TrendcatcherStrategy
│   │   └── trendcatcher.py                   # Exit-aware target labeling
│   │
│   ├── gui/
│   │   ├── __init__.py                       # Package marker
│   │   ├── main_window.py                    # 5-tab cockpit (NTCPMainWindow)
│   │   ├── widgets.py                        # LogWidget, EquityCurveWidget, MetricsTable
│   │   ├── workers.py                        # QThread workers (Training, Backtest, Export)
│   │   └── style.qss                         # Pure-black dark theme stylesheet
│   │
│   ├── data/
│   │   └── NTCP_DATA_BTCUSD_M1.csv          # Sample OHLCV data (M1)
│   │
│   └── logs/
│       └── session_log.txt                   # Session log file (synced by LogWidget)
│
└── .claude/
    └── settings.local.json                   # Claude Code permissions
```

### File Count Summary
| Category | Count |
|---|---|
| Python source files | 12 |
| MQL5 source files (.mq5) | 2 |
| Compiled binaries (.ex5) | 2 |
| Stylesheet (.qss) | 1 |
| Documentation (.md) | 4 |
| Data / logs | 2 |
| **Total** | **23** |

---

## 3. The 'Trendcatcher' Strategy Module

**File:** `python/strategies/trendcatcher.py` — Class `TrendcatcherStrategy`

### 3.1 Signal Logic — MA-Fan Concept
The Trendcatcher strategy does **not** generate entry signals in the Python pipeline. Instead, it provides **exit-condition-aware target labels** that teach the neural network to understand when a trend is likely to end. The actual entry signal logic (MA-Fan: price position, N-MA slopes, and ordering) is intended for the MQL5 EA side.

The **Signal Depth N** (`signal_ma_count`, default=4) determines how many of the fastest MAs from the spectrum are considered "trigger-relevant" for entry decisions on the EA side. With the default spectrum `[5, 8, 13, 21, ...]`, the first 4 MAs (periods 5, 8, 13, 21) form the signal boundary.

### 3.2 Exit Logic — 'Option 2' (Fastest MA Slope Reversal)

**Method:** `compute_exit_bars(closes, ma_fastest_values) -> np.ndarray`

For each bar `T`, the exit distance is the number of bars until the **slope of the fastest MA** (period 5 by default) reverses sign:

```
1. Compute slope[i] = MA_fastest[i] - MA_fastest[i-1]   (via np.diff)
2. For each bar T:
   a. ref_sign = sign(slope[T])    (+1 if slope >= 0, else -1)
   b. Scan forward from T+1 up to T + 2*max(TARGET_HORIZONS)
   c. exit_dist[T] = first j where sign(slope[j]) != ref_sign
   d. If no reversal found: exit_dist[T] = 2*max(TARGET_HORIZONS) = 160
```

This exit distance is then used to compute **effective horizons**:
```
eff_h = min(H, exit_dist[T])     where H ∈ {3, 5, 10, 20, 40, 80}
```

### 3.3 Target Generation

**Method:** `generate_targets(closes, highs, lows, ma_fastest) -> (targets, names)`

For each horizon `H` and bar `T`, with `eff_h = max(1, min(H, exit_dist[T]))`:

| Target | Formula | Description |
|---|---|---|
| `tgt_mfe_{H}` | `max(highs[T+1 : T+1+eff_h]) / close[T] - 1.0` | Maximum Favorable Excursion |
| `tgt_mae_{H}` | `min(lows[T+1 : T+1+eff_h]) / close[T] - 1.0` | Maximum Adverse Excursion |
| `tgt_mom_{H}` | `close[T+eff_h] / close[T] - 1.0` | Momentum at effective horizon end |
| `tgt_exit_{H}` | `1.0 if exit_dist[T] <= H else 0.0` | Binary exit flag (classification) |

**Output shape:** `(n_bars, 24)` — 6 horizons × 4 targets per horizon.

### 3.4 Parameters
| Parameter | Default | Role |
|---|---|---|
| `signal_ma_count` | 4 | Number of fastest MAs used for signal boundary (EA side) |
| `TARGET_HORIZONS` | `[3, 5, 10, 20, 40, 80]` | Forward-scan horizon lengths (in M5 bars) |
| `ma_spectrum` | Dynamic (from config) | Full MA spectrum passed from DataManager |

---

## 4. DataManager & Feature Engineering

**File:** `python/data_manager.py` — Class `NTCPDataManager`

### 4.1 Dynamic MA Spectrum

**Function:** `generate_ma_spectrum(ma_min, ma_max, ma_count) -> list[int]`

Generates **logarithmically-spaced** MA periods:
```
period[i] = ma_min * exp(i / (ma_count - 1) * ln(ma_max / ma_min))
```
Results are rounded to unique integers with minimum value 2.

| Parameter | Default | Description |
|---|---|---|
| `ma_min` | 5 | Shortest MA period |
| `ma_max` | 500 | Longest MA period |
| `ma_count` | 11 | Number of MA periods |

**Default spectrum:** `[5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 500]` (matches Fibonacci-like spacing)

### 4.2 STF (Super-Timeframe) Developing Bar

Instead of loading a separate H1 CSV, the system aggregates M5 bars on-the-fly using a configurable `stf_factor`:

```
stf_factor = 12  →  12 M5 bars = 1 H1 bar
stf_factor = 6   →   6 M5 bars = 1 M30 bar
stf_factor = 24  →  24 M5 bars = 1 H2 bar
```

**Developing Bar Pattern (snapshot/rollback):**
- On `pos_in_group == 0` (first M5 bar of the STF group): **snapshot** — save indicator state
- On `pos_in_group > 0` (subsequent M5 bars): **rollback** to snapshot, then re-apply with updated developing bar OHLCV

This mirrors the `tcMA` class from `moving_average.mqh` in the MQL5 codebase.

### 4.3 Sequence Logic — Lookback K

The `NTCPDataset` creates sliding windows of shape `[lookback, num_features]`:

```
For index idx:
  seq = features[idx : idx + lookback]      # shape [K, F]
  tgt = targets[idx + lookback - 1]          # shape [T]
```

| Parameter | Default | Description |
|---|---|---|
| `lookback` (K) | 20 | Number of historical time steps per sample |
| Batch shape | `[B, K, F]` | B=batch, K=lookback, F=feature count |

### 4.4 OnlineIndicator — Python Port of tcMA

**Class:** `OnlineIndicator` — Windowed SMA + Welford online standard deviation

```
__slots__: length, count, window_sum, circ_buf, circ_idx, ma, m2, running_sum, std_dev
           + snapshot mirrors of all above
```

**Algorithm:**
1. **Windowed SMA:** Circular buffer of size `length`. On each update, subtract the oldest value and add the new one. `ma = window_sum / min(count+1, length)`.
2. **Welford's Online StdDev:** Tracks running mean and M2 (sum of squared deviations). `std_dev = sqrt(M2 / n)`.
3. **Snapshot/Rollback:** Before each update, either saves state (new bar) or restores state (developing bar re-computation).

### 4.5 OnlineIndicatorBank

**Class:** `OnlineIndicatorBank` — Groups all indicators for one timeframe:

| Component | Count | Description |
|---|---|---|
| MA indicators | 11 (configurable) | One `OnlineIndicator` per spectrum period |
| Volume indicator | 1 | `OnlineIndicator(TICK_SMA_PERIOD=20)` |
| RSI | 1 | Wilder's smoothing, period=14 |
| ATR | 1 | Classic ATR, period=14 |

Two banks exist per pipeline run: one for **M5** and one for **STF**.

### 4.6 Complete Feature List

The pipeline generates features in two phases: a **bar-by-bar loop** and a **post-loop** batch computation.

#### M5 Features (per-bar loop)
| Group Key | Features | Count | Normalization |
|---|---|---|---|
| `m5_ma` | `m5_ma_{p}` for each p in spectrum | 11 | `(MA / close) - 1.0` |
| `m5_sd` | `m5_sd_{p}` for each p in spectrum | 11 | `SD / close` |
| `m5_slope` | `m5_slope_{p}` for each p in spectrum | 11 | `(MA_cur - MA_prev) / close` |
| `m5_misc` | `m5_rsi`, `m5_atr`, `m5_vol_zscore` | 3 | RSI/100, ATR/close, (vol-SMA)/SD |
| `m5_candle` | `m5_body`, `m5_wick_upper`, `m5_wick_lower` | 3 | `/ (high - low + ε)` |
| `m5_range` | `m5_squeeze`, `m5_dist_day_high`, `m5_dist_day_low` | 3 | Relative to close |
| `m5_time` | `m5_hour_sin`, `m5_hour_cos`, `m5_dow_sin`, `m5_dow_cos` | 4 | `sin/cos(2π·t/T)` |

#### M5 Features (post-loop)
| Group Key | Features | Count | Method |
|---|---|---|---|
| `m5_hurst` | `m5_hurst` | 1 | Rolling R/S analysis, window=100 |
| `m5_regime` | `m5_regime_0` ... `m5_regime_3` | 4 | GMM one-hot, 4 clusters |

#### STF Features (per-bar loop)
| Group Key | Features | Count | Normalization |
|---|---|---|---|
| `stf_ma` | `stf_ma_{p}` for each p in spectrum | 11 | `(MA / m5_close) - 1.0` |
| `stf_sd` | `stf_sd_{p}` for each p in spectrum | 11 | `SD / m5_close` |
| `stf_slope` | `stf_slope_{p}` for each p in spectrum | 11 | `(MA_cur - MA_prev) / m5_close` |
| `stf_misc` | `stf_rsi`, `stf_atr`, `stf_vol_zscore`, `stf_close`, `stf_high`, `stf_low` | 6 | Mixed (RSI/100, rel. to close) |
| `stf_squeeze` | `stf_squeeze` | 1 | `(MA_fast - MA_slow) / close` |

#### STF Features (post-loop)
| Group Key | Features | Count | Method |
|---|---|---|---|
| `stf_hurst` | `stf_hurst` | 1 | Rolling R/S on developing-bar close |

#### Feature Count Summary
| Category | Count |
|---|---|
| M5 loop features | 46 |
| STF loop features | 40 |
| M5 post-loop (hurst) | 1 |
| M5 post-loop (regime) | 4 |
| STF post-loop (hurst) | 1 |
| **Total (all groups active)** | **92** |

### 4.7 Stationarity — Exact Normalization Formulas

| Feature | Formula | Ensures |
|---|---|---|
| Moving Averages | `(MA_n / Close_current) - 1.0` | Price-independent relative deviation |
| Standard Deviations | `SD_n / Close_current` | Scale-invariant volatility measure |
| RSI | `RSI / 100.0` | Bounded [0, 1] |
| ATR | `ATR_14 / Close_current` | Relative volatility |
| Candle Body | `(Close - Open) / (High - Low + ε)` | Range-normalized body size |
| Wick Upper | `(High - max(Open, Close)) / (High - Low + ε)` | Range-normalized upper shadow |
| Wick Lower | `(min(Open, Close) - Low) / (High - Low + ε)` | Range-normalized lower shadow |
| Squeeze | `(MA_fastest - MA_slowest) / (Close + ε)` | Relative MA spread |
| Daily High Distance | `(rolling_24h_high / Close) - 1.0` | Relative position |
| Daily Low Distance | `(rolling_24h_low / Close) - 1.0` | Relative position |
| Volume Z-Score | `(last_vol - SMA_20_vol) / (SD_vol + ε)` | Standardized volume |
| Hour | `sin(2π · hour/24)`, `cos(2π · hour/24)` | Cyclic encoding |
| Day of Week | `sin(2π · dow/7)`, `cos(2π · dow/7)` | Cyclic encoding |
| Hurst Exponent | R/S analysis: `log(R/S) / log(window)` | Mean-reversion vs trend measure |
| Market Regime | GMM one-hot (4 clusters on MA+SD features) | Discrete regime classification |
| STF OHLC | `(stf_value / m5_close) - 1.0` | Cross-timeframe relative position |

**Post-pipeline normalization:** Per-feature mean/std are computed over all valid (non-NaN) rows and stored in `scaling_params` for export to MQL5.

### 4.8 Hurst Exponent Computation

**Function:** `compute_hurst_exponent(series, window=100) -> np.ndarray`

Rolling R/S (Rescaled Range) analysis:
```
For each position i from window to n:
  1. segment = series[i-window : i]
  2. mean = mean(segment)
  3. cumdev = cumsum(segment - mean)
  4. R = max(cumdev) - min(cumdev)
  5. S = std(segment, ddof=1)
  6. H = log(R/S) / log(window)    if S > ε, else 0.5
```

Interpretation: H < 0.5 = mean-reverting, H = 0.5 = random walk, H > 0.5 = trending.

### 4.9 Market Regime Clustering

**Function:** `compute_market_regimes(feature_subset, n_clusters=4) -> (one_hot, centroids)`

- **Algorithm:** Gaussian Mixture Model (GMM) with diagonal covariance
- **Input:** M5 MA + SD feature columns (22 dimensions)
- **Output:** One-hot encoded cluster assignments `[n_bars, 4]` + centroid matrix `[4, 22]`
- **Parameters:** `max_iter=100`, `random_state=42`, `n_init=3`
- **Centroids** are exported to `scaling_params.mqh` for MQL5-side regime classification

### 4.10 Target Clipping

After the strategy generates raw targets, the DataManager clips at configurable percentiles (default: 1st/99th) to suppress news-spike outliers. Binary `tgt_exit_*` columns are excluded from clipping. Clip bounds are stored in `scaling_params["target_clip_bounds"]`.

### 4.11 Feature Group Filtering

The GUI allows selective activation of feature groups. When `active_feature_groups` is non-empty, only matching columns are retained. This enables ablation studies and feature importance analysis.

**Available groups:** `m5_ma`, `m5_sd`, `m5_slope`, `m5_misc`, `m5_candle`, `m5_range`, `m5_time`, `m5_hurst`, `m5_regime`, `stf_ma`, `stf_sd`, `stf_slope`, `stf_misc`, `stf_squeeze`, `stf_hurst`

---

## 5. ML Engine & Forward-Scan Labeling

### 5.1 Model Architecture

**File:** `python/model.py` — Class `NTCPModel(nn.Module)`

```
Input: [batch, lookback, features]     e.g. [256, 20, 92]
  │
  ▼
RNN (GRU or LSTM)
  ├── input_size = num_features (dynamic, set after feature engineering)
  ├── hidden_size = 128 (default)
  ├── num_layers = 2 (default)
  ├── dropout = 0.2 (between layers, only if num_layers > 1)
  └── batch_first = True
  │
  ▼
Take last time step: output[:, -1, :]     → [batch, hidden_size]
  │
  ▼
Dropout(0.2)
  │
  ▼
Linear(hidden_size, NUM_TARGETS)          → [batch, 18]
  │
  ▼
Output: [batch, 18]
```

**NUM_TARGETS** = `len(TARGET_HORIZONS) * 3 = 6 * 3 = 18`

Note: The model outputs 18 regression targets (MFE, MAE, Momentum per horizon). The 6 binary `tgt_exit_*` columns from the strategy are **not** included in the model's output — the model is trained on the first 18 columns only.

### 5.2 Target Layout

| Index | Target | Horizon |
|---|---|---|
| 0 | `tgt_mfe_3` | 3 bars |
| 1 | `tgt_mae_3` | 3 bars |
| 2 | `tgt_mom_3` | 3 bars |
| 3 | `tgt_mfe_5` | 5 bars |
| 4 | `tgt_mae_5` | 5 bars |
| 5 | `tgt_mom_5` | 5 bars |
| 6 | `tgt_mfe_10` | 10 bars |
| 7 | `tgt_mae_10` | 10 bars |
| 8 | `tgt_mom_10` | 10 bars |
| 9 | `tgt_mfe_20` | 20 bars |
| 10 | `tgt_mae_20` | 20 bars |
| 11 | `tgt_mom_20` | 20 bars |
| 12 | `tgt_mfe_40` | 40 bars |
| 13 | `tgt_mae_40` | 40 bars |
| 14 | `tgt_mom_40` | 40 bars |
| 15 | `tgt_mfe_80` | 80 bars |
| 16 | `tgt_mae_80` | 80 bars |
| 17 | `tgt_mom_80` | 80 bars |

### 5.3 Forward-Scan Labeling (Exit-Aware)

The TrendcatcherStrategy computes labels relative to either the **exit point** (fastest MA slope reversal) or the **nominal horizon**, whichever comes first:

```
For bar T, horizon H:
  1. exit_dist[T] = bars until fastest MA slope reverses sign
  2. eff_h = max(1, min(H, exit_dist[T]))
  3. MFE  = max(highs[T+1 : T+1+eff_h]) / close[T] - 1.0
  4. MAE  = min(lows[T+1 : T+1+eff_h])  / close[T] - 1.0
  5. Mom  = close[T+eff_h] / close[T] - 1.0
```

This means:
- If the trend exits **before** the horizon, targets reflect only the pre-exit portion
- If the trend survives **past** the horizon, targets reflect the full horizon
- The `tgt_exit_{H}` flag indicates which case applies (1.0 = early exit)

### 5.4 Training Loop

**File:** `python/trainer.py` — Class `NTCPTrainer`

| Feature | Implementation |
|---|---|
| **Loss function** | MSELoss (regression) |
| **Optimizer** | AdamW |
| **Learning rate** | 1e-3 default, ReduceLROnPlateau (patience=5, factor=0.5) |
| **Early stopping** | Patience=10 epochs on validation loss |
| **AMP** | `torch.amp.autocast("cuda")` + `GradScaler` when CUDA available |
| **TF32** | Enabled on Ampere+ GPUs (`matmul.allow_tf32`, `cudnn.allow_tf32`) |
| **Data split** | Chronological (no shuffle for val): last `val_split`% as validation |
| **DataLoader** | `num_workers=min(4, cpu_count)` when CUDA, `pin_memory=True`, `persistent_workers=True` |

### 5.5 STF Factor Range Training

**Method:** `train_factor_range()` — Iterates over STF factors `[stf_factor_min..stf_factor_max]` (default 6..24) and runs the full pipeline (data loading → feature engineering → training) for each factor independently. After all factors, selects the one with the lowest validation loss.

### 5.6 TrainingResult

**Class:** `TrainingResult` — Container for a single factor's training run:

| Field | Type | Description |
|---|---|---|
| `stf_factor` | int | The STF aggregation factor used |
| `train_losses` | list[float] | Per-epoch training losses |
| `val_losses` | list[float] | Per-epoch validation losses |
| `best_val_loss` | float | Best validation loss achieved |
| `best_epoch` | int | Epoch of best validation loss |
| `model_state` | dict | `state_dict()` of best model (CPU tensors) |
| `feature_cols` | list[str] | Ordered feature column names |
| `target_cols` | list[str] | Ordered target column names |
| `scaling_params` | dict | Feature means/stds, clip bounds, regime centroids |
| `num_features` | int | Feature dimensionality |
| `dataset_size` | int | Number of valid samples |

### 5.7 Backtest Engine

**File:** `python/backtest.py` — Class `NTCPBacktester`

**Strategy:** CRV (Chance-Risk Ratio) filter on the shortest horizon (3 bars):
```
For each sample i:
  predicted_crv = pred_mfe_3[i] / |pred_mae_3[i]|
  if predicted_crv >= crv_threshold (default 1.5):
    take trade, PnL = actual_momentum_3[i]
```

**Metrics computed:**
| Metric | Description |
|---|---|
| `equity_curve` | Cumulative PnL array |
| `profit_factor` | gross_profit / gross_loss |
| `max_drawdown` | Largest peak-to-trough decline |
| `win_rate` | Winning trades / total trades |
| `total_trades` | Number of trades taken |
| `per_target_r2` | R² per target column |
| `per_target_mse` | MSE per target column |
| `target_check` | Comprehensive per-target stats (R², MSE, MAE, Correlation, Pred/Actual mean/std, Ratio) |

---

## 6. The Bridge — `scaling_params.mqh`

**File:** `python/exporter.py` — Class `NTCPExporter`
**Output path:** `../scaling_params.mqh` (relative to `python/`, resolves to `ntcp/../scaling_params.mqh` = `trade-commander/scaling_params.mqh`)

### 6.1 Purpose
The MQH header file is the **Single Source of Truth** bridge between the Python training pipeline and the MQL5 Expert Advisor. It contains every constant the EA needs to replicate the exact feature pipeline at inference time.

### 6.2 Status
**Not yet generated** — the file is created by the Export tab's "Export ONNX + MQH" button after a model has been trained. No pre-existing `scaling_params.mqh` exists in the repository.

### 6.3 Generated Structure

```mql5
//+------------------------------------------------------------------+
//| NTCP Scaling Parameters v1.0.0
//| Auto-generated 2026-02-09 12:00:00
//| DO NOT EDIT — regenerate from Python exporter.
//+------------------------------------------------------------------+
#property strict

#define NTCP_VERSION "1.0.0"
#define NTCP_STF_FACTOR 12
#define NTCP_LOOKBACK 20
#define NTCP_NUM_FEATURES 92
#define NTCP_NUM_TARGETS 24
#define NTCP_TICK_SMA_PERIOD 20

// MA Spectrum
const int NTCP_MA_SPECTRUM[] = {5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 500};
#define NTCP_MA_COUNT 11

// Target Horizons
const int NTCP_TARGET_HORIZONS[] = {3, 5, 10, 20, 40, 80};
#define NTCP_HORIZON_COUNT 6

// Feature order (must match Python pipeline)
const string NTCP_FEATURE_NAMES[] = {
    "m5_ma_5",
    "m5_ma_8",
    ...
};

// Target names
const string NTCP_TARGET_NAMES[] = {
    "tgt_mfe_3",
    "tgt_mae_3",
    ...
};

// Target clipping bounds (1st/99th percentile)
const double NTCP_CLIP_TGT_MFE_3_LO = ...;
const double NTCP_CLIP_TGT_MFE_3_HI = ...;
...

// Per-feature normalization: mean values
const double NTCP_FEATURE_MEAN[] = {
    0.0012345678,  // m5_ma_5
    ...
};

// Per-feature normalization: standard deviation
const double NTCP_FEATURE_STD[] = {
    0.0045678901,  // m5_ma_5
    ...
};

// GMM regime centroids [cluster][feature]
#define NTCP_REGIME_CLUSTERS 4
#define NTCP_REGIME_FEATURES 22
const double NTCP_REGIME_CENTROIDS[][22] = {
    {0.001, 0.002, ...},
    {0.003, 0.004, ...},
    ...
};

// Active feature groups used during training
const string NTCP_ACTIVE_FEATURE_GROUPS[] = {"m5_ma", "m5_sd", ...};
```

### 6.4 Metadata Exported

| Section | Content |
|---|---|
| **Architecture** | Version, STF factor, lookback, feature/target counts |
| **Spectrum** | MA periods array, MA count |
| **Horizons** | Target horizon array, horizon count |
| **Feature Order** | Ordered string array (guarantees index alignment with ONNX input) |
| **Target Names** | Ordered string array (maps ONNX output indices to semantics) |
| **Clip Bounds** | Per-target lower/upper percentile bounds |
| **Normalization** | Per-feature mean and standard deviation arrays |
| **Regime Centroids** | GMM cluster centers for real-time regime classification |
| **Feature Groups** | Which groups were active during training |

### 6.5 ONNX Export

- **Format:** ONNX opset 17
- **Input:** `input` — shape `[batch_size, lookback, num_features]` (dynamic batch axis)
- **Output:** `output` — shape `[batch_size, NUM_TARGETS]`
- **Path:** Configurable, default `model.onnx`

---

## 7. GUI Cockpit Specification

**File:** `python/gui/main_window.py` — Class `NTCPMainWindow(QMainWindow)`

### 7.1 Theme
- **Background:** Pure black `#000000`
- **Font:** `Consolas` / `Courier New` monospace, 12px
- **Accent color:** `#4488ff` (selected tabs, focused inputs, checkboxes, progress bars)
- **Stylesheet:** `gui/style.qss` — 288 lines of pure QSS covering all Qt widgets
- **Start button:** Dark green (`#003300` bg, `#44cc44` text)
- **Stop button:** Dark red (`#330000` bg, `#cc4444` text)
- **Log widget:** Near-black `#050505`, 11px monospace

### 7.2 Layout Architecture

```
┌─────────────────────────────────────────────┐
│           QMainWindow (1000×700 min)        │
│ ┌─────────────────────────────────────────┐ │
│ │          QSplitter (Vertical)           │ │
│ │ ┌─────────────────────────────────────┐ │ │
│ │ │   QTabWidget (5 Tabs) — 500px       │ │ │
│ │ └─────────────────────────────────────┘ │ │
│ │ ┌─────────────────────────────────────┐ │ │
│ │ │   LogWidget (footer) — 200px        │ │ │
│ │ └─────────────────────────────────────┘ │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

### 7.3 Tab 1: Data & Split

| Widget | Type | Description |
|---|---|---|
| CSV Path | QLineEdit + Browse button | File picker for M5 CSV export |
| Load Data | QPushButton | Loads CSV, shows bar count and date range |
| Data Info | QLabel | `{bars} bars | {date_start} to {date_end}` |
| Val % | QSlider (5–40) | Validation split percentage, default 15% |
| Split Count | QLabel | `(~{train_bars} train / ~{val_bars} val)` |

### 7.4 Tab 2: Features & Factors

| Widget | Type | Description |
|---|---|---|
| Min Factor | QSpinBox (2–60) | STF factor range start, default 6 |
| Max Factor | QSpinBox (2–60) | STF factor range end, default 24 |
| Lookback | QSpinBox (5–100) | Sequence length K, default 20 |
| Feature Group Checkboxes | 17× QCheckBox | Toggle individual feature groups |
| MACD Fast/Slow/Signal | QSpinBox | MACD parameters (referenced but MACD features may be in development) |
| Hurst Window | QSpinBox (10–500) | Rolling window for Hurst computation, default 100 |
| Regime Clusters | QSpinBox (2–10) | Number of GMM clusters, default 4 |
| Rolling Window | QSpinBox (0–1M) | Data trimming window, 0 = no limit |

### 7.5 Tab 3: Training & Hardware

| Widget | Type | Description |
|---|---|---|
| CUDA checkbox | QCheckBox | Auto-detected, shows GPU name |
| Epochs | QSpinBox (1–1000) | Max training epochs, default 100 |
| Batch Size | QSpinBox (16–2048) | Mini-batch size, default 256 |
| Learning Rate | QDoubleSpinBox | 1e-5 to 0.1, default 0.001 |
| Hidden Size | QSpinBox (16–1024) | RNN hidden dimension, default 128 |
| Layers | QSpinBox (1–8) | RNN depth, default 2 |
| Dropout | QDoubleSpinBox (0–0.8) | Regularization, default 0.2 |
| Patience | QSpinBox (1–50) | Early stopping patience, default 10 |
| Cell Type | QRadioButton | GRU (default) or LSTM |
| Start Training | QPushButton (green) | Launches TrainingWorker QThread |
| Stop | QPushButton (red) | Requests graceful training stop |
| Progress | QProgressBar | Epoch progress 0–100% |

### 7.6 Tab 4: Backtest & Validation

| Widget | Type | Description |
|---|---|---|
| Factor selector | QComboBox | Populated after training with all factor results |
| CRV Threshold | QDoubleSpinBox (0.5–10) | MFE/MAE filter, default 1.5 |
| Run Backtest | QPushButton | Launches BacktestWorker QThread |
| Equity Curve | EquityCurveWidget (pyqtgraph) | Blue line on black bg, zero-line dashed |
| Metrics Table | MetricsTable | Key-value display (trades, WR, PF, DD, R²) |
| Target Check Table | QTableWidget (9 cols) | Per-target: R², MSE, MAE, Corr, Pred/Actual mean/std, Ratio |

### 7.7 Tab 5: Export

| Widget | Type | Description |
|---|---|---|
| Version | QLineEdit | Semantic version string, default "1.0.0" |
| ONNX Path | QLineEdit + Browse | Output path for ONNX model |
| MQH Path | QLineEdit + Browse | Output path for scaling_params.mqh |
| Export ONNX + MQH | QPushButton | Launches ExportWorker QThread |

### 7.8 Trace Window (LogWidget)

**File:** `python/gui/widgets.py` — Class `LogWidget(QWidget)`

The LogWidget serves as a persistent trace/log window that occupies the bottom portion of the main window (splitter, ~200px default height).

| Feature | Implementation |
|---|---|
| **Timestamps** | `[HH:MM:SS]` prefix on every message |
| **Color coding** | Info: `#aaaaaa`, Warn: `#ffaa44`, Error: `#ff4444` |
| **Auto-scroll** | `ensureCursorVisible()` after each append |
| **File sync** | Every message appended to `logs/session_log.txt` |
| **Clear button** | Clears the text area (does not clear the log file) |
| **Read-only** | `QTextEdit.setReadOnly(True)` |

**Log mirroring:** All workers (Training, Backtest, Export) emit `log_signal(str, str)` which connects to `LogWidget.log()`. The trainer's internal `_log()` method also writes to Python's `logging` module.

### 7.9 QThread Workers

**File:** `python/gui/workers.py`

| Worker | Signals | Function |
|---|---|---|
| `TrainingWorker` | `log_signal(msg, level)`, `progress_signal(epoch, total)`, `loss_signal(train, val)`, `finished_signal(list[TrainingResult])` | Runs `train_factor_range()` or `train_single_factor()` |
| `BacktestWorker` | `log_signal(msg, level)`, `finished_signal(BacktestMetrics)` | Runs `NTCPBacktester.run()` |
| `ExportWorker` | `log_signal(msg, level)`, `finished_signal(bool)` | Runs `NTCPExporter.export_onnx()` + `export_mqh()` |

---

## 8. MQL5 Data Exporter

**File:** `tools/ntcp_exporter.mq5`

### 8.1 Purpose
Exports chart data from MetaTrader 5 directly to the Python `data/` folder, bypassing the MT5 sandbox via WinAPI (`kernel32.dll`).

### 8.2 Implementation
- **WinAPI functions:** `CreateFileW`, `WriteFile`, `CloseHandle`, `CreateDirectoryW`
- **Output path:** `{TERMINAL_PATH}\MQL5\Experts\trade-commander\ntcp\python\data\NTCP_DATA_{SYMBOL}_{TF}.csv`
- **Format:** Comma-separated CSV with header: `Date,Open,High,Low,Close,TickVolume`
- **Chunking:** 1000 lines per `WriteFile` call for performance
- **UI:** Dark-themed button in chart corner ("Sync {SYMBOL} {TF} to Python")
- **Auto-sync:** Optional `InpAutoSync` flag to re-export on every new bar
- **Parameters:** `InpBarsToExport` (0 = all available), `InpAutoSync` (false)

### 8.3 Visual Feedback
- Idle: Dark button with play icon
- Exporting: Orange background
- Success: Green background with checkmark + bar count
- Failure: Red background with X icon
- Auto-resets after 3 seconds

---

## 9. Configuration Reference

### 9.1 DataConfig
```python
@dataclass
class DataConfig:
    m5_path: str | Path = ""
    stf_factor: int = 12           # M5 bars per STF bar
    lookback: int = 20             # Sequence length K
    clip_percentile: float = 1.0   # Target clipping (1st/99th)
    rolling_window_size: int = 0   # 0 = use all data
    hurst_window: int = 100        # R/S analysis window
    regime_n_clusters: int = 4     # GMM clusters
    ma_min: int = 5                # Shortest MA period
    ma_max: int = 500              # Longest MA period
    ma_count: int = 11             # Number of MA periods
    signal_ma_count: int = 4       # Signal boundary (EA side)
    strategy: str = "trendcatcher" # Strategy module name
    active_feature_groups: list[str] = []  # Empty = all groups
```

### 9.2 ModelConfig
```python
@dataclass
class ModelConfig:
    input_size: int = 0            # Set dynamically from features
    hidden_size: int = 128         # RNN hidden dimension
    num_layers: int = 2            # RNN depth
    dropout: float = 0.2           # Dropout rate
    cell_type: Literal["GRU", "LSTM"] = "GRU"
```

### 9.3 TrainConfig
```python
@dataclass
class TrainConfig:
    epochs: int = 100              # Max epochs
    batch_size: int = 256          # Mini-batch size
    lr: float = 1e-3               # Initial learning rate
    val_split: float = 0.15        # Validation fraction
    patience: int = 10             # Early stopping patience
    use_cuda: bool = True          # GPU acceleration
    stf_factor_min: int = 6        # Factor range start
    stf_factor_max: int = 24       # Factor range end
```

### 9.4 ExportConfig
```python
@dataclass
class ExportConfig:
    version: str = "1.0.0"         # Semantic version
    onnx_path: str | Path = "model.onnx"
    mqh_path: str | Path = "../scaling_params.mqh"
```

### 9.5 Global Constants
```python
MA_SPECTRUM = [5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 500]
TARGET_HORIZONS = [3, 5, 10, 20, 40, 80]
TICK_SMA_PERIOD = 20
EPSILON = 1e-9
RSI_PERIOD = 14
ATR_PERIOD = 14
DAY_BARS = 288       # M5 bars in 24 hours
HURST_WINDOW = 100
REGIME_N_CLUSTERS = 4
```

---

## 10. Execution & Run Instructions

### 10.1 Running the GUI
```bash
# From ntcp/ directory:
python -m python.main

# OR from ntcp/python/ directory:
python main.py
```

### 10.2 Typical Workflow
1. **MT5 side:** Attach `ntcp_exporter` EA to chart, click "Sync" button to export CSV
2. **Tab 1 (Data):** Browse and load the exported CSV, set validation split
3. **Tab 2 (Features):** Configure MA spectrum, STF factor range, enable/disable feature groups
4. **Tab 3 (Training):** Set hyperparameters, click "Start Training"
5. **Tab 4 (Backtest):** Select best factor, run backtest, inspect equity curve and per-target metrics
6. **Tab 5 (Export):** Export ONNX model + `scaling_params.mqh` for the EA

---

*Document auto-generated from source code scan — February 2026*
*NTCP v1.0.0 — Neural Trading Consultant Platform*
