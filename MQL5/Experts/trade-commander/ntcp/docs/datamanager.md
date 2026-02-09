# NTCP Implementation Spec: DataManager & Preprocessing Pipeline

## Role & Context
You are an expert ML Engineer and Python Developer. Your task is to implement the `DataManager` module for the NTCP (Neural Trading Consultant Plattform). This module is responsible for turning raw MT5 CSV exports into stationarity-normalized tensors for PyTorch.

## Technical Constraints
- **Language:** Python 3.10+
- **Style:** Clean, modular, production-ready. 
- **Documentation:** Self-explanatory code, English comments only.
- **Hardware Optimization:** Use Vectorized operations (NumPy/Pandas) for high-speed preprocessing.

## 1. Data Loading & Alignment (MTF)
- Load two CSV files: Primary (M5) and Anchor (H1).
- **Alignment:** Merge H1 data onto M5 timestamps. 
- **Anti-Look-Ahead:** Shift Anchor (H1) data by 1 interval (T-1) so that only "closed" candle information is available for the current M5 bar.

## 2. Normalization Protocol (Stationarity)
Implement the following transformations to ensure price-independence:

### A. Inputs (Features)
| Feature Group | Transformation Formula |
| :--- | :--- |
| **Moving Averages (MA)** | `(MA_n / Close) - 1.0` |
| **Standard Deviations (SD)** | `SD_n / Close` |
| **Oscillators (RSI, ADX)** | `Value / 100.0` |
| **Volatility (ATR)** | `ATR_14 / Close` |
| **Candle Body** | `(Close - Open) / (High - Low + 1e-9)` |
| **Wicks/Shadows** | `(High - Max(Open, Close)) / (High - Low + 1e-9)` |
| **Volume (Ticks)** | `Ticks / SMA(Ticks, 20)` |
| **Cyclic Time** | `sin/cos(2 * PI * hour / 24)` and `day / 7` |

### B. Targets (Labels)
Calculate for horizons $n \in \{3, 5, 10, 20\}$:
- **Momentum:** `(Price_T+n / Close_entry) - 1.0`
- **MFE (Max Run-up):** `(Max_High_in_n / Close_entry) - 1.0`
- **MAE (Max Run-down):** `(Min_Low_in_n / Close_entry) - 1.0`

## 3. Sequence Generation
- Create a Sliding Window / Sequence Generator.
- **Lookback (N):** 20 to 30 steps (parameterizable).
- **Output Shape:** `[Batch, Sequence_Length, Feature_Count]`.

## 4. Implementation Steps for Cursor
1. **Create `data_manager.py`:** Define a class `NTCPDataManager`.
2. **Method `load_and_sync()`:** Handle CSV import and H1-shifting.
3. **Method `normalize_features()`:** Apply the table logic above.
4. **Method `generate_pytorch_dataset()`:** Return a `torch.utils.data.Dataset` object.
5. **Validation:** Implement a small check to ensure no `NaN` values remain after normalization (handle division by zero).

## 5. Security & Commercial Logic
- Ensure the scaling factors (Mean/Std if used, or fixed bounds) can be exported to a JSON file for the MQL5 EA to ensure "Single Source of Truth".