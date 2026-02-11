# Task: Generate Comprehensive Project Documentation (Status: February 2026)

Please generate a single, comprehensive Markdown document named `NTCP_FULL_PROJECT_STATE.md` that captures the entire architecture, logic, and configuration of the Neural Trading Consultant Plattform (NTCP).

The document must include:

## 1. Project Overview & Hardware
- **Context:** Commercial High-End Trading System for Digistore24.
- **Hardware Stack:** AMD Ryzen 9 7950X3D (Multiprocessing) and NVIDIA RTX 5080 (CUDA).
- **Environment:** Windows 11 Pro / Ubuntu 22.04 VM, Python 3.10+, PyQt6, PyTorch.

## 2. Directory Structure & File Manifest
- A complete tree mapping of `MQL5/Experts/trade-commander/ntcp/`, including `/python/`, `/tools/`, and `/strategies/`.

## 3. The 'Trendcatcher' Strategy Module
- **Signal Logic:** Description of the MA-Fan (Price position, N-MA slopes, and ordering).
- **Exit Logic:** Detailed explanation of 'Option 2' (Fastest MA slope reversal).
- **Parameters:** Role of N (Signal depth) and how it's integrated.

## 4. DataManager & Feature Engineering
- **Dynamic MA Spectrum:** Logic for Min/Max MA, MA Count, and spacing.
- **Sequence Logic:** Explanation of Lookback K (Batch History) and how it forms the NN input.
- **Indicators:** List of all active features (MAs, Slopes, StdDev, Squeeze, Volume Z-Score, Hurst, Regime).
- **Stationarity:** Exact normalization formulas used for inputs.

## 5. ML Engine & Forward-Scan Labeling
- **Target Logic:** Multi-target horizons (3, 5, 10, 20, 40, 80).
- **Forward-Scan:** How the labels (Momentum, MFE, MAE) are calculated relative to $T_{exit}$ or the Horizon.
- **Model Architecture:** GRU-based multi-task network (Regression + Exit-Flag Classification).

## 6. The Bridge (scaling_params.mqh)
- Structure and purpose of the bridge file in the root directory.
- How Mean/StdDev and metadata (K, F, N, Spectrum) are exported.

## 7. GUI Cockpit Specification
- Description of the Dark Theme (#000000) and all current Tabs (Data, Features, Training, Backtest, Export).
- Implementation of the Trace Window and log mirroring.

Please scan all existing scripts to ensure the documentation reflects the EXACT current implementation.