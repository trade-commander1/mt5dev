"""
MAFilter NN App — 3-page layout: Data Manager, Backtest & Train NN, Final Backtest.
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QDoubleSpinBox,
    QComboBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QColor, QFont

from .app_config import get_config_path, load_config, save_config, get_default_config, APP_VERSION
from .backtest_stats import compute_backtest_metrics, compute_equity_curve
from .csv_loader import (
    load_ohlcv_csv,
    df_to_arrays,
    parse_timeframe,
    resample_ohlcv,
    split_by_day_offset,
)
from .equity_curve_widget import EquityCurveWidget
from .mafilter_engine import EAParams, run_backtest
from .symbol_config import SymbolConfig
from .trace_log import LogFile, TraceWidget
from .trainer import train_filter, load_filter_model, filter_trades_by_nn

# Standard sizes for consistent layout (all input fields same height; result columns spaced)
STANDARD_INPUT_HEIGHT = 30
INPUT_PADDING = 5
COLUMN_SPACING = 40


def _get_training_device_options() -> tuple[list[str], int]:
    """Return (list of device display strings, default index). CUDA devices first if available, then CPU."""
    try:
        import torch
        options = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                options.append(f"CUDA:{i} ({name})")
        options.append("CPU")
        default_index = 0 if (options and options[0].startswith("CUDA")) else len(options) - 1
        return options, default_index
    except Exception:
        return ["CPU"], 0


def _point_from_mintick(mintick: float) -> float:
    return mintick


def _scale_ea_params_for_timeframe(params: EAParams, scale_factor: float) -> EAParams:
    """
    Scale EA parameters for higher timeframes so the strategy can generate trades.
    On 5M/15M/1H, slope/laminar/bandwidth behave differently; we relax entry thresholds
    and reduce NH so historical averages are over a comparable time span.
    """
    if scale_factor <= 1.0:
        return params
    sqrt_s = math.sqrt(scale_factor)
    # Stronger scaling: laminar threshold down (easier to pass), BW factor up (allow more)
    min_laminar = max(0.15, params.min_laminar_level / scale_factor)
    max_bw = min(10.0, params.max_bandwidth_factor * scale_factor)
    # Slope strength: on higher TF, require smaller multiple of historical avg slope
    min_slope = max(0.1, params.min_slope_factor / sqrt_s)
    # NH: use fewer bars so "history" is comparable (e.g. 1000 5M bars = 3.5 days; 447 5M = 1.5 days)
    nh_scaled = max(100, int(params.nh / sqrt_s))
    return EAParams(
        lot_size=params.lot_size,
        min_len=params.min_len,
        max_len=params.max_len,
        nbr_ma=params.nbr_ma,
        min_slope_factor=min_slope,
        min_laminar_level=min_laminar,
        max_bandwidth_factor=max_bw,
        nh=nh_scaled,
        std_dev_factor=params.std_dev_factor,
        exit_option=params.exit_option,
        magic_number=params.magic_number,
    )


class Worker(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, str)  # percent 0-100, message
    equity_point = Signal(int, float)  # bar_index, cumulative_equity (for live curve)
    trace = Signal(str, str)  # level, message (for trace window)

    def __init__(
        self,
        csv_path: str,
        params: EAParams,
        symbol: SymbolConfig,
        good_threshold: float,
        epochs: int,
        batch_size: int,
        patience: int,
        out_dir: Optional[str],
        model_basename: Optional[str] = None,
        data_tf_str: str = "1M",
        system_tf_str: str = "1M",
        day_offset: int = 20,
        nn_type: str = "Dense (Feedforward)",
        architecture: str = "Medium",
        custom_hidden_sizes: Optional[tuple[int, ...]] = None,
        dropout: float = 0.2,
        use_dropout: bool = True,
        device: str = "CPU",
        do_training: bool = True,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.params = params
        self.symbol = symbol
        self.good_threshold = good_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.out_dir = out_dir
        self.model_basename = model_basename
        self.data_tf_str = data_tf_str
        self.system_tf_str = system_tf_str
        self.day_offset = day_offset
        self.nn_type = nn_type
        self.architecture = architecture
        self.custom_hidden_sizes = custom_hidden_sizes
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.device = device
        self.do_training = do_training
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._stop_requested = False
            self.progress.emit(0, "Loading CSV...")
            if self._stop_requested:
                self.error.emit("Stopped by user.")
                return
            df = load_ohlcv_csv(self.csv_path)
            if self._stop_requested:
                self.error.emit("Stopped by user.")
                return
            native_count = len(df)
            self.trace.emit("info", f"Native bars: {native_count}, period: {self.data_tf_str}")
            # Diagnostic: first 3 native bars
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                self.trace.emit(
                    "info",
                    f"  Native bar {i}: Time={row.get('DateTime', '')} O={row['Open']:.5f} H={row['High']:.5f} L={row['Low']:.5f} C={row['Close']:.5f} V={row.get('Volume', 0):.0f}",
                )
            data_tf_min = parse_timeframe(self.data_tf_str)
            system_tf_min = parse_timeframe(self.system_tf_str)
            if system_tf_min < data_tf_min:
                self.error.emit("Target bar period must be >= data file period.")
                return
            if system_tf_min % data_tf_min != 0:
                self.error.emit(
                    f"Target period ({self.system_tf_str}) must be an integer multiple of data file period ({self.data_tf_str})."
                )
                return
            if system_tf_min > data_tf_min:
                df = resample_ohlcv(df, data_tf_min, system_tf_min)
            if self._stop_requested:
                self.error.emit("Stopped by user.")
                return
            target_count = len(df)
            ratio = native_count / target_count if target_count else 0
            self.trace.emit("info", f"Target bars: {target_count}, period: {self.system_tf_str} (conversion ratio: {ratio:.1f}:1)")
            # Diagnostic: first 3 converted bars
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                self.trace.emit(
                    "info",
                    f"  Target bar {i}: Time={row.get('DateTime', '')} O={row['Open']:.5f} H={row['High']:.5f} L={row['Low']:.5f} C={row['Close']:.5f} V={row.get('Volume', 0):.0f}",
                )
            df_train, _df_app, _split_ts, _last = split_by_day_offset(df, self.day_offset)
            train_bars = len(df_train)
            self.trace.emit("info", f"Training part: {train_bars} bars (split by day offset)")
            if train_bars < 100:
                self.error.emit("Training part has too few bars after day offset split.")
                return
            bars_needed = self.params.nh + 2
            if train_bars < bars_needed:
                self.error.emit(
                    f"Training bars ({train_bars}) < required ({bars_needed}). "
                    "Use more data, lower NH (BW historical window), or a shorter target period."
                )
                return
            o, h, l, c, vol, spread = df_to_arrays(df_train)
            point = _point_from_mintick(self.symbol.mintick)
            scale_factor = system_tf_min / data_tf_min if data_tf_min else 1.0
            backtest_params = _scale_ea_params_for_timeframe(self.params, scale_factor)
            if scale_factor > 1.0:
                self.trace.emit(
                    "info",
                    f"Timeframe scaling ({self.system_tf_str}): MinLaminarLevel {self.params.min_laminar_level} -> {backtest_params.min_laminar_level:.3f}, "
                    f"MaxBandwidthFactor {self.params.max_bandwidth_factor} -> {backtest_params.max_bandwidth_factor:.3f}, "
                    f"MinSlopeFactor {self.params.min_slope_factor} -> {backtest_params.min_slope_factor:.3f}, NH {self.params.nh} -> {backtest_params.nh}",
                )
            self.progress.emit(0, "Backtesting...")

            def equity_cb(bar: int, cum: float) -> None:
                self.equity_point.emit(bar, cum)

            def stop_cb() -> bool:
                return self._stop_requested

            _total_eval_bars = max(1, train_bars - bars_needed)

            def backtest_progress_cb(current: int, total: int) -> None:
                if total <= 0:
                    return
                p = int((current + 1) / total * 50)
                self.progress.emit(min(p, 50), f"Backtesting... bar {current + 1}/{total}")

            trades = run_backtest(
                o, h, l, c,
                backtest_params,
                tick_value=self.symbol.tick_value,
                point=point,
                lot_size=self.symbol.default_lots,
                equity_callback=equity_cb,
                stop_callback=stop_cb,
                progress_callback=backtest_progress_cb,
            )
            if self._stop_requested:
                self.error.emit("Stopped by user.")
                return
            if len(trades) < 5:
                scale = system_tf_min // data_tf_min if data_tf_min else 1
                self.trace.emit(
                    "warning",
                    f"Too few trades ({len(trades)}). Training bars: {train_bars}, required: {bars_needed}. "
                    f"Target period: {self.system_tf_str}.",
                )
                if scale > 1:
                    self.trace.emit(
                        "warning",
                        "For higher timeframes (e.g. 5M, 15M), EA params tuned for 1M may be too strict. "
                        "Try: lower MinLaminarLevel (e.g. 0.4–0.6), higher MaxBandwidthFactor (e.g. 2.5–3.5), or use 1M.",
                    )
                self.error.emit(
                    f"Too few trades ({len(trades)}). Need more data or different params. "
                    f"(Training bars: {train_bars}, required for strategy: {bars_needed})"
                )
                return
            if not self.do_training:
                self.progress.emit(100, "Backtest done.")
                first_ts = df_train["DateTime"].iloc[0]
                last_ts = df["DateTime"].iloc[-1]
                self.finished.emit({
                    "n_bars": len(df_train),
                    "n_trades": len(trades),
                    "n_good": sum(1 for t in trades if t.profit > self.good_threshold),
                    "net_profit": sum(t.profit for t in trades if t.profit > 0) - abs(sum(t.profit for t in trades if t.profit <= 0)),
                    "net_loss": abs(sum(t.profit for t in trades if t.profit <= 0)),
                    "trades": trades,
                    "model": None,
                    "scaling": None,
                    "out_dir": self.out_dir,
                    "first_date": str(first_ts),
                    "split_date": str(_split_ts),
                    "last_date": str(last_ts),
                })
                return
            self.progress.emit(50, "Training filter...")
            self.trace.emit(
                "info",
                f"Training NN started: epochs={self.epochs}, batch_size={self.batch_size}, "
                f"device={self.device}, out_dir={self.out_dir or 'none'}",
            )

            def progress_cb(epoch: int, total: int) -> None:
                p = 50 + int(50 * epoch / total) if total else 50
                self.progress.emit(min(p, 99), f"Training... Epoch {epoch}/{total}")

            model, scaling, _ = train_filter(
                trades,
                good_threshold=self.good_threshold,
                epochs=self.epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                out_dir=Path(self.out_dir) if self.out_dir else None,
                model_basename=self.model_basename,
                stop_callback=stop_cb,
                progress_callback=progress_cb,
                nn_type=self.nn_type,
                architecture=self.architecture,
                custom_hidden_sizes=self.custom_hidden_sizes,
                dropout=self.dropout,
                use_dropout=self.use_dropout,
                device=self.device,
            )
            if self._stop_requested:
                self.error.emit("Stopped by user.")
                return
            self.progress.emit(100, "Done.")
            wins = [t.profit for t in trades if t.profit > 0]
            losses = [t.profit for t in trades if t.profit <= 0]
            net_profit_sum = sum(wins)
            net_loss_abs = abs(sum(losses))
            first_ts = df_train["DateTime"].iloc[0]
            last_ts = df["DateTime"].iloc[-1]
            result = {
                "n_bars": len(df_train),
                "n_trades": len(trades),
                "n_good": sum(1 for t in trades if t.profit > self.good_threshold),
                "net_profit": net_profit_sum,
                "net_loss": net_loss_abs,
                "trades": trades,
                "model": model,
                "scaling": scaling,
                "out_dir": self.out_dir,
                "first_date": str(first_ts),
                "split_date": str(_split_ts),
                "last_date": str(last_ts),
            }
            self.finished.emit(result)
        except Exception as e:
            tb = traceback.format_exc()
            self.trace.emit("error", f"Backtest/train error: {e}\n{tb}")
            self.error.emit(str(e))


class TrainOnlyWorker(QObject):
    """Run train_filter on existing backtest trades (no backtest). Enables testing multiple NN configs on same trades."""
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, str)
    trace = Signal(str, str)

    def __init__(
        self,
        trades: list,
        n_bars: int,
        good_threshold: float,
        epochs: int,
        batch_size: int,
        patience: int,
        out_dir: Optional[str],
        model_basename: Optional[str] = None,
        nn_type: str = "Dense (Feedforward)",
        architecture: str = "Medium",
        custom_hidden_sizes: Optional[tuple[int, ...]] = None,
        dropout: float = 0.2,
        use_dropout: bool = True,
        device: str = "CPU",
    ) -> None:
        super().__init__()
        self.trades = trades
        self.n_bars = n_bars
        self.good_threshold = good_threshold
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.out_dir = out_dir
        self.model_basename = model_basename
        self.nn_type = nn_type
        self.architecture = architecture
        self.custom_hidden_sizes = custom_hidden_sizes
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.device = device
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self._stop_requested = False
            self.progress.emit(0, "Training NN...")
            self.trace.emit(
                "info",
                f"Train NN (on cached backtest): epochs={self.epochs}, batch_size={self.batch_size}, "
                f"device={self.device}, out_dir={self.out_dir or 'none'}",
            )

            def stop_cb() -> bool:
                return self._stop_requested

            def progress_cb(epoch: int, total: int) -> None:
                if total <= 0:
                    return
                p = int(100 * (epoch + 1) / total)
                self.progress.emit(min(p, 99), f"Training... Epoch {epoch + 1}/{total}")

            model, scaling, _ = train_filter(
                self.trades,
                good_threshold=self.good_threshold,
                epochs=self.epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                out_dir=Path(self.out_dir) if self.out_dir else None,
                model_basename=self.model_basename,
                stop_callback=stop_cb,
                progress_callback=progress_cb,
                nn_type=self.nn_type,
                architecture=self.architecture,
                custom_hidden_sizes=self.custom_hidden_sizes,
                dropout=self.dropout,
                use_dropout=self.use_dropout,
                device=self.device,
            )
            if self._stop_requested:
                self.error.emit("Stopped by user.")
                return
            self.progress.emit(100, "Done.")
            self.finished.emit({
                "n_bars": self.n_bars,
                "n_trades": len(self.trades),
                "trades": self.trades,
                "model": model,
                "scaling": scaling,
                "out_dir": self.out_dir,
            })
        except Exception as e:
            tb = traceback.format_exc()
            self.trace.emit("error", f"Train NN error: {e}\n{tb}")
            self.error.emit(str(e))


class FinalBacktestWorker(QObject):
    """Run baseline (no filter) then filtered (with NN) on same test data; emit both for comparison."""
    finished = Signal(object)  # baseline_metrics, baseline_equity, baseline_trades, filtered_*, n_bars
    error = Signal(str)
    progress = Signal(int, str)
    equity_point = Signal(int, float)  # unused; we send full curves at end
    trace = Signal(str, str)

    def __init__(
        self,
        csv_path: str,
        params: EAParams,
        symbol: SymbolConfig,
        data_tf_str: str,
        system_tf_str: str,
        day_offset: int,
        model_dir: Optional[str],
        model_basename: Optional[str] = None,
        in_memory_model: Optional[tuple] = None,
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.params = params
        self.symbol = symbol
        self.data_tf_str = data_tf_str
        self.system_tf_str = system_tf_str
        self.day_offset = day_offset
        self.model_dir = model_dir
        self.model_basename = model_basename
        self.in_memory_model = in_memory_model
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            self.progress.emit(0, "Loading data...")
            df = load_ohlcv_csv(self.csv_path)
            data_tf_min = parse_timeframe(self.data_tf_str)
            system_tf_min = parse_timeframe(self.system_tf_str)
            if system_tf_min < data_tf_min:
                self.error.emit("System timeframe must be >= data file timeframe.")
                return
            if system_tf_min > data_tf_min:
                df = resample_ohlcv(df, data_tf_min, system_tf_min)
            _df_train, df_app, _split_ts, _last = split_by_day_offset(df, self.day_offset)
            if len(df_app) < 10:
                self.error.emit("Application part has too few bars.")
                return
            o, h, l, c, vol, spread = df_to_arrays(df_app)
            n_bars = len(c)
            point = _point_from_mintick(self.symbol.mintick)
            scale_factor = system_tf_min / data_tf_min if data_tf_min else 1.0
            backtest_params = _scale_ea_params_for_timeframe(self.params, scale_factor)

            def stop_cb() -> bool:
                return self._stop_requested

            # Phase 1: Baseline backtest (test data, NO filter) — 0–50%
            self.progress.emit(5, "Running Baseline Backtest (1/2)...")
            def baseline_progress(current: int, total: int) -> None:
                if total <= 0:
                    return
                pct = 5 + int(45 * (current + 1) / total)
                self.progress.emit(min(pct, 50), "Running Baseline Backtest (1/2)...")

            baseline_trades = run_backtest(
                o, h, l, c,
                backtest_params,
                tick_value=self.symbol.tick_value,
                point=point,
                lot_size=self.symbol.default_lots,
                equity_callback=None,
                stop_callback=stop_cb,
                progress_callback=baseline_progress,
            )
            if self._stop_requested:
                self.error.emit("Stopped by user.")
                return
            baseline_metrics = compute_backtest_metrics(baseline_trades, n_bars)
            baseline_equity = compute_equity_curve(baseline_trades, n_bars)

            # Phase 2: Filtered backtest (same test data, WITH NN) — 50–100%
            self.progress.emit(50, "Running Filtered Backtest (2/2)...")
            if self.in_memory_model is not None:
                model, scaling = self.in_memory_model
                filtered_trades = filter_trades_by_nn(baseline_trades, model, scaling)
                self.trace.emit("info", "Using NN model from memory (trained this session).")
            elif self.model_dir and Path(self.model_dir).exists():
                model, scaling = load_filter_model(Path(self.model_dir), self.model_basename)
                filtered_trades = filter_trades_by_nn(baseline_trades, model, scaling)
                self.trace.emit("info", f"NN model loaded from: {self.model_dir}")
            else:
                self.trace.emit(
                    "warning",
                    "No NN model: none in memory and Output directory is empty or path does not exist. "
                    "Filtered backtest = Baseline (results will be identical). "
                    "Run 'Backtest and Train NN' first, or set Output directory to a folder with a saved model.",
                )
                filtered_trades = baseline_trades
            def filtered_progress(current: int, total: int) -> None:
                if total <= 0:
                    return
                pct = 50 + int(50 * (current + 1) / total)
                self.progress.emit(min(pct, 99), "Running Filtered Backtest (2/2)...")
            # No second run_backtest; we already have baseline_trades and filtered_trades
            filtered_metrics = compute_backtest_metrics(filtered_trades, n_bars)
            filtered_equity = compute_equity_curve(filtered_trades, n_bars)

            self.progress.emit(100, "Final Backtest Complete")
            self.finished.emit({
                "baseline_metrics": baseline_metrics,
                "baseline_equity_curve": baseline_equity,
                "baseline_trades": baseline_trades,
                "filtered_metrics": filtered_metrics,
                "filtered_equity_curve": filtered_equity,
                "filtered_trades": filtered_trades,
                "n_bars": n_bars,
            })
        except Exception as e:
            self.trace.emit("error", f"Final backtest error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"MAFilter NN Trainer v{APP_VERSION}")
        # Width +10% for side-by-side comparison; height for full results
        self.setMinimumSize(946, 1200)
        self.resize(946, 1200)
        self._df = None
        self._trades = []
        self._worker: Optional[QThread] = None
        self._final_worker: Optional[QThread] = None
        self._log_file = LogFile(get_config_path().parent)
        self._training_backtest_metrics: Optional[dict] = None
        self._trained_model = None
        self._trained_scaling: Optional[dict] = None
        self._last_backtest_result: Optional[dict] = None
        self._train_worker: Optional[QThread] = None
        self._train_worker_obj: Optional[QObject] = None

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_tab_data_manager(), "Data Manager")
        self.tabs.addTab(self._build_tab_backtest_train(), "Backtest and Train NN")
        self.tabs.addTab(self._build_tab_final_backtest(), "Final Backtest")
        main_layout.addWidget(self.tabs)

        self.trace = TraceWidget()
        self.trace.setPlaceholderText("Trace — user actions and system events")
        main_layout.addWidget(QLabel("Trace"))
        main_layout.addWidget(self.trace)

        self._form_inputs = ()
        self._form_inputs_p3 = ()
        self._form_inputs_data = ()

        # Application-wide: standard input height and vertical text centering
        self.setStyleSheet("""
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                min-height: 30px;
                padding: 0px 5px;
            }
        """)

        # Load saved config after UI is built
        QTimer.singleShot(0, self._load_and_apply_config)

    def _load_and_apply_config(self) -> None:
        # Start log file so it exists for the whole session (trace + file)
        log_path = self._log_file.start()
        config, err = load_config()
        if err:
            self._log(f"Config: {err}, using defaults.", "info")
            self.trace.append("Config file corrupted or invalid, using defaults.", "warning")
        else:
            self._log(f"Configuration loaded from {get_config_path()}", "info")
        self._log(f"Log file: {log_path}", "info")
        self._apply_config(config)
        dev_text = self.device_combo.currentText()
        if dev_text.startswith("CUDA"):
            self._log(f"CUDA device detected: {dev_text}", "info")
        self._log(f"Device set to: {dev_text}", "info")

    def _build_config_from_ui(self) -> dict:
        """Build config dict from current UI state."""
        cfg = get_default_config()
        cfg["window"] = {
            "width": self.width(),
            "height": self.height(),
            "position_x": self.x(),
            "position_y": self.y(),
        }
        cfg["data_manager"] = {
            "last_data_file": self.csv_path_edit.text().strip(),
            "native_bar_period": self.data_file_timeframe.currentText(),
            "target_bar_period": self.system_timeframe.currentText(),
            "day_offset": self.day_offset_slider.value(),
            "symbol_name": self.symbol_name.text().strip(),
            "mintick": self.mintick.value(),
            "tick_value": self.tick_value.value(),
            "default_lots": self.default_lots.value(),
        }
        cfg["backtest_train"] = {
            "ea_parameters": {
                "lot_size": self.lot_size.value(),
                "min_len": self.min_len.value(),
                "max_len": self.max_len.value(),
                "nbr_ma": self.nbr_ma.value(),
                "min_slope_factor": self.min_slope_factor.value(),
                "min_laminar_level": self.min_laminar_level.value(),
                "max_bandwidth_factor": self.max_bandwidth_factor.value(),
                "nh": self.nh.value(),
                "std_dev_factor": self.std_dev_factor.value(),
                "exit_option": self.exit_option.currentIndex(),
                "magic_number": self.magic_number.value(),
            },
            "training": {
                "good_threshold": self.good_threshold.value(),
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "patience": self.patience_spin.value(),
            },
            "nn_config": {
                "nn_type": self.nn_type_combo.currentText(),
                "architecture": self.arch_combo.currentText(),
                "custom_arch": self.custom_arch_check.isChecked(),
                "custom_layers": self.custom_layers_spin.value(),
                "custom_sizes": self.custom_sizes_edit.text().strip(),
                "dropout": self.dropout_slider.value() / 100.0,
                "use_dropout": self.use_dropout_check.isChecked(),
            },
            "lookback_config": {
                "enabled": self.lookback_enable_check.isChecked(),
                "length": self.lookback_length_slider.value(),
                "include_ohlcv": self.lookback_ohlc_check.isChecked(),
                "include_laminar": self.lookback_laminar_check.isChecked(),
                "include_bw": self.lookback_bw_check.isChecked(),
                "include_stddev": self.lookback_stddev_check.isChecked(),
                "include_slopes": self.lookback_slopes_check.isChecked(),
                "normalization": self.norm_method_combo.currentText(),
            },
            "output_dir": self.out_dir_edit.text().strip(),
            "device": self.device_combo.currentText(),
        }
        return cfg

    def _apply_config(self, config: dict) -> None:
        """Apply loaded config to UI. Handles missing/invalid values."""
        # Window
        if "window" in config:
            w = config["window"]
            width = w.get("width", 946)
            height = w.get("height", 1200)
            self.resize(width, height)
            if "position_x" in w and "position_y" in w:
                self.move(w["position_x"], w["position_y"])

        # Data Manager
        if "data_manager" in config:
            dm = config["data_manager"]
            path = dm.get("last_data_file", "")
            if path and Path(path).exists():
                self.csv_path_edit.setText(path)
            elif path:
                self._log(f"Previous data file not found: {path}", "info")
            idx = self.data_file_timeframe.findText(dm.get("native_bar_period", "1M"))
            if idx >= 0:
                self.data_file_timeframe.setCurrentIndex(idx)
            idx = self.system_timeframe.findText(dm.get("target_bar_period", "1M"))
            if idx >= 0:
                self.system_timeframe.setCurrentIndex(idx)
            self.day_offset_slider.setValue(max(1, min(100, dm.get("day_offset", 20))))
            if dm.get("symbol_name") is not None:
                self.symbol_name.setText(dm.get("symbol_name", ""))
            if "mintick" in dm:
                self.mintick.setValue(float(dm["mintick"]))
            if "tick_value" in dm:
                self.tick_value.setValue(float(dm["tick_value"]))
            if "default_lots" in dm:
                self.default_lots.setValue(float(dm["default_lots"]))

        # Backtest and Train
        if "backtest_train" in config:
            bt = config["backtest_train"]
            ea = bt.get("ea_parameters", {})
            if ea:
                self.lot_size.setValue(float(ea.get("lot_size", 0.1)))
                self.min_len.setValue(int(ea.get("min_len", 5)))
                self.max_len.setValue(int(ea.get("max_len", 21)))
                self.nbr_ma.setValue(int(ea.get("nbr_ma", 4)))
                self.min_slope_factor.setValue(float(ea.get("min_slope_factor", 1.0)))
                self.min_laminar_level.setValue(float(ea.get("min_laminar_level", 0.8)))
                self.max_bandwidth_factor.setValue(float(ea.get("max_bandwidth_factor", 1.0)))
                self.nh.setValue(int(ea.get("nh", 1000)))
                self.std_dev_factor.setValue(float(ea.get("std_dev_factor", 1.0)))
                self.exit_option.setCurrentIndex(int(ea.get("exit_option", 0)))
                self.magic_number.setValue(int(ea.get("magic_number", 20260210)))
                self._sync_ea_p2_to_p3()
            tr = bt.get("training", {})
            if tr:
                self.good_threshold.setValue(float(tr.get("good_threshold", 0.0)))
                self.epochs_spin.setValue(int(tr.get("epochs", 100)))
                self.batch_spin.setValue(int(tr.get("batch_size", 64)))
                self.patience_spin.setValue(int(tr.get("patience", 15)))
            nn = bt.get("nn_config", {})
            if nn:
                idx = self.nn_type_combo.findText(nn.get("nn_type", "Dense (Feedforward)"))
                if idx >= 0:
                    self.nn_type_combo.setCurrentIndex(idx)
                idx = self.arch_combo.findText(nn.get("architecture", "Medium"))
                if idx >= 0:
                    self.arch_combo.setCurrentIndex(idx)
                self.custom_arch_check.setChecked(bool(nn.get("custom_arch", False)))
                self.custom_layers_spin.setValue(int(nn.get("custom_layers", 3)))
                self.custom_sizes_edit.setText(str(nn.get("custom_sizes", "256,128,64")))
                drop = float(nn.get("dropout", 0.2))
                self.dropout_slider.setValue(int(round(drop * 100)))
                self.use_dropout_check.setChecked(bool(nn.get("use_dropout", True)))
            lb = bt.get("lookback_config", {})
            if lb:
                self.lookback_enable_check.setChecked(bool(lb.get("enabled", True)))
                self.lookback_length_slider.setValue(max(10, min(200, int(lb.get("length", 50)))))
                self.lookback_ohlc_check.setChecked(bool(lb.get("include_ohlcv", True)))
                self.lookback_laminar_check.setChecked(bool(lb.get("include_laminar", False)))
                self.lookback_bw_check.setChecked(bool(lb.get("include_bw", False)))
                self.lookback_stddev_check.setChecked(bool(lb.get("include_stddev", False)))
                self.lookback_slopes_check.setChecked(bool(lb.get("include_slopes", False)))
                idx = self.norm_method_combo.findText(lb.get("normalization", "Dynamic (per window)"))
                if idx >= 0:
                    self.norm_method_combo.setCurrentIndex(idx)
            default_output_dir = str(get_config_path().parent / "models")
            self.out_dir_edit.setText(bt.get("output_dir", "") or default_output_dir)
            if "device" in bt:
                idx = self.device_combo.findText(bt["device"])
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)

    def closeEvent(self, event) -> None:
        """Save config on exit."""
        try:
            cfg = self._build_config_from_ui()
            save_config(cfg)
            self._log(f"Configuration saved to {get_config_path()}", "info")
        except Exception as e:
            self._log(f"Failed to save config: {e}", "error")
        super().closeEvent(event)

    def _build_ea_params_group(self, parent: Optional[QWidget] = None) -> tuple[QGroupBox, dict]:
        """Build EA Parameters group with identical layout/size/fonts. Returns (group_box, widgets_dict)."""
        box = QGroupBox("EA Parameters")
        if parent:
            box.setParent(parent)
        layout = QFormLayout()
        layout.setVerticalSpacing(12)

        def add_row(label: str, widget: QWidget) -> None:
            widget.setFixedHeight(STANDARD_INPUT_HEIGHT)
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
            layout.addRow(label, widget)

        lot_size = QDoubleSpinBox()
        lot_size.setRange(0.01, 100)
        lot_size.setValue(0.1)
        lot_size.setDecimals(2)
        add_row("Lot size", lot_size)
        min_len = QSpinBox()
        min_len.setRange(2, 1000)
        min_len.setValue(5)
        add_row("MinLen (shortest MA)", min_len)
        max_len = QSpinBox()
        max_len.setRange(2, 10000)
        max_len.setValue(21)
        add_row("MaxLen (longest MA)", max_len)
        nbr_ma = QSpinBox()
        nbr_ma.setRange(2, 20)
        nbr_ma.setValue(4)
        add_row("NbrMa (number of MAs)", nbr_ma)
        min_slope_factor = QDoubleSpinBox()
        min_slope_factor.setRange(0.01, 10)
        min_slope_factor.setValue(1.0)
        min_slope_factor.setSingleStep(0.1)
        add_row("MinSlopeFactor", min_slope_factor)
        min_laminar_level = QDoubleSpinBox()
        min_laminar_level.setRange(0.01, 1.0)
        min_laminar_level.setValue(0.8)
        min_laminar_level.setSingleStep(0.05)
        add_row("MinLaminarLevel", min_laminar_level)
        max_bandwidth_factor = QDoubleSpinBox()
        max_bandwidth_factor.setRange(0.1, 10)
        max_bandwidth_factor.setValue(1.0)
        add_row("MaxBandwidthFactor", max_bandwidth_factor)
        nh = QSpinBox()
        nh.setRange(10, 50000)
        nh.setValue(1000)
        add_row("NH (BW historical window)", nh)
        std_dev_factor = QDoubleSpinBox()
        std_dev_factor.setRange(0.1, 5)
        std_dev_factor.setValue(1.0)
        add_row("StdDevFactor", std_dev_factor)
        exit_option = QComboBox()
        exit_option.addItems(["0 = StdDev exit", "1 = Slope exit"])
        add_row("ExitOption", exit_option)
        magic_number = QSpinBox()
        magic_number.setRange(0, 2**31 - 1)
        magic_number.setValue(20260210)
        add_row("MagicNumber", magic_number)
        box.setLayout(layout)
        widgets = {
            "lot_size": lot_size, "min_len": min_len, "max_len": max_len, "nbr_ma": nbr_ma,
            "min_slope_factor": min_slope_factor, "min_laminar_level": min_laminar_level,
            "max_bandwidth_factor": max_bandwidth_factor, "nh": nh, "std_dev_factor": std_dev_factor,
            "exit_option": exit_option, "magic_number": magic_number,
        }
        return box, widgets

    def _build_tab_data_manager(self) -> QWidget:
        """Page 1: Data Manager — file, native/target TF, day offset, dates, symbol."""
        w = QWidget()
        layout = QVBoxLayout(w)
        row = QHBoxLayout()
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("Path to CSV (DateTime,Open,High,Low,Close,Volume,Spread)")
        row.addWidget(self.csv_path_edit)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_csv)
        row.addWidget(browse_btn)
        layout.addLayout(row)
        tf_row = QFormLayout()
        self.data_file_timeframe = QComboBox()
        self.data_file_timeframe.addItems(["1M", "5M", "15M", "30M", "1H", "4H", "1D"])
        tf_row.addRow("Native bar period (source file)", self.data_file_timeframe)
        self.system_timeframe = QComboBox()
        self.system_timeframe.addItems(["1M", "5M", "15M", "30M", "1H", "4H", "1D"])
        tf_row.addRow("Target bar period (backtest/training)", self.system_timeframe)
        layout.addLayout(tf_row)
        offset_row = QFormLayout()
        self.day_offset_slider = QSlider(Qt.Orientation.Horizontal)
        self.day_offset_slider.setRange(1, 100)
        self.day_offset_slider.setValue(20)
        self.day_offset_label = QLabel("20 days")
        self.day_offset_slider.valueChanged.connect(lambda v: self.day_offset_label.setText(f"{v} days"))
        offset_row.addRow("Day offset from end", self.day_offset_slider)
        offset_row.addRow("", self.day_offset_label)
        layout.addLayout(offset_row)
        self.date_first_label = QLabel("First date: —")
        self.date_split_label = QLabel("Split date: —")
        self.date_last_label = QLabel("Last date: —")
        layout.addWidget(self.date_first_label)
        layout.addWidget(self.date_split_label)
        layout.addWidget(self.date_last_label)
        sym_group = QGroupBox("Symbol")
        sym_layout = QFormLayout()
        self.symbol_name = QLineEdit()
        self.symbol_name.setPlaceholderText("e.g. EURUSD")
        sym_layout.addRow("Symbol name", self.symbol_name)
        self.mintick = QDoubleSpinBox()
        self.mintick.setDecimals(5)
        self.mintick.setRange(0.00001, 1)
        self.mintick.setValue(0.01)
        sym_layout.addRow("Mintick", self.mintick)
        self.tick_value = QDoubleSpinBox()
        self.tick_value.setDecimals(4)
        self.tick_value.setRange(0.0001, 100000)
        self.tick_value.setValue(1.0)
        sym_layout.addRow("Tick value", self.tick_value)
        self.default_lots = QDoubleSpinBox()
        self.default_lots.setDecimals(2)
        self.default_lots.setRange(0.01, 100)
        self.default_lots.setValue(0.1)
        sym_layout.addRow("Default order size (lots)", self.default_lots)
        sym_group.setLayout(sym_layout)
        layout.addWidget(sym_group)
        layout.addStretch()
        self._form_inputs_data = (
            self.csv_path_edit,
            self.data_file_timeframe,
            self.system_timeframe,
            self.day_offset_slider,
            self.symbol_name,
            self.mintick,
            self.tick_value,
            self.default_lots,
        )
        return w

    def _build_tab_backtest_train(self) -> QWidget:
        """Page 2: EA Parameters, Training params, Start/Stop, progress, equity curve."""
        w = QWidget()
        layout = QVBoxLayout(w)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # ——— EA Parameters (shared builder for identical layout with Page 3) ———
        ea_group, ea_widgets = self._build_ea_params_group(scroll_content)
        self.lot_size = ea_widgets["lot_size"]
        self.min_len = ea_widgets["min_len"]
        self.max_len = ea_widgets["max_len"]
        self.nbr_ma = ea_widgets["nbr_ma"]
        self.min_slope_factor = ea_widgets["min_slope_factor"]
        self.min_laminar_level = ea_widgets["min_laminar_level"]
        self.max_bandwidth_factor = ea_widgets["max_bandwidth_factor"]
        self.nh = ea_widgets["nh"]
        self.std_dev_factor = ea_widgets["std_dev_factor"]
        self.exit_option = ea_widgets["exit_option"]
        self.magic_number = ea_widgets["magic_number"]
        scroll_layout.addWidget(ea_group)

        # ——— Training ———
        train_group = QGroupBox("Training")
        train_layout = QFormLayout()
        self.good_threshold = QDoubleSpinBox()
        self.good_threshold.setRange(-100000, 100000)
        self.good_threshold.setValue(0.0)
        self.good_threshold.setDecimals(2)
        train_layout.addRow("Good trade threshold (profit >)", self.good_threshold)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 2000)
        self.epochs_spin.setValue(100)
        train_layout.addRow("Epochs", self.epochs_spin)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(4, 512)
        self.batch_spin.setValue(64)
        train_layout.addRow("Batch size", self.batch_spin)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 50)
        self.patience_spin.setValue(15)
        train_layout.addRow("Early stopping patience", self.patience_spin)
        train_group.setLayout(train_layout)
        scroll_layout.addWidget(train_group)

        # ——— Neural Network Settings ———
        nn_group = QGroupBox("Neural Network Settings")
        nn_layout = QFormLayout()
        nn_layout.setVerticalSpacing(6)
        self.nn_type_combo = QComboBox()
        self.nn_type_combo.addItems(["Dense (Feedforward)", "LSTM", "GRU"])
        nn_layout.addRow("NN type", self.nn_type_combo)
        self.arch_combo = QComboBox()
        self.arch_combo.addItems(["Small", "Medium", "Large"])
        nn_layout.addRow("Architecture", self.arch_combo)
        self.custom_arch_check = QCheckBox("Use custom architecture")
        self.custom_arch_check.setChecked(False)
        nn_layout.addRow("", self.custom_arch_check)
        self.custom_layers_spin = QSpinBox()
        self.custom_layers_spin.setRange(1, 10)
        self.custom_layers_spin.setValue(3)
        nn_layout.addRow("Custom: layers", self.custom_layers_spin)
        self.custom_sizes_edit = QLineEdit()
        self.custom_sizes_edit.setPlaceholderText("e.g. 256,128,64")
        self.custom_sizes_edit.setText("256,128,64")
        nn_layout.addRow("Custom: layer sizes", self.custom_sizes_edit)
        self.dropout_slider = QSlider(Qt.Orientation.Horizontal)
        self.dropout_slider.setRange(0, 50)
        self.dropout_slider.setValue(20)
        self.dropout_label = QLabel("0.20")
        self.dropout_slider.valueChanged.connect(lambda v: self.dropout_label.setText(f"{v/100:.2f}"))
        nn_layout.addRow("Dropout", self.dropout_slider)
        nn_layout.addRow("", self.dropout_label)
        self.use_dropout_check = QCheckBox("Use dropout")
        self.use_dropout_check.setChecked(True)
        nn_layout.addRow("", self.use_dropout_check)
        nn_group.setLayout(nn_layout)
        scroll_layout.addWidget(nn_group)

        # ——— Training Device ———
        device_group = QGroupBox("Training Device")
        device_layout = QFormLayout()
        self.device_combo = QComboBox()
        _device_opts, _device_default = _get_training_device_options()
        self.device_combo.addItems(_device_opts)
        self.device_combo.setCurrentIndex(_device_default)
        device_layout.addRow("Compute Device:", self.device_combo)
        device_group.setLayout(device_layout)
        scroll_layout.addWidget(device_group)

        # ——— Lookback Window Settings ———
        lb_group = QGroupBox("Lookback Window Settings")
        lb_layout = QFormLayout()
        lb_layout.setVerticalSpacing(6)
        self.lookback_enable_check = QCheckBox("Enable lookback window")
        self.lookback_enable_check.setChecked(True)
        lb_layout.addRow("", self.lookback_enable_check)
        self.lookback_length_slider = QSlider(Qt.Orientation.Horizontal)
        self.lookback_length_slider.setRange(10, 200)
        self.lookback_length_slider.setValue(50)
        self.lookback_length_label = QLabel("50 bars")
        self.lookback_length_slider.valueChanged.connect(lambda v: self.lookback_length_label.setText(f"{v} bars"))
        lb_layout.addRow("Length (bars)", self.lookback_length_slider)
        lb_layout.addRow("", self.lookback_length_label)
        self.lookback_ohlc_check = QCheckBox("OHLC + Volume (required when lookback enabled)")
        self.lookback_ohlc_check.setChecked(True)
        lb_layout.addRow("", self.lookback_ohlc_check)
        self.lookback_laminar_check = QCheckBox("laminar_level in lookback")
        self.lookback_laminar_check.setChecked(False)
        lb_layout.addRow("", self.lookback_laminar_check)
        self.lookback_bw_check = QCheckBox("bw_factor in lookback")
        self.lookback_bw_check.setChecked(False)
        lb_layout.addRow("", self.lookback_bw_check)
        self.lookback_stddev_check = QCheckBox("std_dev_ratio in lookback")
        self.lookback_stddev_check.setChecked(False)
        lb_layout.addRow("", self.lookback_stddev_check)
        self.lookback_slopes_check = QCheckBox("All slope ratios in lookback")
        self.lookback_slopes_check.setChecked(False)
        lb_layout.addRow("", self.lookback_slopes_check)
        self.norm_method_combo = QComboBox()
        self.norm_method_combo.addItems(["Dynamic (per window)", "Global (entire dataset)"])
        lb_layout.addRow("Normalization", self.norm_method_combo)
        lb_group.setLayout(lb_layout)
        scroll_layout.addWidget(lb_group)

        self.out_dir_edit = QLineEdit()
        default_models_dir = str(get_config_path().parent / "models")
        self.out_dir_edit.setPlaceholderText(f"Model saved as model_type_MAFilter.pt")
        self.out_dir_edit.setText(default_models_dir)
        out_row = QFormLayout()
        out_row.addRow("Output directory", self.out_dir_edit)
        scroll_layout.addLayout(out_row)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        self.backtest_btn = QPushButton("Backtest")
        self.backtest_btn.setStyleSheet("background-color: #1E90FF; color: white;")
        self.backtest_btn.clicked.connect(self._run_backtest)
        btn_row.addWidget(self.backtest_btn)
        self.train_nn_btn = QPushButton("Train NN")
        self.train_nn_btn.setStyleSheet("background-color: #228B22; color: white;")
        self.train_nn_btn.setEnabled(False)
        self.train_nn_btn.clicked.connect(self._run_train_nn)
        btn_row.addWidget(self.train_nn_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: #8B0000; color: white;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_run)
        btn_row.addWidget(self.stop_btn)
        layout.addLayout(btn_row)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addWidget(QLabel("Live equity curve"))
        equity_frame_train = QFrame()
        equity_frame_train.setStyleSheet("background-color: #000000;")
        equity_frame_train.setMinimumHeight(180)
        self.equity_curve_train = EquityCurveWidget(equity_frame_train)
        train_equity_layout = QVBoxLayout(equity_frame_train)
        train_equity_layout.setContentsMargins(0, 0, 0, 0)
        train_equity_layout.addWidget(self.equity_curve_train)
        layout.addWidget(equity_frame_train)

        self._form_inputs = (
            self.lot_size, self.min_len, self.max_len, self.nbr_ma,
            self.min_slope_factor, self.min_laminar_level, self.max_bandwidth_factor,
            self.nh, self.std_dev_factor, self.exit_option, self.magic_number,
            self.good_threshold, self.epochs_spin, self.batch_spin, self.patience_spin,
            self.nn_type_combo, self.arch_combo, self.custom_layers_spin, self.custom_sizes_edit,
            self.dropout_slider, self.device_combo, self.out_dir_edit,
            self.lookback_length_slider, self.norm_method_combo,
        )
        return w

    def _build_tab_final_backtest(self) -> QWidget:
        """Page 3: EA params (synced), Start Final Backtest, equity curve, results panel."""
        w = QWidget()
        layout = QVBoxLayout(w)
        # EA Parameters: same widget builder as Page 2 for pixel-perfect consistency
        ea_group_p3, ea_widgets_p3 = self._build_ea_params_group(w)
        ea_group_p3.setTitle("EA Parameters (synced with Backtest and Train NN)")
        self.lot_size_p3 = ea_widgets_p3["lot_size"]
        self.min_len_p3 = ea_widgets_p3["min_len"]
        self.max_len_p3 = ea_widgets_p3["max_len"]
        self.nbr_ma_p3 = ea_widgets_p3["nbr_ma"]
        self.min_slope_factor_p3 = ea_widgets_p3["min_slope_factor"]
        self.min_laminar_level_p3 = ea_widgets_p3["min_laminar_level"]
        self.max_bandwidth_factor_p3 = ea_widgets_p3["max_bandwidth_factor"]
        self.nh_p3 = ea_widgets_p3["nh"]
        self.std_dev_factor_p3 = ea_widgets_p3["std_dev_factor"]
        self.exit_option_p3 = ea_widgets_p3["exit_option"]
        self.magic_number_p3 = ea_widgets_p3["magic_number"]
        layout.addWidget(ea_group_p3)

        btn_row = QHBoxLayout()
        self.run_final_btn = QPushButton("Start Final Backtest")
        self.run_final_btn.setStyleSheet("background-color: #1E90FF; color: white;")
        self.run_final_btn.clicked.connect(self._run_final_backtest)
        btn_row.addWidget(self.run_final_btn)
        self.stop_final_btn = QPushButton("Stop")
        self.stop_final_btn.setStyleSheet("background-color: #8B0000; color: white;")
        self.stop_final_btn.setEnabled(False)
        self.stop_final_btn.clicked.connect(self._stop_final_run)
        btn_row.addWidget(self.stop_final_btn)
        layout.addLayout(btn_row)

        layout.addWidget(QLabel("Live equity curve"))
        equity_frame_final = QFrame()
        equity_frame_final.setStyleSheet("background-color: #000000;")
        equity_frame_final.setMinimumHeight(180)
        self.equity_curve_final = EquityCurveWidget(equity_frame_final)
        final_equity_layout = QVBoxLayout(equity_frame_final)
        final_equity_layout.setContentsMargins(0, 0, 0, 0)
        final_equity_layout.addWidget(self.equity_curve_final)
        layout.addWidget(equity_frame_final)

        layout.addWidget(QLabel("Final Backtest Results — Baseline vs Filtered (scrollable)"))
        self._results_table = QTableWidget()
        self._results_table.setColumnCount(4)
        self._results_table.setHorizontalHeaderLabels(["Metric", "Baseline (No Filter)", "Filtered (NN)", "Change"])
        self._results_table.horizontalHeader().setStretchLastSection(True)
        self._results_table.setAlternatingRowColors(True)
        self._results_table.setStyleSheet(
            "font-family: monospace; font-size: 9pt; background-color: #2b2b2b; color: #ffffff;"
        )
        self._results_table.setMinimumHeight(400)
        results_scroll = QScrollArea()
        results_scroll.setWidget(self._results_table)
        results_scroll.setWidgetResizable(True)
        layout.addWidget(results_scroll)

        self._form_inputs_p3 = (
            self.lot_size_p3, self.min_len_p3, self.max_len_p3, self.nbr_ma_p3,
            self.min_slope_factor_p3, self.min_laminar_level_p3, self.max_bandwidth_factor_p3,
            self.nh_p3, self.std_dev_factor_p3, self.exit_option_p3, self.magic_number_p3,
        )
        self._sync_ea_p2_to_p3()
        self.lot_size.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.min_len.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.max_len.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.nbr_ma.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.min_slope_factor.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.min_laminar_level.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.max_bandwidth_factor.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.nh.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.std_dev_factor.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.exit_option.currentIndexChanged.connect(lambda i: self._sync_ea_p2_to_p3())
        self.magic_number.valueChanged.connect(lambda v: self._sync_ea_p2_to_p3())
        self.lot_size_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        self.min_len_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        self.max_len_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        self.nbr_ma_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        self.min_slope_factor_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        self.min_laminar_level_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        self.max_bandwidth_factor_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        self.nh_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        self.std_dev_factor_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        self.exit_option_p3.currentIndexChanged.connect(lambda i: self._sync_ea_p3_to_p2())
        self.magic_number_p3.valueChanged.connect(lambda v: self._sync_ea_p3_to_p2())
        return w

    def _sync_ea_p2_to_p3(self) -> None:
        """When Page 2 EA params change, update Page 3."""
        for w in (
            self.lot_size_p3, self.min_len_p3, self.max_len_p3, self.nbr_ma_p3,
            self.min_slope_factor_p3, self.min_laminar_level_p3, self.max_bandwidth_factor_p3,
            self.nh_p3, self.std_dev_factor_p3, self.magic_number_p3,
        ):
            w.blockSignals(True)
        self.exit_option_p3.blockSignals(True)
        self.lot_size_p3.setValue(self.lot_size.value())
        self.min_len_p3.setValue(self.min_len.value())
        self.max_len_p3.setValue(self.max_len.value())
        self.nbr_ma_p3.setValue(self.nbr_ma.value())
        self.min_slope_factor_p3.setValue(self.min_slope_factor.value())
        self.min_laminar_level_p3.setValue(self.min_laminar_level.value())
        self.max_bandwidth_factor_p3.setValue(self.max_bandwidth_factor.value())
        self.nh_p3.setValue(self.nh.value())
        self.std_dev_factor_p3.setValue(self.std_dev_factor.value())
        self.exit_option_p3.setCurrentIndex(self.exit_option.currentIndex())
        self.magic_number_p3.setValue(self.magic_number.value())
        for w in (
            self.lot_size_p3, self.min_len_p3, self.max_len_p3, self.nbr_ma_p3,
            self.min_slope_factor_p3, self.min_laminar_level_p3, self.max_bandwidth_factor_p3,
            self.nh_p3, self.std_dev_factor_p3, self.magic_number_p3,
        ):
            w.blockSignals(False)
        self.exit_option_p3.blockSignals(False)

    def _sync_ea_p3_to_p2(self) -> None:
        """When Page 3 EA params change, update Page 2."""
        for w in (
            self.lot_size, self.min_len, self.max_len, self.nbr_ma,
            self.min_slope_factor, self.min_laminar_level, self.max_bandwidth_factor,
            self.nh, self.std_dev_factor, self.magic_number,
        ):
            w.blockSignals(True)
        self.exit_option.blockSignals(True)
        self.lot_size.setValue(self.lot_size_p3.value())
        self.min_len.setValue(self.min_len_p3.value())
        self.max_len.setValue(self.max_len_p3.value())
        self.nbr_ma.setValue(self.nbr_ma_p3.value())
        self.min_slope_factor.setValue(self.min_slope_factor_p3.value())
        self.min_laminar_level.setValue(self.min_laminar_level_p3.value())
        self.max_bandwidth_factor.setValue(self.max_bandwidth_factor_p3.value())
        self.nh.setValue(self.nh_p3.value())
        self.std_dev_factor.setValue(self.std_dev_factor_p3.value())
        self.exit_option.setCurrentIndex(self.exit_option_p3.currentIndex())
        self.magic_number.setValue(self.magic_number_p3.value())
        for w in (
            self.lot_size, self.min_len, self.max_len, self.nbr_ma,
            self.min_slope_factor, self.min_laminar_level, self.max_bandwidth_factor,
            self.nh, self.std_dev_factor, self.magic_number,
        ):
            w.blockSignals(False)
        self.exit_option.blockSignals(False)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        QTimer.singleShot(0, self._apply_standard_input_heights)

    def _apply_standard_input_heights(self) -> None:
        """Set every input field to STANDARD_INPUT_HEIGHT and center text vertically."""
        for w in self._form_inputs_data + self._form_inputs + self._form_inputs_p3:
            w.setFixedHeight(STANDARD_INPUT_HEIGHT)
            if isinstance(w, QLineEdit):
                w.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open OHLCV CSV",
            "",
            "CSV (*.csv);;All (*)",
        )
        if path:
            self.csv_path_edit.setText(path)

    def _log(self, msg: str, level: str = "info") -> None:
        self.trace.append(msg, level=level)
        self._log_file.write(level, msg)

    def _log_backtest_metrics_to_trace(self, m: dict, title: str) -> None:
        """Emit full backtest stats to trace with color: GREEN positive, ORANGE loss, CYAN neutral."""
        net = m.get("net_profit", 0)
        z = m.get("z_score", 0)
        self._log(f"========== {title} ==========", "result")
        self._log(f"Net Profit: ${net:,.2f}", "positive" if net > 0 else "warning")
        self._log(f"Total Profit: ${m.get('total_profit', 0):,.2f}", "positive")
        tl = m.get("total_loss", 0)
        self._log(f"Total Loss: -${tl:,.2f}", "warning")
        self._log(f"Profit Factor: {m.get('profit_factor', 0):.2f}", "positive")
        self._log(f"Sharpe Ratio: {m.get('sharpe_ratio', 0):.2f}", "positive")
        self._log(f"Z-Score: {z:.2f}", "positive" if z > 0 else "warning")
        self._log(f"Max. Drawdown (absolute): -${m.get('max_drawdown_abs', 0):,.2f}", "warning")
        self._log(f"Max. Drawdown (relative): {m.get('max_drawdown_pct', 0):.2f}%", "warning")
        self._log(f"# Trades: {int(m.get('total_trades', 0))}", "result")
        self._log(f"# Long: {int(m.get('n_long', 0))}", "result")
        self._log(f"# Short: {int(m.get('n_short', 0))}", "result")
        self._log(f"# Long Won: {int(m.get('n_long_won', 0))}", "positive")
        self._log(f"# Long Lost: {int(m.get('n_long_lost', 0))}", "warning")
        self._log(f"# Short Won: {int(m.get('n_short_won', 0))}", "positive")
        self._log(f"# Short Lost: {int(m.get('n_short_lost', 0))}", "warning")
        self._log(f"Avg. Win Trade: ${m.get('avg_win_trade', 0):,.2f}", "positive")
        self._log(f"Avg. Loss Trade: -${abs(m.get('avg_loss_trade', 0)):,.2f}", "warning")
        self._log(f"Avg. Win Long Trade: ${m.get('avg_win_long_trade', 0):,.2f}", "positive")
        self._log(f"Avg. Loss Long Trade: -${abs(m.get('avg_loss_long_trade', 0)):,.2f}", "warning")
        self._log(f"Avg. Win Short Trade: ${m.get('avg_win_short_trade', 0):,.2f}", "positive")
        self._log(f"Avg. Loss Short Trade: -${abs(m.get('avg_loss_short_trade', 0)):,.2f}", "warning")
        self._log(f"Largest Win Trade: ${m.get('largest_win_trade', 0):,.2f}", "positive")
        ll = m.get("largest_loss_trade", 0)
        self._log(f"Largest Loss Trade: ${ll:,.2f}", "warning")
        self._log("=========================================================", "result")

    def _ea_params(self) -> EAParams:
        return EAParams(
            lot_size=self.lot_size.value(),
            min_len=self.min_len.value(),
            max_len=self.max_len.value(),
            nbr_ma=self.nbr_ma.value(),
            min_slope_factor=self.min_slope_factor.value(),
            min_laminar_level=self.min_laminar_level.value(),
            max_bandwidth_factor=self.max_bandwidth_factor.value(),
            nh=self.nh.value(),
            std_dev_factor=self.std_dev_factor.value(),
            exit_option=self.exit_option.currentIndex(),
            magic_number=self.magic_number.value(),
        )

    def _symbol_config(self) -> SymbolConfig:
        return SymbolConfig(
            name=self.symbol_name.text().strip() or "SYMBOL",
            mintick=self.mintick.value(),
            tick_value=self.tick_value.value(),
            default_lots=self.default_lots.value(),
        )

    def _run_backtest(self) -> None:
        path = self.csv_path_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Error", "Select a valid CSV file.")
            return
        data_tf = self.data_file_timeframe.currentText()
        system_tf = self.system_timeframe.currentText()
        if parse_timeframe(system_tf) < parse_timeframe(data_tf):
            QMessageBox.warning(
                self, "Error",
                "Target bar period must be >= native period.",
            )
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(
                self,
                "Please wait",
                "Previous backtest is still stopping. Please wait and try again.",
            )
            return
        if self._train_worker is not None and self._train_worker.isRunning():
            QMessageBox.warning(
                self,
                "Please wait",
                "Train NN is still running. Please wait and try again.",
            )
            return
        self._cleanup_worker()
        out_dir = self.out_dir_edit.text().strip() or None
        self.backtest_btn.setEnabled(False)
        self.train_nn_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting backtest...")
        self.equity_curve_train.clear()
        self._log("Starting backtest (training part of data)...", "info")

        self._worker = QThread()
        nn_type = self.nn_type_combo.currentText()
        arch = self.arch_combo.currentText()
        if self.custom_arch_check.isChecked():
            arch = "custom"
            raw = self.custom_sizes_edit.text().strip()
            custom_sizes = tuple(int(x.strip()) for x in raw.split(",") if x.strip().isdigit())
            if not custom_sizes:
                custom_sizes = (256, 128, 64)
        else:
            custom_sizes = None
        dropout_val = self.dropout_slider.value() / 100.0
        use_dropout = self.use_dropout_check.isChecked()
        model_basename = f"{nn_type.split()[0] if nn_type else 'Dense'}_MAFilter"

        self._worker_obj = Worker(
            csv_path=path,
            params=self._ea_params(),
            symbol=self._symbol_config(),
            good_threshold=self.good_threshold.value(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            patience=self.patience_spin.value(),
            out_dir=out_dir,
            model_basename=model_basename,
            data_tf_str=data_tf,
            system_tf_str=system_tf,
            day_offset=self.day_offset_slider.value(),
            nn_type=nn_type,
            architecture=arch,
            custom_hidden_sizes=custom_sizes,
            dropout=dropout_val,
            use_dropout=use_dropout,
            device=self.device_combo.currentText(),
            do_training=False,
        )
        self._worker_obj.moveToThread(self._worker)
        self._worker.started.connect(self._worker_obj.run)
        self._worker_obj.finished.connect(self._on_backtest_finished)
        self._worker_obj.error.connect(self._on_backtest_error)
        self._worker_obj.progress.connect(self._on_progress)
        self._worker_obj.trace.connect(self._on_worker_trace)
        self._worker_obj.equity_point.connect(self.equity_curve_train.append_point)
        self._worker_obj.finished.connect(self._worker.quit)
        self._worker_obj.error.connect(self._worker.quit)
        self._worker.start()

    def _run_train_nn(self) -> None:
        if self._last_backtest_result is None:
            QMessageBox.warning(self, "Error", "Run Backtest first to get trades.")
            return
        if self._train_worker is not None and self._train_worker.isRunning():
            QMessageBox.warning(self, "Please wait", "Train NN is already running.")
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Please wait", "Backtest is still running.")
            return
        self._cleanup_train_worker()
        trades = self._last_backtest_result["trades"]
        n_bars = self._last_backtest_result["n_bars"]
        out_dir = self.out_dir_edit.text().strip() or None
        nn_type = self.nn_type_combo.currentText()
        arch = self.arch_combo.currentText()
        if self.custom_arch_check.isChecked():
            arch = "custom"
            raw = self.custom_sizes_edit.text().strip()
            custom_sizes = tuple(int(x.strip()) for x in raw.split(",") if x.strip().isdigit())
            if not custom_sizes:
                custom_sizes = (256, 128, 64)
        else:
            custom_sizes = None
        model_basename = f"{nn_type.split()[0] if nn_type else 'Dense'}_MAFilter"

        self.backtest_btn.setEnabled(False)
        self.train_nn_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Training NN...")

        self._train_worker = QThread()
        self._train_worker_obj = TrainOnlyWorker(
            trades=trades,
            n_bars=n_bars,
            good_threshold=self.good_threshold.value(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            patience=self.patience_spin.value(),
            out_dir=out_dir,
            model_basename=model_basename,
            nn_type=nn_type,
            architecture=arch,
            custom_hidden_sizes=custom_sizes,
            dropout=self.dropout_slider.value() / 100.0,
            use_dropout=self.use_dropout_check.isChecked(),
            device=self.device_combo.currentText(),
        )
        self._train_worker_obj.moveToThread(self._train_worker)
        self._train_worker.started.connect(self._train_worker_obj.run)
        self._train_worker_obj.finished.connect(self._on_train_nn_finished)
        self._train_worker_obj.error.connect(self._on_train_nn_error)
        self._train_worker_obj.progress.connect(self._on_progress)
        self._train_worker_obj.trace.connect(self._on_worker_trace)
        self._train_worker_obj.finished.connect(self._train_worker.quit)
        self._train_worker_obj.error.connect(self._train_worker.quit)
        self._train_worker.start()

    def _on_worker_trace(self, level: str, msg: str) -> None:
        self._log(msg, level)

    def _cleanup_worker(self) -> None:
        """Disconnect backtest worker, wait, release."""
        if self._worker is None or self._worker_obj is None:
            return
        try:
            self._worker.started.disconnect(self._worker_obj.run)
        except (RuntimeError, TypeError):
            pass
        try:
            self._worker_obj.finished.disconnect(self._on_backtest_finished)
            self._worker_obj.error.disconnect(self._on_backtest_error)
            self._worker_obj.progress.disconnect(self._on_progress)
            self._worker_obj.trace.disconnect(self._on_worker_trace)
            self._worker_obj.equity_point.disconnect(self.equity_curve_train.append_point)
            self._worker_obj.finished.disconnect(self._worker.quit)
            self._worker_obj.error.disconnect(self._worker.quit)
        except (RuntimeError, TypeError):
            pass
        self._worker.quit()
        if self._worker.isRunning():
            self._worker.wait(2000)
        self._worker = None
        self._worker_obj = None

    def _cleanup_train_worker(self) -> None:
        """Disconnect train NN worker, wait, release."""
        if self._train_worker is None or self._train_worker_obj is None:
            return
        try:
            self._train_worker.started.disconnect(self._train_worker_obj.run)
        except (RuntimeError, TypeError):
            pass
        try:
            self._train_worker_obj.finished.disconnect(self._on_train_nn_finished)
            self._train_worker_obj.error.disconnect(self._on_train_nn_error)
            self._train_worker_obj.progress.disconnect(self._on_progress)
            self._train_worker_obj.trace.disconnect(self._on_worker_trace)
            self._train_worker_obj.finished.disconnect(self._train_worker.quit)
            self._train_worker_obj.error.disconnect(self._train_worker.quit)
        except (RuntimeError, TypeError):
            pass
        self._train_worker.quit()
        if self._train_worker.isRunning():
            self._train_worker.wait(2000)
        self._train_worker = None
        self._train_worker_obj = None

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.progress_label.setText(f"{percent}% — {message}")
        # Log phase changes; skip per-bar backtest spam, log epoch updates occasionally
        if "Backtesting" in message and "bar " in message:
            return  # avoid flooding trace with every bar
        if "Training" in message and "Epoch" in message:
            return  # avoid per-epoch spam (progress bar shows it)
        self._log(message)

    def _stop_run(self) -> None:
        if self._worker is not None and self._worker.isRunning() and self._worker_obj is not None:
            self._worker_obj.request_stop()
            self.stop_btn.setText("Stopping...")
            self._log("Stop requested (backtest). Worker will finish current step and stop.", "info")
        elif self._train_worker is not None and self._train_worker.isRunning() and self._train_worker_obj is not None:
            self._train_worker_obj.request_stop()
            self.stop_btn.setText("Stopping...")
            self._log("Stop requested (Train NN). Worker will finish current step and stop.", "info")

    def _on_backtest_finished(self, result: dict) -> None:
        self.backtest_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self._last_backtest_result = result
        self.train_nn_btn.setEnabled(True)
        if "first_date" in result:
            self.date_first_label.setText(f"First date: {result['first_date']}")
        if "split_date" in result:
            self.date_split_label.setText(f"Split date: {result['split_date']}")
        if "last_date" in result:
            self.date_last_label.setText(f"Last date: {result['last_date']}")
        n_bars = result["n_bars"]
        n_trades = result["n_trades"]
        n_good = result["n_good"]
        net_profit = result.get("net_profit", 0.0)
        net_loss = result.get("net_loss", 0.0)
        self._log(
            f"Backtest done. Bars: {n_bars}, Trades: {n_trades}, Good: {n_good}, Bad: {n_trades - n_good} | "
            f"Net Profit: {net_profit:.2f}, Net Loss: {net_loss:.2f}",
            "result",
        )
        self._log("Train NN is now enabled. Adjust NN settings and click Train NN to train (or train multiple configs).", "info")
        training_metrics = compute_backtest_metrics(result["trades"], result["n_bars"])
        self._training_backtest_metrics = training_metrics
        self._log_backtest_metrics_to_trace(training_metrics, "Backtest Results (Training Phase)")
        # Update equity curve from backtest result
        equity_curve = compute_equity_curve(result["trades"], n_bars)
        points = [(i, eq) for i, eq in enumerate(equity_curve)]
        self.equity_curve_train.set_points(points)
        self._cleanup_worker()

    def _on_backtest_error(self, err: str) -> None:
        self.backtest_btn.setEnabled(True)
        self.train_nn_btn.setEnabled(self._last_backtest_result is not None)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self._log(f"Backtest error: {err}", "error")
        QMessageBox.critical(self, "Backtest Error", err)
        self._cleanup_worker()

    def _on_train_nn_finished(self, result: dict) -> None:
        self.backtest_btn.setEnabled(True)
        self.train_nn_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self._trained_model = result.get("model")
        self._trained_scaling = result.get("scaling")
        if self._trained_model is not None:
            self._log("Model kept in memory for Final Backtest (same session).", "info")
        if result.get("out_dir"):
            self._log(f"Model saved to: {result['out_dir']}", "info")
        self._log("Train NN done. You can change NN settings and run Train NN again, or run Final Backtest.", "result")
        self._cleanup_train_worker()

    def _on_train_nn_error(self, err: str) -> None:
        self.backtest_btn.setEnabled(True)
        self.train_nn_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self._log(f"Train NN error: {err}", "error")
        QMessageBox.critical(self, "Train NN Error", err)
        self._cleanup_train_worker()

    def _on_error(self, err: str) -> None:
        """Legacy error handler; backtest now uses _on_backtest_error."""
        self.backtest_btn.setEnabled(True)
        self.train_nn_btn.setEnabled(self._last_backtest_result is not None)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
        self._log(f"Error: {err}", "error")
        QMessageBox.critical(self, "Error", err)
        self._cleanup_worker()

    def _run_final_backtest(self) -> None:
        path = self.csv_path_edit.text().strip()
        if not path or not Path(path).exists():
            QMessageBox.warning(self, "Error", "Select a valid CSV file (Data Manager).")
            return
        if self._final_worker is not None and self._final_worker.isRunning():
            QMessageBox.warning(
                self,
                "Please wait",
                "Previous final backtest is still stopping. Please wait and try again.",
            )
            return
        self._cleanup_final_worker()
        out_dir = self.out_dir_edit.text().strip() or None
        in_memory = (self._trained_model is not None and self._trained_scaling is not None)
        if not in_memory and (not out_dir or not Path(out_dir).exists()):
            reply = QMessageBox.warning(
                self,
                "No model",
                "No NN model in memory and Output directory is empty or path does not exist.\n\n"
                "Baseline and Filtered results will be identical.\n\n"
                "To use the NN filter: run 'Backtest and Train NN' first (model is kept in memory), "
                "or set Output directory to a folder with a saved model.\n\n"
                "Run anyway (Baseline only)?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
        # Model basename for loading from disk: model_type_Strategy_Name
        nn_type_text = self.nn_type_combo.currentText()
        model_type_short = nn_type_text.split()[0] if nn_type_text else "Dense"
        model_basename = f"{model_type_short}_MAFilter"
        in_memory_model = (self._trained_model, self._trained_scaling) if in_memory else None
        self.run_final_btn.setEnabled(False)
        self.run_final_btn.setText("Final Backtest...")
        self.run_final_btn.setStyleSheet("background-color: #00BFFF; color: white;")
        self.stop_final_btn.setEnabled(True)
        self.stop_final_btn.setStyleSheet("background-color: #B22222; color: white;")
        self.equity_curve_final.clear()
        self._log("========== Final Backtest Started ==========", "info")
        self._log("Running Baseline (1/2) then Filtered (2/2) on same test data...", "info")

        self._final_worker = QThread()
        self._final_worker_obj = FinalBacktestWorker(
            csv_path=path,
            params=self._ea_params(),
            symbol=self._symbol_config(),
            data_tf_str=self.data_file_timeframe.currentText(),
            system_tf_str=self.system_timeframe.currentText(),
            day_offset=self.day_offset_slider.value(),
            model_dir=out_dir,
            model_basename=model_basename,
            in_memory_model=in_memory_model,
        )
        self._final_worker_obj.moveToThread(self._final_worker)
        self._final_worker.started.connect(self._final_worker_obj.run)
        self._final_worker_obj.finished.connect(self._on_final_finished)
        self._final_worker_obj.error.connect(self._on_final_error)
        self._final_worker_obj.progress.connect(self._on_final_progress)
        self._final_worker_obj.trace.connect(self._on_final_worker_trace)
        self._final_worker_obj.finished.connect(self._final_worker.quit)
        self._final_worker_obj.error.connect(self._final_worker.quit)
        self._final_worker.start()

    def _on_final_worker_trace(self, level: str, msg: str) -> None:
        self._log(msg, level)

    def _cleanup_final_worker(self) -> None:
        if self._final_worker is None or self._final_worker_obj is None:
            return
        try:
            self._final_worker.started.disconnect(self._final_worker_obj.run)
        except (RuntimeError, TypeError):
            pass
        try:
            self._final_worker_obj.finished.disconnect(self._on_final_finished)
            self._final_worker_obj.error.disconnect(self._on_final_error)
            self._final_worker_obj.progress.disconnect(self._on_final_progress)
            self._final_worker_obj.trace.disconnect(self._on_final_worker_trace)
            self._final_worker_obj.finished.disconnect(self._final_worker.quit)
            self._final_worker_obj.error.disconnect(self._final_worker.quit)
        except (RuntimeError, TypeError):
            pass
        self._final_worker.quit()
        if self._final_worker.isRunning():
            self._final_worker.wait(2000)
        self._final_worker = None
        self._final_worker_obj = None

    def _stop_final_run(self) -> None:
        if self._final_worker_obj is not None and self._final_worker is not None and self._final_worker.isRunning():
            self._final_worker_obj.request_stop()
            self.stop_final_btn.setText("Stopping...")
            self._log("Stop requested (final backtest). Worker will finish current step and stop.", "info")
            # Do NOT block GUI with wait() or call terminate() - both can cause crashes.

    def _on_final_progress(self, percent: int, message: str) -> None:
        self.progress_label.setText(f"{percent}% — {message}")

    def _populate_final_results_table(
        self,
        baseline_metrics: dict,
        filtered_metrics: dict,
        baseline_trades: list,
        filtered_trades: list,
    ) -> None:
        """Fill 4-column comparison table: Metric | Baseline | Filtered | Change. Includes NN Filter Analysis and Confusion Matrix."""
        green = "color: #00FF00;"
        orange = "color: #FFA500;"
        white = "color: #ffffff;"
        higher_better = {
            "net_profit", "total_profit", "profit_factor", "sharpe_ratio", "z_score",
            "n_long_won", "n_short_won", "avg_win_trade", "avg_win_long_trade", "avg_win_short_trade",
            "largest_win_trade", "win_rate",
        }
        lower_better = {
            "total_loss", "max_drawdown_abs", "max_drawdown_pct",
            "n_long_lost", "n_short_lost", "avg_loss_trade", "avg_loss_long_trade", "avg_loss_short_trade",
            "largest_loss_trade",
        }
        currency_keys = {
            "net_profit", "total_profit", "total_loss", "avg_win_trade", "avg_loss_trade",
            "avg_win_long_trade", "avg_loss_long_trade", "avg_win_short_trade", "avg_loss_short_trade",
            "largest_win_trade", "largest_loss_trade", "max_drawdown_abs",
        }
        percent_keys = {"max_drawdown_pct", "win_rate"}
        int_keys = {"total_trades", "n_long", "n_short", "n_long_won", "n_long_lost", "n_short_won", "n_short_lost"}

        def fmt_baseline(key: str, m: dict) -> str:
            val = m.get(key, 0)
            if key == "total_loss":
                return f"-${val:,.2f}"
            if key == "max_drawdown_abs":
                return f"-${val:,.2f}"
            if key in currency_keys:
                return f"${val:,.2f}"
            if key in percent_keys:
                return f"{val:.1f}%" if key == "win_rate" else f"{val:.2f}%"
            if key in int_keys:
                return str(int(val))
            if isinstance(val, float):
                return f"{val:.2f}" if abs(val) < 1e6 else f"{val:.4f}"
            return str(val)

        def cell_style(val: float, is_loss_metric: bool) -> str:
            if is_loss_metric:
                return orange if val != 0 else white
            return green if val > 0 else (orange if val < 0 else white)

        # Win rate computed
        def win_rate(m: dict) -> float:
            n = m.get("total_trades", 0) or 0
            if n == 0:
                return 0.0
            w = m.get("n_long_won", 0) + m.get("n_short_won", 0)
            return w / n * 100.0

        baseline_metrics = dict(baseline_metrics)
        filtered_metrics = dict(filtered_metrics)
        baseline_metrics["win_rate"] = win_rate(baseline_metrics)
        filtered_metrics["win_rate"] = win_rate(filtered_metrics)

        loss_keys = {"total_loss", "max_drawdown_abs", "max_drawdown_pct", "n_long_lost", "n_short_lost", "avg_loss_trade", "avg_loss_long_trade", "avg_loss_short_trade", "largest_loss_trade"}

        metric_spec = [
            ("Net Profit", "net_profit"), ("Total Profit", "total_profit"), ("Total Loss", "total_loss"),
            ("Profit Factor", "profit_factor"), ("Sharpe Ratio", "sharpe_ratio"), ("Z-Score", "z_score"),
            ("Max DD (absolute)", "max_drawdown_abs"), ("Max DD (relative)", "max_drawdown_pct"),
            ("# Trades", "total_trades"), ("# Long", "n_long"), ("# Short", "n_short"),
            ("# Long Won", "n_long_won"), ("# Long Lost", "n_long_lost"),
            ("# Short Won", "n_short_won"), ("# Short Lost", "n_short_lost"),
            ("Avg. Win Trade", "avg_win_trade"), ("Avg. Loss Trade", "avg_loss_trade"),
            ("Avg. Win Long", "avg_win_long_trade"), ("Avg. Loss Long", "avg_loss_long_trade"),
            ("Avg. Win Short", "avg_win_short_trade"), ("Avg. Loss Short", "avg_loss_short_trade"),
            ("Largest Win", "largest_win_trade"), ("Largest Loss", "largest_loss_trade"),
            ("Win Rate", "win_rate"),
        ]

        rows: list[tuple[str, str, str, str, str, str, str]] = []  # metric, baseline_txt, filtered_txt, change_txt, baseline_style, filtered_style, change_style

        for label, key in metric_spec:
            b_val = baseline_metrics.get(key, 0)
            f_val = filtered_metrics.get(key, 0)
            b_txt = fmt_baseline(key, baseline_metrics)
            f_txt = fmt_baseline(key, filtered_metrics)
            is_loss = key in loss_keys
            # Change %
            if isinstance(b_val, (int, float)) and isinstance(f_val, (int, float)):
                if key in lower_better:
                    # improvement when filtered is less bad (e.g. -500 vs -892) -> (filtered - baseline)/|baseline|
                    if abs(b_val) < 1e-12:
                        ch_txt = "N/A"
                        ch_style = white
                    else:
                        pct = (f_val - b_val) / abs(b_val) * 100.0
                        ch_txt = f"{pct:+.1f}%"
                        ch_style = green if pct > 0 else (orange if pct < 0 else white)
                elif key in higher_better:
                    if abs(b_val) < 1e-12:
                        ch_txt = "N/A" if abs(f_val) < 1e-12 else "+100%"
                        ch_style = white if ch_txt == "N/A" else green
                    else:
                        pct = (f_val - b_val) / abs(b_val) * 100.0
                        ch_txt = f"{pct:+.1f}%"
                        ch_style = green if pct > 0 else (orange if pct < 0 else white)
                else:
                    # neutral
                    if abs(b_val) < 1e-12:
                        ch_txt = "N/A"
                    else:
                        pct = (f_val - b_val) / abs(b_val) * 100.0
                        ch_txt = f"{pct:+.1f}%"
                    ch_style = white
            else:
                ch_txt = "—"
                ch_style = white
            b_style = cell_style(b_val if isinstance(b_val, (int, float)) else 0, is_loss)
            f_style = cell_style(f_val if isinstance(f_val, (int, float)) else 0, is_loss)
            rows.append((label, b_txt, f_txt, ch_txt, b_style, f_style, ch_style))

        # NN Filter Analysis
        filtered_set = {(t.exit_bar, t.profit, t.direction) for t in filtered_trades}
        n_base = len(baseline_trades)
        n_filt = len(filtered_trades)
        profitable_filtered_out = sum(1 for t in baseline_trades if t.profit > 0 and (t.exit_bar, t.profit, t.direction) not in filtered_set)
        loss_filtered = sum(1 for t in baseline_trades if t.profit <= 0 and (t.exit_bar, t.profit, t.direction) not in filtered_set)
        profitable_passed = sum(1 for t in filtered_trades if t.profit > 0)
        loss_passed = sum(1 for t in filtered_trades if t.profit <= 0)
        filtered_out = n_base - n_filt
        pct_filtered = (filtered_out / n_base * 100.0) if n_base else 0.0
        rows.append(("", "", "", "", white, white, white))
        rows.append(("— NN Filter Analysis —", "", "", "", white, white, white))
        rows.append(("Trades Filtered Out", "N/A", str(filtered_out), f"{pct_filtered:.1f}% of total", white, white, white))
        rows.append(("  - Profitable (FN)", "N/A", str(profitable_filtered_out), f"{profitable_filtered_out/max(1,filtered_out)*100:.1f}%" if filtered_out else "N/A", white, orange if profitable_filtered_out > 0.10 * max(1, sum(1 for t in baseline_trades if t.profit > 0)) else white, white))
        rows.append(("  - Loss (TN)", "N/A", str(loss_filtered), f"{loss_filtered/max(1,filtered_out)*100:.1f}%" if filtered_out else "N/A", white, green, white))
        rows.append(("Trades Let Through", "N/A", str(n_filt), f"{(n_filt/n_base*100):.1f}% of total" if n_base else "N/A", white, white, white))
        rows.append(("  - Profitable (TP)", "N/A", str(profitable_passed), f"{profitable_passed/max(1,n_filt)*100:.1f}%" if n_filt else "N/A", white, green, white))
        rows.append(("  - Loss (FP)", "N/A", str(loss_passed), f"{loss_passed/max(1,n_filt)*100:.1f}%" if n_filt else "N/A", white, orange if loss_passed > 0.20 * max(1, sum(1 for t in baseline_trades if t.profit <= 0)) else white, white))

        # Confusion Matrix
        tp, fp = profitable_passed, loss_passed
        fn, tn = profitable_filtered_out, loss_filtered
        rows.append(("", "", "", "", white, white, white))
        rows.append(("— Confusion Matrix —", "", "", "", white, white, white))
        rows.append(("", "Actually Profitable", "Actually Loss", "", white, white, white))
        rows.append(("NN: Trade", f"{tp} (TP)", f"{fp} (FP)", "", white, white, white))
        rows.append(("NN: No-Trade", f"{fn} (FN)", f"{tn} (TN)", "", white, white, white))
        total_cm = tp + tn + fp + fn
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
        accuracy = ((tp + tn) / total_cm * 100.0) if total_cm > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append(("", "", "", "", white, white, white))
        rows.append(("Precision", "N/A", f"{precision:.1f}%", "(TP/(TP+FP))", white, white, white))
        rows.append(("Recall (Sensitivity)", "N/A", f"{recall:.1f}%", "(TP/(TP+FN))", white, white, white))
        rows.append(("Specificity", "N/A", f"{specificity:.1f}%", "(TN/(TN+FP))", white, white, white))
        rows.append(("Accuracy", "N/A", f"{accuracy:.1f}%", "((TP+TN)/Total)", white, white, white))
        rows.append(("F1 Score", "N/A", f"{f1:.1f}%", "Harmonic mean", white, white, white))

        self._results_table.setRowCount(len(rows))
        for r, (label, b_txt, f_txt, ch_txt, b_style, f_style, ch_style) in enumerate(rows):
            self._results_table.setItem(r, 0, QTableWidgetItem(label))
            self._results_table.setItem(r, 1, QTableWidgetItem(b_txt))
            self._results_table.setItem(r, 2, QTableWidgetItem(f_txt))
            self._results_table.setItem(r, 3, QTableWidgetItem(ch_txt))
            for c in range(4):
                it = self._results_table.item(r, c)
                if it:
                    it.setForeground(QColor("#ffffff"))
            it1 = self._results_table.item(r, 1)
            if it1 and b_style != white:
                it1.setForeground(QColor("#00FF00" if "00FF00" in b_style else "#FFA500"))
            it2 = self._results_table.item(r, 2)
            if it2 and f_style != white:
                it2.setForeground(QColor("#00FF00" if "00FF00" in f_style else "#FFA500"))
            it3 = self._results_table.item(r, 3)
            if it3 and ch_style != white:
                it3.setForeground(QColor("#00FF00" if "00FF00" in ch_style else "#FFA500"))

    def _on_final_finished(self, result: dict) -> None:
        self.run_final_btn.setEnabled(True)
        self.run_final_btn.setText("Start Final Backtest")
        self.run_final_btn.setStyleSheet("background-color: #1E90FF; color: white;")
        self.stop_final_btn.setEnabled(False)
        self.stop_final_btn.setText("Stop")
        self.stop_final_btn.setStyleSheet("background-color: #8B0000; color: white;")
        baseline_metrics = result["baseline_metrics"]
        filtered_metrics = result["filtered_metrics"]
        baseline_trades = result["baseline_trades"]
        filtered_trades = result["filtered_trades"]
        n_bars = result["n_bars"]

        # Dual equity curve
        baseline_equity = result.get("baseline_equity_curve", [])
        filtered_equity = result.get("filtered_equity_curve", [])
        if baseline_equity and filtered_equity:
            base_pts = [(i, eq) for i, eq in enumerate(baseline_equity)]
            filt_pts = [(i, eq) for i, eq in enumerate(filtered_equity)]
            self.equity_curve_final.set_points_dual(base_pts, filt_pts)
        else:
            self.equity_curve_final.clear()

        self._populate_final_results_table(baseline_metrics, filtered_metrics, baseline_trades, filtered_trades)

        self._log("Final backtest complete.", "result")
        self._log_file.write("result", f"Baseline Net Profit: ${baseline_metrics.get('net_profit', 0):,.2f}")
        self._log_file.write("result", f"Filtered Net Profit: ${filtered_metrics.get('net_profit', 0):,.2f}")
        self._log_file.write("result", "Baseline vs Filtered comparison completed.")
        self._cleanup_final_worker()

    def _on_final_error(self, err: str) -> None:
        self.run_final_btn.setEnabled(True)
        self.run_final_btn.setText("Start Final Backtest")
        self.run_final_btn.setStyleSheet("background-color: #1E90FF; color: white;")
        self.stop_final_btn.setEnabled(False)
        self.stop_final_btn.setText("Stop")
        self.stop_final_btn.setStyleSheet("background-color: #8B0000; color: white;")
        self._log(f"Final backtest error: {err}", "error")
        QMessageBox.critical(self, "Error", err)
        self._cleanup_final_worker()


def main() -> None:
    app = QApplication([])
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
