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
    QDoubleSpinBox,
    QComboBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtGui import QFont

from .app_config import get_config_path, load_config, save_config, get_default_config
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
        data_tf_str: str = "1M",
        system_tf_str: str = "1M",
        day_offset: int = 20,
        nn_type: str = "Dense (Feedforward)",
        architecture: str = "Medium",
        custom_hidden_sizes: Optional[tuple[int, ...]] = None,
        dropout: float = 0.2,
        use_dropout: bool = True,
        device: str = "CPU",
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
        self.data_tf_str = data_tf_str
        self.system_tf_str = system_tf_str
        self.day_offset = day_offset
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
            self.progress.emit(50, "Training filter...")

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


class FinalBacktestWorker(QObject):
    """Run backtest on application part with NN filter; emit equity and metrics."""
    finished = Signal(object)  # dict with metrics + filtered_trades + equity_curve
    error = Signal(str)
    progress = Signal(int, str)
    equity_point = Signal(int, float)
    trace = Signal(str, str)  # level, message

    def __init__(
        self,
        csv_path: str,
        params: EAParams,
        symbol: SymbolConfig,
        data_tf_str: str,
        system_tf_str: str,
        day_offset: int,
        model_dir: Optional[str],
    ) -> None:
        super().__init__()
        self.csv_path = csv_path
        self.params = params
        self.symbol = symbol
        self.data_tf_str = data_tf_str
        self.system_tf_str = system_tf_str
        self.day_offset = day_offset
        self.model_dir = model_dir
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
            self.progress.emit(20, "Running final backtest...")
            point = _point_from_mintick(self.symbol.mintick)
            scale_factor = system_tf_min / data_tf_min if data_tf_min else 1.0
            backtest_params = _scale_ea_params_for_timeframe(self.params, scale_factor)

            def equity_cb(bar: int, cum: float) -> None:
                self.equity_point.emit(bar, cum)

            def stop_cb() -> bool:
                return self._stop_requested

            all_trades = run_backtest(
                o, h, l, c,
                backtest_params,
                tick_value=self.symbol.tick_value,
                point=point,
                lot_size=self.symbol.default_lots,
                equity_callback=equity_cb,
                stop_callback=stop_cb,
            )
            if self._stop_requested:
                self.error.emit("Stopped by user.")
                return
            if not self.model_dir or not Path(self.model_dir).exists():
                filtered_trades = all_trades
            else:
                self.progress.emit(70, "Applying NN filter...")
                model, scaling = load_filter_model(Path(self.model_dir))
                filtered_trades = filter_trades_by_nn(all_trades, model, scaling)
            metrics = compute_backtest_metrics(filtered_trades, n_bars)
            equity_curve = compute_equity_curve(filtered_trades, n_bars)
            self.progress.emit(100, "Done.")
            self.finished.emit({
                "metrics": metrics,
                "filtered_trades": filtered_trades,
                "equity_curve": equity_curve,
                "n_bars": n_bars,
            })
        except Exception as e:
            self.trace.emit("error", f"Final backtest error: {e}\n{traceback.format_exc()}")
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("MAFilter NN — Trade Filter")
        # Extra height for EA params and results (902 + 100 = 1002)
        self.setMinimumSize(860, 1002)
        self.resize(860, 1002)
        self._df = None
        self._trades = []
        self._worker: Optional[QThread] = None
        self._final_worker: Optional[QThread] = None
        self._log_file = LogFile()

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
        config, err = load_config()
        if err:
            self._log(f"Config: {err}, using defaults.", "info")
            self.trace.append("Config file corrupted or invalid, using defaults.", "warning")
        else:
            self._log(f"Configuration loaded from {get_config_path()}", "info")
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
            width = w.get("width", 860)
            height = w.get("height", 902)
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
            if "output_dir" in bt:
                self.out_dir_edit.setText(bt.get("output_dir", "") or "")
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
        self.out_dir_edit.setPlaceholderText("Directory to save model (best.pt, scaling.json) — required for Final Backtest")
        out_row = QFormLayout()
        out_row.addRow("Output directory", self.out_dir_edit)
        scroll_layout.addLayout(out_row)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Start Backtest and Train NN")
        self.run_btn.setStyleSheet("background-color: #1E90FF; color: white;")
        self.run_btn.clicked.connect(self._run_train)
        btn_row.addWidget(self.run_btn)
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

        layout.addWidget(QLabel("Final backtest results"))
        # Three-column layout: more compact vertically, labels left, values right; monospace; color by sign
        results_row = QHBoxLayout()
        col1_layout = QGridLayout()
        col2_layout = QGridLayout()
        col3_layout = QGridLayout()
        _result_row_h = 18
        for g in (col1_layout, col2_layout, col3_layout):
            g.setVerticalSpacing(4)
        self._result_labels = {}
        col1_spec = [
            ("Net Profit:", "net_profit"), ("Total Profit:", "total_profit"), ("Total Loss:", "total_loss"),
            ("Profit Factor:", "profit_factor"), ("Sharpe Ratio:", "sharpe_ratio"), ("Z-Score:", "z_score"),
            ("Max. DD (abs):", "max_drawdown_abs"), ("Max. DD (rel):", "max_drawdown_pct"),
        ]
        col2_spec = [
            ("# Long Won:", "n_long_won"), ("# Long Lost:", "n_long_lost"),
            ("# Short Won:", "n_short_won"), ("# Short Lost:", "n_short_lost"),
            ("Total # Trades:", "total_trades"), ("# Long:", "n_long"), ("# Short:", "n_short"),
        ]
        col3_spec = [
            ("Avg. Win Trade:", "avg_win_trade"), ("Avg. Loss Trade:", "avg_loss_trade"),
            ("Avg. Win Long:", "avg_win_long_trade"), ("Avg. Loss Long:", "avg_loss_long_trade"),
            ("Avg. Win Short:", "avg_win_short_trade"), ("Avg. Loss Short:", "avg_loss_short_trade"),
            ("Largest Win:", "largest_win_trade"), ("Largest Loss:", "largest_loss_trade"),
        ]

        def add_result_column(grid: QGridLayout, spec: list[tuple[str, str]]) -> None:
            for i, (label_text, key) in enumerate(spec):
                name_lbl = QLabel(label_text)
                name_lbl.setMinimumHeight(_result_row_h)
                val_lbl = QLabel("—")
                val_lbl.setMinimumHeight(_result_row_h)
                val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                val_lbl.setStyleSheet("font-family: monospace;")
                self._result_labels[key] = val_lbl
                grid.addWidget(name_lbl, i, 0)
                grid.addWidget(val_lbl, i, 1)

        add_result_column(col1_layout, col1_spec)
        add_result_column(col2_layout, col2_spec)
        add_result_column(col3_layout, col3_spec)
        col1_w = QWidget()
        col1_w.setLayout(col1_layout)
        col2_w = QWidget()
        col2_w.setLayout(col2_layout)
        col3_w = QWidget()
        col3_w.setLayout(col3_layout)
        results_row.addWidget(col1_w)
        results_row.addSpacing(COLUMN_SPACING)
        d1 = QFrame()
        d1.setFrameShape(QFrame.Shape.VLine)
        d1.setStyleSheet("color: #444;")
        results_row.addWidget(d1)
        results_row.addSpacing(COLUMN_SPACING)
        results_row.addWidget(col2_w)
        results_row.addSpacing(COLUMN_SPACING)
        d2 = QFrame()
        d2.setFrameShape(QFrame.Shape.VLine)
        d2.setStyleSheet("color: #444;")
        results_row.addWidget(d2)
        results_row.addSpacing(COLUMN_SPACING)
        results_row.addWidget(col3_w)
        results_row.addStretch()
        results_container = QWidget()
        results_container.setLayout(results_row)
        layout.addWidget(results_container)

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

    def _run_train(self) -> None:
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
                "Previous run is still stopping. Please wait a moment and try again.",
            )
            return
        self._cleanup_worker()
        out_dir = self.out_dir_edit.text().strip() or None
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Backtest and Train...")
        self.run_btn.setStyleSheet("background-color: #00BFFF; color: white;")
        self.stop_btn.setEnabled(True)
        self.stop_btn.setStyleSheet("background-color: #B22222; color: white;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting...")
        self.equity_curve_train.clear()
        log_path = self._log_file.start()
        self._log(f"Log file: {log_path}", "info")
        self._log("Starting: load CSV → backtest (training part) → train NN...", "info")
        device_text = self.device_combo.currentText()
        self._log(f"Training on device: {device_text}", "info")
        self._log(
            f"NN: type={self.nn_type_combo.currentText()}, arch={self.arch_combo.currentText()}, "
            f"dropout={self.dropout_slider.value()/100:.2f}, lookback={self.lookback_enable_check.isChecked()}, "
            f"length={self.lookback_length_slider.value()}, norm={self.norm_method_combo.currentText()}",
            "info",
        )

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

        self._worker_obj = Worker(
            csv_path=path,
            params=self._ea_params(),
            symbol=self._symbol_config(),
            good_threshold=self.good_threshold.value(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            patience=self.patience_spin.value(),
            out_dir=out_dir,
            data_tf_str=data_tf,
            system_tf_str=system_tf,
            day_offset=self.day_offset_slider.value(),
            nn_type=nn_type,
            architecture=arch,
            custom_hidden_sizes=custom_sizes,
            dropout=dropout_val,
            use_dropout=use_dropout,
            device=device_text,
        )
        self._worker_obj.moveToThread(self._worker)
        self._worker.started.connect(self._worker_obj.run)
        self._worker_obj.finished.connect(self._on_finished)
        self._worker_obj.error.connect(self._on_error)
        self._worker_obj.progress.connect(self._on_progress)
        self._worker_obj.trace.connect(self._on_worker_trace)
        self._worker_obj.equity_point.connect(self.equity_curve_train.append_point)
        self._worker_obj.finished.connect(self._worker.quit)
        self._worker_obj.error.connect(self._worker.quit)
        self._worker.start()

    def _on_worker_trace(self, level: str, msg: str) -> None:
        self._log(msg, level)

    def _cleanup_worker(self) -> None:
        """Disconnect signals, wait for thread to finish, release references. Prevents crash on next run."""
        if self._worker is None or self._worker_obj is None:
            return
        try:
            self._worker.started.disconnect(self._worker_obj.run)
        except (RuntimeError, TypeError):
            pass
        try:
            self._worker_obj.finished.disconnect(self._on_finished)
            self._worker_obj.error.disconnect(self._on_error)
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
        if self._worker_obj is not None and self._worker is not None and self._worker.isRunning():
            self._worker_obj.request_stop()
            self.stop_btn.setText("Stopping...")
            self._log("Stop requested. Worker will finish current step and stop.", "info")
            # Do NOT block GUI with wait() or call terminate() - both can cause crashes.
            # Worker checks _stop_requested in backtest loop and training batches and will emit error("Stopped by user.").

    def _on_finished(self, result: dict) -> None:
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Start Backtest and Train NN")
        self.run_btn.setStyleSheet("background-color: #1E90FF; color: white;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.stop_btn.setStyleSheet("background-color: #8B0000; color: white;")
        self.progress_bar.setVisible(False)
        self.progress_label.setText("")
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
            f"Done. Bars: {n_bars}, Trades: {n_trades}, Good: {n_good}, Bad: {n_trades - n_good} | "
            f"Net Profit: {net_profit:.2f}, Net Loss: {net_loss:.2f}",
            "result",
        )
        if result.get("out_dir"):
            self._log(f"Model saved to: {result['out_dir']}", "info")
        self._cleanup_worker()

    def _on_error(self, err: str) -> None:
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Start Backtest and Train NN")
        self.run_btn.setStyleSheet("background-color: #1E90FF; color: white;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stop")
        self.stop_btn.setStyleSheet("background-color: #8B0000; color: white;")
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
        self.run_final_btn.setEnabled(False)
        self.run_final_btn.setText("Final Backtest...")
        self.run_final_btn.setStyleSheet("background-color: #00BFFF; color: white;")
        self.stop_final_btn.setEnabled(True)
        self.stop_final_btn.setStyleSheet("background-color: #B22222; color: white;")
        self.equity_curve_final.clear()
        self._log("Starting final backtest (application part, NN filter applied)...", "info")

        self._final_worker = QThread()
        self._final_worker_obj = FinalBacktestWorker(
            csv_path=path,
            params=self._ea_params(),
            symbol=self._symbol_config(),
            data_tf_str=self.data_file_timeframe.currentText(),
            system_tf_str=self.system_timeframe.currentText(),
            day_offset=self.day_offset_slider.value(),
            model_dir=out_dir,
        )
        self._final_worker_obj.moveToThread(self._final_worker)
        self._final_worker.started.connect(self._final_worker_obj.run)
        self._final_worker_obj.finished.connect(self._on_final_finished)
        self._final_worker_obj.error.connect(self._on_final_error)
        self._final_worker_obj.progress.connect(self._on_final_progress)
        self._final_worker_obj.trace.connect(self._on_final_worker_trace)
        self._final_worker_obj.equity_point.connect(self.equity_curve_final.append_point)
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
            self._final_worker_obj.equity_point.disconnect(self.equity_curve_final.append_point)
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

    def _on_final_finished(self, result: dict) -> None:
        self.run_final_btn.setEnabled(True)
        self.run_final_btn.setText("Start Final Backtest")
        self.run_final_btn.setStyleSheet("background-color: #1E90FF; color: white;")
        self.stop_final_btn.setEnabled(False)
        self.stop_final_btn.setText("Stop")
        self.stop_final_btn.setStyleSheet("background-color: #8B0000; color: white;")
        metrics = result["metrics"]
        # Color scheme: positive = green/cyan, negative = red/orange, neutral = white
        positive_style = "color: #00FF00; font-family: monospace;"
        negative_style = "color: #FFA500; font-family: monospace;"
        neutral_style = "color: #000000; font-family: monospace;"
        currency_keys = {
            "net_profit", "total_profit", "total_loss", "avg_win_trade", "avg_loss_trade",
            "avg_win_long_trade", "avg_loss_long_trade", "avg_win_short_trade", "avg_loss_short_trade",
            "largest_win_trade", "largest_loss_trade", "max_drawdown_abs",
        }
        percent_keys = {"max_drawdown_pct"}
        int_keys = {
            "total_trades", "n_long", "n_short", "n_long_won", "n_long_lost",
            "n_short_won", "n_short_lost",
        }
        for key, lbl in self._result_labels.items():
            val = metrics.get(key, 0)
            if key in currency_keys:
                text = f"${val:,.2f}" if isinstance(val, (int, float)) else str(val)
            elif key in percent_keys:
                text = f"{val:.2f}%" if isinstance(val, (int, float)) else str(val)
            elif key in int_keys:
                text = f"{int(val)}" if isinstance(val, (int, float)) else str(val)
            elif isinstance(val, float):
                text = f"{val:.4f}" if (abs(val) < 1e-4 or abs(val) > 1e4) else f"{val:.2f}"
            else:
                text = str(val)
            lbl.setText(text)
            if isinstance(val, (int, float)):
                if val > 0 and key in (
                    "net_profit", "total_profit", "profit_factor", "sharpe_ratio", "z_score",
                    "avg_win_trade", "avg_win_long_trade", "avg_win_short_trade",
                    "largest_win_trade", "n_long_won", "n_short_won", "total_trades", "n_long", "n_short",
                ):
                    lbl.setStyleSheet(positive_style)
                elif val < 0:
                    lbl.setStyleSheet(negative_style)
                else:
                    lbl.setStyleSheet(neutral_style)
            else:
                lbl.setStyleSheet(neutral_style)
        equity_curve = result.get("equity_curve", [])
        if equity_curve:
            points = [(i, eq) for i, eq in enumerate(equity_curve)]
            self.equity_curve_final.set_points(points)
        self._log("Final backtest completed.", "result")
        for k, v in metrics.items():
            self._log_file.write("result", f"{k}: {v}")
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
