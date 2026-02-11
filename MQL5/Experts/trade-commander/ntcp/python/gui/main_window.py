"""
NTCP Main Window — PyQt6 cockpit with 5 tabs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
import torch
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..backtest import BacktestMetrics
from ..config import (
    HURST_WINDOW,
    REGIME_N_CLUSTERS,
    DataConfig,
    ExportConfig,
    ModelConfig,
    TrainConfig,
)
from ..data_manager import NTCPDataManager, NTCPDataset
from ..trainer import TrainingResult
from .widgets import EquityCurveWidget, LogWidget, MetricsTable
from .workers import BacktestWorker, ExportWorker, TrainingWorker

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent.parent / "logs"


class NTCPMainWindow(QMainWindow):
    """Main application window with 5 tabs + persistent log footer."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("NTCP — Neural Trading Consultant Platform")
        self.setMinimumSize(1000, 700)

        # State
        self._training_results: list[TrainingResult] = []
        self._current_result: Optional[TrainingResult] = None
        self._val_dataset: Optional[NTCPDataset] = None
        self._training_worker: Optional[TrainingWorker] = None
        self._backtest_worker: Optional[BacktestWorker] = None
        self._export_worker: Optional[ExportWorker] = None
        self._data_bar_count: int = 0
        self._last_dm: Optional[NTCPDataManager] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # Tabs
        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab_data(), "Data & Split")
        self._tabs.addTab(self._build_tab_features(), "Features & Factors")
        self._tabs.addTab(self._build_tab_training(), "Training & Hardware")
        self._tabs.addTab(self._build_tab_backtest(), "Backtest & Validation")
        self._tabs.addTab(self._build_tab_export(), "Export")
        splitter.addWidget(self._tabs)

        # Log footer
        self._log = LogWidget(log_dir=LOG_DIR)
        splitter.addWidget(self._log)
        splitter.setSizes([500, 200])

        # Set initial visibility for strategy-dependent widgets
        self._on_strategy_changed(0)
        self._log.log("NTCP initialized.")

    # ------------------------------------------------------------------
    # Tab 1: Data & Split
    # ------------------------------------------------------------------

    def _build_tab_data(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # CSV file picker
        grp = QGroupBox("M5 CSV Data")
        g = QGridLayout(grp)

        g.addWidget(QLabel("CSV Path:"), 0, 0)
        self._csv_path = QLineEdit()
        self._csv_path.setPlaceholderText("Select M5 CSV export...")
        g.addWidget(self._csv_path, 0, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_csv)
        g.addWidget(browse_btn, 0, 2)

        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self._load_data)
        g.addWidget(load_btn, 1, 0)

        self._data_info_label = QLabel("No data loaded.")
        g.addWidget(self._data_info_label, 1, 1, 1, 2)

        layout.addWidget(grp)

        # Symbol data
        sym_grp = QGroupBox("Symbol Data")
        sym_layout = QGridLayout(sym_grp)

        sym_layout.addWidget(QLabel("Spread:"), 0, 0)
        self._spread_spin = QDoubleSpinBox()
        self._spread_spin.setRange(0.0, 100.0)
        self._spread_spin.setDecimals(2)
        self._spread_spin.setSingleStep(0.01)
        self._spread_spin.setValue(0.20)
        sym_layout.addWidget(self._spread_spin, 0, 1)

        sym_layout.addWidget(QLabel("Point:"), 0, 2)
        self._point_spin = QDoubleSpinBox()
        self._point_spin.setRange(0.00001, 100.0)
        self._point_spin.setDecimals(5)
        self._point_spin.setSingleStep(0.00001)
        self._point_spin.setValue(0.01)
        sym_layout.addWidget(self._point_spin, 0, 3)

        sym_layout.addWidget(QLabel("Tick Value $:"), 0, 4)
        self._tick_value_spin = QDoubleSpinBox()
        self._tick_value_spin.setRange(0.001, 10000.0)
        self._tick_value_spin.setDecimals(3)
        self._tick_value_spin.setSingleStep(0.1)
        self._tick_value_spin.setValue(1.00)
        sym_layout.addWidget(self._tick_value_spin, 0, 5)

        sym_layout.addWidget(QLabel("Slippage (pts):"), 1, 0)
        self._slippage_spin = QDoubleSpinBox()
        self._slippage_spin.setRange(0.0, 100.0)
        self._slippage_spin.setDecimals(1)
        self._slippage_spin.setSingleStep(1.0)
        self._slippage_spin.setValue(0.0)
        sym_layout.addWidget(self._slippage_spin, 1, 1)

        layout.addWidget(sym_grp)

        # Validation split
        split_grp = QGroupBox("Validation Split")
        sg = QHBoxLayout(split_grp)
        sg.addWidget(QLabel("Val %:"))
        self._val_slider = QSlider(Qt.Orientation.Horizontal)
        self._val_slider.setRange(5, 40)
        self._val_slider.setValue(15)
        self._val_slider.setTickInterval(5)
        self._val_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._val_slider.valueChanged.connect(self._update_val_label)
        sg.addWidget(self._val_slider)
        self._val_label = QLabel("15%")
        sg.addWidget(self._val_label)
        self._split_count_label = QLabel("")
        sg.addWidget(self._split_count_label)
        layout.addWidget(split_grp)

        layout.addStretch()
        return tab

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select M5 CSV", "", "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self._csv_path.setText(path)

    def _load_data(self) -> None:
        path = self._csv_path.text().strip()
        if not path:
            self._log.log("No CSV path specified.", "warn")
            return

        try:
            cfg = DataConfig(m5_path=path)
            dm = NTCPDataManager(cfg)
            dm.load()
            info = dm.get_info()
            self._data_bar_count = info["bars"]
            self._data_info_label.setText(
                f"{info['bars']} bars | "
                f"{info['date_start']} to {info['date_end']}"
            )
            self._log.log(f"Loaded {info['bars']} M5 bars.")
            self._update_split_count()
        except Exception as e:
            self._log.log(f"Load error: {e}", "error")

    def _update_val_label(self, val: int) -> None:
        self._val_label.setText(f"{val}%")
        self._update_split_count()

    def _update_split_count(self) -> None:
        if self._data_bar_count > 0:
            val_pct = self._val_slider.value() / 100.0
            val_bars = int(self._data_bar_count * val_pct)
            train_bars = self._data_bar_count - val_bars
            self._split_count_label.setText(
                f"(~{train_bars} train / ~{val_bars} val)"
            )

    def _on_strategy_changed(self, index: int) -> None:
        """Show/hide MA Filter parameter group based on selected strategy."""
        strategy = self._strategy_combo.currentData()
        is_maf = strategy == "mafilter"
        self._mafilter_grp.setVisible(is_maf)
        # Backtest tab: swap CRV (trendcatcher) ↔ Cls Confidence (mafilter)
        if hasattr(self, "_crv_label"):
            self._crv_label.setVisible(not is_maf)
            self._crv_spin.setVisible(not is_maf)
            self._cls_thresh_label.setVisible(is_maf)
            self._cls_thresh_spin.setVisible(is_maf)

    # ------------------------------------------------------------------
    # Tab 2: Features & Factors
    # ------------------------------------------------------------------

    def _build_tab_features(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # STF Factor range
        factor_grp = QGroupBox("STF Factor Range")
        fg = QGridLayout(factor_grp)

        fg.addWidget(QLabel("Min Factor:"), 0, 0)
        self._stf_min = QSpinBox()
        self._stf_min.setRange(2, 60)
        self._stf_min.setValue(6)
        fg.addWidget(self._stf_min, 0, 1)

        fg.addWidget(QLabel("Max Factor:"), 0, 2)
        self._stf_max = QSpinBox()
        self._stf_max.setRange(2, 60)
        self._stf_max.setValue(24)
        fg.addWidget(self._stf_max, 0, 3)

        fg.addWidget(QLabel("Lookback:"), 1, 0)
        self._lookback = QSpinBox()
        self._lookback.setRange(5, 100)
        self._lookback.setValue(20)
        fg.addWidget(self._lookback, 1, 1)

        fg.addWidget(QLabel("Base TF (min):"), 1, 2)
        self._base_tf = QSpinBox()
        self._base_tf.setRange(1, 1440)
        self._base_tf.setValue(1)
        fg.addWidget(self._base_tf, 1, 3)

        layout.addWidget(factor_grp)

        # Feature groups (checkboxes) — in scrollable area
        feat_grp = QGroupBox("Feature Groups")
        fl = QVBoxLayout(feat_grp)

        self._feat_checks: dict[str, QCheckBox] = {}
        groups = [
            ("M5 Moving Averages", "m5_ma", True),
            ("M5 Standard Deviations", "m5_sd", True),
            ("M5 Slopes", "m5_slope", True),
            ("M5 RSI / ATR / Vol Z-Score", "m5_misc", True),
            ("M5 Candle Geometry", "m5_candle", True),
            ("M5 Squeeze / Daily H-L", "m5_range", True),
            ("M5 Cyclic Time", "m5_time", True),
            ("M5 Hurst Exponent", "m5_hurst", True),
            ("M5 Market Regime", "m5_regime", True),
            ("STF Moving Averages", "stf_ma", True),
            ("STF Standard Deviations", "stf_sd", True),
            ("STF Slopes", "stf_slope", True),
            ("STF RSI / ATR / Vol / OHLC", "stf_misc", True),
            ("STF Squeeze", "stf_squeeze", True),
            ("STF Hurst Exponent", "stf_hurst", True),
        ]
        for label, key, default in groups:
            cb = QCheckBox(label)
            cb.setChecked(default)
            self._feat_checks[key] = cb
            fl.addWidget(cb)

        layout.addWidget(feat_grp)

        # MA Spectrum
        ma_grp = QGroupBox("MA Spectrum")
        mg = QGridLayout(ma_grp)

        mg.addWidget(QLabel("Min MA:"), 0, 0)
        self._ma_min = QSpinBox()
        self._ma_min.setRange(2, 100)
        self._ma_min.setValue(5)
        mg.addWidget(self._ma_min, 0, 1)

        mg.addWidget(QLabel("Max MA:"), 0, 2)
        self._ma_max = QSpinBox()
        self._ma_max.setRange(10, 2000)
        self._ma_max.setValue(500)
        mg.addWidget(self._ma_max, 0, 3)

        mg.addWidget(QLabel("MA Count:"), 1, 0)
        self._ma_count = QSpinBox()
        self._ma_count.setRange(2, 30)
        self._ma_count.setValue(11)
        mg.addWidget(self._ma_count, 1, 1)

        mg.addWidget(QLabel("Signal MAs (N):"), 1, 2)
        self._signal_ma_count = QSpinBox()
        self._signal_ma_count.setRange(1, 20)
        self._signal_ma_count.setValue(4)
        mg.addWidget(self._signal_ma_count, 1, 3)

        layout.addWidget(ma_grp)

        # Strategy
        strat_grp = QGroupBox("Strategy")
        strat_layout = QVBoxLayout(strat_grp)
        strat_row = QHBoxLayout()
        strat_row.addWidget(QLabel("Strategy:"))
        self._strategy_combo = QComboBox()
        self._strategy_combo.addItem("Trendcatcher", "trendcatcher")
        self._strategy_combo.addItem("MA Filter", "mafilter")
        self._strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strat_row.addWidget(self._strategy_combo)
        strat_row.addStretch()
        strat_layout.addLayout(strat_row)

        # MA Filter parameters (shown/hidden based on strategy selection)
        self._mafilter_grp = QGroupBox("MA Filter Parameters")
        mf = QGridLayout(self._mafilter_grp)

        mf.addWidget(QLabel("NH (BW SMA window):"), 0, 0)
        self._mf_nh = QSpinBox()
        self._mf_nh.setRange(10, 10000)
        self._mf_nh.setValue(1000)
        mf.addWidget(self._mf_nh, 0, 1)

        mf.addWidget(QLabel("Min Slope Factor:"), 0, 2)
        self._mf_fac_slope = QDoubleSpinBox()
        self._mf_fac_slope.setRange(0.0, 5.0)
        self._mf_fac_slope.setDecimals(2)
        self._mf_fac_slope.setSingleStep(0.1)
        self._mf_fac_slope.setValue(0.5)
        mf.addWidget(self._mf_fac_slope, 0, 3)

        mf.addWidget(QLabel("Fslope Thresh:"), 1, 0)
        self._mf_fslope_thresh = QDoubleSpinBox()
        self._mf_fslope_thresh.setRange(0.0, 1.0)
        self._mf_fslope_thresh.setDecimals(2)
        self._mf_fslope_thresh.setSingleStep(0.05)
        self._mf_fslope_thresh.setValue(0.8)
        mf.addWidget(self._mf_fslope_thresh, 1, 1)

        mf.addWidget(QLabel("Max BW Factor:"), 1, 2)
        self._mf_max_bw_factor = QDoubleSpinBox()
        self._mf_max_bw_factor.setRange(0.1, 10.0)
        self._mf_max_bw_factor.setDecimals(2)
        self._mf_max_bw_factor.setSingleStep(0.1)
        self._mf_max_bw_factor.setValue(1.5)
        mf.addWidget(self._mf_max_bw_factor, 1, 3)

        mf.addWidget(QLabel("StdDev Factor:"), 2, 0)
        self._mf_stddev_factor = QDoubleSpinBox()
        self._mf_stddev_factor.setRange(0.0, 5.0)
        self._mf_stddev_factor.setDecimals(2)
        self._mf_stddev_factor.setSingleStep(0.1)
        self._mf_stddev_factor.setValue(1.0)
        mf.addWidget(self._mf_stddev_factor, 2, 1)

        mf.addWidget(QLabel("Exit Option:"), 2, 2)
        self._mf_exit_option = QComboBox()
        self._mf_exit_option.addItem("A: StdDev Exit", 0)
        self._mf_exit_option.addItem("B: Fslope Exit", 1)
        mf.addWidget(self._mf_exit_option, 2, 3)

        self._mafilter_grp.setVisible(False)
        strat_layout.addWidget(self._mafilter_grp)
        layout.addWidget(strat_grp)

        # Feature parameters
        param_grp = QGroupBox("Feature Parameters")
        pg = QGridLayout(param_grp)

        pg.addWidget(QLabel("Hurst Window:"), 0, 0)
        self._hurst_window = QSpinBox()
        self._hurst_window.setRange(10, 500)
        self._hurst_window.setValue(HURST_WINDOW)
        pg.addWidget(self._hurst_window, 0, 1)

        pg.addWidget(QLabel("Regime Clusters:"), 0, 2)
        self._regime_clusters = QSpinBox()
        self._regime_clusters.setRange(2, 10)
        self._regime_clusters.setValue(REGIME_N_CLUSTERS)
        pg.addWidget(self._regime_clusters, 0, 3)

        pg.addWidget(QLabel("Rolling Window:"), 1, 0)
        self._rolling_window = QSpinBox()
        self._rolling_window.setRange(0, 1_000_000)
        self._rolling_window.setSingleStep(10000)
        self._rolling_window.setValue(0)
        self._rolling_window.setSpecialValueText("No limit")
        pg.addWidget(self._rolling_window, 1, 1)

        layout.addWidget(param_grp)
        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # Tab 3: Training & Hardware
    # ------------------------------------------------------------------

    def _build_tab_training(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Hardware
        hw_grp = QGroupBox("Hardware")
        hl = QGridLayout(hw_grp)

        self._cuda_check = QCheckBox("Use CUDA")
        cuda_available = torch.cuda.is_available()
        self._cuda_check.setChecked(cuda_available)
        self._cuda_check.setEnabled(cuda_available)
        hl.addWidget(self._cuda_check, 0, 0)

        gpu_info = torch.cuda.get_device_name(0) if cuda_available else "No CUDA device"
        hl.addWidget(QLabel(f"GPU: {gpu_info}"), 0, 1)

        layout.addWidget(hw_grp)

        # Hyperparameters
        hp_grp = QGroupBox("Hyperparameters")
        hg = QGridLayout(hp_grp)

        hg.addWidget(QLabel("Epochs:"), 0, 0)
        self._epochs = QSpinBox()
        self._epochs.setRange(1, 1000)
        self._epochs.setValue(100)
        hg.addWidget(self._epochs, 0, 1)

        hg.addWidget(QLabel("Batch Size:"), 0, 2)
        self._batch = QSpinBox()
        self._batch.setRange(16, 2048)
        self._batch.setSingleStep(16)
        self._batch.setValue(256)
        hg.addWidget(self._batch, 0, 3)

        hg.addWidget(QLabel("Learning Rate:"), 1, 0)
        self._lr = QDoubleSpinBox()
        self._lr.setRange(0.00001, 0.1)
        self._lr.setDecimals(5)
        self._lr.setSingleStep(0.0001)
        self._lr.setValue(0.0001)
        hg.addWidget(self._lr, 1, 1)

        hg.addWidget(QLabel("Hidden Size:"), 1, 2)
        self._hidden = QSpinBox()
        self._hidden.setRange(16, 1024)
        self._hidden.setSingleStep(16)
        self._hidden.setValue(128)
        hg.addWidget(self._hidden, 1, 3)

        hg.addWidget(QLabel("Layers:"), 2, 0)
        self._layers = QSpinBox()
        self._layers.setRange(1, 8)
        self._layers.setValue(2)
        hg.addWidget(self._layers, 2, 1)

        hg.addWidget(QLabel("Dropout:"), 2, 2)
        self._dropout = QDoubleSpinBox()
        self._dropout.setRange(0.0, 0.8)
        self._dropout.setDecimals(2)
        self._dropout.setSingleStep(0.05)
        self._dropout.setValue(0.2)
        hg.addWidget(self._dropout, 2, 3)

        hg.addWidget(QLabel("Patience:"), 3, 0)
        self._patience = QSpinBox()
        self._patience.setRange(1, 50)
        self._patience.setValue(10)
        hg.addWidget(self._patience, 3, 1)

        hg.addWidget(QLabel("Cell Type:"), 3, 2)
        self._gru_radio = QRadioButton("GRU")
        self._gru_radio.setChecked(True)
        self._lstm_radio = QRadioButton("LSTM")
        cell_row = QHBoxLayout()
        cell_row.addWidget(self._gru_radio)
        cell_row.addWidget(self._lstm_radio)
        cell_w = QWidget()
        cell_w.setLayout(cell_row)
        hg.addWidget(cell_w, 3, 3)

        layout.addWidget(hp_grp)

        # Controls
        ctrl_grp = QGroupBox("Training Control")
        cl = QHBoxLayout(ctrl_grp)

        self._start_btn = QPushButton("Start Training")
        self._start_btn.setObjectName("startButton")
        self._start_btn.clicked.connect(self._start_training)
        cl.addWidget(self._start_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setObjectName("stopButton")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)
        cl.addWidget(self._stop_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear_all)
        cl.addWidget(self._clear_btn)

        self._progress = QProgressBar()
        self._progress.setValue(0)
        cl.addWidget(self._progress)

        layout.addWidget(ctrl_grp)

        # Live training monitor
        self._monitor_plot = pg.PlotWidget()
        self._monitor_plot.setBackground("#000000")
        self._monitor_plot.showGrid(x=True, y=True, alpha=0.15)
        self._monitor_plot.setTitle(
            "Pred vs Actual — MFE 10-bar", color="#888888",
        )
        self._monitor_plot.setLabel("left", "Value (x100)")
        self._monitor_plot.setLabel("bottom", "Sample (sorted)")
        self._monitor_plot.addLegend(offset=(10, 10))
        self._actual_curve = self._monitor_plot.plot(
            [], pen=pg.mkPen(color="#44ff44", width=1.5), name="Actual",
        )
        self._pred_curve = self._monitor_plot.plot(
            [], pen=pg.mkPen(color="#44ddff", width=1.5), name="Predicted",
        )
        self._monitor_plot.addLine(
            y=0, pen=pg.mkPen(color="#333333", width=1, style=Qt.PenStyle.DashLine),
        )
        layout.addWidget(self._monitor_plot, 1)

        return tab

    def _get_active_feature_groups(self) -> list[str]:
        """Return list of active feature groups from checkboxes.
        If all are checked, return empty list (meaning 'all')."""
        checked = [key for key, cb in self._feat_checks.items() if cb.isChecked()]
        if len(checked) == len(self._feat_checks):
            return []  # all checked = no filtering
        return checked

    def _gather_configs(self) -> tuple[DataConfig, ModelConfig, TrainConfig]:
        data_cfg = DataConfig(
            m5_path=self._csv_path.text().strip(),
            base_tf_minutes=self._base_tf.value(),
            stf_factor=self._stf_min.value(),
            lookback=self._lookback.value(),
            rolling_window_size=self._rolling_window.value(),
            hurst_window=self._hurst_window.value(),
            regime_n_clusters=self._regime_clusters.value(),
            ma_min=self._ma_min.value(),
            ma_max=self._ma_max.value(),
            ma_count=self._ma_count.value(),
            signal_ma_count=self._signal_ma_count.value(),
            strategy=self._strategy_combo.currentData(),
            mafilter_nh=self._mf_nh.value(),
            mafilter_fac_slope=self._mf_fac_slope.value(),
            mafilter_fslope_thresh=self._mf_fslope_thresh.value(),
            mafilter_max_bw_factor=self._mf_max_bw_factor.value(),
            mafilter_stddev_factor=self._mf_stddev_factor.value(),
            mafilter_exit_option=self._mf_exit_option.currentData(),
            active_feature_groups=self._get_active_feature_groups(),
            spread=self._spread_spin.value(),
            slippage=self._slippage_spin.value(),
            point=self._point_spin.value(),
            tick_value=self._tick_value_spin.value(),
        )
        model_cfg = ModelConfig(
            hidden_size=self._hidden.value(),
            num_layers=self._layers.value(),
            dropout=self._dropout.value(),
            cell_type="GRU" if self._gru_radio.isChecked() else "LSTM",
        )
        train_cfg = TrainConfig(
            epochs=self._epochs.value(),
            batch_size=self._batch.value(),
            lr=self._lr.value(),
            val_split=self._val_slider.value() / 100.0,
            patience=self._patience.value(),
            use_cuda=self._cuda_check.isChecked(),
            stf_factor_min=self._stf_min.value(),
            stf_factor_max=self._stf_max.value(),
        )
        return data_cfg, model_cfg, train_cfg

    def _start_training(self) -> None:
        csv_path = self._csv_path.text().strip()
        if not csv_path:
            self._log.log("Set CSV path first.", "warn")
            return

        data_cfg, model_cfg, train_cfg = self._gather_configs()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setValue(0)

        # Clear monitor plot
        self._actual_curve.setData([])
        self._pred_curve.setData([])

        self._training_worker = TrainingWorker(data_cfg, model_cfg, train_cfg)
        self._training_worker.log_signal.connect(self._log.log)
        self._training_worker.progress_signal.connect(self._on_train_progress)
        self._training_worker.sample_signal.connect(self._on_train_sample)
        self._training_worker.finished_signal.connect(self._on_train_finished)
        self._training_worker.start()
        self._log.log("Training started.")

    def _stop_training(self) -> None:
        if self._training_worker:
            self._training_worker.request_stop()
            self._log.log("Stop requested...")

    def _clear_all(self) -> None:
        """Reset all results, graphs, tables, and logs."""
        self._training_results.clear()
        self._current_result = None
        self._val_dataset = None
        self._last_dm = None

        # Monitor plot
        self._actual_curve.setData([])
        self._pred_curve.setData([])
        self._monitor_plot.setTitle("Pred vs Actual", color="#888888")

        # Progress
        self._progress.setValue(0)

        # Backtest tab
        self._factor_combo.clear()
        self._equity_widget.set_data(None)
        self._metrics_table._table.setRowCount(0)
        self._target_check_table.setRowCount(0)

        # Log
        self._log.clear()
        self._log.log("Cleared all results.")

    def _on_train_progress(self, epoch: int, total: int) -> None:
        pct = int(epoch / max(total, 1) * 100)
        self._progress.setValue(pct)

    def _on_train_sample(self, epoch, reg_actuals, reg_preds, cls_actuals, cls_probs) -> None:
        """Update live monitor plot — confidence plot or regression plot."""
        strategy = self._strategy_combo.currentData()

        if strategy == "mafilter":
            # Confidence plot: long signal probability vs actual label
            probs = cls_probs[:, 0]  # long confidence
            labels = cls_actuals[:, 0]  # long actual
            idx = np.argsort(probs)
            self._pred_curve.setData(probs[idx])
            self._actual_curve.setData(labels[idx])
            self._monitor_plot.setTitle(
                f"Cls Confidence — Long Signal (Epoch {epoch})", color="#888888",
            )
            self._monitor_plot.setLabel("left", "Probability")
            self._monitor_plot.setLabel("bottom", "Sample (sorted by confidence)")
        else:
            # Regression: MFE 10-bar pred vs actual
            idx = np.argsort(reg_actuals)
            self._actual_curve.setData(reg_actuals[idx])
            self._pred_curve.setData(reg_preds[idx])
            self._monitor_plot.setTitle(
                f"Pred vs Actual — MFE 10-bar (Epoch {epoch})", color="#888888",
            )
            self._monitor_plot.setLabel("left", "Value (x100)")
            self._monitor_plot.setLabel("bottom", "Sample (sorted by actual)")

    def _on_train_finished(self, results: list) -> None:
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(100)

        self._training_results = results
        if results:
            best = min(results, key=lambda r: r.best_val_loss)
            self._current_result = best
            self._log.log(
                f"Training done. Best factor={best.stf_factor}, "
                f"val_loss={best.best_val_loss:.6f}"
            )
            # Populate factor selector in backtest tab
            self._factor_combo.clear()
            for r in results:
                self._factor_combo.addItem(
                    f"Factor {r.stf_factor} (val={r.best_val_loss:.6f})",
                    r,
                )
        else:
            self._log.log("Training returned no results.", "warn")

    # ------------------------------------------------------------------
    # Tab 4: Backtest & Validation
    # ------------------------------------------------------------------

    def _build_tab_backtest(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Controls
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Factor:"))
        self._factor_combo = QComboBox()
        ctrl.addWidget(self._factor_combo)

        self._crv_label = QLabel("CRV Threshold:")
        self._crv_spin = QDoubleSpinBox()
        self._crv_spin.setRange(0.5, 10.0)
        self._crv_spin.setDecimals(2)
        self._crv_spin.setValue(1.0)
        ctrl.addWidget(self._crv_label)
        ctrl.addWidget(self._crv_spin)

        self._cls_thresh_label = QLabel("Cls Confidence:")
        self._cls_thresh_spin = QDoubleSpinBox()
        self._cls_thresh_spin.setRange(0.0, 1.0)
        self._cls_thresh_spin.setDecimals(2)
        self._cls_thresh_spin.setSingleStep(0.05)
        self._cls_thresh_spin.setValue(0.50)
        ctrl.addWidget(self._cls_thresh_label)
        ctrl.addWidget(self._cls_thresh_spin)

        ctrl.addWidget(QLabel("Lot Size:"))
        self._lot_size_spin = QDoubleSpinBox()
        self._lot_size_spin.setRange(0.01, 100.0)
        self._lot_size_spin.setDecimals(2)
        self._lot_size_spin.setSingleStep(0.01)
        self._lot_size_spin.setValue(0.10)
        ctrl.addWidget(self._lot_size_spin)

        self._show_dollar_check = QCheckBox("Show in $")
        ctrl.addWidget(self._show_dollar_check)

        self._bt_btn = QPushButton("Run Backtest")
        self._bt_btn.clicked.connect(self._run_backtest)
        ctrl.addWidget(self._bt_btn)
        layout.addLayout(ctrl)

        # Results area: equity + summary on top, target-check table below
        top_split = QSplitter(Qt.Orientation.Horizontal)

        self._equity_widget = EquityCurveWidget()
        top_split.addWidget(self._equity_widget)

        self._metrics_table = MetricsTable()
        top_split.addWidget(self._metrics_table)
        top_split.setSizes([600, 400])

        # Vertical split: equity+metrics on top, target-check on bottom
        vert_split = QSplitter(Qt.Orientation.Vertical)
        vert_split.addWidget(top_split)

        # Target-check table
        tc_grp = QGroupBox("Per-Target Check")
        tc_layout = QVBoxLayout(tc_grp)
        self._target_check_table = QTableWidget()
        self._target_check_table.setColumnCount(9)
        self._target_check_table.setHorizontalHeaderLabels([
            "Target", "R2", "MSE", "MAE", "Corr",
            "Pred Mean", "Actual Mean", "Pred Std", "Ratio",
        ])
        self._target_check_table.horizontalHeader().setStretchLastSection(True)
        self._target_check_table.setAlternatingRowColors(True)
        tc_layout.addWidget(self._target_check_table)
        vert_split.addWidget(tc_grp)
        vert_split.setSizes([400, 200])

        layout.addWidget(vert_split)
        return tab

    def _run_backtest(self) -> None:
        idx = self._factor_combo.currentIndex()
        if idx < 0 or not self._training_results:
            self._log.log("Train a model first.", "warn")
            return

        result: TrainingResult = self._factor_combo.itemData(idx)
        if result.model_state is None:
            self._log.log("No model checkpoint available.", "warn")
            return

        # Rebuild validation dataset for this factor
        data_cfg, _, _ = self._gather_configs()
        data_cfg.stf_factor = result.stf_factor
        self._log.log("Building validation dataset...")

        try:
            dm = NTCPDataManager(data_cfg)
            dataset = dm.run_pipeline(log_callback=lambda m: self._log.log(m))
            self._last_dm = dm
            n = len(dataset)
            val_size = int(n * self._val_slider.value() / 100.0)
            train_size = n - val_size
            lookback = data_cfg.lookback

            # Extract validation portion
            val_feats = dataset.features[train_size:].numpy()
            val_tgts = dataset.targets[train_size:].numpy()
            val_cls_tgts = dataset.cls_targets[train_size:].numpy()
            val_ds = NTCPDataset(val_feats, val_tgts, lookback=lookback,
                                 cls_targets=val_cls_tgts)

            # Entry prices aligned to validation samples
            val_entry_prices = None
            if dm.raw_closes is not None:
                val_entry_prices = dm.raw_closes[
                    train_size + lookback - 1 : train_size + val_size + lookback - 1
                ]

            model_cfg = ModelConfig(
                input_size=result.num_features,
                hidden_size=self._hidden.value(),
                num_layers=self._layers.value(),
                dropout=self._dropout.value(),
                cell_type="GRU" if self._gru_radio.isChecked() else "LSTM",
            )

            show_dollars = self._show_dollar_check.isChecked()
            self._backtest_worker = BacktestWorker(
                model_state=result.model_state,
                model_cfg=model_cfg,
                dataset=val_ds,
                target_cols=result.target_cols,
                crv_threshold=(self._cls_thresh_spin.value()
                              if self._strategy_combo.currentData() == "mafilter"
                              else self._crv_spin.value()),
                use_cuda=self._cuda_check.isChecked(),
                spread=self._spread_spin.value(),
                slippage=self._slippage_spin.value(),
                point=self._point_spin.value(),
                tick_value=self._tick_value_spin.value(),
                lot_size=self._lot_size_spin.value(),
                show_in_dollars=show_dollars,
                entry_prices=val_entry_prices,
                strategy=self._strategy_combo.currentData(),
            )
            self._backtest_worker.log_signal.connect(self._log.log)
            self._backtest_worker.finished_signal.connect(self._on_backtest_finished)
            self._backtest_worker.start()

        except Exception as e:
            self._log.log(f"Backtest setup error: {e}", "error")

    def _on_backtest_finished(self, metrics: Optional[BacktestMetrics]) -> None:
        if metrics is None:
            return

        # Update equity chart label
        if metrics.show_in_dollars:
            self._equity_widget.set_y_label("Equity ($)")
        else:
            self._equity_widget.set_y_label("Equity")

        self._equity_widget.set_data(metrics.equity_curve.tolist())

        def _fmt(val: float) -> str:
            if metrics.show_in_dollars:
                return f"${val:,.2f}"
            return f"{val:.6f}"

        display = {
            "Total Trades": str(metrics.total_trades),
            "Long Trades": str(metrics.long_trades),
            "Short Trades": str(metrics.short_trades),
            "Winners": str(metrics.num_winners),
            "Losers": str(metrics.num_losers),
            "Avg Winner": _fmt(metrics.avg_winner),
            "Avg Loser": _fmt(metrics.avg_loser),
            "Win Rate": f"{metrics.win_rate*100:.1f}%",
            "Profit Factor": f"{metrics.profit_factor:.3f}",
            "Total PnL": _fmt(metrics.total_pnl),
            "Max Drawdown": _fmt(metrics.max_drawdown),
        }
        # Add per-target R^2
        for name, val in metrics.per_target_r2.items():
            display[f"R2 {name}"] = f"{val:.4f}"

        self._metrics_table.set_metrics(display)

        # Populate target-check table
        tc = metrics.target_check
        self._target_check_table.setRowCount(len(tc))
        for row_i, (tgt_name, stats) in enumerate(tc.items()):
            self._target_check_table.setItem(row_i, 0, QTableWidgetItem(tgt_name))
            self._target_check_table.setItem(row_i, 1, QTableWidgetItem(f"{stats['r2']:.4f}"))
            self._target_check_table.setItem(row_i, 2, QTableWidgetItem(f"{stats['mse']:.6f}"))
            self._target_check_table.setItem(row_i, 3, QTableWidgetItem(f"{stats['mae']:.6f}"))
            self._target_check_table.setItem(row_i, 4, QTableWidgetItem(f"{stats['correlation']:.4f}"))
            self._target_check_table.setItem(row_i, 5, QTableWidgetItem(f"{stats['pred_mean']:.6f}"))
            self._target_check_table.setItem(row_i, 6, QTableWidgetItem(f"{stats['actual_mean']:.6f}"))
            self._target_check_table.setItem(row_i, 7, QTableWidgetItem(f"{stats['pred_std']:.6f}"))
            self._target_check_table.setItem(row_i, 8, QTableWidgetItem(f"{stats['ratio']:.4f}"))

    # ------------------------------------------------------------------
    # Tab 5: Export
    # ------------------------------------------------------------------

    def _build_tab_export(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        grp = QGroupBox("Export Settings")
        g = QGridLayout(grp)

        g.addWidget(QLabel("Version:"), 0, 0)
        self._version_input = QLineEdit("1.0.0")
        g.addWidget(self._version_input, 0, 1)

        g.addWidget(QLabel("ONNX Path:"), 1, 0)
        self._onnx_path = QLineEdit("model.onnx")
        g.addWidget(self._onnx_path, 1, 1)
        onnx_browse = QPushButton("Browse...")
        onnx_browse.clicked.connect(lambda: self._browse_save(self._onnx_path, "ONNX (*.onnx)"))
        g.addWidget(onnx_browse, 1, 2)

        g.addWidget(QLabel("MQH Path:"), 2, 0)
        self._mqh_path = QLineEdit("../scaling_params.mqh")
        g.addWidget(self._mqh_path, 2, 1)
        mqh_browse = QPushButton("Browse...")
        mqh_browse.clicked.connect(lambda: self._browse_save(self._mqh_path, "MQH (*.mqh)"))
        g.addWidget(mqh_browse, 2, 2)

        layout.addWidget(grp)

        export_btn = QPushButton("Export ONNX + MQH")
        export_btn.clicked.connect(self._do_export)
        layout.addWidget(export_btn)

        layout.addStretch()
        return tab

    def _browse_save(self, line_edit: QLineEdit, filter_str: str) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As", line_edit.text(), filter_str,
        )
        if path:
            line_edit.setText(path)

    def _do_export(self) -> None:
        if not self._current_result or self._current_result.model_state is None:
            self._log.log("No trained model to export.", "warn")
            return

        result = self._current_result
        export_cfg = ExportConfig(
            version=self._version_input.text().strip(),
            onnx_path=self._onnx_path.text().strip(),
            mqh_path=self._mqh_path.text().strip(),
        )

        model_cfg = ModelConfig(
            input_size=result.num_features,
            hidden_size=self._hidden.value(),
            num_layers=self._layers.value(),
            dropout=self._dropout.value(),
            cell_type="GRU" if self._gru_radio.isChecked() else "LSTM",
        )

        # Store active feature groups in scaling_params for MQH export
        active_groups = self._get_active_feature_groups()
        if active_groups:
            result.scaling_params["active_feature_groups"] = active_groups

        self._export_worker = ExportWorker(
            export_cfg=export_cfg,
            model_state=result.model_state,
            model_cfg=model_cfg,
            lookback=self._lookback.value(),
            feature_cols=result.feature_cols,
            target_cols=result.target_cols,
            scaling_params=result.scaling_params,
            stf_factor=result.stf_factor,
        )
        self._export_worker.log_signal.connect(self._log.log)
        self._export_worker.finished_signal.connect(
            lambda ok: self._log.log("Export complete." if ok else "Export failed.", "info" if ok else "error")
        )
        self._export_worker.start()
