"""
NTCP Main Window — PyQt6 cockpit with 5 tabs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

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
    MA_SPECTRUM,
    MACD_FAST_PERIOD,
    MACD_SLOW_PERIOD,
    MACD_SIGNAL_PERIOD,
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

        layout.addWidget(factor_grp)

        # Feature groups (checkboxes) — in scrollable area
        feat_grp = QGroupBox("Feature Groups")
        fl = QVBoxLayout(feat_grp)

        self._feat_checks: dict[str, QCheckBox] = {}
        groups = [
            ("M5 Moving Averages (11)", "m5_ma", True),
            ("M5 Standard Deviations (11)", "m5_sd", True),
            ("M5 Slopes (11)", "m5_slope", True),
            ("M5 RSI / ATR / Vol Z-Score", "m5_misc", True),
            ("M5 MACD (line/signal/hist)", "m5_macd", True),
            ("M5 Candle Geometry", "m5_candle", True),
            ("M5 Squeeze / Daily H-L", "m5_range", True),
            ("M5 Cyclic Time", "m5_time", True),
            ("M5 Hurst Exponent", "m5_hurst", True),
            ("M5 Market Regime (4)", "m5_regime", True),
            ("STF Moving Averages (11)", "stf_ma", True),
            ("STF Standard Deviations (11)", "stf_sd", True),
            ("STF Slopes (11)", "stf_slope", True),
            ("STF RSI / ATR / Vol / OHLC", "stf_misc", True),
            ("STF MACD (line/signal/hist)", "stf_macd", True),
            ("STF Squeeze", "stf_squeeze", True),
            ("STF Hurst Exponent", "stf_hurst", True),
        ]
        for label, key, default in groups:
            cb = QCheckBox(label)
            cb.setChecked(default)
            self._feat_checks[key] = cb
            fl.addWidget(cb)

        layout.addWidget(feat_grp)

        # Feature parameters group
        param_grp = QGroupBox("Feature Parameters")
        pg = QGridLayout(param_grp)

        pg.addWidget(QLabel("MACD Fast:"), 0, 0)
        self._macd_fast = QSpinBox()
        self._macd_fast.setRange(2, 100)
        self._macd_fast.setValue(MACD_FAST_PERIOD)
        pg.addWidget(self._macd_fast, 0, 1)

        pg.addWidget(QLabel("MACD Slow:"), 0, 2)
        self._macd_slow = QSpinBox()
        self._macd_slow.setRange(2, 200)
        self._macd_slow.setValue(MACD_SLOW_PERIOD)
        pg.addWidget(self._macd_slow, 0, 3)

        pg.addWidget(QLabel("MACD Signal:"), 1, 0)
        self._macd_signal = QSpinBox()
        self._macd_signal.setRange(2, 100)
        self._macd_signal.setValue(MACD_SIGNAL_PERIOD)
        pg.addWidget(self._macd_signal, 1, 1)

        pg.addWidget(QLabel("Hurst Window:"), 1, 2)
        self._hurst_window = QSpinBox()
        self._hurst_window.setRange(10, 500)
        self._hurst_window.setValue(HURST_WINDOW)
        pg.addWidget(self._hurst_window, 1, 3)

        pg.addWidget(QLabel("Regime Clusters:"), 2, 0)
        self._regime_clusters = QSpinBox()
        self._regime_clusters.setRange(2, 10)
        self._regime_clusters.setValue(REGIME_N_CLUSTERS)
        pg.addWidget(self._regime_clusters, 2, 1)

        pg.addWidget(QLabel("Rolling Window:"), 2, 2)
        self._rolling_window = QSpinBox()
        self._rolling_window.setRange(0, 1_000_000)
        self._rolling_window.setSingleStep(10000)
        self._rolling_window.setValue(0)
        self._rolling_window.setSpecialValueText("No limit")
        pg.addWidget(self._rolling_window, 2, 3)

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
        self._lr.setValue(0.001)
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

        self._progress = QProgressBar()
        self._progress.setValue(0)
        cl.addWidget(self._progress)

        layout.addWidget(ctrl_grp)
        layout.addStretch()
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
            stf_factor=self._stf_min.value(),
            lookback=self._lookback.value(),
            rolling_window_size=self._rolling_window.value(),
            hurst_window=self._hurst_window.value(),
            regime_n_clusters=self._regime_clusters.value(),
            macd_fast=self._macd_fast.value(),
            macd_slow=self._macd_slow.value(),
            macd_signal=self._macd_signal.value(),
            active_feature_groups=self._get_active_feature_groups(),
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

        self._training_worker = TrainingWorker(data_cfg, model_cfg, train_cfg)
        self._training_worker.log_signal.connect(self._log.log)
        self._training_worker.progress_signal.connect(self._on_train_progress)
        self._training_worker.finished_signal.connect(self._on_train_finished)
        self._training_worker.start()
        self._log.log("Training started.")

    def _stop_training(self) -> None:
        if self._training_worker:
            self._training_worker.request_stop()
            self._log.log("Stop requested...")

    def _on_train_progress(self, epoch: int, total: int) -> None:
        pct = int(epoch / max(total, 1) * 100)
        self._progress.setValue(pct)

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

        self._crv_spin = QDoubleSpinBox()
        self._crv_spin.setRange(0.5, 10.0)
        self._crv_spin.setDecimals(2)
        self._crv_spin.setValue(1.5)
        ctrl.addWidget(QLabel("CRV Threshold:"))
        ctrl.addWidget(self._crv_spin)

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
            n = len(dataset)
            val_size = int(n * self._val_slider.value() / 100.0)
            train_size = n - val_size

            # Extract validation portion
            val_feats = dataset.features[train_size:].numpy()
            val_tgts = dataset.targets[train_size:].numpy()
            val_ds = NTCPDataset(val_feats, val_tgts, lookback=data_cfg.lookback)

            model_cfg = ModelConfig(
                input_size=result.num_features,
                hidden_size=self._hidden.value(),
                num_layers=self._layers.value(),
                dropout=self._dropout.value(),
                cell_type="GRU" if self._gru_radio.isChecked() else "LSTM",
            )

            self._backtest_worker = BacktestWorker(
                model_state=result.model_state,
                model_cfg=model_cfg,
                dataset=val_ds,
                target_cols=result.target_cols,
                crv_threshold=self._crv_spin.value(),
                use_cuda=self._cuda_check.isChecked(),
            )
            self._backtest_worker.log_signal.connect(self._log.log)
            self._backtest_worker.finished_signal.connect(self._on_backtest_finished)
            self._backtest_worker.start()

        except Exception as e:
            self._log.log(f"Backtest setup error: {e}", "error")

    def _on_backtest_finished(self, metrics: Optional[BacktestMetrics]) -> None:
        if metrics is None:
            return

        self._equity_widget.set_data(metrics.equity_curve.tolist())

        display = {
            "Total Trades": str(metrics.total_trades),
            "Win Rate": f"{metrics.win_rate*100:.1f}%",
            "Profit Factor": f"{metrics.profit_factor:.3f}",
            "Max Drawdown": f"{metrics.max_drawdown:.6f}",
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
