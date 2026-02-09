"""
NTCP GUI Workers — QThread workers for training and backtesting.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from ..backtest import BacktestMetrics, NTCPBacktester
from ..config import DataConfig, ExportConfig, ModelConfig, TrainConfig
from ..data_manager import NTCPDataManager, NTCPDataset
from ..exporter import NTCPExporter
from ..trainer import NTCPTrainer, TrainingResult


class TrainingWorker(QThread):
    """Background worker for model training."""

    log_signal = pyqtSignal(str, str)         # (message, level)
    progress_signal = pyqtSignal(int, int)     # (current_epoch, total_epochs)
    loss_signal = pyqtSignal(float, float)     # (train_loss, val_loss)
    finished_signal = pyqtSignal(list)          # list[TrainingResult]

    def __init__(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainConfig,
        single_factor: Optional[int] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.single_factor = single_factor
        self._trainer: Optional[NTCPTrainer] = None

    def run(self) -> None:
        def log_cb(msg: str) -> None:
            self.log_signal.emit(msg, "info")

        def progress_cb(epoch: int, total: int, train_loss: float, val_loss: float) -> None:
            self.progress_signal.emit(epoch, total)
            self.loss_signal.emit(train_loss, val_loss)

        self._trainer = NTCPTrainer(
            self.data_cfg, self.model_cfg, self.train_cfg,
            log_callback=log_cb,
        )

        try:
            if self.single_factor is not None:
                result = self._trainer.train_single_factor(
                    self.single_factor, progress_callback=progress_cb,
                )
                self.finished_signal.emit([result])
            else:
                results = self._trainer.train_factor_range(
                    progress_callback=progress_cb,
                )
                self.finished_signal.emit(results)
        except Exception as e:
            self.log_signal.emit(f"Training error: {e}", "error")
            self.finished_signal.emit([])

    def request_stop(self) -> None:
        if self._trainer:
            self._trainer.request_stop()


class BacktestWorker(QThread):
    """Background worker for backtesting."""

    log_signal = pyqtSignal(str, str)
    finished_signal = pyqtSignal(object)  # BacktestMetrics

    def __init__(
        self,
        model_state: dict,
        model_cfg: ModelConfig,
        dataset: NTCPDataset,
        target_cols: list[str],
        crv_threshold: float = 1.5,
        use_cuda: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model_state = model_state
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.target_cols = target_cols
        self.crv_threshold = crv_threshold
        self.use_cuda = use_cuda

    def run(self) -> None:
        try:
            self.log_signal.emit("Running backtest...", "info")
            bt = NTCPBacktester(
                self.model_state, self.model_cfg, self.crv_threshold,
            )
            metrics = bt.run(
                self.dataset, self.target_cols, use_cuda=self.use_cuda,
            )
            self.log_signal.emit(
                f"Backtest complete: {metrics.total_trades} trades, "
                f"WR={metrics.win_rate*100:.1f}%, PF={metrics.profit_factor:.2f}",
                "info",
            )
            self.finished_signal.emit(metrics)
        except Exception as e:
            self.log_signal.emit(f"Backtest error: {e}", "error")
            self.finished_signal.emit(None)


class ExportWorker(QThread):
    """Background worker for ONNX + MQH export."""

    log_signal = pyqtSignal(str, str)
    finished_signal = pyqtSignal(bool)

    def __init__(
        self,
        export_cfg: ExportConfig,
        model_state: dict,
        model_cfg: ModelConfig,
        lookback: int,
        feature_cols: list[str],
        target_cols: list[str],
        scaling_params: dict,
        stf_factor: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.export_cfg = export_cfg
        self.model_state = model_state
        self.model_cfg = model_cfg
        self.lookback = lookback
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.scaling_params = scaling_params
        self.stf_factor = stf_factor

    def run(self) -> None:
        try:
            exporter = NTCPExporter(self.export_cfg)
            onnx_path = exporter.export_onnx(
                self.model_state, self.model_cfg, self.lookback,
            )
            self.log_signal.emit(f"ONNX exported: {onnx_path}", "info")

            mqh_path = exporter.export_mqh(
                self.feature_cols, self.target_cols,
                self.scaling_params, self.stf_factor, self.lookback,
            )
            self.log_signal.emit(f"MQH exported: {mqh_path}", "info")
            self.finished_signal.emit(True)
        except Exception as e:
            self.log_signal.emit(f"Export error: {e}", "error")
            self.finished_signal.emit(False)
