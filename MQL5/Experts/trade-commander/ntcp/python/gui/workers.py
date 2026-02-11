"""
NTCP GUI Workers — QThread workers for training and backtesting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
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
    sample_signal = pyqtSignal(int, object, object, object, object)  # (epoch, reg_actuals, reg_preds, cls_actuals, cls_probs)
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

        def sample_cb(epoch, actuals, predictions, cls_actuals, cls_probs) -> None:
            self.sample_signal.emit(epoch, actuals, predictions, cls_actuals, cls_probs)

        self._trainer = NTCPTrainer(
            self.data_cfg, self.model_cfg, self.train_cfg,
            log_callback=log_cb,
        )

        try:
            if self.single_factor is not None:
                result = self._trainer.train_single_factor(
                    self.single_factor, progress_callback=progress_cb,
                    sample_callback=sample_cb,
                )
                self.finished_signal.emit([result])
            else:
                results = self._trainer.train_factor_range(
                    progress_callback=progress_cb,
                    sample_callback=sample_cb,
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
        spread: float = 0.0,
        slippage: float = 0.0,
        point: float = 0.01,
        tick_value: float = 1.0,
        lot_size: float = 0.10,
        show_in_dollars: bool = False,
        entry_prices: np.ndarray | None = None,
        strategy: str = "trendcatcher",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model_state = model_state
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.target_cols = target_cols
        self.crv_threshold = crv_threshold
        self.use_cuda = use_cuda
        self.spread = spread
        self.slippage = slippage
        self.point = point
        self.tick_value = tick_value
        self.lot_size = lot_size
        self.show_in_dollars = show_in_dollars
        self.entry_prices = entry_prices
        self.strategy = strategy

    def run(self) -> None:
        try:
            self.log_signal.emit("Running backtest...", "info")
            bt = NTCPBacktester(
                self.model_state, self.model_cfg, self.crv_threshold,
                spread=self.spread,
                slippage=self.slippage,
                point=self.point,
                tick_value=self.tick_value,
                lot_size=self.lot_size,
                show_in_dollars=self.show_in_dollars,
                strategy=self.strategy,
            )
            metrics = bt.run(
                self.dataset, self.target_cols, use_cuda=self.use_cuda,
                entry_prices=self.entry_prices,
            )
            self.log_signal.emit(
                f"Backtest complete: {metrics.total_trades} trades "
                f"({metrics.long_trades}L/{metrics.short_trades}S), "
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
