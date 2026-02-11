"""
NTCP GUI Widgets — LogWidget, EquityCurve, MetricsTable.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pyqtgraph as pg


# ---------------------------------------------------------------------------
# LogWidget
# ---------------------------------------------------------------------------

class LogWidget(QWidget):
    """
    Persistent log area with timestamps, color-coded messages,
    and optional file sync.
    """

    LOG_COLORS = {
        "info": QColor("#aaaaaa"),
        "warn": QColor("#ffaa44"),
        "error": QColor("#ff4444"),
    }

    def __init__(self, log_dir: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self._log_path: Path | None = None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            self._log_path = log_dir / "session_log.txt"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._text = QTextEdit()
        self._text.setObjectName("logWidget")
        self._text.setReadOnly(True)
        layout.addWidget(self._text)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clear)
        btn_row.addWidget(self._clear_btn)
        layout.addLayout(btn_row)

    def log(self, message: str, level: str = "info") -> None:
        """Append a timestamped, color-coded message."""
        ts = datetime.now().strftime("[%H:%M:%S]")
        full = f"{ts} {message}"

        color = self.LOG_COLORS.get(level, self.LOG_COLORS["info"])
        fmt = QTextCharFormat()
        fmt.setForeground(color)

        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(full + "\n", fmt)
        self._text.setTextCursor(cursor)
        self._text.ensureCursorVisible()

        # File sync
        if self._log_path:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(full + "\n")

    def clear(self) -> None:
        self._text.clear()


# ---------------------------------------------------------------------------
# EquityCurveWidget
# ---------------------------------------------------------------------------

class EquityCurveWidget(QWidget):
    """PyQtGraph-based equity curve plot."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot = pg.PlotWidget()
        self._plot.setBackground("#000000")
        self._plot.showGrid(x=True, y=True, alpha=0.15)
        self._plot.setLabel("left", "Equity")
        self._plot.setLabel("bottom", "Trade #")
        layout.addWidget(self._plot)

    def set_y_label(self, label: str) -> None:
        self._plot.setLabel("left", label)

    def set_data(self, equity: list[float] | None) -> None:
        self._plot.clear()
        if equity and len(equity) > 1:
            pen = pg.mkPen(color="#4488ff", width=1.5)
            self._plot.plot(equity, pen=pen)

            # Zero line
            zero_pen = pg.mkPen(color="#333333", width=1, style=Qt.PenStyle.DashLine)
            self._plot.addLine(y=0, pen=zero_pen)


# ---------------------------------------------------------------------------
# MetricsTable
# ---------------------------------------------------------------------------

class MetricsTable(QWidget):
    """Table displaying backtest metrics."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._table = QTableWidget()
        self._table.setColumnCount(2)
        self._table.setHorizontalHeaderLabels(["Metric", "Value"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

    def set_metrics(self, metrics: dict[str, str]) -> None:
        self._table.setRowCount(len(metrics))
        for i, (key, val) in enumerate(metrics.items()):
            self._table.setItem(i, 0, QTableWidgetItem(key))
            self._table.setItem(i, 1, QTableWidgetItem(str(val)))
