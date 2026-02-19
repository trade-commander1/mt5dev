"""
Trace window (colored messages, timestamps) and log file.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import QPlainTextEdit


class TraceWidget(QPlainTextEdit):
    """Read-only trace window: black background, colored text, timestamps to seconds."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("background-color: #000000; color: #ffffff; font-family: Consolas; font-size: 9pt;")
        self.setMinimumHeight(120)
        self._colors = {
            "warning": QColor("#ffa500"),   # Orange
            "error": QColor("#ff0000"),     # Red
            "result": QColor("#00ffff"),    # Cyan
            "info": QColor("#ffffff"),      # White
            "positive": QColor("#00ff00"), # Green (positive/profitable)
        }

    def append_with_level(self, level: str, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] <{level.upper()}> {message}\n"
        color = self._colors.get(level.lower(), self._colors["info"])
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(line, fmt)
        self.ensureCursorVisible()

    def append(self, text: str, level: str = "info") -> None:
        """Append line; if text has no newline, add one."""
        if text and not text.endswith("\n"):
            text = text + "\n"
        for line in text.rstrip("\n").split("\n"):
            if line:
                self.append_with_level(level, line)


class LogFile:
    """Write log messages to StrategyBacktestNN_YYYYMMDD_HHMMSS.log."""

    def __init__(self, log_dir: Optional[Path] = None) -> None:
        self._path: Optional[Path] = None
        self._dir = Path(log_dir) if log_dir else Path(".")
        self._fp = None

    def start(self) -> Path:
        """Create new log file; return path."""
        self.close()
        name = f"StrategyBacktestNN_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self._path = self._dir / name
        self._fp = open(self._path, "w", encoding="utf-8")
        return self._path

    def write(self, level: str, message: str) -> None:
        if self._fp is None:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._fp.write(f"[{ts}] <{level.upper()}> {message}\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    @property
    def path(self) -> Optional[Path]:
        return self._path
