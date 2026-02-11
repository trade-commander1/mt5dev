"""
Live equity curve widget: black background, cyan line, axis labels and scaling.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import QWidget


# Chart styling: black background, cyan line, dark gray grid, white text
CHART_BG = "#000000"
CHART_LINE = "#00FFFF"
CHART_GRID = "#333333"
CHART_TEXT = "#FFFFFF"

# Margins for axis labels (left = Y labels, bottom = X label)
MARGIN_LEFT = 48
MARGIN_RIGHT = 12
MARGIN_TOP = 8
MARGIN_BOTTOM = 28


class EquityCurveWidget(QWidget):
    """Draw equity curve with X-axis (Bar/Trade #), Y-axis (Equity), and tick labels."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(120)
        self.setStyleSheet(f"background-color: {CHART_BG};")
        self.setAutoFillBackground(True)
        self._points: list[tuple[int, float]] = []
        self._percentage_mode = False
        self._initial_equity: float = 0.0

    def set_percentage_mode(self, on: bool, initial_equity: float = 0.0) -> None:
        self._percentage_mode = on
        self._initial_equity = initial_equity
        self.update()

    def clear(self) -> None:
        self._points.clear()
        self.update()

    def append_point(self, bar: int, equity: float) -> None:
        self._points.append((bar, equity))
        self.update()

    def set_points(self, points: list[tuple[int, float]]) -> None:
        self._points = list(points)
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        w, h = self.width(), self.height()
        if w < 4 or h < 4:
            painter.end()
            return
        plot_w = w - MARGIN_LEFT - MARGIN_RIGHT
        plot_h = h - MARGIN_TOP - MARGIN_BOTTOM
        if plot_w < 10 or plot_h < 10:
            painter.end()
            return

        def to_x(x: int) -> float:
            return MARGIN_LEFT + (x - x_min) / x_range * plot_w

        def to_y(y: float) -> float:
            return MARGIN_TOP + (y_max - y) / (y_max - y_min) * plot_h

        x_min = x_max = 0
        y_min = y_max = 0.0
        ys_pct = []

        if self._points:
            xs = [p[0] for p in self._points]
            ys = [p[1] for p in self._points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            if self._percentage_mode and self._initial_equity != 0:
                ys_pct = [(y - self._initial_equity) / abs(self._initial_equity) * 100 for y in ys]
                y_min, y_max = min(ys_pct), max(ys_pct)
            else:
                ys_pct = ys
            if y_max <= y_min:
                y_max = y_min + 1
            x_range = x_max - x_min or 1

            # Grid (dark gray)
            grid_pen = QPen(QColor(CHART_GRID), 1)
            painter.setPen(grid_pen)
            for i in range(1, 5):
                gx = MARGIN_LEFT + plot_w * i // 4
                painter.drawLine(int(gx), MARGIN_TOP, int(gx), MARGIN_TOP + plot_h)
            for i in range(1, 4):
                gy = MARGIN_TOP + plot_h * i // 3
                painter.drawLine(MARGIN_LEFT, int(gy), MARGIN_LEFT + plot_w, int(gy))

            # Equity line (cyan)
            line_pen = QPen(QColor(CHART_LINE), 1.5)
            painter.setPen(line_pen)
            for i in range(1, len(self._points)):
                x1 = to_x(self._points[i - 1][0])
                y1 = to_y(ys_pct[i - 1] if self._percentage_mode and self._initial_equity != 0 else self._points[i - 1][1])
                x2 = to_x(self._points[i][0])
                y2 = to_y(ys_pct[i] if self._percentage_mode and self._initial_equity != 0 else self._points[i][1])
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Axis labels and tick values (white text)
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(CHART_TEXT))

        # Y-axis label (rotated)
        painter.save()
        painter.translate(12, MARGIN_TOP + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-50, 0, "Equity")
        painter.restore()

        # Y tick labels (3–4 values)
        if self._points and (y_max - y_min) > 1e-12:
            for i in range(4):
                frac = i / 3.0
                val = y_min + (y_max - y_min) * frac
                gy = MARGIN_TOP + plot_h * (1 - frac)
                if abs(val) >= 1000:
                    txt = f"{val/1000:.1f}k"
                elif abs(val) < 0.01 and val != 0:
                    txt = f"{val:.2e}"
                else:
                    txt = f"{val:.2f}"
                painter.drawText(4, int(gy + 4), txt)
        else:
            painter.drawText(4, MARGIN_TOP + plot_h // 2, "0")

        # X-axis label
        painter.drawText(MARGIN_LEFT + plot_w // 2 - 40, h - 6, "Bar / Trade #")

        # X tick labels
        if self._points and x_max > x_min:
            for i in range(5):
                xi = x_min + (x_max - x_min) * i // 4
                gx = MARGIN_LEFT + plot_w * i // 4
                painter.drawText(int(gx - 12), h - 10, str(xi))

        painter.end()
