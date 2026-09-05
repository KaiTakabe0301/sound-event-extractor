"""Time-synced custom widgets: spectrogram view and score timeline."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QWidget

BG_COLOR = QColor("#161616")
TEXT_COLOR = QColor("#dddddd")
PLAYHEAD_COLOR = QColor("#e53935")
SCORE_COLOR = QColor("#4caf50")
THRESHOLD_COLOR = QColor("#ff9800")
AXIS_COLOR = QColor("#999999")
MARK_COLOR = QColor("#ffd54f")
MARK_FILL_COLOR = QColor(255, 213, 79, 50)

# plot margins reserved for the axes
MARGIN_LEFT = 58
MARGIN_TOP = 6
MARGIN_RIGHT = 10
MARGIN_BOTTOM = 30

_TIME_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200]


def _tick_step(duration: float) -> int:
    """Pick a tick interval that yields at most ~8 time ticks."""
    for step in _TIME_STEPS:
        if duration / step <= 8:
            return step
    return 14400


def _format_tick(seconds: int) -> str:
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{sec:02d}" if hours else f"{minutes}:{sec:02d}"


class TimeStrip(QWidget):
    """Base widget mapping x position to time; click requests a seek."""

    seekRequested = Signal(float)  # seconds

    def __init__(self, height: int) -> None:
        super().__init__()
        self.setFixedHeight(height)
        self._duration = 0.0
        self._position = 0.0
        self._mark_in: float | None = None
        self._mark_out: float | None = None

    def set_duration(self, seconds: float) -> None:
        if seconds > 0:
            self._duration = seconds
            self.update()

    def set_position(self, seconds: float) -> None:
        self._position = seconds
        self.update()

    def set_marks(self, mark_in: float | None, mark_out: float | None) -> None:
        """Show the in/out points of the segment being annotated by hand."""
        self._mark_in = mark_in
        self._mark_out = mark_out
        self.update()

    def _time_to_x(self, rect: QRect, seconds: float) -> int:
        return rect.x() + int(seconds / self._duration * rect.width())

    def _draw_marks(self, painter: QPainter) -> None:
        if self._duration <= 0 or (self._mark_in is None and self._mark_out is None):
            return
        rect = self._plot_rect()
        if self._mark_in is not None and self._mark_out is not None:
            left = self._time_to_x(rect, min(self._mark_in, self._mark_out))
            right = self._time_to_x(rect, max(self._mark_in, self._mark_out))
            span = QRect(left, rect.top(), max(right - left, 1), rect.height())
            painter.fillRect(span, MARK_FILL_COLOR)
        painter.setPen(QPen(MARK_COLOR, 1, Qt.PenStyle.DashLine))
        for mark in (self._mark_in, self._mark_out):
            if mark is not None:
                x = self._time_to_x(rect, mark)
                painter.drawLine(x, rect.top(), x, rect.bottom())

    def _plot_rect(self) -> QRect:
        return QRect(
            MARGIN_LEFT,
            MARGIN_TOP,
            max(self.width() - MARGIN_LEFT - MARGIN_RIGHT, 1),
            max(self.height() - MARGIN_TOP - MARGIN_BOTTOM, 1),
        )

    def mousePressEvent(self, event) -> None:
        rect = self._plot_rect()
        if self._duration > 0 and rect.width() > 0:
            ratio = min(max((event.position().x() - rect.x()) / rect.width(), 0.0), 1.0)
            self.seekRequested.emit(ratio * self._duration)

    def _draw_playhead(self, painter: QPainter) -> None:
        if self._duration > 0:
            rect = self._plot_rect()
            x = rect.x() + int(self._position / self._duration * rect.width())
            painter.setPen(QPen(PLAYHEAD_COLOR, 2))
            painter.drawLine(x, rect.top(), x, rect.bottom())

    def _draw_axes(
        self, painter: QPainter, y_title: str, y_ticks: list[tuple[float, str]]
    ) -> None:
        """Draw the plot frame, y-axis ticks/title, and the time (x) axis."""
        rect = self._plot_rect()
        painter.setPen(QPen(AXIS_COLOR, 1))
        painter.drawLine(rect.x(), rect.top(), rect.x(), rect.bottom())
        painter.drawLine(rect.x(), rect.bottom(), rect.right(), rect.bottom())
        for ratio, label in y_ticks:  # (position from the top, label)
            y = rect.top() + int(ratio * (rect.height() - 1))
            painter.drawLine(rect.x() - 3, y, rect.x(), y)
            painter.drawText(
                QRect(0, y - 7, rect.x() - 6, 14),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                label,
            )
        painter.save()
        painter.translate(11, rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRect(-60, -8, 120, 16), Qt.AlignmentFlag.AlignCenter, y_title)
        painter.restore()
        if self._duration > 0:
            t = 0
            while t <= self._duration:
                x = rect.x() + int(t / self._duration * (rect.width() - 1))
                painter.drawLine(x, rect.bottom(), x, rect.bottom() + 3)
                painter.drawText(
                    QRect(x - 35, rect.bottom() + 4, 70, 12),
                    Qt.AlignmentFlag.AlignHCenter,
                    _format_tick(t),
                )
                t += _tick_step(self._duration)
        painter.drawText(
            QRect(rect.x(), self.height() - 14, rect.width(), 14),
            Qt.AlignmentFlag.AlignHCenter,
            "時間",
        )

    def _draw_placeholder(self, painter: QPainter, text: str) -> None:
        painter.setPen(TEXT_COLOR)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)


class SpectrogramView(TimeStrip):
    """Grayscale STFT spectrogram, 0 Hz at the bottom, 8 kHz at the top."""

    def __init__(self) -> None:
        super().__init__(176)
        self._image: QImage | None = None
        self._image_bytes: bytes | None = None

    def set_data(self, image: np.ndarray, duration: float) -> None:
        height, width = image.shape
        self._image_bytes = image.tobytes()  # QImage does not copy; keep alive
        self._image = QImage(
            self._image_bytes, width, height, width, QImage.Format.Format_Grayscale8
        )
        self.set_duration(duration)

    def clear(self) -> None:
        self._image = None
        self._image_bytes = None
        self._position = 0.0
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BG_COLOR)
        if self._image is None:
            self._draw_placeholder(painter, "スペクトログラム(解析後に表示)")
        else:
            painter.drawImage(self._plot_rect(), self._image)
            self._draw_axes(
                painter,
                "周波数 (kHz)",
                [(0.0, "8"), (0.25, "6"), (0.5, "4"), (0.75, "2"), (1.0, "0")],
            )
        self._draw_marks(painter)
        self._draw_playhead(painter)
        painter.end()


class ScoreTimeline(TimeStrip):
    """Per-frame detection score curve with the threshold line."""

    def __init__(self) -> None:
        super().__init__(124)
        self._scores: np.ndarray | None = None
        self._hop = 0.48
        self._threshold = 0.05
        self._polygon: QPolygon | None = None
        self._polygon_key: tuple[int, int] | None = None

    def set_data(
        self, scores: np.ndarray, hop_sec: float, threshold: float, duration: float
    ) -> None:
        self._scores = scores
        self._hop = hop_sec
        self._threshold = threshold
        self._polygon_key = None
        self.set_duration(duration)

    def clear(self) -> None:
        self._scores = None
        self._polygon_key = None
        self._position = 0.0
        self.update()

    def begin_stream(self, hop_sec: float, threshold: float, duration: float) -> None:
        """Prepare to receive incremental score chunks during playback."""
        self._scores = np.zeros(0, dtype=np.float32)
        self._hop = hop_sec
        self._threshold = threshold
        self._polygon_key = None
        self.set_duration(duration)

    def append_scores(self, chunk: np.ndarray) -> None:
        if self._scores is None:
            self._scores = np.zeros(0, dtype=np.float32)
        self._scores = np.concatenate([self._scores, chunk])
        self._polygon_key = None
        self.update()

    def score_at(self, seconds: float) -> float | None:
        if self._scores is None or self._scores.size == 0 or self._hop <= 0:
            return None
        idx = int(seconds / self._hop)
        if 0 <= idx < self._scores.size:
            return float(self._scores[idx])
        return None

    def _build_polygon(self, rect: QRect) -> QPolygon:
        centers = np.arange(self._scores.size) * self._hop + self._hop
        xs = rect.x() + (centers / self._duration * (rect.width() - 1)).astype(int)
        ys = rect.top() + (
            (1.0 - np.clip(self._scores, 0.0, 1.0)) * (rect.height() - 1)
        ).astype(int)
        return QPolygon([QPoint(int(x), int(y)) for x, y in zip(xs, ys)])

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BG_COLOR)
        rect = self._plot_rect()
        if self._scores is None or self._scores.size == 0 or self._duration <= 0:
            self._draw_placeholder(painter, "検出スコア(解析後に表示)")
        else:
            pen = QPen(THRESHOLD_COLOR)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            threshold_y = rect.top() + int((1.0 - self._threshold) * (rect.height() - 1))
            painter.drawLine(rect.x(), threshold_y, rect.right(), threshold_y)
            if self._polygon_key != (self.width(), self.height()):
                self._polygon = self._build_polygon(rect)
                self._polygon_key = (self.width(), self.height())
            painter.setPen(QPen(SCORE_COLOR, 1))
            painter.drawPolyline(self._polygon)
            frontier = self._scores.size * self._hop
            if frontier < self._duration - self._hop:  # streaming still in progress
                x = rect.x() + int(frontier / self._duration * rect.width())
                painter.setPen(QPen(QColor("#666666"), 1))
                painter.drawLine(x, rect.top(), x, rect.bottom())
            self._draw_axes(painter, "スコア", [(0.0, "1.0"), (0.5, "0.5"), (1.0, "0.0")])
            painter.setPen(TEXT_COLOR)
            painter.drawText(rect.x() + 6, rect.top() + 12, f"しきい値 {self._threshold:.2f}")
        self._draw_marks(painter)
        self._draw_playhead(painter)
        painter.end()
