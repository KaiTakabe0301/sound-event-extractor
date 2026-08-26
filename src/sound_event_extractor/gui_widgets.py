"""Time-synced custom widgets: spectrogram view and score timeline."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QWidget

BG_COLOR = QColor("#161616")
TEXT_COLOR = QColor("#dddddd")
PLAYHEAD_COLOR = QColor("#e53935")
SCORE_COLOR = QColor("#4caf50")
THRESHOLD_COLOR = QColor("#ff9800")


class TimeStrip(QWidget):
    """Base widget mapping x position to time; click requests a seek."""

    seekRequested = Signal(float)  # seconds

    def __init__(self, height: int) -> None:
        super().__init__()
        self.setFixedHeight(height)
        self._duration = 0.0
        self._position = 0.0

    def set_duration(self, seconds: float) -> None:
        if seconds > 0:
            self._duration = seconds
            self.update()

    def set_position(self, seconds: float) -> None:
        self._position = seconds
        self.update()

    def mousePressEvent(self, event) -> None:
        if self._duration > 0 and self.width() > 0:
            ratio = min(max(event.position().x() / self.width(), 0.0), 1.0)
            self.seekRequested.emit(ratio * self._duration)

    def _draw_playhead(self, painter: QPainter) -> None:
        if self._duration > 0:
            x = int(self._position / self._duration * self.width())
            painter.setPen(QPen(PLAYHEAD_COLOR, 2))
            painter.drawLine(x, 0, x, self.height())

    def _draw_placeholder(self, painter: QPainter, text: str) -> None:
        painter.setPen(TEXT_COLOR)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)


class SpectrogramView(TimeStrip):
    """Grayscale STFT spectrogram, 0 Hz at the bottom, 8 kHz at the top."""

    def __init__(self) -> None:
        super().__init__(150)
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
            painter.drawImage(self.rect(), self._image)
            painter.setPen(TEXT_COLOR)
            painter.drawText(4, 12, "8kHz")
            painter.drawText(4, self.height() // 2 + 4, "4kHz")
            painter.drawText(4, self.height() - 4, "0")
        self._draw_playhead(painter)
        painter.end()


class ScoreTimeline(TimeStrip):
    """Per-frame detection score curve with the threshold line."""

    def __init__(self) -> None:
        super().__init__(100)
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

    def _build_polygon(self, width: int, height: int) -> QPolygon:
        centers = np.arange(self._scores.size) * self._hop + self._hop
        xs = (centers / self._duration * width).astype(int)
        ys = ((1.0 - np.clip(self._scores, 0.0, 1.0)) * (height - 1)).astype(int)
        return QPolygon([QPoint(int(x), int(y)) for x, y in zip(xs, ys)])

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BG_COLOR)
        width, height = self.width(), self.height()
        if self._scores is None or self._scores.size == 0 or self._duration <= 0:
            self._draw_placeholder(painter, "検出スコア(解析後に表示)")
        else:
            pen = QPen(THRESHOLD_COLOR)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            threshold_y = int((1.0 - self._threshold) * (height - 1))
            painter.drawLine(0, threshold_y, width, threshold_y)
            if self._polygon_key != (width, height):
                self._polygon = self._build_polygon(width, height)
                self._polygon_key = (width, height)
            painter.setPen(QPen(SCORE_COLOR, 1))
            painter.drawPolyline(self._polygon)
            frontier = self._scores.size * self._hop
            if frontier < self._duration - self._hop:  # streaming still in progress
                x = int(frontier / self._duration * width)
                painter.setPen(QPen(QColor("#666666"), 1))
                painter.drawLine(x, 0, x, height)
            painter.setPen(TEXT_COLOR)
            painter.drawText(4, 12, f"しきい値 {self._threshold:.2f}")
        self._draw_playhead(painter)
        painter.end()
