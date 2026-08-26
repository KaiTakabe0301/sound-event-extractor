"""Widgets showing the instantaneous spectrum and scores at the playhead."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QWidget

from .audio import SAMPLE_RATE

BG_COLOR = QColor("#161616")
TEXT_COLOR = QColor("#dddddd")
MUTED_COLOR = QColor("#777777")
SPECTRUM_COLOR = QColor("#40c4ff")
BAR_ACTIVE = QColor("#4caf50")
BAR_INACTIVE = QColor("#607d8b")
THRESHOLD_COLOR = QColor("#ff9800")

FFT_SIZE = 4096  # ~256 ms window around the playhead
DB_FLOOR = -90.0


class InstantSpectrumView(QWidget):
    """FFT magnitude spectrum (0-8 kHz) of the moment being played."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(170)
        self.setMinimumWidth(300)
        self._waveform: np.ndarray | None = None
        self._window = np.hanning(FFT_SIZE)
        self._db: np.ndarray | None = None
        self._peak_hz = 0.0

    def set_waveform(self, waveform: np.ndarray) -> None:
        self._waveform = waveform
        self.update()

    def clear(self) -> None:
        self._waveform = None
        self._db = None
        self.update()

    def set_position(self, seconds: float) -> None:
        if self._waveform is None:
            return
        center = int(seconds * SAMPLE_RATE)
        start = min(max(center - FFT_SIZE // 2, 0), max(self._waveform.size - FFT_SIZE, 0))
        chunk = self._waveform[start : start + FFT_SIZE]
        if chunk.size < FFT_SIZE:
            chunk = np.pad(chunk, (0, FFT_SIZE - chunk.size))
        magnitude = np.abs(np.fft.rfft(chunk * self._window)) * 2 / self._window.sum()
        self._db = np.clip(20.0 * np.log10(magnitude + 1e-10), DB_FLOOR, 0.0)
        peak = int(magnitude[1:].argmax()) + 1  # skip the DC bin
        self._peak_hz = peak * SAMPLE_RATE / FFT_SIZE
        self.update()

    def _column_levels(self, n_cols: int) -> np.ndarray:
        """Max-pool the dB bins into one level per pixel column."""
        edges = np.linspace(0, self._db.size, n_cols + 1).astype(int)
        return np.array(
            [
                self._db[a:b].max() if b > a else self._db[min(a, self._db.size - 1)]
                for a, b in zip(edges[:-1], edges[1:])
            ]
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BG_COLOR)
        w, h = self.width(), self.height()
        if self._db is None:
            painter.setPen(TEXT_COLOR)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "周波数スペクトル(動画読み込み後、再生位置に追従)",
            )
            painter.end()
            return
        plot_h = h - 16
        painter.setPen(MUTED_COLOR)
        for k in range(5):  # gridlines at 0/2/4/6/8 kHz
            x = int(k / 4 * (w - 1))
            painter.drawLine(x, 0, x, plot_h)
            painter.drawText(min(x + 2, w - 24), h - 3, f"{k * 2}k")
        levels = self._column_levels(max(w, 1))
        ys = (levels / DB_FLOOR * (plot_h - 1)).astype(int)
        painter.setPen(QPen(SPECTRUM_COLOR, 1))
        painter.drawPolyline(QPolygon([QPoint(x, int(y)) for x, y in enumerate(ys)]))
        painter.setPen(TEXT_COLOR)
        painter.drawText(w - 130, 14, f"ピーク {self._peak_hz:.0f} Hz")
        painter.end()


class InstantScoreView(QWidget):
    """Matched-label score and top-5 class scores at the moment being played."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(170)
        self.setMinimumWidth(280)
        self._hop = 0.48
        self._label = ""
        self._threshold = 0.3
        self._combined: np.ndarray | None = None
        self._tops: list = []
        self._index = -1

    def begin_stream(self, hop_sec: float, label: str, threshold: float) -> None:
        self._hop = hop_sec
        self._label = label
        self._threshold = threshold
        self._combined = np.zeros(0, dtype=np.float32)
        self._tops = []
        self._index = -1
        self.update()

    def append(self, combined: np.ndarray, tops: list) -> None:
        if self._combined is None:
            self._combined = np.zeros(0, dtype=np.float32)
        self._combined = np.concatenate([self._combined, combined])
        self._tops.extend(tops)
        self.update()

    def clear(self) -> None:
        self._combined = None
        self._tops = []
        self._index = -1
        self.update()

    def set_position(self, seconds: float) -> None:
        index = int(seconds / self._hop) if self._hop > 0 else -1
        if index != self._index:
            self._index = index
            self.update()

    @staticmethod
    def _bar(painter: QPainter, x: int, y: int, width: int, height: int, ratio: float, color: QColor) -> None:
        painter.fillRect(x, y, int(width * min(max(ratio, 0.0), 1.0)), height, color)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), BG_COLOR)
        w = self.width()
        has_frame = (
            self._combined is not None
            and 0 <= self._index < len(self._tops)
            and self._index < self._combined.size
        )
        if not has_frame:
            painter.setPen(TEXT_COLOR)
            text = (
                "現在位置のスコア(解析後に表示)"
                if not self._tops
                else "この位置はまだ解析されていません"
            )
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)
            painter.end()
            return
        score = float(self._combined[self._index])
        x, bar_w = 8, w - 16
        painter.setPen(TEXT_COLOR)
        painter.drawText(x, 16, f"{self._label}: {score:.3f}")
        color = BAR_ACTIVE if score >= self._threshold else BAR_INACTIVE
        self._bar(painter, x, 22, bar_w, 10, score, color)
        tick_x = x + int(bar_w * self._threshold)
        painter.setPen(QPen(THRESHOLD_COLOR, 1))
        painter.drawLine(tick_x, 20, tick_x, 34)
        painter.setPen(MUTED_COLOR)
        painter.drawText(x, 50, "この瞬間の上位クラス:")
        y = 56
        name_w = min(220, w // 2)
        for name, value in self._tops[self._index]:
            painter.setPen(TEXT_COLOR)
            painter.drawText(x, y + 10, f"{name[:26]} {value:.3f}")
            self._bar(painter, x + name_w + 8, y + 3, max(bar_w - name_w - 16, 40), 8, value, BAR_INACTIVE)
            y += 21
        painter.end()
