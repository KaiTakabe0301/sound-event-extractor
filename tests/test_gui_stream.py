"""Offscreen tests for streaming score display (no model required)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from sound_event_extractor.gui_widgets import ScoreTimeline, SpectrogramView


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_streaming_append_and_score_at(app) -> None:
    strip = ScoreTimeline()
    strip.begin_stream(0.48, 0.3, 9.6)  # 20 frames total
    assert strip.score_at(0.0) is None
    strip.append_scores(np.array([0.1, 0.9], dtype=np.float32))
    strip.append_scores(np.array([0.5], dtype=np.float32))
    assert strip.score_at(0.5) == pytest.approx(0.9)  # frame 1
    assert strip.score_at(1.0) == pytest.approx(0.5)  # frame 2
    assert strip.score_at(5.0) is None  # beyond the analyzed frontier
    strip.grab()  # exercise the paint path with partial data


def test_clear_resets(app) -> None:
    strip = ScoreTimeline()
    strip.begin_stream(0.48, 0.3, 5.0)
    strip.append_scores(np.array([0.4], dtype=np.float32))
    strip.clear()
    assert strip.score_at(0.0) is None
    strip.grab()

    spec = SpectrogramView()
    spec.set_data(np.zeros((16, 8), dtype=np.uint8), 5.0)
    spec.clear()
    spec.grab()
