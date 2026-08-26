"""Unit tests for spectrogram computation (numpy only, no Qt)."""

import numpy as np
import pytest

from sound_event_extractor.audio import SAMPLE_RATE
from sound_event_extractor.spectrogram import N_FFT, compute_spectrogram


def test_shape_duration_and_peak_frequency() -> None:
    # 2 s of a 1 kHz sine: the brightest row must sit near the 1 kHz bin.
    t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    waveform = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    image, duration = compute_spectrogram(waveform)
    assert duration == pytest.approx(2.0)
    n_bins = N_FFT // 2 + 1
    assert image.shape[0] == n_bins
    assert image.dtype == np.uint8
    # Row 0 is Nyquist (8 kHz), the last row is 0 Hz.
    expected_row = (n_bins - 1) - round(1000 / 8000 * (n_bins - 1))
    peak_row = int(image[:, image.shape[1] // 2].argmax())
    assert abs(peak_row - expected_row) <= 2


def test_short_input_is_padded() -> None:
    image, duration = compute_spectrogram(np.zeros(100, dtype=np.float32))
    assert image.shape[1] >= 1
    assert duration == pytest.approx(100 / SAMPLE_RATE)


def test_column_cap() -> None:
    waveform = np.zeros(SAMPLE_RATE * 60, dtype=np.float32)
    image, _ = compute_spectrogram(waveform, max_cols=200)
    assert image.shape[1] <= 200
