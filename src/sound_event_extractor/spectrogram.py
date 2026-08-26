"""STFT spectrogram computation for the GUI (numpy only, no Qt here)."""

from __future__ import annotations

import numpy as np

from .audio import SAMPLE_RATE

N_FFT = 1024
MIN_HOP = 512
MAX_COLS = 2400  # cap the image width so 30-min videos stay lightweight
DB_RANGE = 80.0


def compute_spectrogram(waveform: np.ndarray, max_cols: int = MAX_COLS) -> tuple[np.ndarray, float]:
    """Compute a display-ready magnitude spectrogram.

    Returns (image, duration_seconds). The image is uint8 with shape
    (freq_bins, time_cols), row 0 = Nyquist (8 kHz), last row = 0 Hz,
    dB-scaled over DB_RANGE below the global peak.
    """
    duration = waveform.size / SAMPLE_RATE
    if waveform.size < N_FFT:
        waveform = np.pad(waveform, (0, N_FFT - waveform.size))
    hop = max(MIN_HOP, (waveform.size - N_FFT) // max_cols + 1)
    starts = np.arange(0, waveform.size - N_FFT + 1, hop)[:max_cols]
    frames = waveform[starts[:, None] + np.arange(N_FFT)] * np.hanning(N_FFT)
    magnitude = np.abs(np.fft.rfft(frames, axis=1))  # (cols, bins)
    db = 20.0 * np.log10(magnitude + 1e-9)
    db -= db.max()
    scaled = np.clip((db + DB_RANGE) / DB_RANGE, 0.0, 1.0)
    image = (scaled * 255).astype(np.uint8).T[::-1]  # high freq on top
    return np.ascontiguousarray(image), duration
