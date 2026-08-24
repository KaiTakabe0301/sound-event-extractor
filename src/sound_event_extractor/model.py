"""YAMNet model loading and chunked inference."""

from __future__ import annotations

import csv
import os
from collections.abc import Callable

import numpy as np

from .audio import SAMPLE_RATE

HUB_URL = "https://tfhub.dev/google/yamnet/1"

# YAMNet scores one frame per 0.48 s over a 0.96 s window.
FRAME_HOP_SEC = 0.48
FRAME_WIN_SEC = 0.96
_HOP = int(SAMPLE_RATE * FRAME_HOP_SEC)  # 7680 samples
_WIN = int(SAMPLE_RATE * FRAME_WIN_SEC)  # 15360 samples

# ~48 s of audio per inference call; chunks tile exactly on the frame grid
# (advance = FRAMES_PER_CHUNK * hop) so frame i is always at i * FRAME_HOP_SEC.
FRAMES_PER_CHUNK = 100

ProgressFn = Callable[[float], None]


class YamNet:
    """Wrapper around the TF-Hub YAMNet model (521 AudioSet classes)."""

    def __init__(self) -> None:
        os.environ.setdefault(
            "TFHUB_CACHE_DIR",
            os.path.expanduser("~/.cache/sound-event-extractor/tfhub"),
        )
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow_hub as hub  # heavy import, kept lazy

        self._model = hub.load(HUB_URL)
        self.class_names = self._load_class_names()

    def _load_class_names(self) -> list[str]:
        path = self._model.class_map_path().numpy().decode("utf-8")
        with open(path, newline="", encoding="utf-8") as fh:
            return [row["display_name"] for row in csv.DictReader(fh)]

    def scores(self, waveform: np.ndarray, progress: ProgressFn | None = None) -> np.ndarray:
        """Return per-frame class scores of shape (n_frames, 521)."""
        chunk_samples = _WIN + (FRAMES_PER_CHUNK - 1) * _HOP
        advance = FRAMES_PER_CHUNK * _HOP
        total = waveform.size
        parts: list[np.ndarray] = []
        pos = 0
        while pos < total:
            chunk = waveform[pos : pos + chunk_samples]
            if chunk.size < _WIN:
                chunk = np.pad(chunk, (0, _WIN - chunk.size))
            chunk_scores, _, _ = self._model(chunk)
            parts.append(chunk_scores.numpy())
            pos += advance
            if progress is not None:
                progress(min(pos / total, 1.0))
        return np.concatenate(parts, axis=0)
