"""Turn per-frame class scores into time segments and CSV rows."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from .model import FRAME_HOP_SEC, FRAME_WIN_SEC

CSV_HEADER = [
    "label",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "start_time",
    "end_time",
    "max_score",
    "mean_score",
]


@dataclass
class Segment:
    start: float
    end: float
    max_score: float
    mean_score: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS.mmm."""
    ms = round(seconds * 1000)
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{hours:d}:{minutes:02d}:{rest / 1000:06.3f}"


def detect_segments(
    scores: np.ndarray,
    class_indices: list[int],
    threshold: float = 0.3,
    merge_gap: float = 1.0,
    min_duration: float = 0.5,
) -> list[Segment]:
    """Threshold per-frame scores and merge active frames into segments.

    Frame i covers [i * hop, i * hop + win] seconds; the per-frame score is
    the max over the selected classes.
    """
    if not class_indices or scores.size == 0:
        return []
    frame_scores = scores[:, class_indices].max(axis=1)
    active = frame_scores >= threshold

    # Runs of consecutive active frames as (first, last) inclusive indices.
    runs: list[list[int]] = []
    for i, on in enumerate(active):
        if on and runs and runs[-1][1] == i - 1:
            runs[-1][1] = i
        elif on:
            runs.append([i, i])

    # Merge runs whose gap (in seconds) is small enough.
    merged: list[list[int]] = []
    for first, last in runs:
        if merged and first * FRAME_HOP_SEC - (merged[-1][1] * FRAME_HOP_SEC + FRAME_WIN_SEC) <= merge_gap:
            merged[-1][1] = last
        else:
            merged.append([first, last])

    segments = []
    for first, last in merged:
        window = frame_scores[first : last + 1]
        seg = Segment(
            start=first * FRAME_HOP_SEC,
            end=last * FRAME_HOP_SEC + FRAME_WIN_SEC,
            max_score=float(window.max()),
            mean_score=float(window.mean()),
        )
        if seg.duration >= min_duration:
            segments.append(seg)
    return segments


def write_csv(path: str, label: str, segments: list[Segment]) -> None:
    """Write segments to a CSV file in chronological order."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        for seg in segments:
            writer.writerow(
                [
                    label,
                    f"{seg.start:.3f}",
                    f"{seg.end:.3f}",
                    f"{seg.duration:.3f}",
                    format_timestamp(seg.start),
                    format_timestamp(seg.end),
                    f"{seg.max_score:.3f}",
                    f"{seg.mean_score:.3f}",
                ]
            )
