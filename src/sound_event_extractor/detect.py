"""Turn per-frame class scores into time segments and CSV rows."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass

import numpy as np

from .model import FRAME_HOP_SEC, FRAME_WIN_SEC

SOURCE_AUTO = "auto"
SOURCE_MANUAL = "manual"

CSV_HEADER = [
    "label",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "start_time",
    "end_time",
    "max_score",
    "mean_score",
    "source",
]


@dataclass
class Segment:
    start: float
    end: float
    max_score: float  # NaN when no frame scores are available (manual, unanalyzed)
    mean_score: float
    source: str = SOURCE_AUTO

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def is_manual(self) -> bool:
        return self.source == SOURCE_MANUAL


def format_score(score: float) -> str:
    return "" if math.isnan(score) else f"{score:.3f}"


def _parse_score(text: str) -> float:
    text = text.strip()
    return float(text) if text else math.nan


def segment_scores(
    frame_scores: np.ndarray | None, start: float, end: float
) -> tuple[float, float]:
    """(max, mean) of the frames overlapping [start, end]; NaN when none are available."""
    if frame_scores is None or frame_scores.size == 0:
        return math.nan, math.nan
    first = max(int(math.ceil((start - FRAME_WIN_SEC) / FRAME_HOP_SEC)), 0)
    last = min(int(end / FRAME_HOP_SEC), frame_scores.size - 1)
    if first > last:
        return math.nan, math.nan
    window = frame_scores[first : last + 1]
    return float(window.max()), float(window.mean())


def make_manual_segment(
    start: float, end: float, frame_scores: np.ndarray | None = None
) -> Segment:
    max_score, mean_score = segment_scores(frame_scores, start, end)
    return Segment(start, end, max_score, mean_score, source=SOURCE_MANUAL)


def insert_segment(segments: list[Segment], segment: Segment) -> list[Segment]:
    """Return a new chronologically ordered list; identical start/end is not duplicated."""
    for existing in segments:
        if existing.start == segment.start and existing.end == segment.end:
            return list(segments)
    return sorted([*segments, segment], key=lambda s: (s.start, s.end))


def merge_results(auto_segments: list[Segment], previous: list[Segment]) -> list[Segment]:
    """Combine fresh analysis results with the manual segments kept from `previous`."""
    merged = list(auto_segments)
    for seg in previous:
        if seg.is_manual:
            merged = insert_segment(merged, seg)
    return merged


def format_timestamp(seconds: float) -> str:
    """Format seconds as H:MM:SS.mmm."""
    ms = round(seconds * 1000)
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    return f"{hours:d}:{minutes:02d}:{rest / 1000:06.3f}"


def combined_frame_scores(scores: np.ndarray, class_indices: list[int]) -> np.ndarray:
    """Max score over the selected classes for each frame."""
    if not class_indices or scores.size == 0:
        return np.zeros(0, dtype=np.float32)
    return scores[:, class_indices].max(axis=1)


def detect_segments(
    scores: np.ndarray,
    class_indices: list[int],
    threshold: float = 0.05,
    merge_gap: float = 1.0,
    min_duration: float = 0.5,
) -> list[Segment]:
    """Threshold per-frame scores and merge active frames into segments.

    Frame i covers [i * hop, i * hop + win] seconds; the per-frame score is
    the max over the selected classes.
    """
    frame_scores = combined_frame_scores(scores, class_indices)
    if frame_scores.size == 0:
        return []
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
                    format_score(seg.max_score),
                    format_score(seg.mean_score),
                    seg.source,
                ]
            )


def read_csv(path: str) -> tuple[str, list[Segment]]:
    """Read a CSV written by `write_csv` back into (label, segments).

    Files from older versions without a `source` column are treated as automatic.
    """
    label = ""
    segments: list[Segment] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        for column in ("start_seconds", "end_seconds"):
            if column not in fields:
                raise ValueError(f"CSV に '{column}' 列がありません: {path}")
        for row in reader:
            label = label or (row.get("label") or "").strip()
            source = (row.get("source") or SOURCE_AUTO).strip().lower()
            if source not in (SOURCE_AUTO, SOURCE_MANUAL):
                raise ValueError(f"source 列の値が不正です: {source!r}")
            segments.append(
                Segment(
                    start=float(row["start_seconds"]),
                    end=float(row["end_seconds"]),
                    max_score=_parse_score(row.get("max_score") or ""),
                    mean_score=_parse_score(row.get("mean_score") or ""),
                    source=source,
                )
            )
    segments.sort(key=lambda s: (s.start, s.end))
    return label, segments


DEBUG_TOP_N = 5


def write_debug_scores(
    path: str,
    class_names: list[str],
    scores: np.ndarray,
    class_indices: list[int],
    label: str,
) -> None:
    """Write per-frame diagnostics: matched-class max plus overall top-N classes."""
    matched = combined_frame_scores(scores, class_indices)
    top_n = min(DEBUG_TOP_N, len(class_names))
    header = ["frame_time_seconds", "frame_time", f"score[{label}]"]
    for rank in range(1, top_n + 1):
        header += [f"top{rank}_class", f"top{rank}_score"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, frame in enumerate(scores):
            t = i * FRAME_HOP_SEC
            row = [
                f"{t:.2f}",
                format_timestamp(t),
                f"{matched[i]:.3f}" if matched.size else "",
            ]
            for j in np.argsort(frame)[::-1][:top_n]:
                row += [class_names[j], f"{frame[j]:.3f}"]
            writer.writerow(row)
