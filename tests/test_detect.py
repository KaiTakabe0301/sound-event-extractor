"""Unit tests for segment detection (no model required)."""

import csv

import numpy as np
import pytest

from sound_event_extractor.detect import (
    CSV_HEADER,
    Segment,
    combined_frame_scores,
    detect_segments,
    format_timestamp,
    insert_segment,
    make_manual_segment,
    merge_results,
    read_csv,
    segment_scores,
    write_csv,
    write_debug_scores,
)
from sound_event_extractor.labels import match_classes

HOP = 0.48
WIN = 0.96


def make_scores(frame_scores: list[float]) -> np.ndarray:
    """Build a (n_frames, 3) score matrix with the target class at index 1."""
    scores = np.zeros((len(frame_scores), 3), dtype=np.float32)
    scores[:, 1] = frame_scores
    return scores


def test_single_segment_times() -> None:
    scores = make_scores([0.0, 0.0, 0.8, 0.9, 0.7, 0.0, 0.0])
    segments = detect_segments(scores, [1], threshold=0.3, merge_gap=0.0)
    assert len(segments) == 1
    seg = segments[0]
    assert seg.start == pytest.approx(2 * HOP)
    assert seg.end == pytest.approx(4 * HOP + WIN)
    assert seg.max_score == pytest.approx(0.9)
    assert seg.mean_score == pytest.approx((0.8 + 0.9 + 0.7) / 3)


def test_nearby_segments_are_merged() -> None:
    # Two runs separated by one inactive frame: gap = 5*HOP - (3*HOP + WIN) = 0
    scores = make_scores([0.8, 0.8, 0.8, 0.8, 0.0, 0.8, 0.8])
    merged = detect_segments(scores, [1], threshold=0.3, merge_gap=1.0)
    assert len(merged) == 1
    separate = detect_segments(scores, [1], threshold=0.3, merge_gap=-1.0)
    assert len(separate) == 2


def test_min_duration_filters_short_segments() -> None:
    scores = make_scores([0.0, 0.8, 0.0])
    assert len(detect_segments(scores, [1], min_duration=0.5)) == 1  # 0.96 s long
    assert len(detect_segments(scores, [1], min_duration=1.0)) == 0


def test_threshold_and_other_classes_ignored() -> None:
    scores = make_scores([0.2, 0.2, 0.2])
    scores[:, 0] = 0.9  # high score on a non-target class must not trigger
    assert detect_segments(scores, [1], threshold=0.3) == []
    assert detect_segments(scores, [], threshold=0.3) == []


def test_format_timestamp() -> None:
    assert format_timestamp(0.0) == "0:00:00.000"
    assert format_timestamp(3.36) == "0:00:03.360"
    assert format_timestamp(3671.5) == "1:01:11.500"


def test_combined_frame_scores() -> None:
    scores = make_scores([0.2, 0.7])
    assert combined_frame_scores(scores, [1]).tolist() == pytest.approx([0.2, 0.7])
    assert combined_frame_scores(scores, []).size == 0


def test_write_debug_scores(tmp_path) -> None:
    scores = np.array([[0.1, 0.9, 0.2], [0.0, 0.1, 0.8]], dtype=np.float32)
    out = tmp_path / "scores.csv"
    write_debug_scores(str(out), ["A", "B", "C"], scores, [1], "test")
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0][:3] == ["frame_time_seconds", "frame_time", "score[test]"]
    assert rows[0][3:5] == ["top1_class", "top1_score"]
    assert len(rows) == 3
    assert rows[1][2] == "0.900"  # matched class (B) score in frame 0
    assert rows[1][3:5] == ["B", "0.900"]  # top1 overall in frame 0
    assert rows[2][3:5] == ["C", "0.800"]  # top1 overall in frame 1


def test_match_classes_alias_and_substring() -> None:
    class_names = ["Speech", "Dog", "Bark", "Growling", "Cat", "Thunder"]
    assert match_classes(class_names, "犬の鳴き声") == [1, 2, 3]
    assert match_classes(class_names, "thunder") == [5]
    assert match_classes(class_names, "存在しない音") == []
    assert match_classes(class_names, "") == []


def test_manual_segment_scores_from_frames() -> None:
    frames = np.array([0.1, 0.2, 0.9, 0.3, 0.0], dtype=np.float32)
    # frames overlapping [1.0, 1.4]: frame 1 (0.48-1.44) and frame 2 (0.96-1.92)
    assert segment_scores(frames, 1.0, 1.4) == pytest.approx((0.9, 0.55))
    assert all(np.isnan(segment_scores(None, 1.0, 1.5)))
    assert all(np.isnan(segment_scores(frames, 10.0, 11.0)))  # beyond analyzed range
    seg = make_manual_segment(1.0, 1.5)
    assert seg.is_manual and np.isnan(seg.max_score)


def test_insert_segment_keeps_order_and_skips_duplicates() -> None:
    auto = [Segment(1.0, 2.0, 0.5, 0.5), Segment(5.0, 6.0, 0.5, 0.5)]
    merged = insert_segment(auto, make_manual_segment(3.0, 4.0))
    assert [s.start for s in merged] == [1.0, 3.0, 5.0]
    assert insert_segment(merged, make_manual_segment(3.0, 4.0)) == merged
    assert len(auto) == 2  # input untouched


def test_merge_results_keeps_manual_only() -> None:
    previous = [Segment(1.0, 2.0, 0.5, 0.5), make_manual_segment(3.0, 4.0)]
    fresh = [Segment(7.0, 8.0, 0.9, 0.9)]
    merged = merge_results(fresh, previous)
    assert [(s.start, s.is_manual) for s in merged] == [(3.0, True), (7.0, False)]


def test_csv_roundtrip_with_manual_rows(tmp_path) -> None:
    segments = [Segment(1.0, 2.0, 0.5, 0.4), make_manual_segment(3.0, 4.5)]
    out = tmp_path / "events.csv"
    write_csv(str(out), "犬の鳴き声", segments)
    with open(out, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == CSV_HEADER
    assert rows[1][-1] == "auto" and rows[2][-1] == "manual"
    assert rows[2][6] == "" and rows[2][7] == ""  # no scores for unanalyzed manual rows

    label, loaded = read_csv(str(out))
    assert label == "犬の鳴き声"
    assert [(s.start, s.end, s.source) for s in loaded] == [
        (1.0, 2.0, "auto"),
        (3.0, 4.5, "manual"),
    ]
    assert loaded[0].max_score == pytest.approx(0.5)
    assert np.isnan(loaded[1].max_score)


def test_read_legacy_csv_without_source_column(tmp_path) -> None:
    legacy = tmp_path / "old.csv"
    legacy.write_text(
        "label,start_seconds,end_seconds,duration_seconds,start_time,end_time,max_score,mean_score\n"
        "dog,3.360,5.280,1.920,0:00:03.360,0:00:05.280,0.912,0.774\n",
        encoding="utf-8",
    )
    label, loaded = read_csv(str(legacy))
    assert label == "dog"
    assert len(loaded) == 1 and not loaded[0].is_manual
    assert loaded[0].mean_score == pytest.approx(0.774)


def test_read_csv_rejects_missing_columns(tmp_path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv(str(bad))
