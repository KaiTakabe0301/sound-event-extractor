"""Unit tests for ffmpeg command construction."""

from sound_event_extractor.audio import DYNAUDNORM_FILTER, _ffmpeg_cmd


def test_normalize_adds_dynaudnorm_filter() -> None:
    cmd = _ffmpeg_cmd("in.mp4", normalize=True)
    assert "-af" in cmd
    assert DYNAUDNORM_FILTER in cmd
    # the filter must come after the input and before the output format args
    assert cmd.index("-i") < cmd.index("-af") < cmd.index("-f")


def test_default_has_no_filter() -> None:
    cmd = _ffmpeg_cmd("in.mp4", normalize=False)
    assert "-af" not in cmd
    assert cmd[-1] == "-"
