"""Audio extraction from video/audio files via ffmpeg."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 16000  # YAMNet expects 16 kHz mono float32


def find_ffmpeg() -> str:
    """Prefer a system ffmpeg; fall back to the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found: install ffmpeg or run `uv sync` to get the bundled binary"
        ) from exc


def extract_waveform(media_path: str) -> np.ndarray:
    """Decode the audio track of a media file to 16 kHz mono float32 in [-1, 1]."""
    cmd = [
        find_ffmpeg(),
        "-hide_banner",
        "-loglevel", "error",
        "-i", media_path,
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-acodec", "pcm_f32le",
        "-f", "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode {media_path!r}: {stderr}")
    waveform = np.frombuffer(proc.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise RuntimeError(f"no audio track found in {media_path!r}")
    return waveform
