"""ffmpeg/ffprobe wrappers: duration, 16 kHz mono WAV, frame sampling."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from cqa.config import get_settings

FRAME_WIDTH = 640
_FRAME_RE = re.compile(r"frame_(\d+)\.jpg$")


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed (rc={r.returncode}): {r.stderr.strip()[-500:]}")
    return r


def probe_duration(src: Path) -> float:
    r = _run(
        [
            get_settings().ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(src),
        ]
    )  # fmt: skip
    return float(r.stdout.strip())


def extract_wav(
    src: Path, out_dir: Path, windows: list[tuple[float, float]] | None = None
) -> list[tuple[Path, float]]:
    """16 kHz mono PCM WAV of the whole file, or one WAV per [start, end] window.
    Returns (wav path, offset in seconds) pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_settings().ffmpeg_bin
    audio_args = ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"]
    if not windows:
        out = out_dir / "audio.wav"
        if not out.exists():
            _run([ffmpeg, "-y", "-v", "error", "-i", str(src), *audio_args, str(out)])
        return [(out, 0.0)]
    results = []
    for start, end in windows:
        out = out_dir / f"audio_{int(start):06d}_{int(end):06d}.wav"
        if not out.exists():
            _run(
                [
                    ffmpeg, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                    "-i", str(src), *audio_args, str(out),
                ]
            )  # fmt: skip
        results.append((out, float(start)))
    return results


def frame_time(path: Path) -> float:
    m = _FRAME_RE.search(path.name)
    if not m:
        raise ValueError(f"not a frame file: {path.name}")
    return float(m.group(1))


def list_frames(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("frame_*.jpg"), key=frame_time)


def sample_frames_fps(src: Path, out_dir: Path, interval: float) -> list[Path]:
    """One decode pass; select keeps the first frame at or after each multiple of interval
    (the fps filter rounds to the nearest output slot and can drop the last frame)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_pattern = out_dir / "tmp_%06d.jpg"
    select = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval - 0.001})'"
    _run(
        [
            get_settings().ffmpeg_bin, "-y", "-v", "error", "-i", str(src), "-an",
            "-vf", f"{select},scale={FRAME_WIDTH}:-2", "-fps_mode", "vfr", "-q:v", "4",
            "-start_number", "0", str(tmp_pattern),
        ]
    )  # fmt: skip
    for p in sorted(out_dir.glob("tmp_*.jpg")):
        k = int(p.stem.split("_")[1])
        p.replace(out_dir / f"frame_{int(k * interval):06d}.jpg")
    return list_frames(out_dir)


def sample_frames_seek(
    src: Path, out_dir: Path, interval: float, duration: float | None = None
) -> list[Path]:
    """One fast input seek per timestamp (keyframe-accurate, decodes very little)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = duration if duration is not None else probe_duration(src)
    ffmpeg = get_settings().ffmpeg_bin
    t = 0.0
    while t < duration:
        out = out_dir / f"frame_{int(t):06d}.jpg"
        if not out.exists():
            _run(
                [
                    ffmpeg, "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(src), "-an",
                    "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "4", str(out),
                ]
            )  # fmt: skip
        t += interval
    return list_frames(out_dir)


def sample_frames(src: Path, out_dir: Path, interval: float | None = None) -> list[Path]:
    """Default strategy is chosen by the M4 benchmark (EXPERIMENTS E5)."""
    interval = interval or get_settings().frame_interval_sec
    existing = list_frames(out_dir) if out_dir.exists() else []
    if existing:
        return existing
    return sample_frames_seek(src, out_dir, interval)
