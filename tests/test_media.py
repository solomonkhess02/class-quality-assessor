"""Media tests on a synthetic ffmpeg test pattern (no real recordings)."""

import shutil
import subprocess
import wave
from pathlib import Path

import pytest

from cqa import media
from cqa.config import get_settings


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    ffmpeg = get_settings().ffmpeg_bin
    if not (Path(ffmpeg).is_file() or shutil.which(ffmpeg)):
        pytest.skip("ffmpeg not available")
    out = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=10:duration=12",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out),
        ],
        check=True,
    )  # fmt: skip
    return out


def test_probe_duration(clip):
    assert media.probe_duration(clip) == pytest.approx(12, abs=0.2)


def test_extract_wav_whole_and_windows(clip, tmp_path):
    [(wav, offset)] = media.extract_wav(clip, tmp_path)
    with wave.open(str(wav)) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1)
        assert w.getnframes() / 16000 == pytest.approx(12, abs=0.2)
    assert offset == 0
    parts = media.extract_wav(clip, tmp_path / "win", [(2, 4), (8, 11)])
    assert [o for _, o in parts] == [2, 8]
    with wave.open(str(parts[1][0])) as w:
        assert w.getnframes() / 16000 == pytest.approx(3, abs=0.1)


@pytest.mark.parametrize("strategy", ["fps", "seek"])
def test_sample_frames(clip, tmp_path, strategy):
    if strategy == "fps":
        frames = media.sample_frames_fps(clip, tmp_path, 5)
    else:
        frames = media.sample_frames_seek(clip, tmp_path, 5)
    assert [media.frame_time(p) for p in frames] == [0, 5, 10]
    import cv2

    assert cv2.imread(str(frames[0])).shape[1] == media.FRAME_WIDTH
