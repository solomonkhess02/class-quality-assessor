"""Benchmark the two frame-sampling strategies on one recording (EXPERIMENTS E5).

Usage: python scripts/bench_frames.py "quality issue/B441"
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402


def main(key: str) -> None:
    s = get_settings()
    item = next(i for i in discover(s.classes_dir) if i.key == key)
    dur = media.probe_duration(item.video)
    print(f"{key}: duration {dur / 60:.1f} min, interval {s.frame_interval_sec:.0f} s")
    for name, fn in [
        ("seek", lambda d: media.sample_frames_seek(item.video, d, s.frame_interval_sec, dur)),
        ("select", lambda d: media.sample_frames_fps(item.video, d, s.frame_interval_sec)),
    ]:
        out = s.cache_dir / "bench" / name
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.perf_counter()
        frames = fn(out)
        print(f"{name:<7} {time.perf_counter() - t0:7.1f} s  frames={len(frames)}")
    shutil.rmtree(s.cache_dir / "bench", ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "quality issue/B441")
