"""Benchmark whisper.cpp models on one 2-minute window (EXPERIMENTS). Prints numbers only.

Usage: python scripts/bench_whisper.py ["average/B435"] [start_s]
"""

from __future__ import annotations

import difflib
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import REPO_ROOT, get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.features.text_stats import window  # noqa: E402
from cqa.ingest import cache_dir_for, sha256_file  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402
from cqa.transcript.whisper import transcribe  # noqa: E402

CLIP_S = 120


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def main(key: str, start: float) -> None:
    s = get_settings()
    item = next(i for i in discover(s.classes_dir) if i.key == key)
    cdir = cache_dir_for(sha256_file(item.video))
    [(wav, offset)] = media.extract_wav(item.video, cdir / "bench_audio", [(start, start + CLIP_S)])
    ref = words(" ".join(t.text for t in window(parse_vtt(item.vtt.read_text("utf-8")), start, start + CLIP_S)))
    configs = [
        ("large-v3-turbo-q5_0", s.whisper_model, 6),
        ("base-q5_1", REPO_ROOT / "models/ggml-base-q5_1.bin", 2),
        ("small-q5_1", REPO_ROOT / "models/ggml-small-q5_1.bin", 2),
        ("large-v3-turbo-q5_0", s.whisper_model, 2),
    ]
    print(f"{key} window {start:.0f}-{start + CLIP_S:.0f} s, VTT words {len(ref)}")
    print(f"{'model':<22}{'threads':>8}{'wall_s':>8}{'RTF':>6}{'segments':>9}{'first_s':>8}{'agree%':>8}  lang")
    for name, model, threads in configs:
        t0 = time.perf_counter()
        turns, lang = transcribe(wav, offset, language="en", model=model, threads=threads)
        wall = time.perf_counter() - t0
        hyp = words(" ".join(t.text for t in turns))
        agree = difflib.SequenceMatcher(None, ref, hyp, autojunk=False).ratio() * 100
        first = turns[0].start_s if turns else float("nan")
        cols = f"{threads:>8}{wall:>8.1f}{wall / CLIP_S:>6.2f}{len(turns):>9}{first:>8.1f}{agree:>8.0f}"
        print(f"{name:<22}{cols}  {lang}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "average/B435", float(sys.argv[2]) if len(sys.argv) > 2 else 180)
