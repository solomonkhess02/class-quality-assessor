"""Malayalam transcription: base-q5_1 vs small-q5_1 with -l ml (no translation), 2 threads (P2.2). No API calls.

Usage: python scripts/bench_malayalam.py [class key] [start_s]
Default: demo/1 (the Malayalam demo; its Zoom VTT fails the quality check), a 5-minute clip from 10:00. Prints
numbers only: wall time, real-time factor, characters, and the share of letters in Malayalam script (U+0D00-U+0D7F).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import REPO_ROOT, get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.ingest import cache_dir_for, sha256_file  # noqa: E402
from cqa.transcript.whisper import transcribe  # noqa: E402

CLIP_S = 300
MODELS = [
    ("base-q5_1", REPO_ROOT / "models" / "ggml-base-q5_1.bin"),
    ("small-q5_1", REPO_ROOT / "models" / "ggml-small-q5_1.bin"),
]


def malayalam_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("ഀ" <= c <= "ൿ" for c in letters) / len(letters) if letters else 0.0


def main(key: str, start: float) -> None:
    s = get_settings()
    item = next(i for i in discover(s.classes_dir, probe_duration=media.probe_duration) if i.key == key)
    cdir = cache_dir_for(sha256_file(item.video))
    [(wav, offset)] = media.extract_wav(item.video, cdir / "bench_audio", [(start, start + CLIP_S)])
    print("| model | language | wall s | x real time | segments | characters | Malayalam script share |")
    print("|---|---|---|---|---|---|---|")
    for name, model in MODELS:
        t0 = time.perf_counter()
        turns, lang = transcribe(wav, offset, language="ml", model=model, threads=2)
        wall = time.perf_counter() - t0
        text = " ".join(t.text for t in turns)
        print(f"| {name} | {lang} | {wall:.0f} | {wall / CLIP_S:.2f} | {len(turns)} | {len(text)} | "
              f"{malayalam_share(text):.0%} |", flush=True)  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo/1", float(sys.argv[2]) if len(sys.argv) > 2 else 600.0)
