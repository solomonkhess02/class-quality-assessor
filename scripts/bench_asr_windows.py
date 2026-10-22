"""Windowed whisper.cpp transcription on labelled classes (P2.2). No API calls. Prints numbers only.

Usage: python scripts/bench_asr_windows.py [class keys ...]
Default: one class per level. Model base-q5_1 at 2 threads (DECISIONS D11). Windows, language detection and
prompt from the ASR_* settings. The transcript is cached locally as
<class cache>/whisper_<language>_<windows>.json (the pipeline's cache); agreement = word-sequence similarity
with the Zoom VTT over the same windows
(both are machine transcripts, so this is agreement, not accuracy).
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
from cqa.ingest import cache_dir_for, sha256_file  # noqa: E402
from cqa.schemas import Transcript  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402
from cqa.transcript.whisper import cache_name, configured_windows, transcribe_wavs  # noqa: E402

DEFAULT = ["excellent/B472", "good/B552", "average/B516", "weak/B482"]
MODEL = REPO_ROOT / "models" / "ggml-base-q5_1.bin"
THREADS = 2


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]+", " ", text.lower()).split()


def main(keys: list[str]) -> None:
    s = get_settings()
    items = {i.key: i for i in discover(s.classes_dir, probe_duration=media.probe_duration)}
    print(f"Windows {s.asr_opening_min:g} + {s.asr_closing_min:g} + {s.asr_middle_windows} x {s.asr_middle_min:g} min, "
          f"language {s.whisper_language}, detect once {s.asr_detect_language_once}, chess prompt {s.asr_chess_prompt}")
    print("| class | duration min | windows | audio min | extract s | whisper s | x real time | language | "
          "words | agreement with VTT |")  # fmt: skip
    print("|" + "---|" * 10)
    for key in keys:
        item = items[key]
        cdir = cache_dir_for(sha256_file(item.video))
        duration = media.probe_duration(item.video)
        windows = configured_windows(duration) or [(0.0, duration)]
        t0 = time.perf_counter()
        wavs = media.extract_wav(item.video, cdir / "asr_windows", windows)
        t_extract = time.perf_counter() - t0
        t0 = time.perf_counter()
        turns, language = transcribe_wavs(wavs, model=MODEL, threads=THREADS)
        t_whisper = time.perf_counter() - t0
        audio_min = sum(b - a for a, b in windows) / 60
        vtt = parse_vtt(item.vtt.read_text(encoding="utf-8"))
        in_win = [t for t in vtt.turns if any(a <= t.start_s < b for a, b in windows)]
        ref, hyp = words(" ".join(t.text for t in in_win)), words(" ".join(t.text for t in turns))
        agreement = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False).ratio()
        cached = Transcript(source="whisper", language=language, duration_s=duration, turns=turns, windows=windows)
        (cdir / cache_name(windows, s.whisper_language)).write_text(cached.model_dump_json(), encoding="utf-8")
        rt = t_whisper / (audio_min * 60)
        print(f"| {key} | {duration / 60:.0f} | {len(windows)} | {audio_min:.0f} | {t_extract:.0f} | {t_whisper:.0f} | "
              f"{rt:.2f} | {language} | {len(hyp)} | {agreement:.0%} |", flush=True)  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
