"""Accuracy of LLM role tagging on whisper transcripts, against the Zoom VTT speakers (P2.2). USES THE API.

Usage: python scripts/role_tag_check.py [class keys ...]
Needs the whisper cache of the current ASR settings (scripts/bench_asr_windows.py). For each class: anonymise
the whisper transcript (names scrubbed, as in the pipeline), tag roles with one DeepSeek call, and compare each
segment's role with the role of the most-overlapping VTT turn. Prints numbers only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.ingest import cache_dir_for, sha256_file  # noqa: E402
from cqa.schemas import Transcript  # noqa: E402
from cqa.transcript.privacy import anonymise  # noqa: E402
from cqa.transcript.role_tagging import reference_roles, tag_roles  # noqa: E402
from cqa.transcript.roles import assign_roles  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402
from cqa.transcript.whisper import cache_name, configured_windows  # noqa: E402

DEFAULT = ["excellent/B472", "good/B552", "average/B516", "weak/B482"]


def main(keys: list[str]) -> None:
    s = get_settings()
    items = {i.key: i for i in discover(s.classes_dir, probe_duration=media.probe_duration)}
    print("| class | segments | with VTT overlap | accuracy | tutor precision | tutor recall | lines skipped |")
    print("|---|---|---|---|---|---|---|")
    for key in keys:
        item = items[key]
        windows = configured_windows(media.probe_duration(item.video))
        cached = cache_dir_for(sha256_file(item.video)) / cache_name(windows, s.whisper_language)
        whisper = Transcript.model_validate_json(cached.read_text(encoding="utf-8"))
        tagged, meta = tag_roles(anonymise(whisper)[0])
        vtt = assign_roles(parse_vtt(item.vtt.read_text(encoding="utf-8")))
        refs = reference_roles(tagged.turns, vtt.turns)
        pairs = [(t.role, r) for t, r in zip(tagged.turns, refs, strict=True) if r in ("tutor", "student")]
        acc = sum(p == r for p, r in pairs) / len(pairs) if pairs else 0.0
        tp = sum(p == r == "tutor" for p, r in pairs)
        prec = tp / max(1, sum(p == "tutor" for p, _ in pairs))
        rec = tp / max(1, sum(r == "tutor" for _, r in pairs))
        print(f"| {key} | {len(tagged.turns)} | {len(pairs)} | {acc:.0%} | {prec:.0%} | {rec:.0%} | "
              f"{meta['role_lines_missing']} |")  # fmt: skip


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
