"""Transcript quality check (cqa/transcript/quality.py) on every Zoom VTT. No API calls. Numbers only.

Usage: python scripts/quality_table.py
Expected: the Malayalam demo (English ASR applied to Malayalam speech, EXPERIMENTS E2) is flagged, the rest pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.transcript.quality import check_quality  # noqa: E402
from cqa.transcript.roles import assign_roles  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402


def main() -> None:
    s = get_settings()
    print("| class | usable | tutor words/min | non-English share | reasons |")
    print("|---|---|---|---|---|")
    for item in discover(s.classes_dir, probe_duration=media.probe_duration):
        if not item.vtt:
            continue
        q = check_quality(assign_roles(parse_vtt(item.vtt.read_text(encoding="utf-8"))))
        share = "-" if q.non_english_share is None else f"{q.non_english_share:.1%}"
        print(f"| {item.key} | {'yes' if q.usable else 'NO'} | {q.words_per_min:.0f} | {share} | {len(q.reasons)} |")


if __name__ == "__main__":
    main()
