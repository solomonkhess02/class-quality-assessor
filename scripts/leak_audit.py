"""Privacy audit of the exact judge input for every class with a VTT. No API calls. Prints counts only.

Usage: python scripts/leak_audit.py [--dump]
  (a) capitalised tokens not in the allowlist, pseudonyms or fixed prompt template
  (b) tokens fuzzy-matching an original display-name token at >= 70 (should be 0)
  (c) [NAME] replacements, and the share of tutor word tokens replaced (over-scrubbing; should be <= ~2%)
  --dump  writes matched (b) tokens to .cache/leak_audit_hits.json for local review; contains real names,
          never print it, paste it or commit it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.features.text_stats import compute_text_stats  # noqa: E402
from cqa.judge.prompts import build_messages  # noqa: E402
from cqa.transcript.privacy import anonymise, leak_counts, scrub_stats  # noqa: E402
from cqa.transcript.roles import assign_roles  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402


def joined(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    rows, dump = [], {}
    for item in discover(s.classes_dir, probe_duration=media.probe_duration):
        if not item.vtt:
            continue
        # Same steps as pipeline.run for a VTT-only class; video metrics are numbers under fixed keys.
        raw = assign_roles(parse_vtt(item.vtt.read_text(encoding="utf-8")))
        transcript, _mapping = anonymise(raw)
        stats = compute_text_stats(transcript, s.opening_window_min, s.closing_window_min)
        win = (s.opening_window_min, s.closing_window_min)
        text = joined(build_messages(transcript, stats, None, *win))
        template = joined(build_messages(transcript.model_copy(update={"turns": []}), stats, None, *win))
        lc = leak_counts(text, template, sorted({t.speaker for t in raw.turns}))
        n_name, tutor_share = scrub_stats(transcript)
        rows.append(
            [item.key, str(len(transcript.turns)), str(lc.caps), str(lc.caps_distinct), str(lc.name_hits),
             str(lc.names_hit), str(n_name), f"{tutor_share:.2%}"]
        )  # fmt: skip
        dump[item.key] = lc.hits

    header = ["class", "turns", "(a) caps", "(a) distinct", "(b) name hits", "(b) names hit", "(c) [NAME]",
              "(c) tutor share"]  # fmt: skip
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    if args.dump:
        out = s.cache_dir / "leak_audit_hits.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dump, indent=1), encoding="utf-8")
        print(f"\nWrote {out} (real names: local review only)")


if __name__ == "__main__":
    main()
