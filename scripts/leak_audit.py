"""Privacy audit of the exact judge input for every class with a VTT. No API calls. Prints counts only.

Usage: python scripts/leak_audit.py [--dump]
  (a) capitalised tokens not in the allowlist, pseudonyms or fixed prompt template
  (b) tokens fuzzy-matching an original display-name token, split at 85: >= 85 must be 0; 70-84 is checked
      against the human labels in .cache/leak_review_labels.json (labelled names left must be 0; DECISIONS D16)
  (c) over-scrub: common English words replaced, as a share of tutor words (target <= 0.5%)
  --dump  writes matched (b) tokens to .cache/leak_audit_hits.json for local review; contains real names,
          never print it, paste it or commit it.
demo/1 is listed but not gated: its transcript is English ASR of Malayalam speech (noise).
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
from cqa.transcript.privacy import FUZZY_MIN, anonymise, leak_counts, over_scrub_share  # noqa: E402
from cqa.transcript.roles import assign_roles  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402

NOT_GATED = {"demo/1"}


def joined(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    labels_path = s.cache_dir / "leak_review_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"] if labels_path.exists() else None
    rows, dump = [], {}
    high_total = names_left_total = 0
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
        high = [h for h in lc.hits if h[2] >= FUZZY_MIN]
        band = [h for h in lc.hits if h[2] < FUZZY_MIN]
        if labels is None:
            names_left = unlabelled = "-"
        else:
            known = dict(map(tuple, labels.get(item.key, [])))
            names_left = sum(known.get(w) == "name" for w, _, _ in band)
            unlabelled = sum(w not in known for w, _, _ in band)
            if item.key not in NOT_GATED:
                names_left_total += names_left
        if item.key not in NOT_GATED:
            high_total += len(high)
        rows.append(
            [item.key + (" (not gated)" if item.key in NOT_GATED else ""), str(len(transcript.turns)),
             str(lc.caps), str(lc.caps_distinct), str(len(high)), str(len(band)), str(names_left), str(unlabelled),
             f"{over_scrub_share(raw, transcript):.2%}"]
        )  # fmt: skip
        dump[item.key] = lc.hits

    header = ["class", "turns", "(a) caps", "(a) distinct", "(b) >= 85", "(b) 70-84", "(b) labelled names left",
              "(b) unlabelled 70-84", "(c) over-scrub"]  # fmt: skip
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print("(b) columns count distinct tokens.")
    if labels is None:
        print(f"\nGate D16: not checked ({labels_path.name} missing); hits >= 85: {high_total}")
    else:
        verdict = "PASS" if high_total == 0 and names_left_total == 0 else "FAIL"
        print(f"\nGate D16: {verdict} (hits >= 85: {high_total}, labelled names left: {names_left_total})")
    if args.dump:
        out = s.cache_dir / "leak_audit_hits.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dump, indent=1), encoding="utf-8")
        print(f"\nWrote {out} (real names: local review only)")


if __name__ == "__main__":
    main()
