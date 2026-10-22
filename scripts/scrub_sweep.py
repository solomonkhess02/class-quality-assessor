"""Sweep the name scrubber against the human labels of leak-audit hits (E11b). No API calls. Counts only.

Usage: python scripts/scrub_sweep.py
Needs .cache/leak_review_labels.json ({"labels": {class key: [[token, "name" | "word"], ...]}}; local only,
contains real names). demo/1 is skipped (its transcript is ASR noise).
  names caught    labelled names no longer anywhere in the transcript sent to the judge
  words scrubbed  labelled ordinary words with at least one occurrence removed
  over-scrub      common English words replaced, as a share of tutor words (max / mean over classes)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.transcript.privacy import _WORD, FUZZY_MIN, anonymise, over_scrub_share  # noqa: E402
from cqa.transcript.roles import assign_roles  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402

SKIP = {"demo/1"}
NO_SKIP = 99.0  # zipf no word reaches


def words(transcript) -> Counter[str]:
    return Counter(w.lower() for t in transcript.turns for w in _WORD.findall(t.text))


def main() -> None:
    s = get_settings()
    labels: dict[str, list[list[str]]] = json.loads(
        (s.cache_dir / "leak_review_labels.json").read_text(encoding="utf-8")
    )["labels"]
    raws = {
        it.key: assign_roles(parse_vtt(it.vtt.read_text(encoding="utf-8")))
        for it in discover(s.classes_dir, probe_duration=media.probe_duration)
        if it.vtt and it.key not in SKIP
    }
    n_names = sum(kind == "name" for k in labels if k in raws for _, kind in labels[k])
    n_words = sum(kind == "word" for k in labels if k in raws for _, kind in labels[k])
    variants = [("E11 (no [NAME] band, no skip)", FUZZY_MIN, NO_SKIP)]
    variants += [(f"fuzzy >= {t}, skip zipf >= {z}", t, z) for t in (70, 75, 80) for z in (3.5, 4.0)]

    print(
        f"| variant | names caught (of {n_names}) | words scrubbed (of {n_words}) | over-scrub max | over-scrub mean |"
    )
    print("|---|---|---|---|---|")
    for name, t, z in variants:
        caught = scrubbed = 0
        shares = []
        for key, raw in raws.items():
            anon, _ = anonymise(raw, name_min=t, common_zipf=z)
            before, after = words(raw), words(anon)
            for tok, kind in labels.get(key, []):
                if kind == "name":
                    caught += after[tok] == 0
                else:
                    scrubbed += after[tok] < before[tok]
            shares.append(over_scrub_share(raw, anon))
        print(f"| {name} | {caught} | {scrubbed} | {max(shares):.2%} | {sum(shares) / len(shares):.2%} |")


if __name__ == "__main__":
    main()
