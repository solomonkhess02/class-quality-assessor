"""Label-free filter (DECISIONS D19, extends D15): how often each severity check fires. No API calls. Numbers only.

Usage: python scripts/check_rates.py --tag a3b_rates
A class counts as firing when the check fires in a majority of the tag's runs. Checks that fire for >= 80% of
the classes cannot tell classes apart; they are printed as LABEL_EXCLUDE additions (shown, not counted).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.config import get_settings  # noqa: E402
from cqa.schemas import Assessment  # noqa: E402

D19_MAX_FIRING = 0.8


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    tag_dir = get_settings().cache_dir / "eval" / args.tag
    votes: dict[str, dict[str, list[bool | None]]] = {}  # check -> class -> per-run result
    for run_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
        for f in run_dir.glob("*.json"):
            a = Assessment.model_validate_json(f.read_text(encoding="utf-8"))
            for cid, fires in a.meta.get("checks", {}).items():
                votes.setdefault(cid, {}).setdefault(f.stem, []).append(fires)
            tech = next((c for c in a.criteria if c.id == "technical_quality"), None)
            if tech is not None and tech.score is not None:
                votes.setdefault("tech_problem (info, not in the label rule)", {}).setdefault(f.stem, []).append(
                    tech.score <= 2
                )
    print("| check | classes firing | classes unknown | classes | firing rate | >= 80% |")
    print("|---|---|---|---|---|---|")
    exclude = []
    for cid, by_class in votes.items():
        n = len(by_class)
        firing = sum(sum(bool(v) for v in runs) * 2 > len(runs) for runs in by_class.values())
        unknown = sum(all(v is None for v in runs) for runs in by_class.values())
        rate = firing / n if n else 0.0
        flag = rate >= D19_MAX_FIRING
        if flag and not cid.startswith("tech_problem"):
            exclude.append(cid)
        print(f"| {cid} | {firing} | {unknown} | {n} | {rate:.0%} | {'yes' if flag else 'no'} |")
    print(f"\nChecks to add to LABEL_EXCLUDE: {','.join(exclude) or '(none)'}")


if __name__ == "__main__":
    main()
