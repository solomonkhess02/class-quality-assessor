"""D15 check: how much each criterion's score varies across classes. No API calls. Numbers only.

Usage: python scripts/criterion_std.py --tag a0_evidence_first
For each criterion: the mean score of each class over the tag's runs, then the population standard deviation of
those means across all classes. technical_quality is also computed with the code scorer (D14) from the cached
transcript statistics and frame metrics, since that is what the label will use once TECH_SCORER=code.
Criteria with std < 0.3 are printed as the LABEL_EXCLUDE list (DECISIONS D15).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.features.text_stats import compute_text_stats  # noqa: E402
from cqa.features.visual import aggregate, cached_frame_metrics  # noqa: E402
from cqa.ingest import cache_dir_for, sha256_file  # noqa: E402
from cqa.judge.rubric import CRITERIA  # noqa: E402
from cqa.schemas import Assessment  # noqa: E402
from cqa.scoring import score_technical  # noqa: E402
from cqa.transcript.privacy import anonymise  # noqa: E402
from cqa.transcript.roles import assign_roles  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402

D15_MIN_STD = 0.3


def class_means(tag_dir: Path) -> dict[str, dict[str, float]]:
    """{criterion: {class file stem: mean score over runs}} for scored criteria."""
    scores: dict[str, dict[str, list[int]]] = {}
    for run_dir in sorted(p for p in tag_dir.iterdir() if p.is_dir()):
        for f in run_dir.glob("*.json"):
            a = Assessment.model_validate_json(f.read_text(encoding="utf-8"))
            for c in a.criteria:
                if c.score is not None and not c.insufficient_evidence:
                    scores.setdefault(c.id, {}).setdefault(f.stem, []).append(c.score)
    return {cid: {k: statistics.mean(v) for k, v in by_class.items()} for cid, by_class in scores.items()}


def code_tech_scores() -> dict[str, int]:
    s = get_settings()
    out = {}
    for item in discover(s.classes_dir, probe_duration=media.probe_duration):
        if not item.vtt:
            continue
        transcript, _ = anonymise(assign_roles(parse_vtt(item.vtt.read_text(encoding="utf-8"))))
        if not transcript.duration_s and item.video:
            transcript = transcript.model_copy(update={"duration_s": media.probe_duration(item.video)})
        stats = compute_text_stats(transcript, s.opening_window_min, s.closing_window_min)
        visual = (
            aggregate(cached_frame_metrics(item.video, cache_dir_for(sha256_file(item.video)))) if item.video else None
        )
        out[item.key] = score_technical(stats, visual, None).score
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    means = class_means(get_settings().cache_dir / "eval" / args.tag)
    rows = []
    for c in CRITERIA:
        vals = list(means.get(c.id, {}).values())
        rows.append((c.id, f"{args.tag} runs", len(vals), statistics.pstdev(vals) if len(vals) > 1 else None))
    tech = code_tech_scores()
    rows.append(("technical_quality", "code scorer (D14)", len(tech), statistics.pstdev(tech.values())))
    print("| criterion | scores from | classes | std | below 0.3 |")
    print("|---|---|---|---|---|")
    for cid, src, n, sd in rows:
        flag = "-" if sd is None else ("yes" if sd < D15_MIN_STD else "no")
        print(f"| {cid} | {src} | {n} | {'-' if sd is None else f'{sd:.2f}'} | {flag} |")
    # technical_quality is judged on its code scores, since the LLM's score is no longer used for it (A1)
    exclude = [cid for cid, src, _, sd in rows if sd is not None and sd < D15_MIN_STD
               and (cid != "technical_quality" or src.startswith("code"))]  # fmt: skip
    print(f"\nLABEL_EXCLUDE={','.join(exclude)}")
    print("Code technical problems per class:", {k: v for k, v in tech.items() if v == 2} or "none")


if __name__ == "__main__":
    main()
