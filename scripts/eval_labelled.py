"""Evaluate on the labelled set using the Zoom VTTs. Writes eval/results.md (class IDs and numbers only).

Usage: python scripts/eval_labelled.py [--skip-visual]
  default        two genuine runs per class: VTT + visual metrics, and VTT only (the judge never sees video numbers)
  --skip-visual  VTT-only run

Reports, per mode: label accuracy for two aggregation rules (weighted mean, problem count) and agreement with the
organiser's per-criterion expectations (DATA_DIR/expected_criteria.json; score >= 3 = good, <= 2 = weak).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import REPO_ROOT, get_settings  # noqa: E402
from cqa.dataset import discover, load_expected  # noqa: E402
from cqa.judge.rubric import CRITERIA  # noqa: E402
from cqa.pipeline import run  # noqa: E402
from cqa.scoring import LABELS  # noqa: E402

SHORT = {
    "opening_rapport": "open",
    "individual_engagement": "engage",
    "communication": "warmth",
    "closure_homework": "close",
    "visual_presence": "visual",
    "technical_quality": "tech",
}


def accuracy(pairs: list[tuple[str, str]]) -> tuple[int, int, int]:
    exact = sum(t == p for t, p in pairs)
    within = sum(abs(LABELS.index(t) - LABELS.index(p)) <= 1 for t, p in pairs)
    return exact, within, len(pairs)


def judged_as(score: int | None, insufficient: bool) -> str | None:
    if score is None or insufficient:
        return None
    return "good" if score >= 3 else "weak"


def run_mode(items, expected, skip_visual: bool) -> dict:
    rows, records = [], []
    for item in items:
        if not item.vtt:
            continue
        print(f"  {item.key} ...", flush=True)
        a, _ = run(None if skip_visual else item.video, item.vtt, skip_visual=skip_visual)
        exp = expected.get(item.key, {})
        cells, agree = [], {}
        for c in CRITERIA:
            cr = next((x for x in a.criteria if x.id == c.id), None)
            score = cr.score if cr else None
            insufficient = cr.insufficient_evidence if cr else True
            pred = judged_as(score, insufficient)
            mark = ""
            if c.id in exp:
                agree[c.id] = None if pred is None else pred == exp[c.id]
                mark = {True: " ✓", False: " ✗", None: " ?"}[agree[c.id]]
            cells.append(("-" if insufficient or score is None else str(score)) + mark)
        kf = a.key_facts
        facts = (
            f"hw={'y' if kf.homework_assigned else 'n'} join={kf.new_joiners} greet={kf.children_greeted_by_name} "
            f"recap={'y' if kf.recap_present else 'n'}"
            if kf
            else "-"
        )
        ev_total = a.meta.get("evidence_total", 0)
        ev = f"{ev_total - a.meta.get('evidence_dropped', 0)}/{ev_total}"
        rows.append(
            [
                item.key,
                item.label or "-",
                a.label + ("*" if a.meta["n_scored"] < 3 else ""),
                a.meta["mean_label"],
                "-" if a.weighted_score is None else f"{a.weighted_score:.2f}",
                a.meta["problem_count_label"],
                str(a.meta["problems"]),
                f"{a.confidence:.2f}",
                *cells,
                facts,
                ev,
            ]
        )
        records.append(
            {
                "key": item.key,
                "true": item.label,
                "mean_label": a.meta["mean_label"],
                "mean": a.weighted_score,
                "problem_label": a.meta["problem_count_label"],
                "problems": a.meta["problems"],
                "scores": {c.id: c.score for c in a.criteria},
                "agree": agree,
                "key_facts": kf.model_dump(exclude={"evidence", "homework_what"}) if kf else None,
                "evidence": [ev_total, a.meta.get("evidence_dropped", 0)],
                "generic": [a.meta.get("generic_observations"), a.meta.get("observations")],
            }
        )
    return {"rows": rows, "records": records}


def summarise(records: list[dict]) -> list[str]:
    labelled = [r for r in records if r["true"]]
    m_ex, m_w1, n = accuracy([(r["true"], r["mean_label"]) for r in labelled])
    p_ex, p_w1, _ = accuracy([(r["true"], r["problem_label"]) for r in labelled])
    lines = [
        f"- Weighted mean rule: exact {m_ex}/{n}, within one level {m_w1}/{n}",
        f"- Problem-count rule: exact {p_ex}/{n}, within one level {p_w1}/{n}",
        "",
        "| criterion | agree | disagree | no score | agreement |",
        "|---|---|---|---|---|",
    ]
    tot_a = tot_d = tot_u = 0
    for c in CRITERIA:
        vals = [r["agree"][c.id] for r in labelled if c.id in r["agree"]]
        a, d, u = vals.count(True), vals.count(False), vals.count(None)
        tot_a, tot_d, tot_u = tot_a + a, tot_d + d, tot_u + u
        pct = f"{a / (a + d):.0%}" if a + d else "-"
        lines.append(f"| {c.id} | {a} | {d} | {u} | {pct} |")
    pct = f"{tot_a / (tot_a + tot_d):.0%}" if tot_a + tot_d else "-"
    lines.append(f"| **all** | {tot_a} | {tot_d} | {tot_u} | {pct} |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-visual", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    items = discover(s.classes_dir, probe_duration=media.probe_duration)
    expected = load_expected(s.data_dir)
    modes = [("VTT only", True)] if args.skip_visual else [("VTT + visual", False), ("VTT only", True)]

    header = ["class", "true", "shipped", "mean label", "mean", "problem label", "problems", "conf",
              *[SHORT[c.id] for c in CRITERIA], "key facts", "evidence"]  # fmt: skip

    def table(rs: list[list[str]]) -> list[str]:
        head = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        return head + ["| " + " | ".join(r) + " |" for r in rs]

    lines = [
        "# Evaluation on the labelled set",
        "",
        f"Judge: {s.judge_model}, design {s.judge_design}, thinking {s.judge_thinking}, "
        f"temperature {s.judge_temperature}. Labels: weak and quality issue -> Poor.",
        "Criterion cells: score 1-4 ('-' = insufficient evidence); ✓ / ✗ = agrees / disagrees with the organiser's "
        "expectation (score >= 3 = good, <= 2 = weak), ? = organiser expectation but no score.",
        "Mean rule: Excellent >= 3.4, Good >= 2.8, Average >= 2.2. Problem rule: criteria scored <= 2; "
        "0 Excellent, 1 Good, 2 Average, 3+ Poor.",
        f"Shipped label uses AGGREGATION={s.aggregation}; * = placeholder (fewer than 3 criteria scored).",
    ]
    dump = {}
    for name, skip in modes:
        print(f"Mode: {name}", flush=True)
        res = run_mode(items, expected, skip)
        dump[name] = res["records"]
        labelled = [r for r in res["rows"] if r[1] != "-"]
        demos = [r for r in res["rows"] if r[1] == "-"]
        lines += ["", f"## {name}", "", *summarise(res["records"]), "", "### Labelled classes", "", *table(labelled),
                  "", "### Demo classes (unlabelled)", "", *table(demos)]  # fmt: skip

    out_dir = REPO_ROOT / "eval"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(dump, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print("\nWrote eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
