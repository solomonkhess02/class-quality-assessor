"""Evaluation harness v2 (docs/EVALUATION.md). Class IDs and numbers only in the outputs.

Usage: python scripts/eval_labelled.py [--runs 3] [--tag baseline] [--skip-visual] [--no-grade]
  --runs N       independent pipeline runs, in parallel (the judge cache key includes the run index)
  --tag NAME     each run's full Assessment is kept under .cache/eval/<tag>/<run>/ and reused if present
  --skip-visual  VTT-only mode (the judge never sees video numbers)
  --no-grade     skip the feedback-recall grader (one DeepSeek call per labelled class per run)

Writes eval/scorecard.md (metric | value mean [min-max] across runs | target | n) and appends one row to
eval/ablations.md with the git commit and judge settings. The grader spot-check file for 3 classes goes to
.cache/eval/<tag>/feedback_spotcheck.md (local only: it quotes the organiser's points).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from leak_audit import audit  # noqa: E402  (scripts/ is on sys.path when run as a script)

from cqa import evaluation as ev  # noqa: E402
from cqa import media  # noqa: E402
from cqa.config import REPO_ROOT, get_settings  # noqa: E402
from cqa.dataset import ClassItem, discover  # noqa: E402
from cqa.features.visual import cached_frame_metrics  # noqa: E402
from cqa.ingest import cache_dir_for, sha256_file  # noqa: E402
from cqa.judge import grader, prompts  # noqa: E402
from cqa.judge.rubric import CRITERIA  # noqa: E402
from cqa.pipeline import run  # noqa: E402
from cqa.schemas import Assessment  # noqa: E402

SPOTCHECK = ["excellent/B472", "average/B516", "weak/B558"]  # one class per level group, run 0
SOURCES = {"T": "transcript-observable", "V": "video-observable", "A": "voice-only", "N": "not in recording"}


def safe(key: str) -> str:
    return key.replace("/", "__").replace(" ", "_")


def load_ground_truth(data_dir: Path) -> dict[str, dict]:
    raw = json.loads((data_dir / "expected_criteria.json").read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict)}


def run_once(run_idx: int, items: list[ClassItem], out: Path, skip_visual: bool) -> dict[str, Assessment]:
    rdir = out / str(run_idx)
    rdir.mkdir(parents=True, exist_ok=True)
    results = {}
    for item in items:
        f = rdir / f"{safe(item.key)}.json"
        if f.exists():
            results[item.key] = Assessment.model_validate_json(f.read_text(encoding="utf-8"))
            continue
        a, _ = run(None if skip_visual else item.video, item.vtt, skip_visual=skip_visual, run_index=run_idx)
        f.write_text(a.model_dump_json(indent=1), encoding="utf-8")
        results[item.key] = a
        print(f"  run {run_idx}: {item.key} -> {a.label} ({a.meta['problems']} problems)", flush=True)
    return results


def grade_once(run_idx: int, assessed: dict[str, Assessment], truth: dict[str, dict], out: Path) -> dict:
    """Per class: list of (points_source, verdict), plus grader meta."""
    gdir = out / str(run_idx) / "feedback"
    gdir.mkdir(parents=True, exist_ok=True)
    graded = {}
    for key, gt in truth.items():
        if key not in assessed or not gt.get("points"):
            continue
        f = gdir / f"{safe(key)}.json"
        if f.exists():
            saved = json.loads(f.read_text(encoding="utf-8"))
            grades, meta = [grader.PointGrade(**g) for g in saved["grades"]], saved["meta"]
        else:
            grades, meta = grader.grade_feedback(assessed[key], gt["points"])
            f.write_text(json.dumps({"grades": [g.model_dump() for g in grades], "meta": meta}, indent=1),
                         encoding="utf-8")  # fmt: skip
        graded[key] = {"pairs": list(zip(gt["points_source"], [g.verdict for g in grades], strict=True)),
                       "grades": grades, "meta": meta}  # fmt: skip
    return graded


def run_metrics(assessed: dict[str, Assessment], truth: dict[str, dict], graded: dict | None) -> dict:
    true = {k: gt["label"] for k, gt in truth.items() if k in assessed}
    pred = {k: assessed[k].label for k in true}
    problems = {k: assessed[k].meta["problems"] for k in true}
    scores = {k: {c.id: (None if c.insufficient_evidence else c.score) for c in assessed[k].criteria} for k in true}
    expected = {k: {c: v for c, v in truth[k].items() if v in ("good", "weak")} for k in true}
    ev_total = sum(assessed[k].meta.get("evidence_total", 0) for k in true)
    ev_dropped = sum(assessed[k].meta.get("evidence_dropped", 0) for k in true)
    obs = sum(assessed[k].meta.get("observations", 0) for k in true)
    generic = sum(assessed[k].meta.get("generic_observations", 0) for k in true)
    pairwise, n_pairs = ev.pairwise_accuracy(true, problems)
    m = {
        "pairwise": pairwise,
        "n_pairs": n_pairs,
        **ev.label_metrics(true, pred),
        "recall": ev.criterion_recall(expected, scores),
        "evidence_validity": (ev_total - ev_dropped) / ev_total if ev_total else None,
        "evidence_n": ev_total,
        "genericness": generic / obs if obs else None,
        "observations_n": obs,
        "confidence": ev.confidence_separation(true, pred, {k: assessed[k].confidence for k in true}),
        "true": true,
        "pred": pred,
        "problems": problems,
        "scores": scores,
    }
    if graded is not None:
        m["feedback"] = ev.feedback_recall(p for k in true if k in graded for p in graded[k]["pairs"])
        m["grader_downgraded"] = sum(g["meta"].get("downgraded", 0) for g in graded.values())
    return m


def fmt(values: list[float | None], kind: str) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return "-"
    mean, lo, hi = ev.spread(vals)
    if kind == "pct":
        return f"{mean:.0%} [{lo:.0%}-{hi:.0%}]"
    if kind == "signed":
        return f"{mean:+.2f} [{lo:+.2f} to {hi:+.2f}]"
    if kind.startswith("count/"):
        return f"{mean:.1f}/{kind[6:]} [{lo:g}-{hi:g}]"
    return f"{mean:.2f} [{lo:.2f}-{hi:.2f}]"


@cache  # read once at startup, so edits made while runs are going don't change the record
def git_commit() -> str:
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                           cwd=REPO_ROOT)  # fmt: skip
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def switches() -> str:
    """Ablation switches in effect (cqa/config.py)."""
    s = get_settings()
    return (
        f"evidence_first={'yes' if s.judge_evidence_first else 'no'}, tech={s.tech_scorer}, "
        f"exclude={s.label_exclude or '-'}, facts_drive={'yes' if s.facts_drive else 'no'}"
    )


@cache
def prompt_hash() -> str:
    text = (
        prompts.system_prompt() + inspect.getsource(prompts.user_prompt) + inspect.getsource(prompts.format_transcript)
    )
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def scorecard(per_run: list[dict], truth_n: dict, args, gate_line: str) -> list[str]:
    s = get_settings()
    n = per_run[0]["n"]

    def col(f: Callable[[dict], float | None]) -> list[float | None]:
        return [f(r) for r in per_run]

    def rec(r: dict, cid: str, kind: str) -> float | None:
        return ev.ratio(r["recall"][cid][kind]) if cid in r["recall"] else None

    rows = [
        ["Pairwise ordering (problem count, ties 0.5)", fmt(col(lambda r: r["pairwise"]), "pct"), ">= 80%",
         f"{per_run[0]['n_pairs']} pairs"],
        ["Exact label", fmt(col(lambda r: r["exact"]), f"count/{n}"), "-", str(n)],
        ["Within one level", fmt(col(lambda r: r["within_one"]), f"count/{n}"), f">= 9/{n}", str(n)],
        ["MAE (levels)", fmt(col(lambda r: r["mae"]), "num"), "<= 0.6", str(n)],
        ["Bias (predicted - true; negative = harsh)", fmt(col(lambda r: r["bias"]), "signed"), "within +-0.3", str(n)],
        ["Weakness recall", fmt(col(lambda r: rec(r, "all", "weak")), "pct"), ">= 75%", str(truth_n["weak"])],
        ["Strength recall", fmt(col(lambda r: rec(r, "all", "good")), "pct"), "-", str(truth_n["good"])],
        ["Balanced criterion accuracy", fmt(col(lambda r: ev.balanced_accuracy(r["recall"]["all"])), "pct"),
         ">= 75%", str(truth_n["weak"] + truth_n["good"])],
    ]  # fmt: skip
    if "feedback" in per_run[0]:
        rows.append(["Feedback recall (all points)", fmt(col(lambda r: ev.ratio(r["feedback"]["all"])), "pct"),
                     ">= 60%", str(per_run[0]["feedback"]["all"][1])])  # fmt: skip
        for code, name in SOURCES.items():
            if code in per_run[0]["feedback"]:
                rows.append([f"Feedback recall, {name} ({code})",
                             fmt(col(lambda r, c=code: ev.ratio(r["feedback"][c])), "pct"), "-",
                             str(per_run[0]["feedback"][code][1])])  # fmt: skip
    stable, n_stable = ev.run_agreement([r["pred"] for r in per_run])
    rng = ev.mean_score_range([r["scores"] for r in per_run])
    rows += [
        ["Evidence validity (quotes verified)", fmt(col(lambda r: r["evidence_validity"]), "pct"), ">= 95%",
         f"{sum(r['evidence_n'] for r in per_run) / len(per_run):.0f} quotes/run"],
        ["Genericness (observations with no speaker, time or quote)", fmt(col(lambda r: r["genericness"]), "pct"),
         "<= 20%", f"{sum(r['observations_n'] for r in per_run) / len(per_run):.0f} observations/run"],
        ["Run agreement (label identical in all runs)", f"{stable}/{n_stable}", f">= 8/{n_stable}",
         f"{len(per_run)} runs"],
        ["Mean criterion score range across runs", "-" if rng is None else f"{rng:.2f}", "-", "class x criterion"],
        ["Confidence on correct labels", fmt(col(lambda r: r["confidence"][0]), "num"), "correct > wrong", "-"],
        ["Confidence on wrong labels", fmt(col(lambda r: r["confidence"][1]), "num"), "", "-"],
        ["Privacy gate (scripts/leak_audit.py)", gate_line.removeprefix("Gate D16: "), "pass", "11 classes"],
    ]  # fmt: skip

    lines = [
        f"# Scorecard: {args.tag}",
        "",
        f"{date.today().isoformat()}, commit {git_commit()}, {len(per_run)} runs, "
        f"{'VTT only' if args.skip_visual else 'VTT + visual'}. Judge {s.judge_model}, design {s.judge_design}, "
        f"aggregation {s.aggregation}, temperature {s.judge_temperature}, thinking {s.judge_thinking}, prompt hash "
        f"{prompt_hash()}.",
        "Values are mean [min-max] across runs. Labels: weak and quality issue -> Poor. Criterion points: score <= 2 "
        "catches a criticised point, >= 3 a praised one; unscored counts as missed.",
        "",
        "| metric | value | target | n |",
        "|---|---|---|---|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]
    if "feedback" in per_run[0]:
        down = [r["grader_downgraded"] for r in per_run]
        lines += ["", f"Feedback grader: DeepSeek, temperature 0, sees only the report and the points. Verdicts whose "
                  f"cited line was not in the report were set to no: {sum(down)} across runs."]  # fmt: skip

    lines += [
        "",
        "## Per criterion",
        "",
        "| criterion | weakness recall | n | strength recall | n |",
        "|---|---|---|---|---|",
    ]
    for c in CRITERIA:
        if c.id not in per_run[0]["recall"]:
            continue
        cell = per_run[0]["recall"][c.id]
        lines.append(
            f"| {c.id} | {fmt(col(lambda r, i=c.id: rec(r, i, 'weak')), 'pct')} | {cell['weak'][1]} | "
            f"{fmt(col(lambda r, i=c.id: rec(r, i, 'good')), 'pct')} | {cell['good'][1]} |"
        )

    conf = ev.confusion((t, p) for r in per_run for k, t in r["true"].items() for p in [r["pred"][k]])
    lines += ["", f"## Confusion matrix (summed over {len(per_run)} runs; rows true, columns predicted)", "",
              "| true \\ predicted | " + " | ".join(ev.LABEL_NAMES) + " |", "|---|" + "---|" * 4]  # fmt: skip
    for t in ev.LABEL_NAMES:
        lines.append(f"| {t} | " + " | ".join(str(conf[(t, p)]) for p in ev.LABEL_NAMES) + " |")

    true = per_run[0]["true"]
    lines += ["", "## Trivial baselines (any real result must beat these)", "",
              "| baseline | exact | within one | MAE | bias | pairwise |", "|---|---|---|---|---|---|"]  # fmt: skip
    for name, b in ev.trivial_baselines(true).items():
        lines.append(
            f"| {name} | {b['exact']:g}/{n} | {b['within_one']:g}/{n} | {b['mae']:.2f} | {b['bias']:+.2f} | "
            f"{b['pairwise']:.0%} |"
        )

    lines += ["", "## Per class", "", "| class | true | label per run | problems per run | confidence per run |",
              "|---|---|---|---|---|"]  # fmt: skip
    for k in sorted(true):
        labels = " / ".join(r["pred"][k] for r in per_run)
        probs = " / ".join(str(r["problems"][k]) for r in per_run)
        confs = " / ".join(f"{r['assessed'][k].confidence:.2f}" for r in per_run)
        lines.append(f"| {k} | {true[k]} | {labels} | {probs} | {confs} |")
    return lines


def append_ablation(per_run: list[dict], args) -> None:
    s = get_settings()
    path = REPO_ROOT / "eval" / "ablations.md"
    header = [
        "# Ablations", "", "One row per configuration (docs/EVALUATION.md). Values are mean [min-max] across runs.", "",
        "| tag | date | commit | mode | runs | model | design | aggregation | temp | thinking | prompt hash | pairwise "
        "| exact | within one | MAE | bias | weakness recall | balanced | feedback recall | settings |",
        "|" + "---|" * 20,
    ]  # fmt: skip
    n = per_run[0]["n"]

    def col(f: Callable[[dict], float | None]) -> list[float | None]:
        return [f(r) for r in per_run]

    row = [
        args.tag, date.today().isoformat(), git_commit(), "VTT only" if args.skip_visual else "VTT + visual",
        str(len(per_run)), s.judge_model, s.judge_design, s.aggregation, str(s.judge_temperature), s.judge_thinking,
        prompt_hash(), fmt(col(lambda r: r["pairwise"]), "pct"), fmt(col(lambda r: r["exact"]), f"count/{n}"),
        fmt(col(lambda r: r["within_one"]), f"count/{n}"), fmt(col(lambda r: r["mae"]), "num"),
        fmt(col(lambda r: r["bias"]), "signed"), fmt(col(lambda r: ev.ratio(r["recall"]["all"]["weak"])), "pct"),
        fmt(col(lambda r: ev.balanced_accuracy(r["recall"]["all"])), "pct"),
        fmt(col(lambda r: ev.ratio(r["feedback"]["all"]) if "feedback" in r else None), "pct"),
        switches(),
    ]  # fmt: skip
    existing = path.read_text(encoding="utf-8").rstrip("\n").splitlines() if path.exists() else header
    path.write_text("\n".join([*existing, "| " + " | ".join(row) + " |"]) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--no-grade", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    truth = load_ground_truth(s.data_dir)
    items = [i for i in discover(s.classes_dir, probe_duration=media.probe_duration) if i.vtt]
    out = s.cache_dir / "eval" / args.tag
    print(f"commit {git_commit()}, prompt hash {prompt_hash()}, {switches()}", flush=True)

    if not args.skip_visual:  # warm the frame cache once, so parallel runs only read it
        print("Warming the frame cache", flush=True)
        for item in items:
            if item.video:
                cached_frame_metrics(item.video, cache_dir_for(sha256_file(item.video)))

    def one(run_idx: int) -> dict:
        assessed = run_once(run_idx, items, out, args.skip_visual)
        graded = None if args.no_grade else grade_once(run_idx, assessed, truth, out)
        return {**run_metrics(assessed, truth, graded), "assessed": assessed, "graded": graded}

    print(f"{args.runs} runs, tag {args.tag}", flush=True)
    with ThreadPoolExecutor(max_workers=args.runs) as pool:
        per_run = list(pool.map(one, range(args.runs)))

    if not args.no_grade and per_run[0]["graded"]:
        md = [f"# Feedback grader spot-check ({args.tag}, run 0)", "",
              "Local only. Check each verdict; report agreement as yes/partial/no matches.", ""]  # fmt: skip
        for key in SPOTCHECK:
            if key in per_run[0]["graded"]:
                gt = truth[key]
                md.append(grader.spotcheck_markdown(key, gt["points"], gt["points_source"],
                                                    per_run[0]["graded"][key]["grades"]))  # fmt: skip
        (out / "feedback_spotcheck.md").write_text("\n".join(md), encoding="utf-8")

    recall_all = per_run[0]["recall"]["all"]
    truth_n = {"weak": recall_all["weak"][1], "good": recall_all["good"][1]}
    _, gate_line, _ = audit()
    lines = scorecard(per_run, truth_n, args, gate_line)
    (REPO_ROOT / "eval").mkdir(exist_ok=True)
    (REPO_ROOT / "eval" / "scorecard.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_ablation(per_run, args)
    print("\n".join(lines))
    print(f"\nWrote eval/scorecard.md, appended eval/ablations.md; assessments under {out}")


if __name__ == "__main__":
    main()
