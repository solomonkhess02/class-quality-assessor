"""Evaluation metrics (docs/EVALUATION.md). Pure functions over class keys, labels and scores; no I/O."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

ORDINAL = {"Poor": 1, "Average": 2, "Good": 3, "Excellent": 4}
LABEL_NAMES = list(ORDINAL)
VERDICT_CREDIT = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def pairwise_accuracy(true: dict[str, str], severity: dict[str, float]) -> tuple[float, int]:
    """Over pairs of classes with different true labels: share where the better class has the lower severity
    (e.g. fewer problems). Ties count 0.5. Returns (accuracy, number of pairs)."""
    credit, n = 0.0, 0
    for a, b in combinations(sorted(true), 2):
        if ORDINAL[true[a]] == ORDINAL[true[b]]:
            continue
        better, worse = (a, b) if ORDINAL[true[a]] > ORDINAL[true[b]] else (b, a)
        n += 1
        if severity[better] < severity[worse]:
            credit += 1
        elif severity[better] == severity[worse]:
            credit += 0.5
    return (credit / n if n else 0.0), n


def label_metrics(true: dict[str, str], pred: dict[str, str]) -> dict[str, float]:
    """exact / within_one (counts), mae, bias (mean predicted - true; negative = harsh), n."""
    diffs = [ORDINAL[pred[k]] - ORDINAL[true[k]] for k in true]
    n = len(diffs)
    return {
        "exact": sum(d == 0 for d in diffs),
        "within_one": sum(abs(d) <= 1 for d in diffs),
        "mae": sum(abs(d) for d in diffs) / n if n else 0.0,
        "bias": sum(diffs) / n if n else 0.0,
        "n": n,
    }


def confusion(pairs: Iterable[tuple[str, str]]) -> dict[tuple[str, str], int]:
    """(true, predicted) -> count."""
    out = {(t, p): 0 for t in LABEL_NAMES for p in LABEL_NAMES}
    for t, p in pairs:
        out[(t, p)] += 1
    return out


def criterion_recall(
    expected: dict[str, dict[str, str]], scores: dict[str, dict[str, int | None]]
) -> dict[str, dict[str, tuple[int, int]]]:
    """Organiser expectations vs scores. A criticised point is caught when scored <= 2, a praised one when
    scored >= 3; an unscored criterion counts as not caught. Returns {criterion | "all": {"weak": (hit, n),
    "good": (hit, n)}}."""
    out: dict[str, dict[str, list[int]]] = {}
    for key, crits in expected.items():
        for cid, verdict in crits.items():
            score = scores.get(key, {}).get(cid)
            hit = score is not None and (score <= 2 if verdict == "weak" else score >= 3)
            for bucket in (cid, "all"):
                cell = out.setdefault(bucket, {"weak": [0, 0], "good": [0, 0]})[verdict]
                cell[0] += hit
                cell[1] += 1
    return {b: {v: (h, n) for v, (h, n) in d.items()} for b, d in out.items()}


def ratio(hit_n: tuple[float, int]) -> float | None:
    hit, n = hit_n
    return hit / n if n else None


def balanced_accuracy(recall: dict[str, tuple[int, int]]) -> float | None:
    parts = [r for r in (ratio(recall["weak"]), ratio(recall["good"])) if r is not None]
    return sum(parts) / len(parts) if parts else None


def feedback_recall(grades: Iterable[tuple[str, str]]) -> dict[str, tuple[float, int]]:
    """(points_source, verdict) pairs -> {"all" | source: (credit, n)}; yes 1, partial 0.5, no 0."""
    out: dict[str, list[float]] = {}
    for source, verdict in grades:
        for bucket in ("all", source):
            cell = out.setdefault(bucket, [0.0, 0])
            cell[0] += VERDICT_CREDIT[verdict]
            cell[1] += 1
    return {k: (c, int(n)) for k, (c, n) in out.items()}


def run_agreement(labels_by_run: list[dict[str, str]]) -> tuple[int, int]:
    """Classes whose label is identical in every run. Returns (stable, n)."""
    keys = set.intersection(*(set(r) for r in labels_by_run)) if labels_by_run else set()
    return sum(len({r[k] for r in labels_by_run}) == 1 for k in keys), len(keys)


def mean_score_range(scores_by_run: list[dict[str, dict[str, int | None]]]) -> float | None:
    """Mean (max - min) of each class x criterion score across runs, over cells scored in every run."""
    ranges = []
    for key in set.intersection(*(set(r) for r in scores_by_run)) if scores_by_run else set():
        for cid in scores_by_run[0][key]:
            vals = [r[key].get(cid) for r in scores_by_run]
            if all(v is not None for v in vals):
                ranges.append(max(vals) - min(vals))  # type: ignore[type-var]
    return sum(ranges) / len(ranges) if ranges else None


def confidence_separation(
    true: dict[str, str], pred: dict[str, str], conf: dict[str, float]
) -> tuple[float | None, float | None]:
    """Mean confidence on correct and on wrong labels."""
    right = [conf[k] for k in true if pred[k] == true[k]]
    wrong = [conf[k] for k in true if pred[k] != true[k]]
    return (sum(right) / len(right) if right else None), (sum(wrong) / len(wrong) if wrong else None)


def trivial_baselines(true: dict[str, str]) -> dict[str, dict[str, float]]:
    """Always-Poor, always-Average and (expected) uniform-random label baselines."""
    out = {}
    for name in ("Poor", "Average"):
        pred = {k: name for k in true}
        out[f'Always "{name}"'] = {
            **label_metrics(true, pred),
            "pairwise": pairwise_accuracy(true, {k: 0 for k in true})[0],
        }
    n = len(true)
    exp_exact = sum(1 / 4 for _ in true)
    exp_within = sum(sum(abs(ORDINAL[t] - p) <= 1 for p in range(1, 5)) / 4 for t in true.values())
    exp_mae = sum(sum(abs(ORDINAL[t] - p) for p in range(1, 5)) / 4 for t in true.values()) / n if n else 0.0
    exp_bias = sum(2.5 - ORDINAL[t] for t in true.values()) / n if n else 0.0
    out["Random label (expected)"] = {
        "exact": exp_exact, "within_one": exp_within, "mae": exp_mae, "bias": exp_bias, "n": n, "pairwise": 0.5,
    }  # fmt: skip
    return out


def spread(values: list[float]) -> tuple[float, float, float]:
    """(mean, min, max) of per-run values."""
    return sum(values) / len(values), min(values), max(values)
