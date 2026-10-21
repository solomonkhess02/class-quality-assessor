"""Fixed, explainable scoring: weighted mean -> label, hard caps, confidence (DECISIONS D2, D7)."""

from __future__ import annotations

from typing import Any

from cqa.config import Settings, get_settings
from cqa.judge.rubric import BY_ID
from cqa.schemas import Assessment, CriterionResult, JudgeOutput, Label, TextStats, VisualReport

LABELS: list[Label] = ["Poor", "Average", "Good", "Excellent"]
THRESHOLDS = [(3.4, "Excellent"), (2.8, "Good"), (2.2, "Average")]  # initial guesses (ARCHITECTURE.md)
FACE_MISSING_CAP_PCT = 40.0  # face missing in more than this share of frames -> at most Average
FACE_RARE_PCT = 40.0  # face visible in fewer frames than this -> extra visual penalty


def label_for(m: float) -> Label:
    for threshold, label in THRESHOLDS:
        if m >= threshold - 1e-9:
            return label  # type: ignore[return-value]
    return "Poor"


def weighted_score(criteria: list[CriterionResult]) -> float | None:
    scored = [c for c in criteria if c.score is not None and not c.insufficient_evidence and c.id in BY_ID]
    total_w = sum(BY_ID[c.id].weight for c in scored)
    if total_w == 0:
        return None
    return sum(BY_ID[c.id].weight * c.score for c in scored) / total_w


MIN_SCORED_FOR_LABEL = 3  # fewer scored criteria -> placeholder label, low confidence
PROBLEM_MAX_SCORE = 2
# Problem count -> label, set from the organiser's framing ("how many problems did this class have"),
# not fitted to the labelled set: 0 Excellent, 1 Good, 2 Average, 3+ Poor.
PROBLEM_LABELS: list[Label] = ["Excellent", "Good", "Average"]


def problem_count_label(criteria: list[CriterionResult]) -> tuple[Label, int]:
    """Alternative aggregation: every scored criterion at <= 2 is one problem (EXPERIMENTS, M7)."""
    n = sum(
        1
        for c in criteria
        if c.score is not None and not c.insufficient_evidence and c.id in BY_ID and c.score <= PROBLEM_MAX_SCORE
    )
    return (PROBLEM_LABELS[n] if n < len(PROBLEM_LABELS) else "Poor"), n


def score_visual(v: VisualReport | None, s: Settings | None = None) -> CriterionResult:
    """Phase 1: OpenCV metrics only. Thresholds live in config and are initial guesses."""
    s = s or get_settings()
    if v is None or v.n_frames == 0:
        return CriterionResult(
            id="visual_presence",
            score=None,
            insufficient_evidence=True,
            observations=["No video frames were analysed."],
        )
    score = 4
    obs = [f"A face is visible in {v.pct_face:.0f}% of {v.n_frames} sampled frames."]
    fixes = []
    if v.pct_face < s.visual_min_face_pct:
        score -= 1
        fixes.append("keep the camera on with your face clearly in frame")
    if v.pct_face < FACE_RARE_PCT:
        score -= 1
    if v.pct_dark > s.visual_max_dark_pct:
        score -= 1
        obs.append(f"{v.pct_dark:.0f}% of frames are dark (low brightness).")
        fixes.append("add a light in front of you")
    if v.pct_black > s.visual_max_black_pct:
        score -= 1
        obs.append(f"{v.pct_black:.0f}% of frames are black (video off or not shared).")
        fixes.append("make sure video is on for the whole class")
    if v.pct_frozen > s.visual_max_frozen_pct:
        score -= 1
        obs.append(f"{v.pct_frozen:.0f}% of frames are unchanged from the previous sample (frozen or static).")
    if score == 4:
        obs.append("Lighting and video look fine in the sampled frames.")
    return CriterionResult(
        id="visual_presence",
        score=max(1, score),
        observations=obs,
        improvement=("Please " + "; ".join(fixes) + ".") if fixes else "",
    )


def confidence(
    criteria: list[CriterionResult],
    evidence_dropped: int,
    evidence_total: int,
    transcript_source: str,
    transcript_usable: bool,
    has_visual: bool,
) -> float:
    excluded_w = sum(BY_ID[c.id].weight for c in criteria if c.insufficient_evidence or c.score is None)
    missing_w = sum(c.weight for cid, c in BY_ID.items() if cid not in {x.id for x in criteria})
    conf = 1.0 - 0.6 * (excluded_w + missing_w)
    if evidence_total:
        conf -= 0.3 * evidence_dropped / evidence_total
    if transcript_source == "whisper":
        conf -= 0.15
    if not transcript_usable:
        conf -= 0.4
    if not has_visual:
        conf -= 0.1
    return round(min(1.0, max(0.0, conf)), 2)


def assess(
    judged: JudgeOutput,
    visual: VisualReport | None,
    stats: TextStats | None,
    transcript_source: str,
    warnings: list[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> Assessment:
    meta = dict(meta or {})
    warnings = list(warnings or [])
    criteria = [c for c in judged.criteria if c.id != "visual_presence"] + [score_visual(visual)]
    order = list(BY_ID)
    criteria.sort(key=lambda c: order.index(c.id) if c.id in order else len(order))

    m = weighted_score(criteria)
    problem_label, problems = problem_count_label(criteria)
    mean_label = label_for(m) if m is not None else "Average"
    n_scored = sum(1 for c in criteria if c.score is not None and not c.insufficient_evidence)
    if n_scored < MIN_SCORED_FOR_LABEL:
        label: Label = "Average"
        warnings.append(f"Only {n_scored} criteria could be scored; the label is a placeholder.")
    else:
        label = problem_label if get_settings().aggregation == "problems" else mean_label

    # The ARCHITECTURE cap ("at most Average") is a warning until screen-share frames are reliably
    # excluded (E6: a Good class shares a board for long stretches and shows a face in only 40% of frames).
    if visual is not None and visual.n_frames and 100 - visual.pct_face > FACE_MISSING_CAP_PCT:
        warnings.append(
            f"No face visible in {100 - visual.pct_face:.0f}% of sampled frames (camera off, or screen share); "
            "not applied as a cap until screen-share detection is validated."
        )
    if not judged.transcript_usable:
        note = f" ({judged.transcript_note})" if judged.transcript_note else ""
        warnings.append(f"The transcript looks unusable{note}; treat this assessment with caution.")
    if visual is None:
        warnings.append("Video was not analysed; visual presence is not scored.")

    conf = confidence(
        criteria,
        meta.get("evidence_dropped", 0),
        meta.get("evidence_total", 0),
        transcript_source,
        judged.transcript_usable,
        visual is not None,
    )
    meta["transcript_source"] = transcript_source
    meta.update(
        {"mean_label": mean_label, "problem_count_label": problem_label, "problems": problems, "n_scored": n_scored}
    )
    return Assessment(
        label=label,
        weighted_score=round(m, 2) if m is not None else None,
        confidence=conf,
        criteria=criteria,
        key_facts=judged.key_facts,
        summary=judged.summary,
        strengths=judged.strengths,
        improvements=judged.improvements,
        warnings=warnings,
        text_stats=stats,
        visual=visual,
        meta=meta,
    )
