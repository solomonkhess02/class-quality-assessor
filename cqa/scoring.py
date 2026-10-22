"""Fixed, explainable scoring: weighted mean -> label, hard caps, confidence (DECISIONS D2, D7)."""

from __future__ import annotations

from typing import Any

from cqa.config import Settings, get_settings
from cqa.judge.rubric import BY_ID
from cqa.schemas import (
    Assessment,
    CriterionResult,
    JudgeOutput,
    KeyFacts,
    Label,
    TextStats,
    VisualReport,
    format_ts,
)

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


TECH_INCIDENTS_PER_HOUR = 3.0  # D14, fixed before results
TECH_FROZEN_CAMERA_PCT = 20.0  # D14


def score_technical(
    stats: TextStats | None, visual: VisualReport | None, llm: CriterionResult | None
) -> CriterionResult:
    """D14: a technical problem only on objective evidence, i.e. >= 3 connectivity incidents per hour of class (found
    in the transcript by code) or >= 20% frozen camera frames. The judge's observations and verified quotes are kept
    as context; its score is not used."""
    if stats is None or stats.duration_min <= 0:
        return CriterionResult(id="technical_quality", observations=["No transcript statistics available."])
    incidents = stats.connectivity_incidents_s
    rate = len(incidents) * 60 / stats.duration_min
    shown = ", ".join(format_ts(t) for t in incidents[:8]) + (" ..." if len(incidents) > 8 else "")
    obs = [f"{len(incidents)} connectivity incidents found in the transcript ({rate:.1f} per hour)"
           + (f" at {shown}." if incidents else ".")]  # fmt: skip
    problems = ["connectivity"] if rate >= TECH_INCIDENTS_PER_HOUR else []
    if visual is not None and visual.n_frames:
        obs.append(f"{visual.pct_frozen_camera:.0f}% of camera frames (not screen share) are frozen.")
        if visual.pct_frozen_camera >= TECH_FROZEN_CAMERA_PCT:
            problems.append("frozen video")
    else:
        obs.append("No video analysed, so frozen video was not checked.")
    improvement = ""
    if problems:
        improvement = (llm.improvement if llm and llm.improvement else "") or (
            "Check the internet connection and camera before class, and keep a wired or stronger connection ready."
        )
    return CriterionResult(
        id="technical_quality",
        score=2 if problems else 3,
        observations=obs + [f"Judge: {o}" for o in (llm.observations if llm else [])],
        evidence=llm.evidence if llm else [],
        improvement=improvement,
    )


def label_excluded(s: Settings | None = None) -> set[str]:
    """D15: criteria shown in the report but left out of the label (LABEL_EXCLUDE; "none" = exclude nothing)."""
    s = s or get_settings()
    return {c.strip() for c in s.label_exclude.split(",") if c.strip() and c.strip().lower() != "none"}


# Ablation A3 (AGGREGATION=severity): severity-tiered, two-sided label rule, written from the organiser's wording
# before the run (docs/EVALUATION.md). Derived from the notes on the same 10 classes, so its accuracy there is
# optimistic.
SEVERITY_CRITICAL_WEIGHT = 10  # in the ordering value, any critical problem outranks all moderate ones


# A3b: a face is underexposed when its luma is more than one stop below middle grey (18% grey = 118 in sRGB,
# 9% = 85). Fixed from this definition before the run, not fitted.
FACE_UNDEREXPOSED_LUMA = 85.0


def severity_checks(
    criteria: list[CriterionResult], kf: KeyFacts | None, visual: VisualReport | None, s: Settings
) -> dict[str, tuple[bool | None, str, str]]:
    """Every severity check as {id: (fires: True | False | None = unknown, tier, description)} (A3b)."""
    score = {c.id: (None if c.insufficient_evidence else c.score) for c in criteria}
    checks: dict[str, tuple[bool | None, str, str]] = {}

    def fact(name: str) -> bool | None:  # fires on a verified "no"; unknown never fires
        state = getattr(kf, name) if kf is not None else "unknown"
        return None if state == "unknown" else state == "no"

    if s.facts_drive:
        # D22: a critical problem needs speaker-labelled evidence; without speakers (whisper path) rapport is the
        # judge's fact from a quote whose speaker is unknown, so it can count as a moderate problem at most
        judged = kf is not None and kf.rapport_method == "judge"
        checks["no_rapport"] = (
            fact("rapport_before_content"),
            "moderate" if judged else "critical",
            "no rapport before content (judged without speakers)" if judged else "no rapport before content",
        )
    else:
        opening = score.get("opening_rapport")
        checks["no_rapport"] = (None if opening is None else opening <= 2, "critical", "no rapport (opening <= 2)")
    has_video = visual is not None and visual.n_frames > 0
    face_low = visual.pct_face_camera < s.visual_min_face_pct if has_video else None  # type: ignore[union-attr]
    face_desc = f"tutor face in only {visual.pct_face_camera:.0f}% of camera frames" if has_video else "face share"  # type: ignore[union-attr]
    checks["face_low"] = (face_low, "critical" if s.face_check_critical else "moderate", face_desc)
    # "no 8x8 background" is a critical problem in the spec but is not measured yet (ablation A4)
    checks["no_homework"] = (fact("homework_assigned"), "moderate", "no homework")
    checks["no_recap"] = (fact("recap_present"), "moderate", "no closing recap")
    energy = score.get("communication")
    checks["low_energy"] = (None if energy is None else energy <= 2, "moderate", "low energy")
    face_luma = visual.median_face_brightness if has_video else None  # type: ignore[union-attr]
    checks["dark_face"] = (
        None if face_luma is None else face_luma < FACE_UNDEREXPOSED_LUMA,
        "moderate",
        f"lighting: face luma {face_luma:.0f} (< {FACE_UNDEREXPOSED_LUMA:.0f})"
        if face_luma is not None
        else "lighting",
    )
    return checks


def severity_findings(
    criteria: list[CriterionResult], kf: KeyFacts | None, visual: VisualReport | None, s: Settings
) -> tuple[list[str], list[str], bool]:
    """(critical problems, moderate problems, positive evidence for Excellent). Checks in LABEL_EXCLUDE (by check id,
    or "communication" for low_energy) are shown in the report but not counted (D15, D19)."""
    excluded = label_excluded(s)
    critical: list[str] = []
    moderate: list[str] = []
    for cid, (fires, tier, desc) in severity_checks(criteria, kf, visual, s).items():
        if not fires or cid in excluded or (cid == "low_energy" and "communication" in excluded):
            continue
        (critical if tier == "critical" else moderate).append(desc)
    # post-hoc (A3b): positive evidence as facts, consistent with A2
    positive = kf is not None and kf.rapport_before_content == "yes" and kf.each_child_asked_individually == "yes"
    return critical, moderate, positive


def severity_label(critical: list[str], moderate: list[str], positive: bool) -> tuple[Label, float, int]:
    """(label, ordering value, margin). Margin = fewest findings that would have to flip to change the label."""
    if critical:
        return "Poor", SEVERITY_CRITICAL_WEIGHT * len(critical) + len(moderate), len(critical)
    n = len(moderate)
    if n >= 2:
        return "Average", n, n - 1
    if n == 1:
        return "Good", 1, 1
    return ("Excellent", 0, 1) if positive else ("Good", 0.5, 1)


def margin_confidence(margin: int) -> float:
    """A3 confidence from the distance to the nearest label boundary: 0.75 one flip away, 1.0 two or more."""
    return 0.5 + 0.25 * min(margin, 2)


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
    s = get_settings()
    meta = dict(meta or {})
    warnings = list(warnings or [])
    criteria = [c for c in judged.criteria if c.id != "visual_presence"] + [score_visual(visual)]
    if s.tech_scorer == "code":
        llm_tech = next((c for c in criteria if c.id == "technical_quality"), None)
        criteria = [c for c in criteria if c.id != "technical_quality"] + [score_technical(stats, visual, llm_tech)]
    order = list(BY_ID)
    criteria.sort(key=lambda c: order.index(c.id) if c.id in order else len(order))

    excluded = label_excluded(s)
    counted = [c for c in criteria if c.id not in excluded]
    m = weighted_score(counted)
    problem_label, problems = problem_count_label(counted)
    mean_label = label_for(m) if m is not None else "Average"
    n_scored = sum(1 for c in counted if c.score is not None and not c.insufficient_evidence)
    found = [c.id for c in counted if c.score is not None and not c.insufficient_evidence and c.score <= 2]
    margin = None
    if s.aggregation == "severity":
        critical, moderate, positive = severity_findings(criteria, judged.key_facts, visual, s)
        severity_lbl, severity, margin = severity_label(critical, moderate, positive)
        found = critical + moderate
        checks = severity_checks(criteria, judged.key_facts, visual, s)
        meta.update(
            {
                "severity": severity,
                "critical": critical,
                "moderate": moderate,
                "severity_label": severity_lbl,
                "checks": {cid: fires for cid, (fires, _, _) in checks.items()},
            }
        )
        if checks["face_low"][0] and not s.face_check_critical:
            warnings.append(
                "Low face share counted as a moderate problem only: the screen-share tag is not yet validated (E6)."
            )
    if n_scored < MIN_SCORED_FOR_LABEL:
        label: Label = "Average"
        warnings.append(f"Only {n_scored} criteria could be scored; the label is a placeholder.")
    else:
        label = {"problems": problem_label, "severity": meta.get("severity_label")}.get(s.aggregation) or mean_label

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
    if margin is not None:  # A3: distance to the nearest label boundary; low whenever the input is unreliable
        conf = margin_confidence(margin)
        if not judged.transcript_usable or n_scored < MIN_SCORED_FOR_LABEL:
            conf = min(conf, 0.25)
    meta["transcript_source"] = transcript_source
    meta.update(
        {
            "mean_label": mean_label,
            "problem_count_label": problem_label,
            "problems": problems,
            "n_scored": n_scored,
            "label_excluded": sorted(excluded),
            "problems_found": found,
            "score_100": round((m - 1) / 3 * 100) if m is not None else None,
            "tech_scorer": s.tech_scorer,
        }
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
