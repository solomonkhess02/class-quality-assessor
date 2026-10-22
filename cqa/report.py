"""What the report page says, as plain data (so it is testable without Streamlit): key reasons, next-level actions,
what could not be checked, reliability notes."""

from __future__ import annotations

from dataclasses import dataclass

from cqa.config import get_settings
from cqa.judge.rubric import BY_ID
from cqa.schemas import Assessment, CriterionResult, Evidence
from cqa.scoring import severity_checks

QUOTE_CHARS = 140
# severity check id -> (criterion whose evidence explains it, key fact whose "no" quote shows it)
CHECK_SOURCES: dict[str, tuple[str, str | None]] = {
    "no_rapport": ("opening_rapport", "rapport_before_content"),
    "face_low": ("visual_presence", None),
    "no_homework": ("closure_homework", "homework_assigned"),
    "no_recap": ("closure_homework", "recap_present"),
    "low_energy": ("communication", None),
    "dark_face": ("visual_presence", None),
}
NEXT_LEVEL = {"Poor": "Average", "Average": "Good", "Good": "Excellent"}


@dataclass
class Reason:
    text: str
    problem: bool
    evidence: Evidence | None = None  # verified quote, if the finding rests on one
    criterion: str = ""


def short(quote: str, n: int = QUOTE_CHARS) -> str:
    quote = " ".join(quote.split())
    return quote if len(quote) <= n else quote[: n - 1].rsplit(" ", 1)[0] + " …"


def first_sentence(text: str) -> str:
    text = " ".join(text.split())
    for end in (". ", "! ", "? "):
        i = text.find(end)
        if 0 < i < 300:
            return text[: i + 1]
    return text


def excluded(a: Assessment) -> set[str]:
    return set(a.meta.get("label_excluded", []))


def _verified(evidence: list[Evidence]) -> Evidence | None:
    return next((e for e in evidence if e.verified), None)


def _criterion(a: Assessment, cid: str) -> CriterionResult | None:
    return next((c for c in a.criteria if c.id == cid), None)


def key_reasons(a: Assessment, n: int = 3) -> list[Reason]:
    """The counted problems behind the label (critical first), then the strongest counted criteria."""
    counted = a.meta.get("critical", []) + a.meta.get("moderate", [])
    reasons: list[Reason] = []
    if counted:
        tier = {d: ("critical" if d in a.meta.get("critical", []) else "moderate") for d in counted}
        checks = severity_checks(a.criteria, a.key_facts, a.visual, get_settings())
        for cid, (_, _, desc) in sorted(checks.items(), key=lambda kv: tier.get(kv[1][2]) != "critical"):
            if desc not in tier:
                continue
            crit_id, fact = CHECK_SOURCES.get(cid, ("", None))
            ev = None
            if fact and a.key_facts:
                ev = _verified([e for e in a.key_facts.evidence if e.fact == fact and e.value == "no"])
            c = _criterion(a, crit_id)
            if ev is None and c is not None and BY_ID.get(crit_id) and BY_ID[crit_id].scored_by == "llm":
                ev = _verified(c.evidence)
            text = f"{tier[desc].capitalize()} problem: {desc}"
            if ev is None and c is not None and c.observations:
                text += f". {c.observations[0]}"
            reasons.append(Reason(text=text, problem=True, evidence=ev, criterion=crit_id))
    strong = sorted(
        (c for c in a.criteria if c.id not in excluded(a) and c.score is not None and c.score >= 3),
        key=lambda c: (-c.score, -(BY_ID[c.id].weight if c.id in BY_ID else 0), not c.evidence),  # type: ignore[operator]
    )
    for c in strong:
        if len(reasons) >= n:
            break
        name = BY_ID[c.id].name if c.id in BY_ID else c.id
        ev = _verified(c.evidence)
        obs = f": {c.observations[0]}" if c.observations and ev is None else ""
        reasons.append(Reason(text=f"Strength: {name} ({c.score}/4){obs}", problem=False, evidence=ev, criterion=c.id))
    return reasons[:n]


def next_level_actions(a: Assessment, n: int = 3) -> list[str]:
    """Improvements tied to the counted problems first, then the judge's overall improvements."""
    problem_criteria = [r.criterion for r in key_reasons(a, n=10) if r.problem]
    candidates = [c.improvement for cid in problem_criteria if (c := _criterion(a, cid)) and c.improvement]
    candidates += a.improvements
    candidates += [c.improvement for c in a.criteria if c.improvement and c.id not in excluded(a)]
    out: list[str] = []
    for s in candidates:
        s = " ".join(s.split())
        if s and s not in out:
            out.append(s)
    return out[:n]


def cannot_check(a: Assessment, speakers_identified: bool) -> list[str]:
    items = [
        "Voice energy and tone of voice: only the words are assessed, not how they sound.",
        "The 8x8 branded background: not checked yet.",
    ]
    if not speakers_identified:
        items.append("Who is speaking: speakers are not identified, so tutor and children are told apart only roughly.")
    if a.visual is not None and a.visual.n_frames:
        items.append(
            "Tutor face on camera is indicative only: screen-share frames are not yet reliably told apart from "
            "a camera that is off."
        )
    else:
        items.append("Video: no recording was analysed, so visual presence and frozen video were not checked.")
    return items


SOURCE_NAMES = {"vtt": "Zoom VTT", "srt": "SRT subtitles", "txt": "plain-text transcript"}


def reliability_notes(a: Assessment, inputs: dict) -> list[str]:
    """Plain-language notes on how far the label can be trusted (replaces the numeric confidence, which was
    higher on wrong labels than on right ones)."""
    notes = []
    if a.meta.get("transcript_source") == "whisper":
        notes.append("Transcript: transcribed locally with whisper from the audio; speakers are not identified.")
    else:
        fmt = SOURCE_NAMES.get(inputs.get("transcript_format") or "vtt", "uploaded transcript")
        who = "identified" if inputs.get("speakers_identified", True) else "not identified"
        notes.append(f"Transcript: {fmt} uploaded with the class; speakers {who}.")
    unavailable = [BY_ID[c.id].name if c.id in BY_ID else c.id for c in a.criteria if c.score is None]
    if unavailable:
        notes.append(f"Not enough evidence to score: {', '.join(unavailable)}.")
    dropped, total = a.meta.get("evidence_dropped", 0), a.meta.get("evidence_total", 0)
    if dropped:
        notes.append(
            f"{dropped} of {total} quotes from the judge could not be found in the transcript and were dropped."
        )
    notes += inputs.get("notes", [])
    notes += a.warnings
    return list(dict.fromkeys(notes))
