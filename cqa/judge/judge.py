"""Run the LLM judge and verify every quote against the transcript.

Two designs (EXPERIMENTS, M6b): "direct" (LLM scores 1-4 per criterion) and "facts" (LLM extracts observable
facts with evidence, code maps facts to scores, cqa/judge/facts.py).
"""

from __future__ import annotations

import re
from typing import TypeVar

from pydantic import BaseModel
from rapidfuzz import fuzz

from cqa.config import get_settings
from cqa.judge import facts as F
from cqa.judge import prompts
from cqa.judge.llm import call_json
from cqa.judge.prompts import speakers_known
from cqa.judge.rubric import LLM_CRITERIA
from cqa.schemas import (
    YES_NO_FACTS,
    CriterionResult,
    Evidence,
    FactEvidence,
    JudgeOutput,
    KeyFacts,
    TextStats,
    Transcript,
    Turn,
    VisualReport,
    format_ts,
    parse_ts,
)

MATCH_MIN = 85
SHORT_QUOTE_CHARS = 12  # shorter quotes must appear verbatim (after normalisation)
GENERIC_MAX_SHARE = 0.30  # re-ask once when more observations than this are generic
_SPECIFIC = re.compile(r"\b(?:TUTOR|S\d+)\b|\d{1,2}:\d{2}|[\"“”']\w|\d")
E = TypeVar("E", bound=Evidence)
M = TypeVar("M", bound=BaseModel)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def _match(quote: str, turns: list[Turn]) -> Turn | None:
    """Best-matching turn for a quote, or None. Also tries pairs of adjacent turns."""
    q = _norm(quote)
    if not q:
        return None
    texts = [_norm(t.text) for t in turns]
    if len(q) < SHORT_QUOTE_CHARS:
        return next((t for t, txt in zip(turns, texts, strict=True) if q in txt), None)
    best, best_score = None, 0.0
    for i, t in enumerate(turns):
        score = fuzz.partial_ratio(q, texts[i])
        if i + 1 < len(turns) and len(q) > len(texts[i]):
            # a quote spanning two turns still matches, but an exact single-turn match wins the tie, so the
            # timestamp is the quoted line itself, not the short turn before it (A3c content_start)
            score = max(score, fuzz.partial_ratio(q, f"{texts[i]} {texts[i + 1]}") - 1)
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= MATCH_MIN else None


def verify_list(evidence: list[E], transcript: Transcript, window_s: float) -> tuple[list[E], int]:
    """Keep quotes found within +-window_s of their timestamp, with the real speaker and turn start time."""
    kept: list[E] = []
    for e in evidence:
        try:
            t = parse_ts(e.timestamp)
        except ValueError:
            continue
        near = [x for x in transcript.turns if x.end_s >= t - window_s and x.start_s <= t + window_s]
        turn = _match(e.quote, near)
        if turn is not None:
            kept.append(
                e.model_copy(update={"verified": True, "speaker": turn.speaker, "timestamp": format_ts(turn.start_s)})
            )
    return kept, len(evidence) - len(kept)


def verify_evidence(judged: JudgeOutput, transcript: Transcript, window_s: float) -> tuple[JudgeOutput, int, int]:
    """Verify every criterion's evidence. Returns (output, dropped, total)."""
    dropped = total = 0
    criteria = []
    for c in judged.criteria:
        kept, d = verify_list(c.evidence, transcript, window_s)
        dropped += d
        total += len(c.evidence)
        criteria.append(c.model_copy(update={"evidence": kept}))
    return judged.model_copy(update={"criteria": criteria}), dropped, total


GREETING_RE = re.compile(r"\b(?:hi|hello|hey|good (?:morning|afternoon|evening)|how are you)\b", re.IGNORECASE)
HOMEWORK_RE = re.compile(r"\b(?:home ?work|assignment|practi[cs]e|at home)\b", re.IGNORECASE)


# A3c, fixed before results: rapport = a real warm-up between the first greeting and the first content line
WARM_UP_MIN_WORDS = 4
RAPPORT_NO_BELOW = (2, 2.0)  # fewer than 2 students talking or under 2 minutes -> "no"
RAPPORT_YES_FROM = (3, 3.0)  # at least 3 students talking and at least 3 minutes -> "yes"


def warm_up(transcript: Transcript, content_start_s: float) -> tuple[float, int] | None:
    """(minutes from the first tutor greeting to the start of content, distinct students with a turn of
    >= WARM_UP_MIN_WORDS words in that window). No tutor greeting before content -> None: a warm-up that cannot be
    located cannot be measured (A3d), e.g. when another account taught the start."""
    greetings = [t.start_s for t in transcript.turns
                 if t.role == "tutor" and t.start_s <= content_start_s and GREETING_RE.search(t.text)]  # fmt: skip
    if not greetings:
        return None
    start = min(greetings)
    talkers = {
        t.speaker
        for t in transcript.turns
        if t.role == "student" and start <= t.start_s < content_start_s and len(t.text.split()) >= WARM_UP_MIN_WORDS
    }
    return round((content_start_s - start) / 60, 1), len(talkers)


def rapport_from_warm_up(minutes: float, students: int) -> str:
    if students < RAPPORT_NO_BELOW[0] or minutes < RAPPORT_NO_BELOW[1]:
        return "no"
    if students >= RAPPORT_YES_FROM[0] and minutes >= RAPPORT_YES_FROM[1]:
        return "yes"
    return "unknown"


def judged_rapport_fact(
    state: str, evidence: list[FactEvidence], content_start_s: float | None, transcript: Transcript
) -> str:
    """E24, guards fixed before results (homework-style: unknown unless the transcript can support the claim).
    The judge's rapport fact counts only with a verified quote of the same value (already checked) that lies in
    the opening, no later than the verified content start. A "no" also needs a verified content start inside the
    transcribed opening window: "straight into content" must be visible."""
    if state not in ("yes", "no"):
        return "unknown"
    opening_end = transcript.windows[0][1] if transcript.windows else transcript.duration_s
    limit = min(content_start_s, opening_end) if content_start_s is not None else opening_end
    quotes = [e for e in evidence if e.fact == "rapport_before_content" and e.value == state]
    if not any(parse_ts(e.timestamp) <= limit + 1 for e in quotes):
        return "unknown"
    if state == "no" and (content_start_s is None or content_start_s > opening_end):
        return "unknown"
    return state


def verify_key_facts(
    kf: KeyFacts, transcript: Transcript, window_s: float, closing_min: float = 10.0
) -> tuple[KeyFacts, int, list[str], dict[str, int]]:
    """Three-state facts (A3b): "yes" / "invited" need a verified quote with value yes, "no" / "not_invited" a
    verified quote with value no; otherwise the fact becomes "unknown". The homework guard moves a "no" to "unknown"
    when the closing window mentions homework. rapport_before_content is measured by code from the warm-up before
    the verified first content line (A3c); without a verified content_start it is "unknown".
    Returns (facts, dropped quotes, facts reset to unknown, guard counts)."""
    kept, dropped = verify_list(kf.evidence, transcript, window_s)
    shown = {(e.fact, e.value) for e in kept}
    updates: dict = {"evidence": kept}
    for fact in YES_NO_FACTS:
        state = getattr(kf, fact)
        if state in ("yes", "no") and (fact, state) not in shown:
            updates[fact] = "unknown"
    if kf.children_greeted_by_name and ("children_greeted_by_name", "yes") not in shown:
        updates["children_greeted_by_name"] = 0
    joiner_value = {"invited": "yes", "not_invited": "no"}.get(kf.new_joiners)
    if joiner_value and ("new_joiners", joiner_value) not in shown:
        updates["new_joiners"] = "unknown"

    starts = [parse_ts(e.timestamp) for e in kept if e.fact == "content_start"]
    judged_rapport = updates.get("rapport_before_content", kf.rapport_before_content)  # after the quote check
    measured = warm_up(transcript, min(starts)) if starts else None
    if not speakers_known(transcript):  # E24: no speakers, no warm-up to measure: the judge's guarded fact
        rapport = judged_rapport_fact(judged_rapport, kept, min(starts) if starts else None, transcript)
        updates.update(content_start=format_ts(min(starts)) if starts else "", warm_up_min=None,
                       warm_up_students=None, rapport_before_content=rapport,
                       rapport_method="judge" if rapport != "unknown" else "none")  # fmt: skip
    elif measured is not None:
        minutes, students = measured
        updates.update(content_start=format_ts(min(starts)), warm_up_min=minutes, warm_up_students=students,
                       rapport_before_content=rapport_from_warm_up(minutes, students),
                       rapport_method="warm_up")  # fmt: skip
    elif starts:  # content start verified, but no tutor greeting before it
        updates.update(content_start=format_ts(min(starts)), warm_up_min=None, warm_up_students=None,
                       rapport_before_content="unknown", rapport_method="none")  # fmt: skip
    else:
        updates.update(content_start="", warm_up_min=None, warm_up_students=None, rapport_before_content="unknown",
                       rapport_method="none")  # fmt: skip

    guards = {"homework_no_to_unknown": 0}
    if updates.get("homework_assigned", kf.homework_assigned) == "no":
        closing_start = transcript.duration_s - closing_min * 60
        if any(HOMEWORK_RE.search(t.text) for t in transcript.turns if t.start_s >= closing_start):
            updates["homework_assigned"] = "unknown"
            guards["homework_no_to_unknown"] += 1
    if updates.get("homework_assigned", kf.homework_assigned) != "yes":
        updates["homework_what"] = ""
    measured = (
        "evidence",
        "homework_what",
        "content_start",
        "warm_up_min",
        "warm_up_students",
        "rapport_before_content",
        "rapport_method",
    )
    reset = [k for k in updates if k not in measured]
    return kf.model_copy(update=updates), dropped, reset, guards


def fact_bounds(kf: KeyFacts) -> dict[str, tuple[int | None, int | None, list[str]]]:
    """Ablation A2: (floor, cap, reasons) per criterion from the verified yes/no facts, so a fact and its criterion
    cannot contradict each other. Rules written before results (docs/EVALUATION.md, A2)."""
    bounds: dict[str, tuple[int | None, int | None, list[str]]] = {}
    # unknown facts never bound a score (A3b); "no" facts carry a verified quote of the opposite
    no_intro = kf.new_joiners == "not_invited"
    if kf.rapport_before_content == "no" or no_intro:
        why = (["no personal chat before teaching"] if kf.rapport_before_content == "no" else []) + (
            ["a new joiner was not invited to introduce themselves"] if no_intro else []
        )
        bounds["opening_rapport"] = (None, 2, why)
    elif kf.rapport_before_content == "yes":
        bounds["opening_rapport"] = (3, None, ["personal chat before teaching (verified quote)"])
    if kf.each_child_asked_individually == "yes":
        bounds["individual_engagement"] = (3, None, ["every child asked individually (verified quote)"])
    elif kf.each_child_asked_individually == "no":
        bounds["individual_engagement"] = (None, 2, ["not every child was asked individually (verified quote)"])
    floor = 3 if kf.homework_assigned == "yes" else None
    cap = None
    why = ["homework assigned (verified quote)"] if floor else []
    if kf.homework_assigned == "no":
        cap = 2 if kf.recap_present == "no" else 3
        why.append("no homework assigned")
    if kf.recap_present == "no":
        cap = min(cap or 3, 3)
        why.append("no closing recap")
    if floor or cap:
        bounds["closure_homework"] = (floor, cap, why)
    return bounds


def apply_fact_bounds(criteria: list[CriterionResult], kf: KeyFacts) -> list[CriterionResult]:
    """Clamp each fact-bound criterion into [floor, cap] (cap wins); an unscored one takes the floor, else the cap."""
    bounds = fact_bounds(kf)
    out = []
    for c in criteria:
        if c.id not in bounds:
            out.append(c)
            continue
        floor, cap, why = bounds[c.id]
        old = None if c.insufficient_evidence else c.score
        new = old if old is not None else (floor or cap)
        if floor is not None and new is not None and new < floor:
            new = floor
        if cap is not None and new is not None and new > cap:
            new = cap
        if new != old:
            verb = "set" if old is None else ("raised" if new > old else "capped")
            note = f"Score {verb} to {new} by verified facts: {'; '.join(why)}."
            c = c.model_copy(
                update={"score": new, "insufficient_evidence": False, "observations": [*c.observations, note]}
            )
        out.append(c)
    return out


def generic_observations(observations: list[str]) -> list[str]:
    """Observations that name no speaker, timestamp, quote or number."""
    return [o for o in observations if not _SPECIFIC.search(o)]


def _call_with_generic_retry(
    messages: list[dict], schema: type[M], observations_of, run_index: int = 0
) -> tuple[M, dict, list[str]]:
    """Call the judge; if too many observations are generic, re-ask once with them listed."""
    out, meta = call_json(messages, schema, run_index)
    obs = observations_of(out)
    generic = generic_observations(obs)
    meta["generic_retry"] = False
    if obs and len(generic) / len(obs) > GENERIC_MAX_SHARE:
        listed = "\n".join(f"- {g}" for g in generic)
        retry = [
            *messages,
            {"role": "assistant", "content": out.model_dump_json()},
            {
                "role": "user",
                "content": "These observations are generic. Rewrite each to say who did what and when, citing "
                f"speaker labels and timestamps from the transcript, and return the full json object again:\n{listed}",
            },
        ]
        out2, meta2 = call_json(retry, schema, run_index)
        obs2 = observations_of(out2)
        generic = generic_observations(obs2)
        out, meta = out2, {**meta2, "generic_retry": True}
        obs = obs2
    meta["observations"] = len(obs)
    meta["generic_observations"] = len(generic)
    return out, meta, generic


def _judge_direct(
    transcript: Transcript, stats: TextStats, visual: VisualReport | None, run_index: int = 0
) -> tuple[JudgeOutput, list[str], dict]:
    s = get_settings()
    messages = prompts.build_messages(transcript, stats, visual, s.opening_window_min, s.closing_window_min)
    raw, meta, _ = _call_with_generic_retry(
        messages, JudgeOutput, lambda o: [x for c in o.criteria for x in c.observations], run_index
    )
    warnings: list[str] = []
    expected = {c.id for c in LLM_CRITERIA}
    if missing := expected - {c.id for c in raw.criteria}:
        warnings.append(f"Judge did not return criteria: {', '.join(sorted(missing))}")
    raw = raw.model_copy(update={"criteria": [c for c in raw.criteria if c.id in expected]})

    verified, dropped, total = verify_evidence(raw, transcript, s.evidence_window_s)
    # A score with no verified evidence does not count toward the label.
    criteria = []
    for c in verified.criteria:
        if c.score is not None and not c.insufficient_evidence and not c.evidence:
            warnings.append(f"{c.id}: no verified evidence, excluded from the score")
            c = c.model_copy(update={"insufficient_evidence": True})
        criteria.append(c)
    verified = verified.model_copy(update={"criteria": criteria})
    if verified.key_facts is not None:
        kf, kf_dropped, reset, guards = verify_key_facts(
            verified.key_facts, transcript, s.evidence_window_s, s.closing_window_min
        )
        verified = verified.model_copy(update={"key_facts": kf})
        total += len(raw.key_facts.evidence) if raw.key_facts else 0
        dropped += kf_dropped
        meta["fact_guards"] = guards
        if kf.rapport_method == "judge":
            warnings.append(
                "Speakers are not identified, so the warm-up before teaching was not measured: rapport was judged "
                f"from the transcript (verified quote): {kf.rapport_before_content}."
            )
        if reset:
            warnings.append(f"Key facts without a verified quote were set to unknown: {', '.join(reset)}")
        if s.facts_drive and verified.transcript_usable:
            verified = verified.model_copy(update={"criteria": apply_fact_bounds(verified.criteria, kf)})
    meta.update({"design": "direct", "evidence_total": total, "evidence_dropped": dropped})
    return verified, warnings, meta


def _judge_facts(
    transcript: Transcript, stats: TextStats, visual: VisualReport | None, run_index: int = 0
) -> tuple[JudgeOutput, list[str], dict]:
    s = get_settings()
    user = prompts.user_prompt(transcript, stats, visual, s.opening_window_min, s.closing_window_min)
    user = user.replace("Assess this class", "Extract the facts for this class")
    messages = [{"role": "system", "content": F.facts_system_prompt()}, {"role": "user", "content": user}]
    out, meta, _ = _call_with_generic_retry(
        messages, F.FactsOutput, lambda o: [x for c in o.criteria for x in c.observations], run_index
    )
    verified, dropped = verify_list(out.evidence, transcript, s.evidence_window_s)
    facts = F.apply_verified_evidence(out, verified)
    class_start = transcript.turns[0].start_s if transcript.turns else 0.0
    judged = F.to_judge_output(out, facts, verified, stats, class_start)
    warnings: list[str] = []
    unsupported = [k for k in F.NEEDS_EVIDENCE if getattr(out.facts, k) and not getattr(facts, k)]
    if unsupported:
        warnings.append(f"Facts reported without verified evidence were set to false: {', '.join(unsupported)}")
    meta.update(
        {
            "design": "facts",
            "evidence_total": len(out.evidence),
            "evidence_dropped": dropped,
            "facts": facts.model_dump(),
        }
    )
    return judged, warnings, meta


def judge(
    transcript: Transcript,
    stats: TextStats,
    visual: VisualReport | None,
    design: str | None = None,
    run_index: int = 0,
) -> tuple[JudgeOutput, list[str], dict]:
    """Call the judge and verify evidence. Returns (verified output, warnings, call meta).
    run_index > 0 makes a separate (uncached) call for repeated evaluation runs."""
    design = design or get_settings().judge_design
    judged, warnings, meta = (_judge_facts if design == "facts" else _judge_direct)(
        transcript, stats, visual, run_index
    )
    if meta["evidence_dropped"]:
        warnings.append(
            f"Dropped {meta['evidence_dropped']} of {meta['evidence_total']} evidence quotes "
            "that could not be found in the transcript"
        )
    if meta["generic_observations"]:
        warnings.append(f"{meta['generic_observations']} of {meta['observations']} observations look generic")
    return judged, warnings, meta
