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
from cqa.judge.rubric import LLM_CRITERIA
from cqa.schemas import (
    Evidence,
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
            score = max(score, fuzz.partial_ratio(q, f"{texts[i]} {texts[i + 1]}"))
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


def verify_key_facts(kf: KeyFacts, transcript: Transcript, window_s: float) -> tuple[KeyFacts, int, list[str]]:
    """Verify key-fact quotes; true / non-zero / "invited" facts without a verified quote are reset.
    Returns (facts, dropped quotes, names of facts that were reset)."""
    kept, dropped = verify_list(kf.evidence, transcript, window_s)
    supported = {e.fact for e in kept}
    updates: dict = {"evidence": kept}
    if kf.homework_assigned and "homework_assigned" not in supported:
        updates.update(homework_assigned=False, homework_what="")
    if kf.recap_present and "recap_present" not in supported:
        updates["recap_present"] = False
    if kf.children_greeted_by_name and "children_greeted_by_name" not in supported:
        updates["children_greeted_by_name"] = 0
    if kf.new_joiners == "invited" and "new_joiners" not in supported:
        updates["new_joiners"] = "none"  # unknown whether a new child joined
    reset = [k for k in updates if k not in ("evidence", "homework_what")]
    return kf.model_copy(update=updates), dropped, reset


def generic_observations(observations: list[str]) -> list[str]:
    """Observations that name no speaker, timestamp, quote or number."""
    return [o for o in observations if not _SPECIFIC.search(o)]


def _call_with_generic_retry(
    messages: list[dict], schema: type[M], observations_of
) -> tuple[M, dict, list[str]]:
    """Call the judge; if too many observations are generic, re-ask once with them listed."""
    out, meta = call_json(messages, schema)
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
        out2, meta2 = call_json(retry, schema)
        obs2 = observations_of(out2)
        generic = generic_observations(obs2)
        out, meta = out2, {**meta2, "generic_retry": True}
        obs = obs2
    meta["observations"] = len(obs)
    meta["generic_observations"] = len(generic)
    return out, meta, generic


def _judge_direct(
    transcript: Transcript, stats: TextStats, visual: VisualReport | None
) -> tuple[JudgeOutput, list[str], dict]:
    s = get_settings()
    messages = prompts.build_messages(transcript, stats, visual, s.opening_window_min, s.closing_window_min)
    raw, meta, _ = _call_with_generic_retry(
        messages, JudgeOutput, lambda o: [x for c in o.criteria for x in c.observations]
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
        kf, kf_dropped, reset = verify_key_facts(verified.key_facts, transcript, s.evidence_window_s)
        verified = verified.model_copy(update={"key_facts": kf})
        total += len(raw.key_facts.evidence) if raw.key_facts else 0
        dropped += kf_dropped
        if reset:
            warnings.append(f"Key facts without a verified quote were reset: {', '.join(reset)}")
    meta.update({"design": "direct", "evidence_total": total, "evidence_dropped": dropped})
    return verified, warnings, meta


def _judge_facts(
    transcript: Transcript, stats: TextStats, visual: VisualReport | None
) -> tuple[JudgeOutput, list[str], dict]:
    s = get_settings()
    user = prompts.user_prompt(transcript, stats, visual, s.opening_window_min, s.closing_window_min)
    user = user.replace("Assess this class", "Extract the facts for this class")
    messages = [{"role": "system", "content": F.facts_system_prompt()}, {"role": "user", "content": user}]
    out, meta, _ = _call_with_generic_retry(
        messages, F.FactsOutput, lambda o: [x for c in o.criteria for x in c.observations]
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
    transcript: Transcript, stats: TextStats, visual: VisualReport | None, design: str | None = None
) -> tuple[JudgeOutput, list[str], dict]:
    """Call the judge and verify evidence. Returns (verified output, warnings, call meta)."""
    design = design or get_settings().judge_design
    judged, warnings, meta = (_judge_facts if design == "facts" else _judge_direct)(transcript, stats, visual)
    if meta["evidence_dropped"]:
        warnings.append(
            f"Dropped {meta['evidence_dropped']} of {meta['evidence_total']} evidence quotes "
            "that could not be found in the transcript"
        )
    if meta["generic_observations"]:
        warnings.append(f"{meta['generic_observations']} of {meta['observations']} observations look generic")
    return judged, warnings, meta
