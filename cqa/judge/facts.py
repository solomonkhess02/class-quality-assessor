"""Extract-then-score judge design: the LLM reports observable facts with evidence,
code maps facts to 1-4 scores with fixed rules derived from the rubric anchors (docs/ARCHITECTURE.md).

Rule thresholds are initial guesses from the anchor wording, not fitted to the labelled classes.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field

from cqa.judge.rubric import LLM_CRITERIA
from cqa.schemas import CriterionResult, Evidence, JudgeOutput, TextStats, parse_ts


class FactEvidence(Evidence):
    fact: str


class CriterionNotes(BaseModel):
    id: str
    observations: list[str] = Field(default_factory=list)
    improvement: str = ""


class ClassFacts(BaseModel):
    content_start: str | None = None  # hh:mm:ss when chess teaching starts
    children_greeted_by_name: int = 0
    plan_stated: bool = False
    new_joiners: Literal["invited", "not_invited", "none"] = "none"
    children_asked_individually: int = 0  # distinct children asked a question directly
    individual_questions: int = 0  # questions addressed to a named child
    praise_count: int = 0
    specific_praise: bool = False
    negative_tone_count: int = 0
    recap_present: bool = False
    homework_assigned: bool = False
    homework_description: str = ""
    proper_goodbye: bool = False
    technical_disruptions: int = 0
    minutes_lost_to_technical: float = 0.0
    transcript_ends_early: bool = False  # closing not visible in the transcript


class FactsOutput(BaseModel):
    transcript_usable: bool = True
    transcript_note: str = ""
    facts: ClassFacts
    evidence: list[FactEvidence] = Field(default_factory=list)
    criteria: list[CriterionNotes] = Field(default_factory=list)
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


# Boolean facts that only count when at least one quote supporting them is verified.
NEEDS_EVIDENCE = ["plan_stated", "recap_present", "homework_assigned", "proper_goodbye", "specific_praise"]

# Which facts' evidence supports which criterion.
FACT_CRITERION = {
    "content_start": "opening_rapport",
    "children_greeted_by_name": "opening_rapport",
    "plan_stated": "opening_rapport",
    "new_joiners": "opening_rapport",
    "children_asked_individually": "individual_engagement",
    "individual_questions": "individual_engagement",
    "praise_count": "communication",
    "specific_praise": "communication",
    "negative_tone_count": "communication",
    "recap_present": "closure_homework",
    "homework_assigned": "closure_homework",
    "proper_goodbye": "closure_homework",
    "technical_disruptions": "technical_quality",
    "minutes_lost_to_technical": "technical_quality",
}

FACTS_EXAMPLE = {
    "transcript_usable": True,
    "transcript_note": "",
    "facts": {
        "content_start": "00:06:30", "children_greeted_by_name": 4, "plan_stated": True,
        "new_joiners": "invited", "children_asked_individually": 5, "individual_questions": 23,
        "praise_count": 14, "specific_praise": True, "negative_tone_count": 0, "recap_present": False,
        "homework_assigned": True, "homework_description": "solve 5 fork puzzles on the portal",
        "proper_goodbye": True, "technical_disruptions": 1, "minutes_lost_to_technical": 0.5,
        "transcript_ends_early": False,
    },
    "evidence": [
        {"fact": "children_greeted_by_name", "timestamp": "00:00:42", "speaker": "TUTOR",
         "quote": "S1, how was your exam yesterday?"},
        {"fact": "homework_assigned", "timestamp": "00:52:10", "speaker": "TUTOR",
         "quote": "for homework solve the five fork puzzles"},
    ],
    "criteria": [
        {"id": "opening_rapport", "observations": ["Chats with S1 and S2 about school for about 6 minutes."],
         "improvement": "Ask the new joiner S5 to introduce themselves before the first puzzle."},
    ],
    "summary": "Two or three sentences on the class overall.",
    "strengths": ["Specific strength with a timestamp."],
    "improvements": ["Specific, actionable suggestion."],
}  # fmt: skip

FACT_DEFINITIONS = """\
- content_start: timestamp of the first line where chess teaching starts (null if unclear).
- children_greeted_by_name: number of distinct children the tutor greets or chats with by name before content_start.
- plan_stated: the tutor tells the children what the class will cover before teaching.
- new_joiners: "invited" if a new child is asked to introduce themselves, "not_invited" if a new child joins
  and is not, "none" if no new child is apparent.
- children_asked_individually: distinct children the tutor asks a question directly (by name or clearly to one child).
- individual_questions: total number of questions addressed to a specific child.
- praise_count: number of times the tutor praises or encourages a child ("well done", "great thinking").
- specific_praise: at least one praise names what was good ("good, you saw the fork on e5").
- negative_tone_count: impatient, dismissive or discouraging remarks by the tutor.
- recap_present: near the end the tutor summarises what was learned.
- homework_assigned: the tutor assigns specific work for after class (put what in homework_description).
- proper_goodbye: the tutor says a clear goodbye to the children at the end.
- technical_disruptions: separate episodes of audio/video/connection problems that interrupt the class.
- minutes_lost_to_technical: approximate total minutes lost to them.
- transcript_ends_early: the transcript stops before the class visibly ends, so closing facts cannot be judged."""


def facts_system_prompt() -> str:
    ids = ", ".join(f'"{c.id}"' for c in LLM_CRITERIA)
    return f"""You are an experienced academic quality reviewer for 8x8, an online chess academy for children.
You review a machine transcript of a live group class (one tutor, several children). Names are pseudonymised:
TUTOR is the tutor, S1, S2, ... are children, UNKNOWN means the speaker is not identified. Expect speech
recognition errors.

Your job is to EXTRACT OBSERVABLE FACTS, not to grade. Report these facts:
{FACT_DEFINITIONS}

Rules:
1. Count carefully from the transcript; do not estimate generously. If something does not happen, report 0/false.
2. For every non-zero or true fact, give 1-3 evidence quotes copied verbatim from the transcript (at most 25 words
   each) with the line's timestamp and speaker, and the fact name. Quotes are verified automatically; unverifiable
   quotes are discarded and a true fact without a verified quote counts as false.
3. For each criterion ({ids}) give 1-3 observations specific to this class (who, what, when) and one concrete,
   actionable improvement. No generic praise.
4. Set "transcript_usable": false (with "transcript_note") if the transcript is mostly unintelligible.

Return a single json object with exactly the keys of this example (content is illustrative only):
{json.dumps(FACTS_EXAMPLE, indent=1)}"""


def warmup_minutes(content_start: str | None, class_start_s: float) -> float | None:
    """Minutes from the first spoken line to the start of chess content."""
    if not content_start:
        return None
    try:
        return max(0.0, (parse_ts(content_start) - class_start_s) / 60)
    except ValueError:
        return None


def score_from_facts(f: ClassFacts, stats: TextStats, class_start_s: float = 0.0) -> dict[str, int | None]:
    """Map facts to 1-4 per criterion using the anchor wording. None = cannot judge."""
    students = max(stats.distinct_students_spoke, f.children_asked_individually, 1)

    # Opening: warm-up chat, children greeted by name, plan, new joiners handled.
    warmup = warmup_minutes(f.content_start, class_start_s)
    opening_points = sum(
        [
            warmup is not None and warmup >= 3,
            f.children_greeted_by_name >= max(2, students // 2),
            f.plan_stated,
            f.new_joiners != "not_invited",
        ]
    )
    opening = max(1, opening_points)

    # Engagement: share of children asked directly, and whether they are asked repeatedly.
    share = f.children_asked_individually / students
    if share >= 0.9 and f.individual_questions >= 2 * students:
        engagement = 4
    elif share >= 0.6:
        engagement = 3
    elif share >= 0.3 or f.individual_questions > 0:
        engagement = 2
    else:
        engagement = 1

    # Warmth: praise rate per 10 minutes, specific praise, negative remarks.
    per10 = f.praise_count / max(stats.duration_min / 10, 1)
    if per10 >= 3 and f.specific_praise:
        warmth = 4
    elif per10 >= 1.5:
        warmth = 3
    elif per10 >= 0.5:
        warmth = 2
    else:
        warmth = 1
    if f.negative_tone_count >= 2:
        warmth = max(1, warmth - 1)

    # Closure: recap, homework, goodbye -> anchors "none / one / two / all three".
    closure = None if f.transcript_ends_early else 1 + sum([f.recap_present, f.homework_assigned, f.proper_goodbye])

    # Technical: disruption episodes and minutes lost.
    if f.technical_disruptions == 0:
        tech = 4
    elif f.technical_disruptions <= 2 and f.minutes_lost_to_technical <= 2:
        tech = 3
    elif f.technical_disruptions <= 5 and f.minutes_lost_to_technical <= 5:
        tech = 2
    else:
        tech = 1

    return {
        "opening_rapport": opening,
        "individual_engagement": engagement,
        "communication": warmth,
        "closure_homework": closure,
        "technical_quality": tech,
    }


def apply_verified_evidence(out: FactsOutput, verified: list[FactEvidence]) -> ClassFacts:
    """Boolean facts without a verified quote are set to False."""
    supported = {e.fact for e in verified}
    updates = {name: False for name in NEEDS_EVIDENCE if getattr(out.facts, name) and name not in supported}
    return out.facts.model_copy(update=updates)


def to_judge_output(
    out: FactsOutput, facts: ClassFacts, verified: list[FactEvidence], stats: TextStats, class_start_s: float = 0.0
) -> JudgeOutput:
    scores = score_from_facts(facts, stats, class_start_s)
    notes = {c.id: c for c in out.criteria}
    criteria = []
    for c in LLM_CRITERIA:
        ev = [Evidence(**e.model_dump(exclude={"fact"})) for e in verified if FACT_CRITERION.get(e.fact) == c.id]
        n = notes.get(c.id)
        score = scores[c.id]
        criteria.append(
            CriterionResult(
                id=c.id,
                score=score if out.transcript_usable else None,
                insufficient_evidence=score is None or not out.transcript_usable,
                observations=n.observations if n else [],
                evidence=ev,
                improvement=n.improvement if n else "",
            )
        )
    return JudgeOutput(
        criteria=criteria,
        summary=out.summary,
        strengths=out.strengths,
        improvements=out.improvements,
        transcript_usable=out.transcript_usable,
        transcript_note=out.transcript_note,
    )
