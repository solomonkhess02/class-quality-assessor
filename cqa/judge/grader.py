"""Feedback-recall grader for evaluation (docs/EVALUATION.md).

Sees only the app's report and the organiser's feedback points, never the transcript. Each "yes"/"partial" must
cite a report line verbatim; a cited line that is not in the report downgrades the verdict to "no".
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from cqa.judge.llm import call_json
from cqa.schemas import Assessment

SUPPORT_MATCH_MIN = 90


class PointGrade(BaseModel):
    point: int  # 1-based index into the organiser's points
    verdict: Literal["yes", "partial", "no"]
    support: str = ""
    verified: bool = False


class GraderOutput(BaseModel):
    grades: list[PointGrade] = Field(default_factory=list)


SYSTEM = """You check whether an automated review of an online children's chess class covers each point that an
expert reviewer made about the same class. You see only the automated report (one finding per line) and the
expert's points.

For each expert point decide:
- "yes": the report clearly makes the same point (same behaviour, same direction: praise vs criticism).
- "partial": the report touches the same behaviour but is vaguer, covers only part of the point, or only
  mentions it as a measurement without the judgement.
- "no": the report does not make the point, or says the opposite.
Judge meaning, not wording. "support" is the single report line that best supports a yes or partial, copied
exactly; leave it empty for no.

Return a json object: {"grades": [{"point": 1, "verdict": "yes", "support": "<report line>"}, ...]} with one
entry per expert point, in order."""


def report_lines(a: Assessment) -> list[str]:
    """The app's output as one finding per line (no transcript quotes)."""
    lines = [f"Summary: {a.summary}"] if a.summary else []
    for c in a.criteria:
        lines += [f"{c.id} observation: {o}" for o in c.observations]
        if c.improvement:
            lines.append(f"{c.id} improvement: {c.improvement}")
    lines += [f"Strength: {x}" for x in a.strengths]
    lines += [f"Improvement: {x}" for x in a.improvements]
    if kf := a.key_facts:
        homework = (
            f"yes ({kf.homework_what})" if kf.homework_assigned and kf.homework_what else str(kf.homework_assigned)
        )
        lines.append(
            f"Key facts: homework assigned: {homework}; new joiners: {kf.new_joiners}; children greeted by name "
            f"before teaching: {kf.children_greeted_by_name}; closing recap: {kf.recap_present}"
        )
    lines += [f"Warning: {w}" for w in a.warnings]
    return lines


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def grade_feedback(a: Assessment, points: list[str]) -> tuple[list[PointGrade], dict]:
    """One call per class at temperature 0. Returns one verified grade per point, in order, and call meta."""
    lines = report_lines(a)
    user = (
        "Automated report:\n"
        + "\n".join(lines)
        + "\n\nExpert points:\n"
        + "\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1))
        + "\n\nGrade every expert point and return the json object."
    )
    out, meta = call_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], GraderOutput,
                          temperature=0.0)  # fmt: skip
    by_point = {g.point: g for g in out.grades}
    normed = [_norm(x) for x in lines]
    grades, downgraded = [], 0
    for i in range(1, len(points) + 1):
        g = by_point.get(i, PointGrade(point=i, verdict="no"))
        if g.verdict != "no":
            s = _norm(g.support)
            ok = bool(s) and any(fuzz.partial_ratio(s, x) >= SUPPORT_MATCH_MIN for x in normed)
            if not ok:
                downgraded += 1
                g = g.model_copy(update={"verdict": "no"})
            g = g.model_copy(update={"verified": ok})
        grades.append(g)
    meta.update({"missing_grades": len(points) - len(by_point.keys() & set(range(1, len(points) + 1))),
                 "downgraded": downgraded, "report_lines": len(lines)})  # fmt: skip
    return grades, meta


def spotcheck_markdown(key: str, points: list[str], sources: list[str], grades: list[PointGrade]) -> str:
    rows = [
        f"## {key}",
        "",
        "| # | source | expert point | verdict | supporting report line |",
        "|---|---|---|---|---|",
    ]
    for i, (p, src, g) in enumerate(zip(points, sources, grades, strict=True), start=1):
        verdict = g.verdict + (" (downgraded: cited line not in report)" if g.verdict == "no" and g.support else "")
        cells = [str(i), src, p, verdict, g.support]
        rows.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    return "\n".join(rows) + "\n"


def dumps(grades: list[PointGrade]) -> str:
    return json.dumps([g.model_dump() for g in grades], indent=1)
