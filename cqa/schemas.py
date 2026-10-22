"""Contracts between pipeline stages (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Role = Literal["tutor", "student", "unknown"]
Label = Literal["Poor", "Average", "Good", "Excellent"]

_TS_RE = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,]\d+)?$")


def format_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def parse_ts(ts: str) -> int:
    m = _TS_RE.match(ts.strip())
    if not m:
        raise ValueError(f"bad timestamp {ts!r}")
    h, mi, s = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s)


class Turn(BaseModel):
    start_s: float
    end_s: float
    speaker: str
    role: Role = "unknown"
    text: str


class Transcript(BaseModel):
    source: Literal["vtt", "whisper"]
    language: str = "unknown"
    duration_s: float = 0.0
    turns: list[Turn] = Field(default_factory=list)
    windows: list[tuple[float, float]] = Field(default_factory=list)  # transcribed [start, end] windows; [] = all


class TextStats(BaseModel):
    duration_min: float
    tutor_talk_share: float
    student_turns_per_min: float
    distinct_students_spoke: int
    tutor_questions_per_min: float
    sparse_punctuation: bool
    longest_tutor_monologue_s: float
    connectivity_mentions: int
    connectivity_incidents_s: list[float] = Field(default_factory=list)  # start times (D14); not sent to the judge
    opening_turns: int
    closing_turns: int


class VisualFrameMetrics(BaseModel):
    t_s: float
    brightness: float
    blur: float
    is_black: bool
    is_frozen: bool
    face_count: int
    flat_ratio: float = 0.0  # share of 8x8 blocks with ~no texture (screen content)
    face_brightness: float | None = None  # mean grey of the largest face box (A3b lighting)
    is_screen_share: bool = False


class VisualReport(BaseModel):
    n_frames: int
    pct_face: float
    pct_dark: float
    pct_black: float
    pct_frozen: float
    median_blur: float
    pct_screen_share: float = 0.0
    pct_face_camera: float = 0.0  # face % over frames that are not screen share or black
    pct_frozen_camera: float = 0.0  # frozen % over the same camera frames (D14); not sent to the judge
    median_face_brightness: float | None = None  # over camera frames with a face; None if < 5 (A3b); code only


class Evidence(BaseModel):
    timestamp: str
    speaker: str
    quote: str
    verified: bool = False

    @field_validator("timestamp")
    @classmethod
    def _normalise_ts(cls, v: str) -> str:
        return format_ts(parse_ts(v))


class CriterionResult(BaseModel):
    id: str
    score: int | None = Field(default=None, ge=1, le=4)
    insufficient_evidence: bool = False
    observations: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    improvement: str = ""

    @model_validator(mode="after")
    def _null_score_is_insufficient(self) -> CriterionResult:
        if self.score is None:
            self.insufficient_evidence = True
        return self


class FactEvidence(Evidence):
    fact: str
    value: Literal["yes", "no"] = "yes"  # what the quote shows: the fact holds, or the opposite (A3b)


FactState = Literal["yes", "no", "unknown"]
YES_NO_FACTS = ("homework_assigned", "recap_present", "rapport_before_content", "each_child_asked_individually")


class KeyFacts(BaseModel):
    """Facts that discriminated between classes in E9. Three states (A3b): "yes" and "no" each need a verified
    quote showing them; otherwise "unknown", which is never treated as a problem."""

    homework_assigned: FactState = "unknown"
    homework_what: str = ""
    new_joiners: Literal["invited", "not_invited", "none", "unknown"] = "none"
    children_greeted_by_name: int = 0
    recap_present: FactState = "unknown"
    # ablation A2 (FACTS_DRIVE): reported only when the switch is on
    rapport_before_content: FactState = "unknown"
    each_child_asked_individually: FactState = "unknown"
    # A3c: the judge quotes the first line of lesson content; code measures the warm-up before it
    content_start: str = ""  # "hh:mm:ss" of the verified first content line, "" if unknown
    warm_up_min: float | None = None  # first tutor greeting -> content start
    warm_up_students: int | None = None  # distinct students with a turn of >= 4 words in that window
    # E24: how rapport_before_content was decided: measured warm-up (speakers known), the judge's verified fact
    # (speakers unknown, e.g. whisper), or neither (unknown)
    rapport_method: Literal["warm_up", "judge", "none"] = "none"
    evidence: list[FactEvidence] = Field(default_factory=list)

    @field_validator(*YES_NO_FACTS, mode="before")
    @classmethod
    def _legacy_bool(cls, v: Any) -> Any:
        return {True: "yes", False: "no"}.get(v, v) if isinstance(v, bool) else v


class JudgeOutput(BaseModel):
    criteria: list[CriterionResult]
    key_facts: KeyFacts | None = None
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    transcript_usable: bool = True
    transcript_note: str = ""


class Assessment(BaseModel):
    label: Label
    weighted_score: float | None
    confidence: float = Field(ge=0.0, le=1.0)
    criteria: list[CriterionResult] = Field(default_factory=list)
    key_facts: KeyFacts | None = None
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    text_stats: TextStats | None = None
    visual: VisualReport | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
