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


class KeyFacts(BaseModel):
    """Yes/no facts that discriminated between classes in E9. Shown in the report; not scored."""

    homework_assigned: bool = False
    homework_what: str = ""
    new_joiners: Literal["invited", "not_invited", "none"] = "none"
    children_greeted_by_name: int = 0
    recap_present: bool = False
    # ablation A2 (FACTS_DRIVE): reported only when the switch is on
    rapport_before_content: bool = False
    each_child_asked_individually: bool = False
    evidence: list[FactEvidence] = Field(default_factory=list)


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
