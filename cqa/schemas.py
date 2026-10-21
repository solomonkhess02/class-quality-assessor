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
    opening_turns: int
    closing_turns: int


class VisualFrameMetrics(BaseModel):
    t_s: float
    brightness: float
    blur: float
    is_black: bool
    is_frozen: bool
    face_count: int


class VisualReport(BaseModel):
    n_frames: int
    pct_face: float
    pct_dark: float
    pct_black: float
    pct_frozen: float
    median_blur: float


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


class JudgeOutput(BaseModel):
    criteria: list[CriterionResult]
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
    summary: str = ""
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    text_stats: TextStats | None = None
    visual: VisualReport | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
