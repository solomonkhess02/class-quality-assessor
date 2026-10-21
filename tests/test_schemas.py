import pytest
from pydantic import ValidationError

from cqa.schemas import (
    Assessment,
    CriterionResult,
    Evidence,
    JudgeOutput,
    Turn,
    format_ts,
    parse_ts,
)


def test_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        CriterionResult(id="communication", score=5)
    with pytest.raises(ValidationError):
        CriterionResult(id="communication", score=0)


def test_null_score_implies_insufficient_evidence():
    c = CriterionResult(id="communication", score=None)
    assert c.insufficient_evidence is True


def test_role_must_be_known():
    with pytest.raises(ValidationError):
        Turn(start_s=0, end_s=1, speaker="X", role="teacher", text="hi")


def test_evidence_timestamp_format():
    assert Evidence(timestamp="00:01:05", speaker="TUTOR", quote="hello").timestamp == "00:01:05"
    assert Evidence(timestamp="1:05", speaker="TUTOR", quote="hello").timestamp == "00:01:05"
    with pytest.raises(ValidationError):
        Evidence(timestamp="soon", speaker="TUTOR", quote="hello")


def test_ts_roundtrip():
    assert format_ts(3725.9) == "01:02:05"
    assert parse_ts("01:02:05") == 3725
    assert parse_ts("02:05") == 125


def test_confidence_bounds():
    with pytest.raises(ValidationError):
        Assessment(label="Good", weighted_score=3.0, confidence=1.5)


def test_judge_output_defaults_and_json_roundtrip():
    j = JudgeOutput.model_validate(
        {
            "criteria": [
                {
                    "id": "opening_rapport",
                    "score": 3,
                    "observations": ["Greets each child."],
                    "evidence": [{"timestamp": "00:00:30", "speaker": "TUTOR", "quote": "hi S1"}],
                    "improvement": "Set the plan.",
                }
            ],
            "summary": "ok",
        }
    )
    assert j.transcript_usable is True
    assert j.criteria[0].evidence[0].verified is False
    assert JudgeOutput.model_validate_json(j.model_dump_json()) == j
