import pytest

from cqa.schemas import CriterionResult, JudgeOutput, VisualReport
from cqa.scoring import label_for, score_visual, weighted_score

GOOD_VISUAL = VisualReport(n_frames=100, pct_face=95, pct_dark=0, pct_black=0, pct_frozen=5, median_blur=200)


def crit(cid: str, score: int | None, insufficient: bool = False) -> CriterionResult:
    return CriterionResult(id=cid, score=score, insufficient_evidence=insufficient)


ALL_LLM = ["opening_rapport", "individual_engagement", "communication", "closure_homework", "technical_quality"]


def judge(scores: dict[str, int | None]) -> JudgeOutput:
    return JudgeOutput(criteria=[crit(k, v) for k, v in scores.items()])


@pytest.mark.parametrize(
    ("m", "label"),
    [(3.4, "Excellent"), (3.39, "Good"), (2.8, "Good"), (2.79, "Average"), (2.2, "Average"), (2.19, "Poor")],
)
def test_label_thresholds(m, label):
    assert label_for(m) == label


def test_weighted_mean_renormalises_over_scored_criteria():
    criteria = [crit("opening_rapport", 4), crit("individual_engagement", 2), crit("communication", None)]
    # (0.20*4 + 0.25*2) / 0.45
    assert weighted_score(criteria) == pytest.approx(1.3 / 0.45)


def test_insufficient_with_score_is_excluded():
    criteria = [crit("opening_rapport", 4), crit("individual_engagement", 1, insufficient=True)]
    assert weighted_score(criteria) == 4


def test_nothing_scored():
    assert weighted_score([crit("opening_rapport", None)]) is None


def test_score_visual():
    assert score_visual(GOOD_VISUAL).score == 4
    no_face = GOOD_VISUAL.model_copy(update={"pct_face": 30})
    assert score_visual(no_face).score <= 2
    dark = GOOD_VISUAL.model_copy(update={"pct_dark": 50, "pct_black": 20})
    assert score_visual(dark).score <= 2
    assert score_visual(None).insufficient_evidence is True


def test_assess_excellent_and_face_cap():
    from cqa.scoring import assess

    j = judge({k: 4 for k in ALL_LLM})
    a = assess(j, GOOD_VISUAL, None, "vtt", meta={"evidence_total": 10, "evidence_dropped": 0})
    assert (a.label, a.weighted_score, a.confidence) == ("Excellent", 4.0, 1.0)
    assert [c.id for c in a.criteria][-2:] == ["visual_presence", "technical_quality"]

    hidden = GOOD_VISUAL.model_copy(update={"pct_face": 50})
    a = assess(j, hidden, None, "vtt")
    assert a.label == "Average"
    assert any("capped" in w for w in a.warnings)


def test_assess_poor_not_raised_by_cap():
    from cqa.scoring import assess

    j = judge({k: 1 for k in ALL_LLM})
    a = assess(j, GOOD_VISUAL.model_copy(update={"pct_face": 10}), None, "vtt")
    assert a.label == "Poor"
    assert not any("capped" in w for w in a.warnings)


def test_assess_unusable_transcript_lowers_confidence():
    from cqa.scoring import assess

    j = judge({k: None for k in ALL_LLM}).model_copy(
        update={"transcript_usable": False, "transcript_note": "wrong language"}
    )
    a = assess(j, None, None, "vtt")
    assert a.weighted_score is None
    assert a.confidence == 0.0
    assert any("unusable" in w for w in a.warnings)
    assert any("placeholder" in w for w in a.warnings)


def test_confidence_penalties():
    from cqa.scoring import assess

    j = judge({k: 3 for k in ALL_LLM})
    base = assess(j, GOOD_VISUAL, None, "vtt").confidence
    whisper = assess(j, GOOD_VISUAL, None, "whisper").confidence
    dropped = assess(j, GOOD_VISUAL, None, "vtt", meta={"evidence_total": 10, "evidence_dropped": 5}).confidence
    assert base == 1.0
    assert whisper < base and dropped < base
