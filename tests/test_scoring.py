import pytest

from cqa import config
from cqa.schemas import CriterionResult, JudgeOutput, VisualReport
from cqa.scoring import label_for, score_visual, weighted_score

GOOD_VISUAL = VisualReport(n_frames=100, pct_face=95, pct_dark=0, pct_black=0, pct_frozen=5, median_blur=200)


def crit(cid: str, score: int | None, insufficient: bool = False) -> CriterionResult:
    return CriterionResult(id=cid, score=score, insufficient_evidence=insufficient)


ALL_LLM = ["opening_rapport", "individual_engagement", "communication", "closure_homework", "technical_quality"]


@pytest.fixture(autouse=True)
def phase1_scoring(monkeypatch):
    """These tests cover the base scoring rules: LLM-scored technical quality, nothing left out of the label
    (the A1 switches are tested in test_ablations.py)."""
    monkeypatch.setenv("TECH_SCORER", "llm")
    monkeypatch.setenv("LABEL_EXCLUDE", "none")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


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

    # face missing in > 40% of frames: warning only (E6), the visual score itself drops to 3
    hidden = GOOD_VISUAL.model_copy(update={"pct_face": 50})
    a = assess(j, hidden, None, "vtt")
    assert a.label == "Excellent"
    assert any("No face visible in 50%" in w for w in a.warnings)


def test_assess_all_ones_is_poor():
    from cqa.scoring import assess

    j = judge({k: 1 for k in ALL_LLM})
    a = assess(j, GOOD_VISUAL.model_copy(update={"pct_face": 10}), None, "vtt")
    assert a.label == "Poor"


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


@pytest.mark.parametrize(("n", "label"), [(0, "Excellent"), (1, "Good"), (2, "Average"), (3, "Poor"), (5, "Poor")])
def test_problem_count_labels(n, label):
    from cqa.scoring import problem_count_label

    scores = {k: (2 if i < n else 3) for i, k in enumerate(ALL_LLM)}
    criteria = [crit(k, v) for k, v in scores.items()]
    assert problem_count_label(criteria) == (label, n)


def test_problem_count_ignores_insufficient_and_counts_visual():
    from cqa.scoring import problem_count_label

    criteria = [crit("opening_rapport", 2), crit("communication", 1, insufficient=True), crit("visual_presence", 1)]
    assert problem_count_label(criteria) == ("Average", 2)


def test_assess_passes_key_facts_and_alt_label():
    from cqa.schemas import KeyFacts
    from cqa.scoring import assess

    j = judge({k: 3 for k in ALL_LLM}).model_copy(update={"key_facts": KeyFacts(homework_assigned=True)})
    a = assess(j, GOOD_VISUAL, None, "vtt")
    assert a.key_facts.homework_assigned is True
    assert a.meta["problem_count_label"] == "Excellent" and a.meta["problems"] == 0


def test_too_few_scored_criteria_gives_placeholder():
    from cqa.scoring import assess

    j = judge({"opening_rapport": 3, "communication": None, "closure_homework": None,
               "individual_engagement": None, "technical_quality": None})  # fmt: skip
    a = assess(j, None, None, "vtt")  # 1 scored criterion, no visual
    assert a.label == "Average"
    assert a.meta["n_scored"] == 1
    assert any("placeholder" in w for w in a.warnings)


def test_aggregation_setting(monkeypatch):
    from cqa import config
    from cqa.scoring import assess

    # mean 2.75 -> Average; problems: tech=2 is the only problem -> Good
    j = judge({"opening_rapport": 3, "individual_engagement": 3, "communication": 3, "closure_homework": 3,
               "technical_quality": 2})  # fmt: skip
    bad_visual = GOOD_VISUAL.model_copy(update={"pct_dark": 50, "pct_black": 20})  # visual score 2
    for agg, expected in [("problems", "Average"), ("mean", "Average")]:
        monkeypatch.setenv("AGGREGATION", agg)
        config.get_settings.cache_clear()
        a = assess(j, bad_visual, None, "vtt")
        assert a.label == expected, agg
        assert a.meta["problems"] == 2
    monkeypatch.setenv("AGGREGATION", "problems")
    config.get_settings.cache_clear()
    assert assess(j, GOOD_VISUAL, None, "vtt").label == "Good"
    config.get_settings.cache_clear()
