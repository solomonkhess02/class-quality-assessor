"""Ablation switches (docs/EVALUATION.md, eval/ablations.md). Synthetic inputs only."""

import json

import pytest

from cqa import config
from cqa.judge import prompts


@pytest.fixture
def env(monkeypatch):
    """Set environment overrides for get_settings and clear its cache around the test."""

    def set_env(**values: str) -> None:
        for k, v in values.items():
            monkeypatch.setenv(k, v)
        config.get_settings.cache_clear()

    yield set_env
    config.get_settings.cache_clear()


def test_evidence_first_puts_score_last(env):
    env(JUDGE_EVIDENCE_FIRST="false")
    assert list(prompts.example()["criteria"][0])[1] == "score"  # baseline: score straight after the id
    env(JUDGE_EVIDENCE_FIRST="true")
    ex = prompts.example()
    for c in ex["criteria"]:
        keys = list(c)
        assert keys[-1] == "score"
        assert keys.index("observations") < keys.index("evidence") < keys.index("score")
    # same content, only the order changes
    assert json.dumps(ex, sort_keys=True) == json.dumps(prompts.EXAMPLE, sort_keys=True)
    sp = prompts.system_prompt()
    assert sp.index('"observations": [') < sp.index('"score": 3')


def _stats(duration_min: float, incidents: list[float]):
    from cqa.schemas import TextStats

    return TextStats(
        duration_min=duration_min, tutor_talk_share=0.6, student_turns_per_min=3, distinct_students_spoke=3,
        tutor_questions_per_min=2, sparse_punctuation=False, longest_tutor_monologue_s=60, connectivity_mentions=0,
        connectivity_incidents_s=incidents, opening_turns=10, closing_turns=10,
    )  # fmt: skip


def _visual(frozen_camera: float):
    from cqa.schemas import VisualReport

    return VisualReport(n_frames=100, pct_face=80, pct_dark=10, pct_black=0, pct_frozen=frozen_camera,
                        median_blur=1000, pct_frozen_camera=frozen_camera)  # fmt: skip


def test_connectivity_mentions_close_together_are_one_incident():
    from cqa.features.text_stats import connectivity_incidents
    from cqa.schemas import Turn

    turns = [
        Turn(start_s=10, end_s=12, speaker="A", text="can you hear me?"),
        Turn(start_s=30, end_s=32, speaker="B", text="you are on mute"),  # 18 s later: same incident
        Turn(start_s=200, end_s=205, speaker="A", text="my internet is lagging"),  # new incident
        Turn(start_s=300, end_s=305, speaker="A", text="the knight forks the rook"),
    ]
    assert connectivity_incidents(turns) == [10, 200]


def test_frozen_share_counts_camera_frames_only():
    from cqa.features.visual import aggregate
    from cqa.schemas import VisualFrameMetrics as F

    def f(frozen, share):
        return F(t_s=0, brightness=120, blur=500, is_black=False, is_frozen=frozen, face_count=1, is_screen_share=share)

    # a frozen shared board does not count; 1 of 2 camera frames is frozen
    v = aggregate([f(True, True), f(True, True), f(True, False), f(False, False)])
    assert v.pct_frozen == 75.0 and v.pct_frozen_camera == 50.0


def test_code_tech_scorer_thresholds():
    from cqa.schemas import CriterionResult
    from cqa.scoring import score_technical

    llm = CriterionResult(id="technical_quality", score=2, observations=["Audio drops twice."])
    ok = score_technical(_stats(60, [100, 900]), _visual(10), llm)  # 2 per hour, 10% frozen
    assert ok.score == 3 and ok.improvement == "" and "Judge: Audio drops twice." in ok.observations
    assert score_technical(_stats(60, [100, 900, 2000]), _visual(10), llm).score == 2  # 3 per hour
    assert score_technical(_stats(120, [100, 900, 2000]), _visual(10), llm).score == 3  # 1.5 per hour
    assert score_technical(_stats(60, []), _visual(20), llm).score == 2  # 20% frozen camera frames
    no_video = score_technical(_stats(60, []), None, None)
    assert no_video.score == 3 and any("not checked" in o for o in no_video.observations)


def test_label_exclude_keeps_criterion_but_not_in_problem_count(env):
    from cqa.schemas import CriterionResult, JudgeOutput
    from cqa.scoring import assess

    judged = JudgeOutput(criteria=[
        CriterionResult(id="opening_rapport", score=3), CriterionResult(id="individual_engagement", score=3),
        CriterionResult(id="communication", score=2), CriterionResult(id="closure_homework", score=3),
        CriterionResult(id="technical_quality", score=2),
    ])  # fmt: skip
    env(LABEL_EXCLUDE="none", TECH_SCORER="llm", AGGREGATION="problems")
    base = assess(judged, _visual(0), _stats(60, []), "vtt")
    env(LABEL_EXCLUDE="communication", TECH_SCORER="code", AGGREGATION="problems")
    a1 = assess(judged, _visual(0), _stats(60, []), "vtt")
    assert base.meta["problems"] == 2 and base.label == "Average"
    assert a1.meta["problems"] == 0 and a1.label == "Excellent"  # communication excluded, tech fine by code
    assert {c.id for c in a1.criteria} == {c.id for c in base.criteria}  # still shown
    assert next(c for c in a1.criteria if c.id == "technical_quality").score == 3


def _kf(**facts):
    from cqa.schemas import KeyFacts

    return KeyFacts(**facts)


def test_fact_bounds_clamp_scores_and_cap_wins():
    from cqa.judge.judge import apply_fact_bounds
    from cqa.schemas import CriterionResult as C

    crits = [C(id="opening_rapport", score=2), C(id="individual_engagement", score=4),
             C(id="closure_homework", score=2), C(id="communication", score=3)]  # fmt: skip
    kf = _kf(rapport_before_content=True, each_child_asked_individually=False, homework_assigned=True)
    out = {c.id: c for c in apply_fact_bounds(crits, kf)}
    assert out["opening_rapport"].score == 3  # raised: rapport before content
    assert out["individual_engagement"].score == 2  # capped: not every child asked
    assert out["closure_homework"].score == 3  # raised by homework, capped at 3 without a recap
    assert out["communication"].score == 3  # not fact-bound
    assert "raised to 3" in out["opening_rapport"].observations[-1]
    # a new joiner left out caps the opening even with rapport (cap wins over floor)
    capped = apply_fact_bounds(
        [C(id="opening_rapport", score=4)], _kf(rapport_before_content=True, new_joiners="not_invited")
    )
    assert capped[0].score == 2


def test_fact_bounds_fill_unscored_and_respect_recap():
    from cqa.judge.judge import apply_fact_bounds
    from cqa.schemas import CriterionResult as C

    out = apply_fact_bounds([C(id="closure_homework", score=None)], _kf(homework_assigned=True, recap_present=True))
    assert out[0].score == 3 and not out[0].insufficient_evidence
    out = apply_fact_bounds([C(id="closure_homework", score=4)], _kf(recap_present=True))
    assert out[0].score == 3  # no homework: at most 3 even with a recap
    out = apply_fact_bounds([C(id="closure_homework", score=3)], _kf())
    assert out[0].score == 2  # neither homework nor recap


def test_new_facts_without_quote_are_reset():
    from cqa.judge.judge import verify_key_facts
    from cqa.schemas import Transcript

    kf = _kf(rapport_before_content=True, each_child_asked_individually=True)
    out, _, reset = verify_key_facts(kf, Transcript(source="vtt"), 90)
    assert not out.rapport_before_content and not out.each_child_asked_individually
    assert set(reset) == {"rapport_before_content", "each_child_asked_individually"}


def test_facts_drive_prompt_switch(env):
    env(FACTS_DRIVE="false")
    off = prompts.system_prompt()
    env(FACTS_DRIVE="true")
    on = prompts.system_prompt()
    assert "rapport_before_content" not in off
    assert "each_child_asked_individually" in on and on.startswith(off.split("Rules:")[0])
