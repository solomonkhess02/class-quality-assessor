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
    env(JUDGE_EVIDENCE_FIRST="false", FACTS_DRIVE="false")
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


def test_tech_incidents_are_problem_phrases_only():
    from cqa.features.text_stats import connectivity_incidents
    from cqa.schemas import Turn

    turns = [
        Turn(start_s=10, end_s=12, speaker="A", text="can you hear me?"),  # routine check, not a problem
        Turn(start_s=20, end_s=22, speaker="B", text="you are on mute, please unmute"),  # mute management
        Turn(start_s=30, end_s=32, speaker="B", text="yes I can see you"),  # affirmative
        Turn(start_s=100, end_s=102, speaker="A", text="I can't hear you, your voice is breaking"),
        Turn(start_s=130, end_s=132, speaker="B", text="my internet is lagging"),  # 28 s later: same incident
        Turn(start_s=400, end_s=405, speaker="A", text="S2 got disconnected, he will rejoin"),
        Turn(start_s=500, end_s=505, speaker="A", text="the knight forks the rook"),
    ]
    assert connectivity_incidents(turns) == [100, 400]


def test_face_box_brightness_and_median_needs_five_frames():
    import numpy as np

    from cqa.features.visual import face_box_brightness, median_or_none

    gray = np.full((100, 100), 200, np.uint8)
    gray[10:30, 10:30] = 60  # the larger face box is dark
    faces = np.array([[10, 10, 20, 20] + [0] * 11, [60, 60, 10, 10] + [0] * 11], np.float32)
    assert face_box_brightness(gray, faces) == 60.0
    assert face_box_brightness(gray, np.empty((0, 15), np.float32)) is None
    assert median_or_none([50, 60, 70, 80]) is None and median_or_none([50, 60, 70, 80, 90]) == 70.0


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
    kf = _kf(rapport_before_content="yes", each_child_asked_individually="no", homework_assigned="yes",
             recap_present="no")  # fmt: skip
    out = {c.id: c for c in apply_fact_bounds(crits, kf)}
    assert out["opening_rapport"].score == 3  # raised: rapport before content
    assert out["individual_engagement"].score == 2  # capped: not every child asked
    assert out["closure_homework"].score == 3  # raised by homework, capped at 3 without a recap
    assert out["communication"].score == 3  # not fact-bound
    assert "raised to 3" in out["opening_rapport"].observations[-1]
    # a new joiner left out caps the opening even with rapport (cap wins over floor)
    capped = apply_fact_bounds(
        [C(id="opening_rapport", score=4)], _kf(rapport_before_content="yes", new_joiners="not_invited")
    )
    assert capped[0].score == 2


def test_fact_bounds_fill_unscored_and_unknown_never_bounds():
    from cqa.judge.judge import apply_fact_bounds
    from cqa.schemas import CriterionResult as C

    out = apply_fact_bounds([C(id="closure_homework", score=None)], _kf(homework_assigned="yes", recap_present="yes"))
    assert out[0].score == 3 and not out[0].insufficient_evidence
    out = apply_fact_bounds([C(id="closure_homework", score=4)], _kf(homework_assigned="no", recap_present="yes"))
    assert out[0].score == 3  # verified no homework: at most 3 even with a recap
    out = apply_fact_bounds([C(id="closure_homework", score=3)], _kf(homework_assigned="no", recap_present="no"))
    assert out[0].score == 2
    crits = [
        C(id="opening_rapport", score=4),
        C(id="individual_engagement", score=1),
        C(id="closure_homework", score=4),
    ]
    assert [c.score for c in apply_fact_bounds(crits, _kf())] == [4, 1, 4]  # all unknown: LLM scores stand


def test_legacy_bool_facts_are_coerced():
    kf = _kf(rapport_before_content=True, each_child_asked_individually=False)
    assert (kf.rapport_before_content, kf.each_child_asked_individually, kf.homework_assigned) == (
        "yes",
        "no",
        "unknown",
    )


def _transcript(*lines):
    from cqa.schemas import Transcript, Turn

    turns = [Turn(start_s=t, end_s=t + 5, speaker=spk, role="tutor" if spk == "TUTOR" else "student", text=txt)
             for t, spk, txt in lines]  # fmt: skip
    return Transcript(source="vtt", duration_s=3600, turns=turns)


def _ev(fact, value, ts, quote):
    from cqa.schemas import FactEvidence

    return FactEvidence(fact=fact, value=value, timestamp=ts, speaker="TUTOR", quote=quote)


def test_three_state_facts_need_matching_verified_quotes():
    from cqa.judge.judge import verify_key_facts

    tr = _transcript((60, "TUTOR", "today we learn the fork, look at the board"), (3300, "TUTOR", "okay bye everyone"))
    kf = _kf(rapport_before_content="no", each_child_asked_individually="yes", recap_present="no",
             evidence=[_ev("rapport_before_content", "no", "00:01:00", "today we learn the fork, look at the board"),
                       _ev("recap_present", "yes", "00:55:00", "okay bye everyone")])  # fmt: skip
    out, _, reset, guards = verify_key_facts(kf, tr, 90)
    assert out.rapport_before_content == "no"  # verified quote of the opposite, nothing before it
    assert out.each_child_asked_individually == "unknown"  # yes without a quote
    assert out.recap_present == "unknown"  # the quote shows yes, not the claimed no
    assert set(reset) == {"each_child_asked_individually", "recap_present"}
    assert guards == {"rapport_no_to_unknown": 0, "homework_no_to_unknown": 0}


def test_guard_greeting_before_the_quote_makes_rapport_unknown():
    from cqa.judge.judge import verify_key_facts

    tr = _transcript((5, "TUTOR", "Hello S1, how was school?"), (60, "TUTOR", "today we learn the fork"))
    kf = _kf(rapport_before_content="no",
             evidence=[_ev("rapport_before_content", "no", "00:01:00", "today we learn the fork")])  # fmt: skip
    out, _, _, guards = verify_key_facts(kf, tr, 90)
    assert out.rapport_before_content == "unknown" and guards["rapport_no_to_unknown"] == 1


def test_guard_homework_word_in_closing_makes_homework_unknown():
    from cqa.judge.judge import verify_key_facts

    tr = _transcript((3200, "TUTOR", "practice these two puzzles at home"), (3500, "TUTOR", "bye everyone"))
    kf = _kf(homework_assigned="no", evidence=[_ev("homework_assigned", "no", "00:58:20", "bye everyone")])
    out, _, _, guards = verify_key_facts(kf, tr, 90, closing_min=10)
    assert out.homework_assigned == "unknown" and guards["homework_no_to_unknown"] == 1
    # guards only ever move "no" to "unknown"
    hw = "practice these two puzzles at home"
    kf_yes = _kf(homework_assigned="yes",
                 evidence=[_ev("homework_assigned", "yes", "00:53:20", hw)])  # fmt: skip
    out_yes, _, _, guards_yes = verify_key_facts(kf_yes, tr, 90)
    assert out_yes.homework_assigned == "yes" and guards_yes["homework_no_to_unknown"] == 0


def test_facts_drive_prompt_switch(env):
    env(FACTS_DRIVE="false")
    off = prompts.system_prompt()
    env(FACTS_DRIVE="true")
    on = prompts.system_prompt()
    assert "rapport_before_content" not in off
    assert "each_child_asked_individually" in on and on.startswith(off.split("Rules:")[0])


def test_severity_label_tiers_and_margin():
    from cqa.scoring import margin_confidence, severity_label

    assert severity_label(["camera"], ["no homework"], True)[:2] == ("Poor", 11)
    assert severity_label([], ["no homework", "no closing recap", "lighting"], False) == ("Average", 3, 2)
    assert severity_label([], ["no homework"], True) == ("Good", 1, 1)
    assert severity_label([], [], True) == ("Excellent", 0, 1)
    assert severity_label([], [], False) == ("Good", 0.5, 1)  # no problems but no positive evidence
    assert margin_confidence(1) == 0.75 and margin_confidence(5) == 1.0


def test_severity_checks_unknown_never_fires_and_face_is_moderate(env):
    from cqa.config import get_settings
    from cqa.schemas import CriterionResult as C
    from cqa.scoring import severity_findings

    crits = [C(id="opening_rapport", score=3), C(id="individual_engagement", score=4), C(id="communication", score=2)]
    visual = _visual(0).model_copy(update={"pct_face_camera": 30.0, "median_face_brightness": 70.0})
    env(FACTS_DRIVE="true", LABEL_EXCLUDE="communication", FACE_CHECK_CRITICAL="false")
    kf = _kf(rapport_before_content="no", recap_present="yes", homework_assigned="unknown")
    critical, moderate, positive = severity_findings(crits, kf, visual, get_settings())
    assert critical == ["no rapport before content"]
    assert moderate == ["tutor face in only 30% of camera frames", "lighting: face luma 70 (< 85)"]
    assert not positive  # needs rapport yes and each child yes
    env(FACE_CHECK_CRITICAL="true")
    assert "tutor face in only 30% of camera frames" in severity_findings(crits, kf, visual, get_settings())[0]
    env(FACE_CHECK_CRITICAL="false", LABEL_EXCLUDE="communication,dark_face,no_rapport")
    critical, moderate, _ = severity_findings(crits, _kf(), visual.model_copy(update={"median_face_brightness": None}),
                                              get_settings())  # fmt: skip
    assert critical == [] and moderate == ["tutor face in only 30% of camera frames"]  # unknown facts never fire


def test_assess_with_severity_rule(env):
    from cqa.schemas import CriterionResult as C
    from cqa.schemas import JudgeOutput
    from cqa.scoring import assess

    env(AGGREGATION="severity", FACTS_DRIVE="true", TECH_SCORER="code", LABEL_EXCLUDE="communication")
    judged = JudgeOutput(
        criteria=[C(id="opening_rapport", score=4), C(id="individual_engagement", score=4),
                  C(id="communication", score=3), C(id="closure_homework", score=4)],
        key_facts=_kf(rapport_before_content="yes", each_child_asked_individually="yes", homework_assigned="yes",
                      recap_present="yes"),
    )  # fmt: skip
    visual = _visual(0).model_copy(update={"pct_face_camera": 95.0})
    a = assess(judged, visual, _stats(60, []), "vtt")
    assert a.label == "Excellent" and a.meta["severity"] == 0 and a.confidence == 0.75
    assert a.meta["problems_found"] == [] and 0 <= a.meta["score_100"] <= 100
    unusable = assess(judged.model_copy(update={"transcript_usable": False}), visual, _stats(60, []), "vtt")
    assert unusable.confidence == 0.25
