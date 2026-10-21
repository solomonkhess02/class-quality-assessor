from cqa.judge.facts import ClassFacts, FactEvidence, FactsOutput, apply_verified_evidence, score_from_facts
from cqa.schemas import TextStats

STATS = TextStats(
    duration_min=50, tutor_talk_share=0.6, student_turns_per_min=5, distinct_students_spoke=5,
    tutor_questions_per_min=3, sparse_punctuation=False, longest_tutor_monologue_s=60,
    connectivity_mentions=0, opening_turns=50, closing_turns=50,
)  # fmt: skip

STRONG = ClassFacts(
    content_start="00:08:00", children_greeted_by_name=5, plan_stated=True, new_joiners="invited",
    children_asked_individually=5, individual_questions=20, praise_count=20, specific_praise=True,
    recap_present=True, homework_assigned=True, proper_goodbye=True,
)  # fmt: skip


def test_strong_class_scores_4():
    assert set(score_from_facts(STRONG, STATS, class_start_s=30).values()) == {4}


def test_weak_class_scores_low():
    weak = ClassFacts(content_start="00:00:40", children_asked_individually=1, individual_questions=2, praise_count=2)
    s = score_from_facts(weak, STATS, class_start_s=30)
    assert s["opening_rapport"] == 1  # only "no new joiner left out" holds
    assert s["individual_engagement"] == 2
    assert s["communication"] == 1  # 0.4 praise per 10 min
    assert s["closure_homework"] == 1
    assert s["technical_quality"] == 4


def test_closure_counts_and_ends_early():
    two = STRONG.model_copy(update={"recap_present": False})
    assert score_from_facts(two, STATS)["closure_homework"] == 3
    cut = STRONG.model_copy(update={"transcript_ends_early": True})
    assert score_from_facts(cut, STATS)["closure_homework"] is None


def test_negative_tone_lowers_warmth():
    rude = STRONG.model_copy(update={"negative_tone_count": 3})
    assert score_from_facts(rude, STATS)["communication"] == 3


def test_unverified_boolean_facts_become_false():
    out = FactsOutput(facts=STRONG)
    verified = [FactEvidence(fact="homework_assigned", timestamp="00:50:00", speaker="TUTOR", quote="homework")]
    f = apply_verified_evidence(out, verified)
    assert f.homework_assigned is True
    assert f.recap_present is False and f.proper_goodbye is False and f.plan_stated is False
