from cqa.judge.judge import verify_evidence
from cqa.schemas import CriterionResult, Evidence, JudgeOutput, Transcript, Turn

TRANSCRIPT = Transcript(
    source="vtt",
    duration_s=600,
    turns=[
        Turn(start_s=10, end_s=15, speaker="TUTOR", role="tutor", text="Hello S1, how was your school today?"),
        Turn(start_s=16, end_s=18, speaker="S1", role="student", text="It was good, ma'am."),
        Turn(start_s=400, end_s=420, speaker="TUTOR", role="tutor", text="For homework, solve puzzles 1 to 5."),
    ],
)


def ev(ts: str, quote: str, speaker: str = "TUTOR") -> Evidence:
    return Evidence(timestamp=ts, speaker=speaker, quote=quote)


def judge_with(*evidence: Evidence) -> JudgeOutput:
    return JudgeOutput(criteria=[CriterionResult(id="opening_rapport", score=3, evidence=list(evidence))])


def test_exact_and_near_quotes_verified():
    out, dropped, total = verify_evidence(
        judge_with(
            ev("00:00:10", "how was your school today"),
            ev("00:01:00", "Hello S1 how was your school today"),  # 50 s off, no punctuation
            ev("00:06:45", "for homework solve puzzles one to 5"),  # small wording drift
        ),
        TRANSCRIPT,
        window_s=90,
    )
    assert (dropped, total) == (0, 3)
    evid = out.criteria[0].evidence
    assert all(e.verified for e in evid)
    # timestamps are snapped to the matching turn
    assert [e.timestamp for e in evid] == ["00:00:10", "00:00:10", "00:06:40"]


def test_wrong_time_or_invented_quote_dropped():
    out, dropped, total = verify_evidence(
        judge_with(
            ev("00:09:00", "how was your school today"),  # real quote, timestamp > 90 s off
            ev("00:00:12", "Great job everyone, amazing work"),  # invented
        ),
        TRANSCRIPT,
        window_s=90,
    )
    assert (dropped, total) == (2, 2)
    assert out.criteria[0].evidence == []


def test_speaker_corrected_from_transcript():
    out, _, _ = verify_evidence(judge_with(ev("00:00:16", "It was good", speaker="TUTOR")), TRANSCRIPT, 90)
    assert out.criteria[0].evidence[0].speaker == "S1"


def test_short_quote_needs_exact_substring():
    out, dropped, _ = verify_evidence(judge_with(ev("00:00:16", "It was bad")), TRANSCRIPT, 90)
    assert dropped == 1


def test_key_facts_without_verified_quote_are_reset():
    from cqa.judge.judge import verify_key_facts
    from cqa.schemas import FactEvidence, KeyFacts

    kf = KeyFacts(
        homework_assigned=True,
        homework_what="puzzles 1 to 5",
        new_joiners="invited",
        children_greeted_by_name=1,
        recap_present=True,
        evidence=[
            FactEvidence(fact="homework_assigned", timestamp="00:06:45", speaker="TUTOR",
                         quote="For homework, solve puzzles 1 to 5"),
            FactEvidence(fact="recap_present", timestamp="00:06:50", speaker="TUTOR",
                         quote="Today we learned about forks and pins"),  # invented
        ],
    )  # fmt: skip
    out, dropped, unsupported = verify_key_facts(kf, TRANSCRIPT, 90)
    assert dropped == 1
    assert out.homework_assigned is True and out.homework_what == "puzzles 1 to 5"
    assert out.recap_present is False
    assert out.new_joiners == "none"
    assert out.children_greeted_by_name == 0
    assert sorted(unsupported) == ["children_greeted_by_name", "new_joiners", "recap_present"]
