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


def test_quote_after_a_short_student_turn_keeps_the_tutor():
    # E26 (live test on B552): partial_ratio aligns the shorter string, so a one-letter student turn ("I.") scored
    # 100 against any quote containing that letter and, coming first, won the tie with the real tutor line
    transcript = Transcript(
        source="vtt",
        duration_s=600,
        turns=[
            Turn(start_s=100, end_s=101, speaker="S2", role="student", text="I."),
            Turn(
                start_s=102,
                end_s=110,
                speaker="TUTOR",
                role="tutor",
                text="This will be your homework, solve the five puzzles before Friday.",
            ),  # fmt: skip
            Turn(start_s=111, end_s=112, speaker="S4", role="student", text="E?"),
            Turn(start_s=113, end_s=118, speaker="TUTOR", role="tutor", text="Next one, I need [NAME] to answer."),
        ],
    )
    out, dropped, _ = verify_evidence(
        judge_with(
            ev("00:01:42", "This will be your homework, solve five puzzles before Friday"),  # small drift
            ev("00:01:53", "next one I need NAME to answer it"),
        ),
        transcript,
        90,
    )
    assert dropped == 0
    assert [(e.speaker, e.timestamp) for e in out.criteria[0].evidence] == [
        ("TUTOR", "00:01:42"),
        ("TUTOR", "00:01:53"),
    ]


def test_quote_spanning_two_turns_goes_to_the_turn_holding_most_of_it():
    transcript = Transcript(
        source="vtt",
        duration_s=600,
        turns=[
            Turn(start_s=200, end_s=202, speaker="S1", role="student", text="I did it."),
            Turn(start_s=203, end_s=210, speaker="TUTOR", role="tutor", text="Great, now the second puzzle is a fork."),
        ],
    )
    out, dropped, _ = verify_evidence(
        judge_with(ev("00:03:20", "I did it. Great, now the second puzzle is a fork")), transcript, 90
    )
    assert dropped == 0 and out.criteria[0].evidence[0].speaker == "TUTOR"


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
    out, dropped, unsupported, _ = verify_key_facts(kf, TRANSCRIPT, 90)
    assert dropped == 1
    assert out.homework_assigned == "yes" and out.homework_what == "puzzles 1 to 5"
    assert out.recap_present == "unknown"  # three-state facts (A3b): no verified quote -> unknown
    assert out.new_joiners == "unknown"
    assert out.children_greeted_by_name == 0
    assert sorted(unsupported) == ["children_greeted_by_name", "new_joiners", "recap_present"]
