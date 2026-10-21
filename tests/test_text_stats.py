import pytest

from cqa.features.text_stats import compute_text_stats, window
from cqa.schemas import Transcript, Turn


def turn(start, end, role, text, speaker=None):
    speaker = speaker or ("TUTOR" if role == "tutor" else "S1")
    return Turn(start_s=start, end_s=end, speaker=speaker, role=role, text=text)


@pytest.fixture
def transcript() -> Transcript:
    # 10-minute class
    turns = [
        turn(0, 60, "tutor", "Hello everyone. How are you? Can you hear me?"),
        turn(60, 70, "student", "Yes.", "S1"),
        turn(70, 100, "tutor", "Great, let us start. What is a fork?"),
        turn(100, 160, "tutor", "A fork attacks two pieces. Remember that."),
        turn(160, 170, "student", "Sorry I was on mute, my network is bad.", "S2"),
        turn(300, 400, "tutor", "Now the homework. Solve five puzzles."),
        turn(560, 570, "student", "Bye!", "S1"),
        turn(570, 600, "tutor", "Bye bye, see you next class."),
    ]
    return Transcript(source="vtt", duration_s=600, turns=turns)


def test_basic_stats(transcript):
    s = compute_text_stats(transcript, opening_min=2, closing_min=1)
    assert s.duration_min == 10
    # tutor 60+30+60+100+30 = 280 s, students 30 s
    assert s.tutor_talk_share == pytest.approx(280 / 310, abs=1e-3)
    assert s.student_turns_per_min == pytest.approx(0.3)
    assert s.distinct_students_spoke == 2
    assert s.tutor_questions_per_min == pytest.approx(0.3)
    assert s.sparse_punctuation is False
    # tutor turns at 70-100 and 100-160 are uninterrupted -> 90 s; 300-400 -> 100 s
    assert s.longest_tutor_monologue_s == 100
    # "Can you hear me", "on mute", "network"
    assert s.connectivity_mentions == 3
    assert s.opening_turns == 4  # starts < 120 s: 0, 60, 70, 100
    assert s.closing_turns == 2  # starts >= 540 s


def test_sparse_punctuation_flag():
    words = " ".join(["word"] * 300)
    t = Transcript(source="vtt", duration_s=600, turns=[turn(0, 600, "tutor", words)])
    s = compute_text_stats(t, 10, 10)
    assert s.sparse_punctuation is True
    assert s.tutor_questions_per_min == 0


def test_full_stops_only_flags_questions_unreliable():
    text = " ".join(["what is this move."] * 40)
    t = Transcript(source="vtt", duration_s=600, turns=[turn(0, 600, "tutor", text)])
    assert compute_text_stats(t, 10, 10).sparse_punctuation is True


def test_empty_transcript():
    s = compute_text_stats(Transcript(source="vtt"), 10, 10)
    assert s.duration_min == 0
    assert s.tutor_talk_share == 0
    assert s.student_turns_per_min == 0


def test_window(transcript):
    assert [t.start_s for t in window(transcript, 60, 170)] == [60, 70, 100, 160]
