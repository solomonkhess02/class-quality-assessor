"""Transcript quality check and ASR windows (P2.2). Synthetic inputs only."""

from cqa.schemas import Transcript, Turn
from cqa.transcript.quality import check_quality
from cqa.transcript.whisper import asr_windows


def _transcript(text: str, minutes: float = 10, language: str = "unknown", role: str = "tutor") -> Transcript:
    turns = [Turn(start_s=i * 30, end_s=i * 30 + 25, speaker="TUTOR", role=role, text=text) for i in range(20)]
    return Transcript(source="vtt", language=language, duration_s=minutes * 60, turns=turns)


def test_english_class_talk_is_usable():
    q = check_quality(_transcript("Okay everyone, look at the knight on f3. Which piece can it fork? Very good."))
    assert q.usable and q.non_english_share is not None and q.non_english_share < 0.1


def test_wrong_language_asr_is_unusable():
    # English recogniser applied to other speech: many tokens are not English words at all
    garbled = "Ningal entha cheyyunne? Nokkoo ivide kuthira enna padam, sheri alle, vegam parayu ok"
    q = check_quality(_transcript(garbled))
    assert not q.usable and q.non_english_share > 0.1
    assert "not English" in q.reasons[0]


def test_almost_no_speech_is_unusable():
    q = check_quality(_transcript("okay", minutes=60))
    assert not q.usable and any("words per minute" in r for r in q.reasons)


def test_non_english_whisper_output_skips_the_english_word_check():
    q = check_quality(_transcript("ശരി, ഈ കുതിര എവിടെ പോകും? നല്ലത്, വളരെ നല്ലത്", language="ml", role="unknown"))
    assert q.non_english_share is None and q.language == "ml"


def test_asr_windows_cover_opening_closing_and_middle():
    w = asr_windows(3000, opening_min=12, closing_min=10, middle_n=3, middle_min=3)
    assert w[0] == (0.0, 720) and w[-1] == (2400, 3000)
    assert len(w) == 5 and all(b - a == 180 for a, b in w[1:-1])
    middles = [(a + b) / 2 for a, b in w[1:-1]]
    assert middles == [1140.0, 1560.0, 1980.0]  # evenly spaced between 12:00 and 40:00


def test_asr_windows_merge_on_short_recordings():
    assert asr_windows(1200, 12, 10, 3, 3) == [(0.0, 1200)]


def test_judge_is_told_which_windows_were_transcribed():
    from cqa.judge.prompts import windows_note

    t = Transcript(source="whisper", duration_s=3000, windows=[(0.0, 720.0), (2400.0, 3000.0)])
    note = windows_note(t)
    assert "00:00:00-00:12:00" in note and "00:40:00-00:50:00" in note
    assert windows_note(Transcript(source="vtt")) == ""


def test_role_tagging_sets_roles_and_leaves_skipped_lines_unknown(monkeypatch):
    from cqa.transcript import role_tagging as rt

    def fake_call_json(messages, schema, run_index=0, temperature=None):
        assert "[NAME]" in messages[1]["content"]  # only anonymised text is sent
        return schema(roles=[{"i": 0, "role": "tutor"}, {"i": 1, "role": "student"}]), {"latency_s": 0.1}

    monkeypatch.setattr(rt, "call_json", fake_call_json)
    t = Transcript(source="whisper", turns=[
        Turn(start_s=0, end_s=5, speaker="UNKNOWN", text="Hello [NAME], what is a fork?"),
        Turn(start_s=6, end_s=8, speaker="UNKNOWN", text="Two pieces attacked"),
        Turn(start_s=9, end_s=12, speaker="UNKNOWN", text="Very good"),
    ])  # fmt: skip
    tagged, meta = rt.tag_roles(t)
    assert [x.role for x in tagged.turns] == ["tutor", "student", "unknown"]
    assert [x.speaker for x in tagged.turns] == ["TUTOR", "STUDENT", "UNKNOWN"]
    assert meta["role_lines_missing"] == 1


def test_reference_roles_use_the_most_overlapping_vtt_turn():
    from cqa.transcript.role_tagging import reference_roles

    vtt = [Turn(start_s=0, end_s=10, speaker="T", role="tutor", text="a"),
           Turn(start_s=10, end_s=14, speaker="S", role="student", text="b")]  # fmt: skip
    segs = [Turn(start_s=a, end_s=b, speaker="UNKNOWN", text="x") for a, b in [(1, 9), (9, 14), (20, 25)]]
    assert reference_roles(segs, vtt) == ["tutor", "student", None]
