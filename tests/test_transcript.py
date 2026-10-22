from pathlib import Path

from cqa.schemas import Transcript, Turn
from cqa.transcript.privacy import anonymise, leak_counts, over_scrub_share, pseudonymise
from cqa.transcript.roles import assign_roles
from cqa.transcript.vtt import merge_turns, parse_vtt

FIX = Path(__file__).parent / "fixtures"
REAL_NAMES = ["Priya", "Raman", "Arjun", "Meera", "Kapoor", "Aarav", "Arav", "iPhone", "iPad"]


def load(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_parse_simple():
    t = parse_vtt(load("simple.vtt"))
    assert t.source == "vtt"
    # cues 1 and 2 are the same speaker 0.5 s apart -> merged
    assert [x.speaker for x in t.turns] == [
        "Priya Raman",
        "Arjun's iPhone",
        "Priya Raman",
        "Meera Kapoor",
        "Priya Raman",
    ]
    first = t.turns[0]
    assert (first.start_s, first.end_s) == (1.0, 6.0)
    assert first.text == "Hello everyone, welcome to class! Arjun, can you hear me?"
    assert t.duration_s == 79.0


def test_crlf_matches_lf():
    lf = load("simple.vtt")
    assert parse_vtt(lf.replace("\n", "\r\n")) == parse_vtt(lf)
    assert parse_vtt("﻿" + lf) == parse_vtt(lf)


def test_parse_messy():
    t = parse_vtt(load("messy.vtt"))
    # mm:ss.mmm timestamps, cue settings, multi-line cue, missing prefix carries speaker
    assert t.turns[0].speaker == "Priya Raman"
    assert t.turns[0].start_s == 1.0
    assert t.turns[0].end_s == 5.0  # cue 2 (no prefix, 0.5 s gap) merged
    assert t.turns[0].text == "Good morning Aarav and welcome back! Did you do the homework?"
    # curly apostrophes normalised
    assert t.turns[1].speaker == "Aarav's iPad"
    assert t.turns[1].text == "I did it, it's done"


def test_merge_gap_threshold():
    turns = [
        Turn(start_s=0, end_s=1, speaker="A", text="a"),
        Turn(start_s=2.4, end_s=3, speaker="A", text="b"),  # 1.4 s gap -> merge
        Turn(start_s=4.5, end_s=5, speaker="A", text="c"),  # 1.5 s gap -> keep
        Turn(start_s=5.1, end_s=6, speaker="B", text="d"),
    ]
    merged = merge_turns(turns, gap_s=1.5)
    assert [(x.speaker, x.text) for x in merged] == [("A", "a b"), ("A", "c"), ("B", "d")]


def test_tutor_is_longest_speaker():
    t = assign_roles(parse_vtt(load("simple.vtt")))
    roles = {x.speaker: x.role for x in t.turns}
    assert roles == {"Priya Raman": "tutor", "Arjun's iPhone": "student", "Meera Kapoor": "student"}


def test_tutor_second_account_is_tutor():
    # synthetic: the tutor joins from a laptop and a phone; the phone account has less talk time than one child
    t = Transcript(
        source="vtt",
        turns=[
            Turn(start_s=0, end_s=60, speaker="Priya Raman", text="Welcome everyone, today we learn forks."),
            Turn(start_s=61, end_s=91, speaker="Meera Kapoor", text="Hello ma'am, I solved the puzzles."),
            Turn(start_s=92, end_s=102, speaker="Priya Raman's iPhone", text="This will be your homework."),
            Turn(start_s=103, end_s=105, speaker="Arjun", text="Okay."),
        ],
    )
    roles = {x.speaker: x.role for x in assign_roles(t).turns}
    assert roles == {"Priya Raman": "tutor", "Meera Kapoor": "student", "Priya Raman's iPhone": "tutor",
                     "Arjun": "student"}  # fmt: skip
    anon, mapping = pseudonymise(assign_roles(t))
    assert mapping["Priya Raman's iPhone"] == "TUTOR" and anon.turns[2].speaker == "TUTOR"
    # a different person with a similar-looking name is not merged
    t2 = t.model_copy(update={"turns": [*t.turns[:2], t.turns[2].model_copy(update={"speaker": "Priyanka Rao"})]})
    assert {x.speaker: x.role for x in assign_roles(t2).turns}["Priyanka Rao"] == "student"


def test_unknown_speaker_stays_unknown():
    t = Transcript(
        source="whisper",
        turns=[Turn(start_s=0, end_s=10, speaker="UNKNOWN", text="hello")],
    )
    assert assign_roles(t).turns[0].role == "unknown"


def test_pseudonymise_speakers_and_text():
    t, mapping = pseudonymise(assign_roles(parse_vtt(load("simple.vtt"))))
    assert mapping == {"Priya Raman": "TUTOR", "Arjun's iPhone": "S1", "Meera Kapoor": "S2"}
    assert [x.speaker for x in t.turns] == ["TUTOR", "S1", "TUTOR", "S2", "TUTOR"]
    assert t.turns[0].text == "Hello everyone, welcome to class! S1, can you hear me?"
    assert t.turns[2].text == "Great S1. S2, what is a fork?"
    # chess vocabulary is not mistaken for a name
    assert "King and the Queen" in t.turns[4].text


def test_pseudonymise_fuzzy_and_device_names():
    t, mapping = pseudonymise(assign_roles(parse_vtt(load("messy.vtt"))))
    assert mapping["Aarav's iPad"] == "S1"
    assert t.turns[0].text.startswith("Good morning S1 and")
    assert "Well done S1!" in t.turns[2].text  # "Arav" fuzzy-matches "Aarav"
    assert "Rook to e4" in t.turns[2].text


def test_no_real_name_survives():
    for fixture in ("simple.vtt", "messy.vtt"):
        for fn in (pseudonymise, anonymise):
            t, _ = fn(assign_roles(parse_vtt(load(fixture))))
            dump = t.model_dump_json()
            for name in REAL_NAMES:
                assert name not in dump, (fixture, fn.__name__, name)


def _vtt(*lines: tuple[str, str]) -> Transcript:
    """VTT transcript with one turn per line; turn length follows text length, so the first speaker is the tutor."""
    turns = [
        Turn(start_s=i * 100, end_s=i * 100 + len(txt), speaker=spk, text=txt) for i, (spk, txt) in enumerate(lines)
    ]
    return assign_roles(Transcript(source="vtt", turns=turns))


def test_vtt_second_pass_scrubs_unlisted_names():
    raw = _vtt(
        ("Priya Raman", "Hello Arjun, welcome. Is Kavya joining today? Kavya, your turn. Move the Knight to f3."),
        ("Arjun's iPhone", "Yes ma'am, I am ready."),
    )
    t, mapping = anonymise(raw)
    assert mapping == {"Priya Raman": "TUTOR", "Arjun's iPhone": "S1"}
    assert t.turns[0].speaker == "TUTOR"
    assert t.turns[0].text == ("Hello S1, welcome. Is [NAME] joining today? [NAME], your turn. Move the Knight to f3.")
    assert t.turns[1].text == "Yes ma'am, I am ready."


def test_lowercase_name_variant_is_pseudonymised():
    t, _ = anonymise(_vtt(("Priya Raman", "well done arav, and aarav's move is next"), ("Aarav's iPad", "ok")))
    assert t.turns[0].text == "well done S1, and S1's move is next"


def test_teaching_sentence_survives():
    for sentence in (
        "Move your knight to f3, then castle on the king side and check the bishop",
        "Move your Knight to f3, then Castle on the King side and Check the Bishop.",
    ):
        t, _ = anonymise(_vtt(("Priya Raman", sentence), ("Meera Kapoor", "ok")))
        assert t.turns[0].text == sentence


def test_leak_counts():
    names = ["Priya Raman", "Arjun's iPhone", "iPad (Meera)"]
    clean, _ = anonymise(_vtt(("Priya Raman", "Great Arjun. Meera, what is a fork?"), ("Arjun's iPhone", "Yes")))
    text = "\n".join(f"{x.speaker}: {x.text}" for x in clean.turns)
    lc = leak_counts(text, "", names)
    assert (lc.caps, lc.name_hits) == (0, 0)
    # a lowercase variant and a name hidden behind a device word are both caught
    leaky = leak_counts("TUTOR: well done arav, and meera too", "", ["Aarav's iPad", "iPad (Meera)"])
    assert leaky.name_hits == 2 and leaky.names_hit == 2
    # chess vocabulary and template words are not counted
    assert leak_counts("Bishop takes the Rook. Rubric says so.", "Rubric", names).caps == 0
    assert leak_counts("Then Kavya moved.", "", names).caps == 1


def test_common_word_resembling_short_name_survives():
    # "need" is 75% similar to "Neel" but is ordinary English (wordfreq zipf ~6)
    t, _ = anonymise(_vtt(("Priya Raman", "you need to move. Need help?"), ("Neel Shah", "ok")))
    assert t.turns[0].text == "you need to move. Need help?"


def test_capitalised_mid_sentence_common_word_name_is_scrubbed():
    # "Need" is 75% similar to "Neel" and ordinary English, but capitalised mid-sentence it reads as the name
    t, _ = anonymise(_vtt(("Priya Raman", "well played Need, now the rook"), ("Neel Shah", "ok")))
    assert t.turns[0].text == "well played [NAME], now the rook"


def test_common_word_name_survives_in_lowercase_and_at_sentence_start():
    for text in ("you need to move", "Need help? Need a hint", "Good. Need to think! Need more time"):
        t, _ = anonymise(_vtt(("Priya Raman", text), ("Neel Shah", "ok")))
        assert t.turns[0].text == text


def test_vocative_common_word_name_at_sentence_start_is_scrubbed():
    for text, expected in (
        ("Need, your move.", "[NAME], your move."),
        ("Well done. Need? Are you there", "Well done. [NAME]? Are you there"),
        ("Need help with the rook", "Need help with the rook"),  # sentence start, no , ? ! after: kept
    ):
        t, _ = anonymise(_vtt(("Priya Raman", text), ("Neel Shah", "ok")))
        assert t.turns[0].text == expected


def test_capitalised_chess_term_resembling_a_name_survives():
    # "Castle" is 83% similar to "Castel" (loose band) but is on the chess allowlist
    t, _ = anonymise(_vtt(("Priya Raman", "now you can Castle on the King side"), ("Castel Roy", "ok")))
    assert t.turns[0].text == "now you can Castle on the King side"


def test_leak_counts_track_capitalised_hits():
    lc = leak_counts("TUTOR: you need it. well played Need", "", ["Neel Shah"])
    assert {w for w, _, _ in lc.hits} == {"need"} and lc.capitalised == {"need"}
    assert leak_counts("TUTOR: you need it", "", ["Neel Shah"]).capitalised == set()


def test_rare_mangled_name_is_scrubbed():
    # "tenvica" is 71% similar to "Tanvika": too far to claim it is S1, close enough to be a name
    t, _ = anonymise(_vtt(("Priya Raman", "good try tenvica, now the rook"), ("Tanvika Rao", "ok")))
    assert t.turns[0].text == "good try [NAME], now the rook"


def test_common_word_display_name_only_scrubbed_when_capitalised():
    raw = _vtt(("Priya Raman", "we will play now, Will, and you will win"), ("Will Smith", "ok"))
    t, _ = anonymise(raw)
    assert t.turns[0].text == "we will play now, S1, and you will win"
    assert leak_counts(t.turns[0].text, "", ["Will Smith"]).name_hits == 0


def test_over_scrub_share():
    raw = _vtt(("Priya Raman", "Kavya, sit down please now"), ("Will Smith", "ok"))
    t, _ = anonymise(raw)
    assert t.turns[0].text == "[NAME], sit down please now"
    assert over_scrub_share(raw, t) == 0.0  # a rare name scrubbed is not over-scrubbing
    # exact display-name match on a fairly common word (zipf 4.27, below WORD_NAME_ZIPF) still counts
    raw = _vtt(("Priya Raman", "at dawn we play and at dawn we win"), ("Dawn Smith", "ok"))
    t, _ = anonymise(raw)
    assert t.turns[0].text == "at S1 we play and at S1 we win"
    assert over_scrub_share(raw, t) == 2 / 9


def test_scrub_names_for_whisper_text():
    from cqa.transcript.privacy import scrub_names

    assert scrub_names("Arjun, what is a fork?") == "[NAME], what is a fork?"
    assert scrub_names("Very good Meera. Now move the Knight.") == "Very good [NAME]. Now move the Knight."
    assert scrub_names("Today we play. I think so.") == "Today we play. I think so."
    assert scrub_names("Where is Kavya?") == "Where is [NAME]?"


def _whisper(*texts: str) -> Transcript:
    return Transcript(
        source="whisper",
        turns=[Turn(start_s=i * 5, end_s=i * 5 + 4, speaker="UNKNOWN", text=x) for i, x in enumerate(texts)],
    )


def test_whisper_path_scrubs_vocatives_and_rare_words_at_sentence_start():
    # synthetic: "Hope" is a name that is also a common word (zipf >= 4); "Kavya" is not common English
    t, mapping = anonymise(
        _whisper(
            "Hope, is it?",  # vocative at a sentence start, common word: scrubbed (E27)
            "Okay. Kavya is ready now.",  # not common English, sentence start: scrubbed (E27)
            "Hope you enjoyed the class.",  # common word at a sentence start, no , ? ! after: kept
            "Very good, Hope!",  # vocative mid-sentence
        )
    )
    assert mapping == {}
    assert [x.text for x in t.turns] == [
        "[NAME], is it?",
        "Okay. [NAME] is ready now.",
        "Hope you enjoyed the class.",
        "Very good, [NAME]!",
    ]


def test_whisper_english_class_scrubs_lowercase_non_english_words():
    # synthetic: "tenvica" (a mangled name) written in lowercase by whisper is not an English word (zipf 0)
    t, _ = anonymise(_whisper("well done tenvica. now the knight takes the pawn").model_copy(update={"language": "en"}))
    assert t.turns[0].text == "well done [NAME]. now the knight takes the pawn"


def test_one_word_sentence_name_is_scrubbed_interjections_and_chess_survive():
    # synthetic: "Hope" is a name that is also a common word, said as a whole sentence (E28)
    lines = [
        "Hope. Okay, your mic is off.",
        "Very good. Hope",  # end of the turn
        "Okay. Checkmate. Right.",
        "Hope you enjoyed it. Wait.",
    ]
    t, _ = anonymise(_whisper(*lines).model_copy(update={"language": "en"}))
    assert [x.text for x in t.turns] == [
        "[NAME]. Okay, your mic is off.",
        "Very good. [NAME]",
        "Okay. Checkmate. Right.",
        "Hope you enjoyed it. Wait.",
    ]
    ml, _ = anonymise(_whisper("Hope. ok").model_copy(update={"language": "ml"}))
    assert ml.turns[0].text == "Hope. ok"  # English classes only


def test_malayalam_class_is_never_scrubbed_as_non_english():
    text = "നന്നായി ചെയ്തു, tenvica. ഇനി കുതിര"  # Malayalam script plus a Latin non-English token
    ml, _ = anonymise(_whisper(text).model_copy(update={"language": "ml"}))
    assert ml.turns[0].text == text  # no "not an English word" scrub for a non-English class
    en, _ = anonymise(_whisper(text).model_copy(update={"language": "en"}))
    assert "നന്നായി ചെയ്തു" in en.turns[0].text and "ഇനി കുതിര" in en.turns[0].text  # script untouched


def test_whisper_path_keeps_chess_and_pseudonyms():
    chess = (
        "Now, Knight to f3. Bishop takes the Rook! Castle King side, then Checkmate with the Queen. "
        "Pawn, Check, and Stalemate are different. Okay, Good, Very nice. Yes, What is En Passant?"
    )
    t, _ = anonymise(_whisper(chess, "TUTOR asks S1 about the Sicilian Defence."))
    assert [x.text for x in t.turns] == [chess, "TUTOR asks S1 about the Sicilian Defence."]


def test_vtt_path_scrubber_unchanged():
    # with display names the second pass keeps its earlier rules: a common word at a sentence start stays
    t, _ = anonymise(_vtt(("Priya Raman", "Hope, is it? Kavya is here."), ("Meera Kapoor", "ok")))
    assert t.turns[0].text == "Hope, is it? Kavya is here."


def test_scrub_keeps_chess_sentences_and_pseudonyms():
    from cqa.transcript.privacy import scrub_names

    chess = (
        "Now, Knight to f3. Bishop takes the Rook! Castle King side, then Checkmate with the Queen. "
        "Pawn, Check, and Stalemate are different. Okay, Good, Very nice. Yes, What is En Passant?"
    )
    assert scrub_names(chess) == chess
    assert scrub_names("TUTOR asks S1 and S2 about the Sicilian Defence.") == (
        "TUTOR asks S1 and S2 about the Sicilian Defence."
    )
    assert scrub_names("Well done, Rohan!") == "Well done, [NAME]!"
