"""No-VTT path (E25): explicit language, language-keyed whisper cache, rapport from the judge's guarded fact when
speakers are unknown. Synthetic transcripts only."""

from cqa.schemas import FactEvidence, KeyFacts, Transcript, Turn

GREETING = "hello everyone how was your day at school"
CHAT = "I went to the park with my brother"
CONTENT = "okay let's start with the revision of forks"


def whisper_class(*lines, windows=((0.0, 480.0), (2400.0, 3000.0))) -> Transcript:
    """Speakers not identified, as on the whisper path without role tagging."""
    turns = [Turn(start_s=t, end_s=t + 5, speaker="UNKNOWN", role="unknown", text=txt) for t, txt in lines]
    return Transcript(source="whisper", duration_s=3000, turns=turns, windows=list(windows))


def ev(fact: str, value: str, ts: str, quote: str) -> FactEvidence:
    return FactEvidence(fact=fact, value=value, timestamp=ts, speaker="UNKNOWN", quote=quote)


def verify(kf: KeyFacts, tr: Transcript) -> KeyFacts:
    from cqa.judge.judge import verify_key_facts

    return verify_key_facts(kf, tr, 90)[0]


def test_judge_rapport_yes_with_verified_quote_before_content():
    tr = whisper_class((20, GREETING), (60, CHAT), (250, CONTENT))
    kf = KeyFacts(
        rapport_before_content="yes",
        evidence=[
            ev("rapport_before_content", "yes", "00:01:00", CHAT),
            ev("content_start", "yes", "00:04:10", CONTENT),
        ],
    )
    out = verify(kf, tr)
    assert (out.rapport_before_content, out.rapport_method, out.content_start) == ("yes", "judge", "00:04:10")
    assert out.warm_up_min is None


def test_judge_rapport_no_needs_content_start_in_opening():
    tr = whisper_class((20, GREETING), (60, CONTENT))
    no_quote = ev("rapport_before_content", "no", "00:01:00", CONTENT)
    with_start = KeyFacts(
        rapport_before_content="no", evidence=[no_quote, ev("content_start", "yes", "00:01:00", CONTENT)]
    )
    assert verify(with_start, tr).rapport_before_content == "no"
    without_start = KeyFacts(rapport_before_content="no", evidence=[no_quote])
    out = verify(without_start, tr)
    assert (out.rapport_before_content, out.rapport_method) == ("unknown", "none")


def test_judge_rapport_unknown_without_quote_or_after_content():
    tr = whisper_class((20, GREETING), (60, CONTENT), (200, CHAT))
    unquoted = KeyFacts(rapport_before_content="yes", evidence=[ev("content_start", "yes", "00:01:00", CONTENT)])
    assert verify(unquoted, tr).rapport_before_content == "unknown"
    late = KeyFacts(
        rapport_before_content="yes",
        evidence=[
            ev("rapport_before_content", "yes", "00:03:20", CHAT),
            ev("content_start", "yes", "00:01:00", CONTENT),
        ],
    )
    assert verify(late, tr).rapport_before_content == "unknown"


def test_speakers_known_still_measures_warm_up():
    from cqa.judge.judge import verify_key_facts

    turns = [
        Turn(start_s=10, end_s=15, speaker="TUTOR", role="tutor", text="Hello everyone, how are you?"),
        Turn(start_s=40, end_s=45, speaker="S1", role="student", text="hello ma'am good evening to you"),
        Turn(start_s=70, end_s=75, speaker="TUTOR", role="tutor", text=CONTENT),
    ]
    tr = Transcript(source="vtt", duration_s=3600, turns=turns)
    kf = KeyFacts(rapport_before_content="yes", evidence=[ev("content_start", "yes", "00:01:10", CONTENT)])
    out = verify_key_facts(kf, tr, 90)[0]
    assert (out.rapport_before_content, out.rapport_method) == ("no", "warm_up")


def test_rapport_rule_only_in_user_prompt_without_speakers():
    from cqa.features.text_stats import compute_text_stats
    from cqa.judge import prompts

    vtt = Transcript(source="vtt", duration_s=3000,
                     turns=[Turn(start_s=0, end_s=5, speaker="TUTOR", role="tutor", text=GREETING)])  # fmt: skip
    stats = compute_text_stats(vtt, 10, 10)
    assert prompts.RAPPORT_RULE not in prompts.user_prompt(vtt, stats, None, 10, 10)
    assert prompts.RAPPORT_RULE in prompts.user_prompt(whisper_class((20, GREETING)), stats, None, 10, 10)
    assert "rapport_before_content" not in prompts.system_prompt().split("key_facts")[0]


def test_whisper_cache_name_keys_language_and_windows():
    from cqa.transcript.whisper import cache_name

    en, ml = cache_name([(0.0, 480.0)], "en"), cache_name([(0.0, 480.0)], "ml")
    assert en != ml and en.startswith("whisper_en_") and en.endswith(".json")
    assert cache_name(None, "en") == "whisper_en_full.json"
