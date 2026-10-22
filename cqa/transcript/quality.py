"""Transcript quality check for both paths (P2.2). An unusable Zoom VTT (e.g. English ASR applied to Malayalam
speech, EXPERIMENTS E2) triggers the whisper fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from wordfreq import zipf_frequency

from cqa.schemas import Transcript

# Fixed from definitions, not fitted (one unusable example exists): unusable when more than one word in ten is not
# an English word at all, or when there is almost no speech.
MAX_NON_ENGLISH_SHARE = 0.10
MIN_WORDS_PER_MIN = 10.0
_LATIN_WORD = re.compile(r"[A-Za-z']+")
ENGLISH = {"en", "english", "unknown"}  # Zoom VTTs carry no language; treat them as claimed English


@dataclass
class TranscriptQuality:
    usable: bool
    words_per_min: float  # tutor words if speakers are known, else all words
    non_english_share: float | None  # share of Latin-script words with no English frequency; None if not English
    language: str
    reasons: list[str] = field(default_factory=list)


def check_quality(t: Transcript) -> TranscriptQuality:
    minutes = (sum(b - a for a, b in t.windows) if t.windows else t.duration_s) / 60
    tutor = [x.text for x in t.turns if x.role == "tutor"] or [x.text for x in t.turns]
    text = " ".join(tutor)
    n_words = len(text.split())
    wpm = n_words / minutes if minutes > 0 else 0.0
    reasons = []
    share = None
    if t.language.lower() in ENGLISH:
        latin = [w.lower() for w in _LATIN_WORD.findall(text)]
        share = sum(zipf_frequency(w, "en") == 0 for w in latin) / len(latin) if latin else 1.0
        if share > MAX_NON_ENGLISH_SHARE:
            reasons.append(f"{share:.0%} of the words are not English (speech recognition in the wrong language?)")
    if wpm < MIN_WORDS_PER_MIN:
        reasons.append(f"only {wpm:.0f} words per minute were transcribed")
    return TranscriptQuality(
        not reasons, round(wpm, 1), None if share is None else round(share, 3), t.language, reasons
    )
