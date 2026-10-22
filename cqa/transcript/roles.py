"""Speaker role assignment: tutor = speaker with the most talk time, plus any second account of the same person."""

from __future__ import annotations

from collections import Counter

from rapidfuzz import fuzz

from cqa.schemas import Transcript
from cqa.transcript.privacy import _base_name

UNKNOWN = "UNKNOWN"
SAME_PERSON_MIN = 85  # display names this similar (device suffix stripped) are the tutor's second account (E26)


def tutor_accounts(transcript: Transcript) -> set[str]:
    """The tutor (most talk time) and speakers whose display name matches the tutor's at >= SAME_PERSON_MIN, e.g.
    "Priya Raman" and "Priya Raman's iPhone" when the tutor joins from two devices."""
    secs = talk_seconds(transcript)
    secs.pop(UNKNOWN, None)
    if not secs:
        return set()
    tutor = secs.most_common(1)[0][0]
    base = _base_name(tutor).lower()
    return {sp for sp in secs if sp == tutor or fuzz.ratio(_base_name(sp).lower(), base) >= SAME_PERSON_MIN}


def talk_seconds(transcript: Transcript) -> Counter[str]:
    secs: Counter[str] = Counter()
    for t in transcript.turns:
        secs[t.speaker] += max(0.0, t.end_s - t.start_s)
    return secs


def assign_roles(transcript: Transcript) -> Transcript:
    tutors = tutor_accounts(transcript)
    turns = []
    for t in transcript.turns:
        if t.speaker == UNKNOWN:
            role = "unknown"
        else:
            role = "tutor" if t.speaker in tutors else "student"
        turns.append(t.model_copy(update={"role": role}))
    return transcript.model_copy(update={"turns": turns})
