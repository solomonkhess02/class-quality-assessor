"""Speaker role assignment: tutor = speaker with the most talk time."""

from __future__ import annotations

from collections import Counter

from cqa.schemas import Transcript

UNKNOWN = "UNKNOWN"


def talk_seconds(transcript: Transcript) -> Counter[str]:
    secs: Counter[str] = Counter()
    for t in transcript.turns:
        secs[t.speaker] += max(0.0, t.end_s - t.start_s)
    return secs


def assign_roles(transcript: Transcript) -> Transcript:
    secs = talk_seconds(transcript)
    secs.pop(UNKNOWN, None)
    tutor = secs.most_common(1)[0][0] if secs else None
    turns = []
    for t in transcript.turns:
        if t.speaker == UNKNOWN:
            role = "unknown"
        else:
            role = "tutor" if t.speaker == tutor else "student"
        turns.append(t.model_copy(update={"role": role}))
    return transcript.model_copy(update={"turns": turns})
