"""Transcript statistics. Evidence for the judge, not a classifier (EXPERIMENTS E2)."""

from __future__ import annotations

import re

from cqa.schemas import TextStats, Transcript, Turn

# "?" counts are unreliable when the ASR punctuates little, or when it ends almost every
# sentence with "." (good/B435: 5% "?" vs >= 21% elsewhere, EXPERIMENTS E4)
SPARSE_PUNCT_PER_100_WORDS = 10.0
MIN_QUESTION_SHARE = 0.10

CONNECTIVITY_RE = re.compile(
    r"can(?:'t|not)? (?:you |u )?hear|not audible|on mute|unmute|network|internet|connection"
    r"|not visible|can(?:'t|not)? see (?:me|you|my|the screen)|(?:screen|video) is (?:stuck|frozen)"
    r"|lagging|voice is breaking|breaking up|disconnected",
    re.IGNORECASE,
)


def window(transcript: Transcript, start_s: float, end_s: float) -> list[Turn]:
    """Turns that start inside [start_s, end_s)."""
    return [t for t in transcript.turns if start_s <= t.start_s < end_s]


def _longest_tutor_run(turns: list[Turn]) -> float:
    longest, run_start, run_end = 0.0, None, None
    for t in turns:
        if t.role == "tutor":
            run_start = t.start_s if run_start is None else run_start
            run_end = t.end_s
            longest = max(longest, run_end - run_start)
        elif t.role == "student":
            run_start = run_end = None
    return longest


def compute_text_stats(transcript: Transcript, opening_min: float, closing_min: float) -> TextStats:
    turns = transcript.turns
    duration_s = transcript.duration_s or max((t.end_s for t in turns), default=0.0)
    minutes = duration_s / 60
    per_min = (lambda n: n / minutes) if minutes > 0 else (lambda n: 0.0)

    tutor = [t for t in turns if t.role == "tutor"]
    students = [t for t in turns if t.role == "student"]
    tutor_secs = sum(t.end_s - t.start_s for t in tutor)
    student_secs = sum(t.end_s - t.start_s for t in students)
    talk = tutor_secs + student_secs

    tutor_text = " ".join(t.text for t in tutor)
    n_words = len(tutor_text.split())
    n_punct = len(re.findall(r"[.?!,]", tutor_text))
    n_ends = len(re.findall(r"[.?!]", tutor_text))
    sparse = n_words > 0 and (
        n_punct * 100 / n_words < SPARSE_PUNCT_PER_100_WORDS
        or (n_ends >= 20 and tutor_text.count("?") / n_ends < MIN_QUESTION_SHARE)
    )

    return TextStats(
        duration_min=round(minutes, 2),
        tutor_talk_share=round(tutor_secs / talk, 3) if talk else 0.0,
        student_turns_per_min=round(per_min(len(students)), 2),
        distinct_students_spoke=len({t.speaker for t in students}),
        tutor_questions_per_min=round(per_min(tutor_text.count("?")), 2),
        sparse_punctuation=sparse,
        longest_tutor_monologue_s=round(_longest_tutor_run(turns), 1),
        connectivity_mentions=sum(len(CONNECTIVITY_RE.findall(t.text)) for t in turns),
        opening_turns=len(window(transcript, 0, opening_min * 60)),
        closing_turns=len(window(transcript, duration_s - closing_min * 60, float("inf"))),
    )
