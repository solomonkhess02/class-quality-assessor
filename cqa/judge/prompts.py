"""Judge prompts. Only pseudonymised transcript text and numeric features go in here."""

from __future__ import annotations

import json

from cqa.judge.rubric import LLM_CRITERIA
from cqa.schemas import TextStats, Transcript, VisualReport, format_ts

EXAMPLE = {
    "transcript_usable": True,
    "transcript_note": "",
    "criteria": [
        {
            "id": "opening_rapport",
            "score": 3,
            "insufficient_evidence": False,
            "observations": [
                "Tutor greets S1 and S2 by name and asks about their school day before starting.",
                "No plan for the class is stated before the first puzzle.",
            ],
            "evidence": [
                {"timestamp": "00:00:42", "speaker": "TUTOR", "quote": "S1, how was your exam yesterday?"},
                {"timestamp": "00:03:10", "speaker": "TUTOR", "quote": "okay let's start with the puzzle"},
            ],
            "improvement": "Before the first puzzle, tell the children the topic and what they will learn today.",
        },
        {"id": "closure_homework", "score": None, "insufficient_evidence": True,
         "observations": ["The transcript ends mid-explanation, so the closing is not visible."],
         "evidence": [], "improvement": ""},
    ],
    "key_facts": {
        "homework_assigned": True,
        "homework_what": "solve 5 fork puzzles on the portal",
        "new_joiners": "none",
        "children_greeted_by_name": 2,
        "recap_present": False,
        "evidence": [
            {"fact": "homework_assigned", "timestamp": "00:52:10", "speaker": "TUTOR",
             "quote": "for homework solve the five fork puzzles"},
            {"fact": "children_greeted_by_name", "timestamp": "00:00:42", "speaker": "TUTOR",
             "quote": "S1, how was your exam yesterday?"},
        ],
    },
    "summary": "Two or three sentences on the class overall.",
    "strengths": ["Specific strength with a timestamp, e.g. names every child in the warm-up (00:00:42)."],
    "improvements": ["Specific, actionable suggestion, e.g. assign one named homework task in the last 5 minutes."],
}  # fmt: skip


def _rubric_text() -> str:
    parts = []
    for c in LLM_CRITERIA:
        anchors = "\n".join(f"    {k} = {v}" for k, v in sorted(c.anchors.items()))
        parts.append(
            f'- id "{c.id}" ({c.name}, weight {c.weight}). Looks good when: {c.looks_good} '
            f"Main evidence: {c.evidence}.\n{anchors}"
        )
    return "\n".join(parts)


def system_prompt() -> str:
    ids = ", ".join(f'"{c.id}"' for c in LLM_CRITERIA)
    return f"""You are an experienced academic quality reviewer for 8x8, an online chess academy for children.
You review recordings of live group classes (one tutor, several children aged roughly 5-14) and grade
how well the tutor ran the class, the way the academy's quality team does.

You receive a machine transcript (Zoom or Whisper speech recognition, so expect recognition errors,
especially on chess terms and children's speech), basic statistics computed from it, and video metrics
computed from sampled frames. Names are pseudonymised: TUTOR is the tutor, S1, S2, ... are children,
UNKNOWN means the speaker is not identified, [NAME] is a name that could not be linked to a speaker.

Rubric. Score each criterion 1-4 using these anchors:
{_rubric_text()}

Rules:
1. Judge only from the transcript and numbers given. Every observation must be specific to this class
   (who did what, when). No generic praise or filler such as "the tutor was good".
2. Support each scored criterion with 1-3 evidence quotes copied verbatim from the transcript lines
   (at most 25 words each), with the timestamp of the line they come from and its speaker label.
   Quotes are checked automatically against the transcript; invented or paraphrased quotes are discarded.
3. If the transcript does not let you judge a criterion (e.g. it is cut off, or the relevant part is
   missing or unintelligible), set "score": null and "insufficient_evidence": true instead of guessing.
4. Absence matters: if the closing window contains no homework, say so and score accordingly.
5. "improvement" is one concrete, actionable suggestion for the tutor per criterion.
6. Set "transcript_usable": false (and explain in "transcript_note") when the transcript is mostly
   unintelligible, e.g. speech recognition applied to the wrong language. Then mark criteria you cannot
   judge as insufficient_evidence.
7. If "sparse_punctuation" is true in the statistics, ignore tutor_questions_per_min (question marks
   are missing from that transcript).
8. Do not score visual presence; it is measured from video frames separately. Use the video metrics only
   as context for technical_quality.
9. Also report "key_facts": homework_assigned (specific work set for after class; say what in homework_what),
   new_joiners ("invited" if a new child is asked to introduce themselves, "not_invited" if a new child joins
   and is not, "none" if no new child is apparent), children_greeted_by_name (distinct children greeted or
   chatted with by name before teaching starts), recap_present (the tutor summarises what was learned near the
   end). Give one verbatim quote in key_facts.evidence for each true or non-zero fact, with its fact name.
   A true fact without a verifiable quote is counted as false.

Return a single json object with exactly these keys: "transcript_usable", "transcript_note",
"criteria" (one entry for each of: {ids}), "key_facts", "summary", "strengths", "improvements".
Example of the format (content is illustrative only):
{json.dumps(EXAMPLE, indent=1)}"""


def format_transcript(transcript: Transcript, opening_min: float, closing_min: float) -> str:
    open_end = opening_min * 60
    close_start = max(transcript.duration_s - closing_min * 60, open_end)
    lines = [f"=== OPENING WINDOW (first {opening_min:g} min) ==="]
    section = "opening"
    for t in transcript.turns:
        if section == "opening" and t.start_s >= open_end:
            lines.append("=== MIDDLE OF CLASS ===")
            section = "middle"
        if section != "closing" and t.start_s >= close_start:
            lines.append(f"=== CLOSING WINDOW (last {closing_min:g} min) ===")
            section = "closing"
        lines.append(f"[{format_ts(t.start_s)}] {t.speaker}: {t.text}")
    return "\n".join(lines)


def user_prompt(
    transcript: Transcript,
    stats: TextStats,
    visual: VisualReport | None,
    opening_min: float,
    closing_min: float,
) -> str:
    visual_text = json.dumps(visual.model_dump()) if visual else "not available"
    return (
        f"Transcript source: {transcript.source}; class length {stats.duration_min:.0f} min.\n\n"
        f"Statistics (json): {json.dumps(stats.model_dump())}\n"
        "(tutor_talk_share is a fraction of talk time; student_turns_per_min counts merged student turns.)\n\n"
        f"Video metrics (json, percentages of sampled frames): {visual_text}\n\n"
        f"Transcript:\n{format_transcript(transcript, opening_min, closing_min)}\n\n"
        "Assess this class and return the json object."
    )


def build_messages(
    transcript: Transcript,
    stats: TextStats,
    visual: VisualReport | None,
    opening_min: float,
    closing_min: float,
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_prompt(transcript, stats, visual, opening_min, closing_min)},
    ]
