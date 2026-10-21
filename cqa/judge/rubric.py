"""Rubric derived from the organiser's feedback (EXPERIMENTS E1, docs/ARCHITECTURE.md)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    looks_good: str
    evidence: str
    weight: float
    scored_by: Literal["llm", "code"]
    anchors: dict[int, str]


CRITERIA: list[Criterion] = [
    Criterion(
        id="opening_rapport",
        name="Opening and rapport",
        looks_good="Warm greeting, chats with children before content, invites new joiners to "
        "introduce themselves, sets the plan for the class.",
        evidence="First ~10 minutes",
        weight=0.20,
        scored_by="llm",
        anchors={
            1: "Starts teaching straight away or after a perfunctory hello; no rapport, no plan.",
            2: "Greets the class but little personal conversation; new joiners not introduced or plan unclear.",
            3: "Warm greeting, some personal chat with several children, plan or topic set before teaching.",
            4: "Greets children by name, genuine chat with most children, new joiners introduced, "
            "clear plan; children are visibly at ease.",
        },
    ),
    Criterion(
        id="individual_engagement",
        name="Individual engagement",
        looks_good="Asks children questions by name, involves every child, children respond often.",
        evidence="Whole transcript and statistics",
        weight=0.25,
        scored_by="llm",
        anchors={
            1: "Lecture style; questions are rare or rhetorical; few or no children speak.",
            2: "Some questions, mostly to the whole group; a few children dominate or many stay silent.",
            3: "Regular questions to named children; most children answer at least once.",
            4: "Every child is drawn in by name repeatedly; frequent back-and-forth; tutor builds on answers.",
        },
    ),
    Criterion(
        id="communication",
        name="Energy and warmth",
        looks_good="Encouraging, praises effort, patient, lively and pleasant tone.",
        evidence="Tone across the transcript",
        weight=0.15,
        scored_by="llm",
        anchors={
            1: "Flat, impatient or discouraging; no praise.",
            2: "Polite but low energy; praise is generic or rare.",
            3: "Friendly and patient; specific praise for effort or good answers.",
            4: "Consistently lively and warm; specific encouragement; handles mistakes positively.",
        },
    ),
    Criterion(
        id="closure_homework",
        name="Closure and homework",
        looks_good="Recap of what was learned, explicit homework assigned, proper goodbye.",
        evidence="Last ~10 minutes",
        weight=0.15,
        scored_by="llm",
        anchors={
            1: "Class just stops; no recap, no homework, no proper goodbye.",
            2: "Only one of recap / homework / goodbye, or they are vague.",
            3: "Two of the three are clear (e.g. homework assigned and a proper goodbye).",
            4: "Clear recap, specific homework explained, warm goodbye with a look ahead to next class.",
        },
    ),
    Criterion(
        id="visual_presence",
        name="Visual presence",
        looks_good="Branded 8x8 background, tutor face visible, camera on, good lighting.",
        evidence="Sampled video frames (OpenCV metrics)",
        weight=0.15,
        scored_by="code",
        anchors={
            1: "Face rarely visible and/or very dark or black video for long stretches.",
            2: "Face missing in a large share of frames or noticeable lighting problems.",
            3: "Face visible most of the time with acceptable lighting.",
            4: "Face visible almost throughout, well lit, sharp video.",
        },
    ),
    Criterion(
        id="technical_quality",
        name="Technical quality",
        looks_good="Clear audio and video, no repeated connectivity problems.",
        evidence="Connectivity phrases in the transcript plus frame metrics",
        weight=0.10,
        scored_by="llm",
        anchors={
            1: "Repeated audio/video/connection problems that disrupt the class.",
            2: "Several problems that cost noticeable time.",
            3: "Minor, quickly resolved problems.",
            4: "No noticeable technical problems.",
        },
    ),
]

BY_ID = {c.id: c for c in CRITERIA}
LLM_CRITERIA = [c for c in CRITERIA if c.scored_by == "llm"]
