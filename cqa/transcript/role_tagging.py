"""LLM role tagging for whisper transcripts (P2.2): TUTOR or STUDENT per segment. Prepared; validated against the
Zoom VTT speakers with scripts/role_tag_check.py before it is switched on (WHISPER_ROLE_TAGGING).

Privacy: call it on the anonymised transcript only (names already scrubbed to [NAME]); no audio is sent.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from cqa.judge.llm import call_json
from cqa.schemas import Transcript, Turn, format_ts

SYSTEM = """You label who is speaking in a machine transcript of an online chess class for children (one adult
tutor, several children). Speaker labels were lost; names are replaced by [NAME]. For every numbered line decide
whether the tutor or a child is speaking. The tutor explains, asks questions, gives instructions and praise; children
answer, ask, or react briefly. Return a json object {"roles": [{"i": 0, "role": "tutor"}, {"i": 1, "role": "student"},
...]} with one entry per line, role "tutor" or "student"."""


class RoleTag(BaseModel):
    i: int
    role: Literal["tutor", "student"]


class RoleTags(BaseModel):
    roles: list[RoleTag] = Field(default_factory=list)


def role_messages(transcript: Transcript) -> list[dict]:
    lines = "\n".join(f"{i}. [{format_ts(t.start_s)}] {t.text}" for i, t in enumerate(transcript.turns))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Transcript lines:\n{lines}"}]


def tag_roles(transcript: Transcript, run_index: int = 0) -> tuple[Transcript, dict]:
    """Assign tutor / student roles and TUTOR / STUDENT speakers; lines the model skipped stay unknown."""
    out, meta = call_json(role_messages(transcript), RoleTags, run_index=run_index, temperature=0.0)
    by_line = {r.i: r.role for r in out.roles}
    turns = []
    for i, t in enumerate(transcript.turns):
        role = by_line.get(i, "unknown")
        speaker = {"tutor": "TUTOR", "student": "STUDENT"}.get(role, t.speaker)
        turns.append(t.model_copy(update={"role": role, "speaker": speaker}))
    meta["role_lines_missing"] = len(transcript.turns) - len(by_line.keys() & set(range(len(transcript.turns))))
    return transcript.model_copy(update={"turns": turns}), meta


def reference_roles(segments: list[Turn], vtt_turns: list[Turn]) -> list[str | None]:
    """Role of the VTT turn that overlaps each whisper segment the most (VTT roles from assign_roles); None when no
    VTT turn overlaps. Used to measure role-tagging accuracy on classes that have both."""
    refs: list[str | None] = []
    for seg in segments:
        best, best_overlap = None, 0.0
        for v in vtt_turns:
            overlap = min(seg.end_s, v.end_s) - max(seg.start_s, v.start_s)
            if overlap > best_overlap:
                best, best_overlap = v.role, overlap
        refs.append(best)
    return refs
