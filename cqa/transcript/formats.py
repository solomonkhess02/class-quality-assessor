"""Transcript files other than a Zoom VTT: SRT, plain-text "Name: text", and VTT/SRT without speaker names.

Everything is normalised to a canonical VTT ("Speaker: text" cues) so the pipeline's VTT path reads it unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from cqa.schemas import Transcript, Turn, format_ts
from cqa.transcript.roles import UNKNOWN
from cqa.transcript.vtt import WINDOWS_NOTE, merge_turns, parse_vtt

TRANSCRIPT_TYPES = ("vtt", "srt", "txt")
WORDS_PER_MIN = 150.0  # plain text without timestamps: estimated speaking rate
# Plain-text line: optional "[hh:mm:ss]" / "hh:mm:ss" / "(mm:ss)" prefix, then "Name: text"
_TXT_RE = re.compile(
    r"^\s*[\[(]?(?P<ts>(?:\d{1,2}:)?\d{1,2}:\d{2})?[\])]?\s*(?:-\s*)?(?P<who>[^:]{1,60}):\s+(?P<text>.+)$"
)
_SRT_INDEX = re.compile(r"^\d+$")
_SRT_TIME = re.compile(r"(\d{1,2}:\d{2}:\d{2}),(\d{1,3})")


class TranscriptFormatError(ValueError):
    """The file cannot be read as a transcript; the message is shown to the user as is."""


@dataclass
class ParsedTranscript:
    transcript: Transcript
    fmt: str  # "vtt" | "srt" | "txt"
    has_speakers: bool
    notes: list[str] = field(default_factory=list)


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1252", errors="replace")


def detect_format(text: str, suffix: str = "") -> str:
    head = text.lstrip("﻿ \t\r\n")[:200]
    if head.upper().startswith("WEBVTT"):
        return "vtt"
    if "-->" in text[:2000]:
        return "srt" if _SRT_TIME.search(text[:2000]) else "vtt"
    return "txt" if suffix.lstrip(".").lower() not in ("vtt", "srt") else suffix.lstrip(".").lower()


def srt_to_vtt(text: str) -> str:
    """SRT differs from VTT in the comma before milliseconds and the numeric cue index lines."""
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if not _SRT_INDEX.match(ln.strip())]
    return "WEBVTT\n\n" + _SRT_TIME.sub(r"\1.\2", "\n".join(lines))


def parse_plain_text(text: str) -> tuple[Transcript, bool]:
    """'Name: text' lines, with or without timestamps. Returns (transcript, timestamps estimated).
    Lines without a speaker prefix continue the previous speaker's turn."""
    rows: list[tuple[float | None, str, str]] = []
    for line in text.lstrip("﻿").splitlines():
        if not line.strip():
            continue
        m = _TXT_RE.match(line)
        who = m.group("who").strip() if m else ""
        if m and len(who.split()) <= 8 and "http" not in who.lower():
            ts = m.group("ts")
            rows.append((_seconds(ts) if ts else None, who, m.group("text").strip()))
        elif rows:
            t, who, prev = rows[-1]
            rows[-1] = (t, who, f"{prev} {line.strip()}")
    if not rows:
        raise TranscriptFormatError(
            'Could not read the transcript: expected lines like "Name: text" (optionally with a [hh:mm:ss] time).'
        )
    estimated = any(t is None for t, _, _ in rows)
    turns: list[Turn] = []
    clock = 0.0
    for i, (t, who, body) in enumerate(rows):
        length = max(2.0, len(body.split()) / WORDS_PER_MIN * 60)
        start = clock if estimated else t  # mixed timed/untimed lines: estimate all, keep them monotonic
        if not estimated and i + 1 < len(rows):
            end = max(start + 0.5, min(start + length, rows[i + 1][0]))  # type: ignore[type-var]
        else:
            end = start + length  # type: ignore[operator]
        turns.append(Turn(start_s=start, end_s=end, speaker=who, text=body))  # type: ignore[arg-type]
        clock = end
    turns = merge_turns(turns)
    return Transcript(source="vtt", duration_s=max(t.end_s for t in turns), turns=turns), estimated


def _seconds(ts: str) -> float:
    parts = [int(p) for p in ts.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_transcript_bytes(data: bytes, suffix: str = "") -> ParsedTranscript:
    """Read a .vtt, .srt or plain-text transcript. Raises TranscriptFormatError with a user-facing message."""
    suffix = suffix.lower()
    if suffix and suffix.lstrip(".") not in TRANSCRIPT_TYPES:
        raise TranscriptFormatError(
            f"Unsupported transcript type '{suffix}'. Please upload a .vtt, .srt or .txt transcript."
        )
    text = _decode(data)
    if not text.strip():
        raise TranscriptFormatError("The transcript file is empty.")
    fmt = detect_format(text, suffix)
    notes: list[str] = []
    try:
        if fmt == "txt":
            transcript, estimated = parse_plain_text(text)
            if estimated:
                notes.append("The text transcript has no timestamps; times are estimated from word counts.")
        else:
            transcript = parse_vtt(srt_to_vtt(text) if fmt == "srt" else text)
    except TranscriptFormatError:
        raise
    except ValueError as e:
        raise TranscriptFormatError(f"Could not read the {fmt.upper()} transcript: {e}.") from e
    if not transcript.turns:
        raise TranscriptFormatError(f"The {fmt.upper()} transcript contains no captions.")
    has_speakers = any(t.speaker != UNKNOWN for t in transcript.turns)
    return ParsedTranscript(transcript=transcript, fmt=fmt, has_speakers=has_speakers, notes=notes)


def to_vtt(transcript: Transcript) -> str:
    """Canonical VTT that cqa.transcript.vtt.parse_vtt reads back to the same turns (and windows)."""
    out = ["WEBVTT", ""]
    if transcript.windows:  # so the judge is told which parts were cut out (prompts.windows_note)
        out += [f"{WINDOWS_NOTE} " + ", ".join(f"{a:.3f}-{b:.3f}" for a, b in transcript.windows), ""]
    for i, t in enumerate(transcript.turns, 1):
        prefix = "" if t.speaker == UNKNOWN else f"{t.speaker}: "
        out += [str(i), f"{_vtt_ts(t.start_s)} --> {_vtt_ts(t.end_s)}", f"{prefix}{t.text}", ""]
    return "\n".join(out)


def _vtt_ts(seconds: float) -> str:
    ms = round(seconds * 1000)
    return f"{format_ts(ms // 1000)}.{ms % 1000:03d}"


def restrict_to_windows(transcript: Transcript, windows: list[tuple[float, float]]) -> Transcript:
    """Keep only the turns that start inside one of the [start, end] windows (recordings over the length limit),
    and record the windows, as the whisper path does."""
    turns = [t for t in transcript.turns if any(a <= t.start_s < b for a, b in windows)]
    return transcript.model_copy(update={"turns": turns, "windows": list(windows)})
