"""Zoom WebVTT parser."""

from __future__ import annotations

import re

from cqa.schemas import Transcript, Turn

_TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_SPEAKER_RE = re.compile(r"^([^:]{1,60}):\s+(.*)$")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def _seconds(ts: str) -> float:
    m = _TIME_RE.fullmatch(ts.strip())
    if not m:
        raise ValueError(f"bad VTT timestamp {ts!r}")
    h, mi, s, ms = m.groups()
    return int(h or 0) * 3600 + int(mi) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def _split_speaker(line: str) -> tuple[str | None, str]:
    m = _SPEAKER_RE.match(line)
    if m:
        prefix = m.group(1).strip()
        if prefix and len(prefix.split()) <= 8 and "http" not in prefix.lower():
            return prefix, m.group(2).strip()
    return None, line.strip()


def merge_turns(turns: list[Turn], gap_s: float = 1.5) -> list[Turn]:
    """Merge consecutive same-speaker turns separated by less than gap_s seconds."""
    out: list[Turn] = []
    for t in turns:
        prev = out[-1] if out else None
        if prev and prev.speaker == t.speaker and t.start_s - prev.end_s < gap_s:
            out[-1] = prev.model_copy(
                update={"end_s": max(prev.end_s, t.end_s), "text": f"{prev.text} {t.text}".strip()}
            )
        else:
            out.append(t.model_copy())
    return out


def parse_vtt(text: str, merge_gap_s: float = 1.5) -> Transcript:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n").translate(_QUOTES)
    turns: list[Turn] = []
    speaker = "UNKNOWN"
    for block in re.split(r"\n\s*\n", text):
        lines = [ln for ln in block.split("\n") if ln.strip()]
        idx = next((i for i, ln in enumerate(lines) if "-->" in ln), None)
        if idx is None:
            continue  # header, NOTE or STYLE block
        start_raw, end_raw = lines[idx].split("-->", 1)
        start, end = _seconds(start_raw), _seconds(end_raw.split()[0])
        body = lines[idx + 1 :]
        if not body:
            continue
        who, first = _split_speaker(body[0])
        if who:
            speaker = who
        cue_text = " ".join([first, *(ln.strip() for ln in body[1:])]).strip()
        if cue_text:
            turns.append(Turn(start_s=start, end_s=end, speaker=speaker, text=cue_text))
    turns = merge_turns(turns, merge_gap_s)
    duration = max((t.end_s for t in turns), default=0.0)
    return Transcript(source="vtt", duration_s=duration, turns=turns)
