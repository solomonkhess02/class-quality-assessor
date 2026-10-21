"""Labelled dataset discovery: pairs each recording with its VTT. Only keys and labels leave this module."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from cqa.transcript.vtt import parse_vtt

FOLDER_LABELS = {
    "excellent": "Excellent",
    "good": "Good",
    "average": "Average",
    "weak": "Poor",
    "quality issue": "Poor",
    "demo": None,
}
_ID_RE = re.compile(r"B\d{3}")


@dataclass
class ClassItem:
    key: str  # "<folder>/<class id>", e.g. "good/B435"; demos are "demo/1", "demo/2"
    folder: str
    label: str | None  # None for demo classes
    technical_flag: bool  # "quality issue" folder
    video: Path | None
    vtt: Path | None


def _vtt_end(vtt: Path) -> float:
    return parse_vtt(vtt.read_text(encoding="utf-8")).duration_s


def discover(classes_dir: Path, probe_duration=None) -> list[ClassItem]:
    """Pair mp4 and vtt per folder by class ID. Files without an ID (demos) are paired by
    closest duration when probe_duration is given, else by file-size order."""
    items: list[ClassItem] = []
    for folder in sorted(p for p in classes_dir.iterdir() if p.is_dir()):
        name = folder.name.lower()
        if name not in FOLDER_LABELS:
            continue
        videos = sorted(folder.glob("*.mp4"), key=lambda p: p.stat().st_size)
        vtts = sorted(folder.glob("*.vtt"), key=lambda p: p.stat().st_size)
        pairs: list[tuple[str, Path | None, Path | None]] = []
        ids = {m.group(0) for p in videos + vtts if (m := _ID_RE.search(p.name))}
        if ids and all(_ID_RE.search(p.name) for p in videos + vtts):
            for cid in sorted(ids):
                v = next((p for p in videos if cid in p.name), None)
                t = next((p for p in vtts if cid in p.name), None)
                pairs.append((cid, v, t))
        else:
            remaining = list(vtts)
            for i, v in enumerate(videos, start=1):
                t = None
                if remaining:
                    if probe_duration is not None:
                        dur = probe_duration(v)
                        t = min(remaining, key=lambda p: abs(_vtt_end(p) - dur))
                    else:
                        t = remaining[0]
                    remaining.remove(t)
                pairs.append((str(i), v, t))
        for cid, v, t in pairs:
            items.append(
                ClassItem(
                    key=f"{name}/{cid}",
                    folder=name,
                    label=FOLDER_LABELS[name],
                    technical_flag=name == "quality issue",
                    video=v,
                    vtt=t,
                )
            )
    return items
