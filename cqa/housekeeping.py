"""Public-deployment housekeeping: one assessment at a time, and no media kept after it finishes."""

from __future__ import annotations

import re
import shutil
import threading
import time
from pathlib import Path

from cqa.config import get_settings
from cqa.ingest import sha256_file

# Module-level, so it survives Streamlit reruns and is shared by every session in the server process
RUN_LOCK = threading.Lock()
KEEP_IN_CLASS_DIR = "assessment*.json"
CLASS_DIR = re.compile(r"[0-9a-f]{16}")  # cqa.ingest.cache_dir_for: first 16 hex chars of the sha256


def _remove(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        p.unlink(missing_ok=True)


def delete_run_files(video: Path | None, vtt: Path | None) -> None:
    """Delete the inputs and everything derived from them (audio, frames, raw transcripts), keeping only the
    assessment JSON (pseudonymised text and numbers). No-op unless DELETE_MEDIA_AFTER_RUN is set."""
    s = get_settings()
    if not s.delete_media_after_run:
        return
    source = video or vtt
    if source is not None and source.exists():
        class_dir = s.cache_dir / sha256_file(source)[:16]  # same key as cqa.ingest.cache_dir_for
        if class_dir.is_dir():
            keep = set(class_dir.glob(KEEP_IN_CLASS_DIR))
            for p in class_dir.iterdir():
                if p not in keep:
                    _remove(p)
    for p in (video, vtt):
        if p is not None:
            p.unlink(missing_ok=True)


def purge_old(now: float | None = None) -> int:
    """Remove top-level cache entries (and LLM cache files) older than CACHE_MAX_AGE_H. Returns the count."""
    s = get_settings()
    if s.cache_max_age_h <= 0 or not s.cache_dir.is_dir():
        return 0
    cutoff = (now or time.time()) - s.cache_max_age_h * 3600
    removed = 0
    for entry in s.cache_dir.iterdir():
        # uploads/, downloads/ and llm/ hold many files of different ages; class dirs go as a whole;
        # anything else (e.g. models) is left alone
        if entry.name in ("uploads", "downloads", "llm"):
            children = list(entry.iterdir())
        elif entry.is_dir() and CLASS_DIR.fullmatch(entry.name):
            children = [entry]
        else:
            continue
        for p in children:
            if p.stat().st_mtime < cutoff:
                _remove(p)
                removed += 1
    return removed
