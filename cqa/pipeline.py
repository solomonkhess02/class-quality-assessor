"""End-to-end orchestration with per-stage caching under .cache/<hash>/."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from cqa import media
from cqa.config import get_settings
from cqa.features.text_stats import compute_text_stats
from cqa.features.visual import aggregate, cached_frame_metrics
from cqa.ingest import cache_dir_for, sha256_file
from cqa.judge.judge import judge
from cqa.schemas import Assessment, JudgeOutput, Transcript, VisualReport
from cqa.scoring import assess
from cqa.transcript.privacy import anonymise
from cqa.transcript.roles import assign_roles
from cqa.transcript.vtt import parse_vtt
from cqa.transcript.whisper import transcribe_media

Progress = Callable[[str, str], None]


def _noop(stage: str, message: str) -> None:
    pass


def load_transcript(video: Path | None, vtt: Path | None, cdir: Path) -> Transcript:
    """Raw transcript (real names; stays in process memory or the local cache)."""
    if vtt is not None:
        return parse_vtt(vtt.read_text(encoding="utf-8"))
    if video is None:
        raise ValueError("Need a recording or a VTT transcript")
    cached = cdir / "whisper.json"
    if cached.exists():
        return Transcript.model_validate_json(cached.read_text(encoding="utf-8"))
    t = transcribe_media(video, cdir / "audio")
    cached.write_text(t.model_dump_json(), encoding="utf-8")
    return t


def run(
    video: Path | None,
    vtt: Path | None,
    progress: Progress = _noop,
    skip_visual: bool = False,
) -> tuple[Assessment, JudgeOutput]:
    """Assess one class. Returns the Assessment and the verified judge output."""
    s = get_settings()
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal t0
        timings[stage] = round(time.perf_counter() - t0, 1)
        t0 = time.perf_counter()

    progress("ingest", "Hashing input")
    source_file = video or vtt
    if source_file is None:
        raise ValueError("Need a recording or a VTT transcript")
    cdir = cache_dir_for(sha256_file(source_file))
    lap("ingest")

    visual: VisualReport | None = None
    if video is not None and not skip_visual:
        progress("visual", "Sampling frames and measuring video quality")
        visual = aggregate(cached_frame_metrics(video, cdir))
        lap("visual")

    if vtt is not None:
        progress("transcript", "Parsing the Zoom transcript")
    else:
        progress("transcript", "Transcribing audio with whisper.cpp (slow on CPU)")
    raw = assign_roles(load_transcript(video, vtt, cdir))
    transcript, _mapping = anonymise(raw)
    if not transcript.duration_s and video is not None:
        transcript = transcript.model_copy(update={"duration_s": media.probe_duration(video)})
    lap("transcript")

    progress("features", "Computing transcript statistics")
    stats = compute_text_stats(transcript, s.opening_window_min, s.closing_window_min)
    lap("features")

    progress("judge", "Asking the LLM judge and verifying its quotes")
    judged, warnings, meta = judge(transcript, stats, visual)
    lap("judge")

    progress("score", "Scoring")
    if raw.source == "whisper":
        warnings.append("No VTT given: transcribed with whisper; speakers are not identified (Phase 1).")
    assessment = assess(judged, visual, stats, transcript.source, warnings, meta)
    assessment.meta["timings_s"] = timings
    (cdir / "assessment.json").write_text(assessment.model_dump_json(indent=1), encoding="utf-8")
    lap("score")
    return assessment, judged
