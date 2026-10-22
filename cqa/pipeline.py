"""End-to-end orchestration with per-stage caching under .cache/<hash>/."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict
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
from cqa.transcript.quality import check_quality
from cqa.transcript.role_tagging import tag_roles
from cqa.transcript.roles import assign_roles
from cqa.transcript.vtt import parse_vtt
from cqa.transcript.whisper import configured_windows, transcribe_media

Progress = Callable[[str, str], None]


def _noop(stage: str, message: str) -> None:
    pass


def whisper_transcript(video: Path, cdir: Path) -> Transcript:
    """whisper.cpp transcript of the configured windows (or the whole file), cached in the class's cache dir."""
    windows = configured_windows(media.probe_duration(video))
    cached = cdir / ("whisper_windows.json" if windows else "whisper.json")
    if cached.exists():
        return Transcript.model_validate_json(cached.read_text(encoding="utf-8"))
    t = transcribe_media(video, cdir / "audio", windows)
    cached.write_text(t.model_dump_json(), encoding="utf-8")
    return t


def load_transcript(video: Path | None, vtt: Path | None, cdir: Path) -> Transcript:
    """Raw transcript (real names; stays in process memory or the local cache)."""
    if vtt is not None:
        return parse_vtt(vtt.read_text(encoding="utf-8"))
    if video is None:
        raise ValueError("Need a recording or a VTT transcript")
    return whisper_transcript(video, cdir)


def run(
    video: Path | None,
    vtt: Path | None,
    progress: Progress = _noop,
    skip_visual: bool = False,
    run_index: int = 0,
) -> tuple[Assessment, JudgeOutput]:
    """Assess one class. Returns the Assessment and the verified judge output.
    run_index > 0: a repeated evaluation run (separate judge call, own assessment file)."""
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
    pre_warnings: list[str] = []
    quality = check_quality(raw)
    if not quality.usable and raw.source == "vtt":
        why = "; ".join(quality.reasons)
        if video is not None:
            progress("transcript", "The Zoom transcript looks unusable; transcribing the audio with whisper.cpp")
            pre_warnings.append(f"The Zoom transcript looks unusable ({why}); the audio was transcribed instead.")
            raw = assign_roles(whisper_transcript(video, cdir))
            quality = check_quality(raw)
        else:
            pre_warnings.append(
                f"The Zoom transcript looks unusable ({why}) and there is no recording to fall back on."
            )
    transcript, _mapping = anonymise(raw)
    role_meta: dict = {}
    if transcript.source == "whisper" and s.whisper_role_tagging:
        progress("transcript", "Labelling tutor and student lines")
        transcript, role_meta = tag_roles(transcript, run_index=run_index)  # anonymised text only
    if not transcript.duration_s and video is not None:
        transcript = transcript.model_copy(update={"duration_s": media.probe_duration(video)})
    lap("transcript")

    progress("features", "Computing transcript statistics")
    stats = compute_text_stats(transcript, s.opening_window_min, s.closing_window_min)
    lap("features")

    progress("judge", "Asking the LLM judge and verifying its quotes")
    judged, warnings, meta = judge(transcript, stats, visual, run_index=run_index)
    lap("judge")

    progress("score", "Scoring")
    warnings = pre_warnings + warnings
    if raw.source == "whisper":
        part = "selected windows" if raw.windows else "the whole recording"
        warnings.append(f"Transcribed with whisper ({part}, language {raw.language}); speakers are not identified.")
    if not quality.usable:
        warnings.append(f"Transcript quality is low: {'; '.join(quality.reasons)}.")
    meta["transcript_quality"] = asdict(quality)
    if role_meta:
        meta["role_tagging"] = {k: role_meta[k] for k in ("role_lines_missing", "latency_s") if k in role_meta}
    assessment = assess(judged, visual, stats, transcript.source, warnings, meta)
    assessment.meta["timings_s"] = timings
    (cdir / (f"assessment_r{run_index}.json" if run_index else "assessment.json")).write_text(
        assessment.model_dump_json(indent=1), encoding="utf-8"
    )
    lap("score")
    return assessment, judged
