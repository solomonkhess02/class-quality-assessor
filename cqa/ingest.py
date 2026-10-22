"""Input handling: uploads, Google Drive links, hashing, per-file cache dirs and input checks."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from cqa import media
from cqa.config import get_settings
from cqa.schemas import format_ts
from cqa.transcript.formats import TranscriptFormatError, parse_transcript_bytes, restrict_to_windows, to_vtt
from cqa.transcript.whisper import asr_windows

CHUNK = 1 << 20
VIDEO_TYPES = (".mp4", ".mkv", ".webm", ".mov")
MAX_DURATION_S = 2 * 3600  # longer recordings: warn and assess the opening, middle and closing windows only
_DRIVE_ID = re.compile(r"(?:/d/|[?&]id=|/folders/)([\w-]{20,})")


class InputError(ValueError):
    """A problem with the user's input; the message is shown in the UI as is (no traceback)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def cache_dir_for(file_hash: str) -> Path:
    d = get_settings().cache_dir / file_hash[:16]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_upload(stream: BinaryIO, suffix: str) -> Path:
    """Copy an uploaded stream into the cache as <hash><suffix> and return its path."""
    uploads = get_settings().cache_dir / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=uploads, suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(stream, tmp, CHUNK)
        tmp_path = Path(tmp.name)
    final = uploads / f"{sha256_file(tmp_path)[:16]}{suffix}"
    if final.exists():
        tmp_path.unlink()
        os.utime(final)  # a re-upload restarts the purge clock (the video is kept for playback)
    else:
        tmp_path.replace(final)
    return final


def check_drive_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"https?://(drive|docs)\.google\.com/", url) or not _DRIVE_ID.search(url):
        raise InputError(
            "That does not look like a Google Drive file link. Use the share link of the file, e.g. "
            "https://drive.google.com/file/d/<id>/view."
        )
    if "/folders/" in url:
        raise InputError("That is a Drive folder link; please share the link of the recording file itself.")
    return url


def download_drive(url: str) -> Path:
    """Download a publicly shared Google Drive file (gdown parses share URLs itself)."""
    import gdown

    url = check_drive_url(url)
    downloads = get_settings().cache_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=downloads) as tmp:
        try:
            out = gdown.download(url, output=tmp + "/", quiet=True)
        except Exception as e:  # gdown raises its own errors for private files, quota limits, network problems
            raise InputError(
                "Could not download the file from Google Drive. Check that sharing is set to "
                f"'Anyone with the link' and that the link points to a file ({type(e).__name__})."
            ) from e
        if not out:
            raise InputError("Google Drive download failed; is the link shared with 'Anyone with the link'?")
        src = Path(out)
        if src.suffix.lower() not in VIDEO_TYPES:
            raise InputError(
                f"The Drive file is a '{src.suffix or 'unknown'}' file; please share a recording "
                f"({', '.join(VIDEO_TYPES)})."
            )
        final = downloads / f"{sha256_file(src)[:16]}{src.suffix.lower()}"
        if not final.exists():
            shutil.move(str(src), final)
    return final


def check_video(path: Path) -> float:
    """Duration in seconds of a readable recording with a video stream; InputError otherwise."""
    if path.suffix.lower() not in VIDEO_TYPES:
        raise InputError(f"Unsupported recording type '{path.suffix}'. Please use {', '.join(VIDEO_TYPES)}.")
    try:
        r = media._run(
            [
                get_settings().ffprobe_bin, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_type:format=duration", "-of", "default=noprint_wrappers=1",
                str(path),
            ],
            timeout=120,
        )  # fmt: skip
        fields = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
        duration = float(fields.get("duration", "nan"))
    except (RuntimeError, ValueError, OSError) as e:
        raise InputError("Could not read the recording; the file may be corrupted or incomplete.") from e
    if fields.get("codec_type") != "video" or not duration > 0:
        raise InputError("The recording has no readable video stream; the file may be corrupted or incomplete.")
    return duration


def long_recording_windows(duration_s: float) -> list[tuple[float, float]]:
    s = get_settings()
    return asr_windows(duration_s, s.asr_opening_min, s.asr_closing_min, s.asr_middle_windows, s.asr_middle_min)


@dataclass
class PreparedInputs:
    """What the pipeline gets, plus what the report needs to say about it."""

    video: Path | None
    vtt: Path | None  # canonical VTT for the pipeline, or None for the whisper path
    duration_s: float | None = None
    transcript_format: str | None = None  # "vtt" | "srt" | "txt" as uploaded
    speakers_identified: bool = False
    notes: list[str] = field(default_factory=list)
    temp_files: list[Path] = field(default_factory=list)  # deleted with the run files


def prepare_inputs(video: Path | None, transcript: Path | None) -> PreparedInputs:
    """Check the recording, read any transcript format, and decide between the VTT and whisper paths."""
    p = PreparedInputs(video=video, vtt=None)
    if video is not None:
        p.duration_s = check_video(video)
        if p.duration_s > MAX_DURATION_S:
            parts = ", ".join(f"{format_ts(a)}-{format_ts(b)}" for a, b in long_recording_windows(p.duration_s))
            p.notes.append(
                f"The recording is {p.duration_s / 3600:.1f} h long (limit 2 h), so only the opening, middle "
                f"samples and closing ({parts}) are assessed from the transcript; video frames cover the whole class."
            )
    if transcript is None:
        return p
    try:
        parsed = parse_transcript_bytes(transcript.read_bytes(), transcript.suffix)
    except TranscriptFormatError as e:
        raise InputError(str(e)) from e
    p.transcript_format = parsed.fmt
    p.notes += parsed.notes
    if not parsed.has_speakers and video is not None:
        p.notes.append(
            "The transcript has no speaker names, so it was not used: the audio is transcribed locally instead "
            "(as without a transcript) and speakers are not identified."
        )
        p.temp_files.append(transcript)
        return p
    if not parsed.has_speakers:
        p.notes.append("The transcript has no speaker names, so tutor and children are not told apart.")
    p.speakers_identified = parsed.has_speakers
    t = parsed.transcript
    if p.duration_s and p.duration_s > MAX_DURATION_S:
        t = restrict_to_windows(t, long_recording_windows(p.duration_s))
    elif parsed.fmt == "vtt":
        p.vtt = transcript  # unchanged Zoom VTT: same cache key as before
        return p
    out = transcript.with_name(f"{transcript.stem}_norm.vtt")
    out.write_text(to_vtt(t), encoding="utf-8")
    p.vtt = out
    p.temp_files.append(transcript)
    return p
