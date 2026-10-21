"""whisper.cpp CLI wrapper. Audio stays local (DECISIONS D3)."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from cqa import media
from cqa.config import get_settings
from cqa.schemas import Transcript, Turn

UNKNOWN = "UNKNOWN"


def parse_whisper_json(data: dict, offset_s: float = 0.0) -> tuple[list[Turn], str]:
    """Parse whisper-cli -oj output into Turns (speaker UNKNOWN) and the detected language."""
    turns = []
    for seg in data.get("transcription", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        off = seg["offsets"]
        turns.append(
            Turn(
                start_s=round(offset_s + off["from"] / 1000, 3),
                end_s=round(offset_s + off["to"] / 1000, 3),
                speaker=UNKNOWN,
                role="unknown",
                text=text,
            )
        )
    language = data.get("result", {}).get("language", "unknown")
    return turns, language


def transcribe(
    wav: Path,
    offset_s: float = 0.0,
    language: str | None = None,
    model: Path | None = None,
    threads: int | None = None,
) -> tuple[list[Turn], str]:
    s = get_settings()
    with tempfile.TemporaryDirectory() as tmp:
        out_base = Path(tmp) / "out"
        cmd = [
            s.whisper_cli, "-m", str(model or s.whisper_model), "-t", str(threads or s.whisper_threads),
            "-l", language or s.whisper_language, "-mc", "0", "-oj", "-np",
            "-of", str(out_base), "-f", str(wav),
        ]  # fmt: skip
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"whisper-cli failed (rc={r.returncode}): {r.stderr.strip()[-500:]}")
        data = json.loads(out_base.with_suffix(".json").read_text(encoding="utf-8"))
    return parse_whisper_json(data, offset_s)


def transcribe_media(
    src: Path, work_dir: Path, windows: list[tuple[float, float]] | None = None
) -> Transcript:
    """Transcribe a whole recording, or only the given [start, end] windows, with correct offsets."""
    turns: list[Turn] = []
    languages: list[str] = []
    for wav, offset in media.extract_wav(src, work_dir, windows):
        t, lang = transcribe(wav, offset)
        turns.extend(t)
        languages.append(lang)
    duration = media.probe_duration(src)
    language = max(set(languages), key=languages.count) if languages else "unknown"
    return Transcript(source="whisper", language=language, duration_s=duration, turns=turns)
