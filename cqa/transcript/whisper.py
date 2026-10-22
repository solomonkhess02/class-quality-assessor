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
# Initial prompt (ASR_CHESS_PROMPT): biases decoding towards the class vocabulary; local only, never sent out
CHESS_PROMPT = (
    "Online chess class for children. Pawn, knight, bishop, rook, queen, king, check, checkmate, stalemate, "
    "castling, en passant, fork, pin, skewer, opening, middlegame, endgame, puzzle, homework."
)


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


def read_whisper_json(path: Path) -> dict:
    """whisper-cli can split a multi-byte character across tokens and write invalid UTF-8; replace those bytes."""
    return json.loads(path.read_bytes().decode("utf-8", errors="replace"))


def transcribe(
    wav: Path,
    offset_s: float = 0.0,
    language: str | None = None,
    model: Path | None = None,
    threads: int | None = None,
    prompt: str = "",
) -> tuple[list[Turn], str]:
    s = get_settings()
    if s.whisper_backend == "py":  # pywhispercpp bindings, no compiled CLI (cqa/transcript/whisper_py.py)
        from cqa.transcript import whisper_py

        return whisper_py.transcribe(wav, offset_s, language=language, model=model, threads=threads, prompt=prompt)
    with tempfile.TemporaryDirectory() as tmp:
        out_base = Path(tmp) / "out"
        cmd = [
            s.whisper_cli, "-m", str(model or s.whisper_model), "-t", str(threads or s.whisper_threads),
            "-l", language or s.whisper_language, "-mc", "0", "-oj", "-np",
            "-of", str(out_base), "-f", str(wav),
        ]  # fmt: skip
        if prompt:
            cmd += ["--prompt", prompt]
        # whisper prints transcribed text to stderr; decode it as UTF-8, not the Windows code page
        r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            raise RuntimeError(f"whisper-cli failed (rc={r.returncode}): {r.stderr.strip()[-500:]}")
        data = read_whisper_json(out_base.with_suffix(".json"))
    return parse_whisper_json(data, offset_s)


def asr_windows(
    duration_s: float, opening_min: float, closing_min: float, middle_n: int, middle_min: float
) -> list[tuple[float, float]]:
    """[start, end] windows: the opening, the closing and middle_n evenly spaced middle samples, merged where they
    overlap (a short recording becomes one window)."""
    opening_end = min(opening_min * 60, duration_s)
    closing_start = max(duration_s - closing_min * 60, 0.0)
    windows = [(0.0, opening_end)]
    gap = closing_start - opening_end
    half = middle_min * 30
    for k in range(1, middle_n + 1):
        centre = opening_end + gap * k / (middle_n + 1)
        windows.append((max(centre - half, 0.0), min(centre + half, duration_s)))
    windows.append((closing_start, duration_s))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(w for w in windows if w[1] > w[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [(round(a, 3), round(b, 3)) for a, b in merged]


def configured_windows(duration_s: float) -> list[tuple[float, float]] | None:
    """Windows from the settings, or None for full transcription (ASR_MODE=full)."""
    s = get_settings()
    if s.asr_mode == "full":
        return None
    return asr_windows(duration_s, s.asr_opening_min, s.asr_closing_min, s.asr_middle_windows, s.asr_middle_min)


def transcribe_media(
    src: Path,
    work_dir: Path,
    windows: list[tuple[float, float]] | None = None,
    model: Path | None = None,
    threads: int | None = None,
    language: str | None = None,
) -> Transcript:
    """Transcribe a whole recording, or only the given [start, end] windows, with correct offsets."""
    wavs = media.extract_wav(src, work_dir, windows)
    turns, detected = transcribe_wavs(wavs, language=language, model=model, threads=threads)
    duration = media.probe_duration(src)
    return Transcript(source="whisper", language=detected, duration_s=duration, turns=turns, windows=windows or [])


def transcribe_wavs(
    wavs: list[tuple[Path, float]],
    language: str | None = None,
    model: Path | None = None,
    threads: int | None = None,
) -> tuple[list[Turn], str]:
    """Transcribe (wav, offset) pairs in order. With ASR_DETECT_LANGUAGE_ONCE and language auto, the language
    detected on the first window (the opening) is used for the rest, so one class is not split across languages."""
    s = get_settings()
    language = language or s.whisper_language
    prompt = CHESS_PROMPT if s.asr_chess_prompt else ""
    turns: list[Turn] = []
    languages: list[str] = []
    for wav, offset in wavs:
        t, lang = transcribe(wav, offset, language=language, model=model, threads=threads, prompt=prompt)
        turns.extend(t)
        languages.append(lang)
        if s.asr_detect_language_once and language == "auto" and lang not in ("", "unknown"):
            language = lang
    detected = max(set(languages), key=languages.count) if languages else "unknown"
    return turns, detected
