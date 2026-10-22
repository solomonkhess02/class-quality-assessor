"""whisper.cpp through the pywhispercpp bindings (WHISPER_BACKEND=py): no compiled CLI needed, e.g. on
Streamlit Community Cloud. Same parameters and output as cqa.transcript.whisper.transcribe. Audio stays local."""

from __future__ import annotations

import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

from cqa.config import get_settings
from cqa.schemas import Turn

UNKNOWN = "UNKNOWN"
SAMPLE_RATE = 16000


@lru_cache(maxsize=2)
def _model(path: str, threads: int):
    from pywhispercpp.model import Model  # optional dependency (Linux wheels)

    return Model(path, n_threads=threads, print_progress=False, print_realtime=False, redirect_whispercpp_logs_to=None)


def read_wav(path: Path) -> np.ndarray:
    """16 kHz mono 16-bit PCM WAV (cqa.media.extract_wav) as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"expected 16 kHz mono 16-bit WAV, got {w.getparams()}")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def transcribe(
    wav: Path,
    offset_s: float = 0.0,
    language: str | None = None,
    model: Path | None = None,
    threads: int | None = None,
) -> tuple[list[Turn], str]:
    """Like whisper-cli -l <language> -t <threads> -mc 0: Turns (speaker UNKNOWN) and the detected language."""
    import _pywhispercpp as pw

    s = get_settings()
    m = _model(str(model or s.whisper_model), threads or s.whisper_threads)
    segments = m.transcribe(read_wav(wav), language=language or s.whisper_language, n_max_text_ctx=0)
    turns = [
        Turn(
            start_s=round(offset_s + seg.t0 / 100, 3),  # pywhispercpp times are in 10 ms units
            end_s=round(offset_s + seg.t1 / 100, 3),
            speaker=UNKNOWN,
            role="unknown",
            text=seg.text.strip(),
        )
        for seg in segments
        if seg.text.strip()
    ]
    return turns, pw.whisper_lang_str(pw.whisper_full_lang_id(m._ctx))
