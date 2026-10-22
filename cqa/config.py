"""Settings loaded from .env. The only place the app reads configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    # LLM judge
    deepseek_api_key: SecretStr = SecretStr("")
    deepseek_base_url: str = "https://api.deepseek.com"
    judge_model: str = "deepseek-flash"
    judge_temperature: float = 0.2
    judge_max_tokens: int = 8000
    judge_timeout_s: float = 180.0
    judge_thinking: str = "disabled"  # "enabled" | "disabled" (DECISIONS D9)
    judge_design: str = "direct"  # "direct" | "facts" (EXPERIMENTS, M6b)
    aggregation: str = "severity"  # "severity" (A3 kept, D18) | "problems" (D12) | "mean"
    # Ablation switches (docs/EVALUATION.md); defaults are the last kept configuration
    judge_evidence_first: bool = True  # A0 (kept, E13): reasons before the score in the judge's example
    tech_scorer: str = "code"  # A1 (kept, E14): "llm" | "code" (DECISIONS D14)
    label_exclude: str = "communication"  # A1 (kept, E14): shown, not in the label (D15); "none" = exclude nothing
    face_check_critical: bool = False  # A3b: low face share is moderate + warning until screen-share tags are validated
    facts_drive: bool = True  # A2 (kept, E15): verified yes/no facts bound opening, engagement, closure scores

    # Binaries (names on PATH by default; full paths on the Windows dev machine)
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    whisper_cli: str = "whisper-cli"
    whisper_backend: str = "cli"  # "cli" (whisper-cli subprocess) | "py" (pywhispercpp bindings, no compile)
    whisper_model: Path = Path("models/ggml-base.bin")
    whisper_threads: int = 2
    whisper_language: str = "en"  # E24: explicit (class language from the UI); auto misread B472's opening as Sinhala
    # No-VTT path (P2.2): transcribe only these windows (opening, closing, evenly spaced middle samples)
    asr_mode: str = "windows"  # "windows" | "full"
    asr_opening_min: float = 8.0  # E22, E24: 8 + 10 + 2 x 2 min (22 min of audio), was 12 + 10 + 3 x 3
    asr_closing_min: float = 10.0  # E24: matches the judge's 10-min closing window (closing_window_min)
    asr_middle_windows: int = 2
    asr_middle_min: float = 2.0
    asr_detect_language_once: bool = False  # language auto: detect on the opening window, reuse for the rest
    asr_chess_prompt: bool = False  # whisper initial prompt with chess vocabulary (cqa/transcript/whisper.py)
    whisper_role_tagging: bool = False  # LLM tags TUTOR / STUDENT per whisper segment; on once validated (P2.2)

    # Data and cache
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")
    # Public deployment: delete media after each assessment and purge old cache entries (off locally)
    delete_media_after_run: bool = False
    cache_max_age_h: float = 0.0  # 0 = never purge

    # Pipeline
    frame_interval_sec: float = 30.0
    face_model: Path = Path("models/face_detection_yunet_2023mar.onnx")
    opening_window_min: float = 10.0
    closing_window_min: float = 10.0
    evidence_window_s: float = 90.0

    # Visual thresholds (initial guesses, check against eval; do not fit to 10 samples)
    visual_min_face_pct: float = 60.0
    visual_max_dark_pct: float = 20.0
    visual_max_black_pct: float = 10.0
    visual_max_frozen_pct: float = 30.0

    @property
    def classes_dir(self) -> Path:
        return self.data_dir / "class recordings"


def _resolve(p: Path) -> Path:
    return p if p.is_absolute() else REPO_ROOT / p


def on_community_cloud() -> bool:
    """Streamlit Community Cloud runs apps from /mount/src/<repo>."""
    return REPO_ROOT.as_posix().startswith("/mount/src/")


def cloud_defaults() -> dict[str, str]:
    """Public-deployment defaults, matching the Docker image: no media kept, no compiled whisper-cli,
    models downloaded at first start into the home cache."""
    models = Path.home() / ".cache" / "cqa-models"
    return {
        "delete_media_after_run": "true",
        "cache_max_age_h": "2",
        "whisper_backend": "py",
        "whisper_model": str(models / "ggml-base-q5_1.bin"),
        "face_model": str(models / "face_detection_yunet_2023mar.onnx"),
    }


def _secrets() -> dict[str, str]:
    """Root-level Streamlit secrets (.streamlit/secrets.toml, or the app's secrets on Community Cloud)."""
    try:
        import streamlit as st

        return {k.lower(): str(v) for k, v in st.secrets.to_dict().items() if not isinstance(v, dict)}
    except Exception:  # no secrets file
        return {}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Precedence: environment (incl. .env) > Streamlit secrets > Community Cloud defaults > field defaults."""
    load_dotenv(REPO_ROOT / ".env")
    values = cloud_defaults() if on_community_cloud() else {}
    environ = dict(os.environ)  # before st.secrets: Streamlit copies root-level secrets into os.environ
    secrets = _secrets()
    for name in Settings.model_fields:
        env = environ.get(name.upper(), "").strip()
        if env:
            values[name] = env
        elif secrets.get(name, "").strip():
            values[name] = secrets[name]
    s = Settings(**values)
    for field in ("whisper_model", "data_dir", "cache_dir", "face_model"):
        setattr(s, field, _resolve(getattr(s, field)))
    return s
