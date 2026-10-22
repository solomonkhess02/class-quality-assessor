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
    whisper_model: Path = Path("models/ggml-base.bin")
    whisper_threads: int = 2
    whisper_language: str = "auto"

    # Data and cache
    data_dir: Path = Path("data")
    cache_dir: Path = Path(".cache")

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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    values = {
        name: os.environ[name.upper()]
        for name in Settings.model_fields
        if os.environ.get(name.upper(), "").strip() != ""
    }
    s = Settings(**values)
    for field in ("whisper_model", "data_dir", "cache_dir", "face_model"):
        setattr(s, field, _resolve(getattr(s, field)))
    return s
