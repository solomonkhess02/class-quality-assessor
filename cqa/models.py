"""Model files: download at first start when the configured path is missing, verified by sha256
(same URLs and hashes as setup.ps1 and the Dockerfile)."""

from __future__ import annotations

import hashlib
import tempfile
import urllib.request
from pathlib import Path

from cqa.config import get_settings

# file name -> (url, sha256)
KNOWN = {
    "ggml-base-q5_1.bin": (
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin",
        "422f1ae452ade6f30a004d7e5c6a43195e4433bc370bf23fac9cc591f01a8898",
    ),
    "face_detection_yunet_2023mar.onnx": (
        # github.com/.../raw/ returns a Git LFS pointer, so use the media URL
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/"
        "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
}


def missing_models() -> list[Path]:
    s = get_settings()
    return [p for p in (s.face_model, s.whisper_model) if not p.exists()]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download(path: Path) -> None:
    """Download a known model to path; the file only appears once its sha256 matches."""
    if path.name not in KNOWN:
        raise FileNotFoundError(f"Model {path} is missing and has no known download")
    url, expected = KNOWN[path.name]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        actual = sha256_of(tmp_path)
        if actual != expected:
            raise RuntimeError(f"{path.name}: sha256 mismatch (got {actual[:12]}..., expected {expected[:12]}...)")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_models() -> list[Path]:
    """Download every missing model. Returns the paths downloaded."""
    missing = missing_models()
    for p in missing:
        download(p)
    return missing
