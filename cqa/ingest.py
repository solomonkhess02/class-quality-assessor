"""Input handling: uploads, Google Drive links, hashing and per-file cache dirs."""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO

from cqa.config import get_settings

CHUNK = 1 << 20


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
    else:
        tmp_path.replace(final)
    return final


def download_drive(url: str) -> Path:
    """Download a publicly shared Google Drive file (gdown parses share URLs itself)."""
    import gdown

    downloads = get_settings().cache_dir / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=downloads) as tmp:
        out = gdown.download(url, output=tmp + "/", quiet=True)
        if not out:
            raise RuntimeError("Google Drive download failed; is the link shared publicly?")
        src = Path(out)
        final = downloads / f"{sha256_file(src)[:16]}{src.suffix.lower() or '.mp4'}"
        if not final.exists():
            shutil.move(str(src), final)
    return final
