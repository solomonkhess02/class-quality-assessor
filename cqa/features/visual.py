"""Per-frame visual metrics with OpenCV (local only; frames never leave the machine)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cqa.config import get_settings
from cqa.media import frame_time, sample_frames
from cqa.schemas import VisualFrameMetrics, VisualReport

BLACK_MAX_BRIGHTNESS = 10.0
DARK_MAX_BRIGHTNESS = 60.0
FROZEN_MAX_MEAN_DIFF = 1.0  # mean abs grey-level difference to the previous sampled frame
FACE_SCORE_THRESHOLD = 0.6
# Screen share: most 8x8 blocks are flat (no sensor noise). Initial guesses, checked by eye (contact sheet).
FLAT_BLOCK_MAX_STD = 2.0
SCREEN_SHARE_MIN_FLAT = 0.5
CACHE_NAME = "visual_frames_v2.json"  # bump when per-frame metrics change


def load_face_detector(model_path: Path) -> cv2.FaceDetectorYN:
    if not model_path.is_file():
        raise FileNotFoundError(f"YuNet model not found: {model_path} (run setup.ps1)")
    return cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), FACE_SCORE_THRESHOLD, 0.3, 100)


def count_faces(img: np.ndarray, detector: cv2.FaceDetectorYN) -> int:
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    return 0 if faces is None else len(faces)


def flat_ratio(gray: np.ndarray, block: int = 8) -> float:
    h, w = (gray.shape[0] // block) * block, (gray.shape[1] // block) * block
    blocks = gray[:h, :w].reshape(h // block, block, w // block, block).astype(np.float32)
    return float((blocks.std(axis=(1, 3)) < FLAT_BLOCK_MAX_STD).mean())


def frame_metrics(
    img: np.ndarray, prev_gray: np.ndarray | None, t_s: float, detector: cv2.FaceDetectorYN | None
) -> tuple[VisualFrameMetrics, np.ndarray]:
    """Metrics for one BGR frame. Returns the grey image for the next frozen check."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    frozen = (
        prev_gray is not None
        and prev_gray.shape == gray.shape
        and float(cv2.absdiff(gray, prev_gray).mean()) < FROZEN_MAX_MEAN_DIFF
    )
    flat = flat_ratio(gray)
    m = VisualFrameMetrics(
        t_s=t_s,
        brightness=round(brightness, 1),
        blur=round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        is_black=brightness < BLACK_MAX_BRIGHTNESS,
        is_frozen=bool(frozen),
        face_count=count_faces(img, detector) if detector is not None else 0,
        flat_ratio=round(flat, 3),
        # dark flat frames are usually "camera off" tiles, not shared content
        is_screen_share=flat >= SCREEN_SHARE_MIN_FLAT and brightness >= DARK_MAX_BRIGHTNESS,
    )
    return m, gray


def analyse_frames(frames: list[Path], face_model: Path) -> list[VisualFrameMetrics]:
    detector = load_face_detector(face_model)
    out: list[VisualFrameMetrics] = []
    prev = None
    for p in frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        m, prev = frame_metrics(img, prev, frame_time(p), detector)
        out.append(m)
    return out


def cached_frame_metrics(video: Path, cdir: Path) -> list[VisualFrameMetrics]:
    """Sample frames into cdir/frames and measure them; cached as cdir/visual_frames.json."""
    s = get_settings()
    cached = cdir / CACHE_NAME
    if cached.exists():
        return [VisualFrameMetrics(**m) for m in json.loads(cached.read_text())]
    frames = sample_frames(video, cdir / "frames", s.frame_interval_sec)
    metrics = analyse_frames(frames, s.face_model)
    cached.write_text(json.dumps([m.model_dump() for m in metrics]))
    return metrics


def aggregate(metrics: list[VisualFrameMetrics]) -> VisualReport:
    n = len(metrics)
    if n == 0:
        return VisualReport(n_frames=0, pct_face=0, pct_dark=0, pct_black=0, pct_frozen=0, median_blur=0)

    def pct(k: int, of: int = n) -> float:
        return round(100 * k / of, 1) if of else 0.0

    camera = [m for m in metrics if not m.is_screen_share and not m.is_black]
    return VisualReport(
        n_frames=n,
        pct_face=pct(sum(m.face_count > 0 for m in metrics)),
        pct_dark=pct(sum(m.brightness < DARK_MAX_BRIGHTNESS for m in metrics)),
        pct_black=pct(sum(m.is_black for m in metrics)),
        pct_frozen=pct(sum(m.is_frozen for m in metrics)),
        median_blur=round(float(np.median([m.blur for m in metrics])), 1),
        pct_screen_share=pct(sum(m.is_screen_share for m in metrics)),
        pct_face_camera=pct(sum(m.face_count > 0 for m in camera), len(camera)),
        pct_frozen_camera=pct(sum(m.is_frozen for m in camera), len(camera)),
    )
