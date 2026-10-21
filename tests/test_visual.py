import numpy as np
import pytest

from cqa.config import get_settings
from cqa.features.visual import aggregate, count_faces, frame_metrics, load_face_detector
from cqa.schemas import VisualFrameMetrics


def img(value: int | None = None, seed: int = 0) -> np.ndarray:
    if value is None:
        return np.random.default_rng(seed).integers(0, 256, (360, 640, 3), dtype=np.uint8)
    return np.full((360, 640, 3), value, dtype=np.uint8)


def test_black_and_uniform_frames():
    m, _ = frame_metrics(img(0), None, 0, None)
    assert m.is_black and m.brightness == 0 and m.blur == 0
    m, _ = frame_metrics(img(128), None, 30, None)
    assert not m.is_black and m.brightness == 128


def test_noise_is_sharp():
    m, _ = frame_metrics(img(seed=1), None, 0, None)
    assert m.blur > 1000


def test_frozen_detection():
    a = img(seed=1)
    _, gray = frame_metrics(a, None, 0, None)
    same, _ = frame_metrics(a.copy(), gray, 30, None)
    diff, _ = frame_metrics(img(seed=2), gray, 30, None)
    assert same.is_frozen is True
    assert diff.is_frozen is False


def test_aggregate():
    ms = [
        VisualFrameMetrics(t_s=0, brightness=5, blur=10, is_black=True, is_frozen=False, face_count=0),
        VisualFrameMetrics(t_s=30, brightness=50, blur=20, is_black=False, is_frozen=True, face_count=1),
        VisualFrameMetrics(t_s=60, brightness=120, blur=30, is_black=False, is_frozen=False, face_count=2),
        VisualFrameMetrics(t_s=90, brightness=130, blur=40, is_black=False, is_frozen=False, face_count=0),
    ]
    r = aggregate(ms)
    assert (r.n_frames, r.pct_face, r.pct_dark, r.pct_black, r.pct_frozen) == (4, 50, 50, 25, 25)
    assert r.median_blur == 25
    assert aggregate([]).n_frames == 0


def test_detector_finds_no_face_in_noise():
    model = get_settings().face_model
    if not model.is_file():
        pytest.skip("YuNet model not downloaded")
    assert count_faces(img(seed=3), load_face_detector(model)) == 0
