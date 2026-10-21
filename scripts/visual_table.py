"""Visual metrics for the labelled recordings (class keys and numbers only, no images).

Frames and per-frame metrics are cached under .cache/<hash>/ where the pipeline reuses them.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.features.visual import aggregate, cached_frame_metrics  # noqa: E402
from cqa.ingest import cache_dir_for, sha256_file  # noqa: E402
from cqa.schemas import VisualFrameMetrics  # noqa: E402


def frame_metrics_cached(video: Path) -> list[VisualFrameMetrics]:
    return cached_frame_metrics(video, cache_dir_for(sha256_file(video)))


def main(include_demo: bool = False) -> None:
    s = get_settings()
    cols = ["frames", "face%", "multi%", "faces", "dark%", "black%", "frozen%", "blur", "secs"]
    print(f"{'key':<20}{'label':<10}" + "".join(f"{c:>8}" for c in cols), flush=True)
    for item in discover(s.classes_dir):
        if not item.video or (item.label is None and not include_demo):
            continue
        t0 = time.perf_counter()
        ms = frame_metrics_cached(item.video)
        r = aggregate(ms)
        multi = 100 * sum(m.face_count >= 2 for m in ms) / max(len(ms), 1)
        mean_faces = sum(m.face_count for m in ms) / max(len(ms), 1)
        vals = [
            r.n_frames, f"{r.pct_face:.0f}", f"{multi:.0f}", f"{mean_faces:.1f}", f"{r.pct_dark:.0f}",
            f"{r.pct_black:.0f}", f"{r.pct_frozen:.0f}", f"{r.median_blur:.0f}", f"{time.perf_counter() - t0:.0f}",
        ]  # fmt: skip
        label = item.label + ("*" if item.technical_flag else "") if item.label else "-"
        print(f"{item.key:<20}{label:<10}" + "".join(f"{v:>8}" for v in vals), flush=True)


if __name__ == "__main__":
    main(include_demo="--demo" in sys.argv)
