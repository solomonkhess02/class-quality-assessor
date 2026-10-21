"""Contact sheet for checking the screen-share tag by eye. Local only: the image stays in .cache/.

  python scripts/contact_sheet.py           build .cache/contact_sheet/sheet.jpg and labels.csv
  python scripts/contact_sheet.py --score   after filling the your_label column (share / camera), print agreement
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.features.visual import cached_frame_metrics  # noqa: E402
from cqa.ingest import cache_dir_for, sha256_file  # noqa: E402

N_PER_TAG = 10
COLS, THUMB_W, THUMB_H, CAPTION_H = 5, 320, 180, 26


def out_dir() -> Path:
    d = get_settings().cache_dir / "contact_sheet"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build(seed: int = 7) -> None:
    s = get_settings()
    candidates: dict[bool, list[tuple[str, Path, float, float]]] = {True: [], False: []}
    for item in discover(s.classes_dir):
        if not item.video:
            continue
        cdir = cache_dir_for(sha256_file(item.video))
        for m in cached_frame_metrics(item.video, cdir):
            if m.is_black:
                continue
            frame = cdir / "frames" / f"frame_{int(m.t_s):06d}.jpg"
            if frame.exists():
                candidates[m.is_screen_share].append((item.key, frame, m.t_s, m.flat_ratio))
    rng = random.Random(seed)
    picked = [(True, c) for c in rng.sample(candidates[True], min(N_PER_TAG, len(candidates[True])))]
    picked += [(False, c) for c in rng.sample(candidates[False], min(N_PER_TAG, len(candidates[False])))]
    rng.shuffle(picked)

    tiles, rows = [], []
    for i, (share, (key, frame, t, flat)) in enumerate(picked, start=1):
        img = cv2.resize(cv2.imread(str(frame)), (THUMB_W, THUMB_H))
        bar = np.zeros((CAPTION_H, THUMB_W, 3), np.uint8)
        text = f"#{i:02d} pred={'SHARE' if share else 'CAMERA'} flat={flat:.2f}"
        cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))
        rows.append({"id": i, "class": key, "t_s": int(t), "pred": "share" if share else "camera",
                     "flat_ratio": flat, "your_label": ""})  # fmt: skip
    while len(tiles) % COLS:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r : r + COLS]) for r in range(0, len(tiles), COLS)])
    d = out_dir()
    cv2.imwrite(str(d / "sheet.jpg"), grid)
    with (d / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(picked)} frames ({sum(p for p, _ in picked)} predicted share) from "
          f"{len(candidates[True]) + len(candidates[False])} non-black frames")  # fmt: skip
    print(f"Sheet:  {(d / 'sheet.jpg').resolve()}")
    print(f"Labels: {(d / 'labels.csv').resolve()}  (fill your_label with share / camera, then --score)")


def score() -> None:
    with (out_dir() / "labels.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["your_label"].strip()]
    if not rows:
        print("No labels filled in yet.")
        return
    agree = sum(r["pred"] == r["your_label"].strip().lower() for r in rows)
    print(f"Agreement: {agree}/{len(rows)}")
    for pred in ("share", "camera"):
        for truth in ("share", "camera"):
            n = sum(r["pred"] == pred and r["your_label"].strip().lower() == truth for r in rows)
            print(f"  pred={pred:<7} you={truth:<7} {n}")
    other = [r for r in rows if r["your_label"].strip().lower() not in ("share", "camera")]
    if other:
        print(f"  other labels: {len(other)}")


if __name__ == "__main__":
    score() if "--score" in sys.argv else build()
