"""Path consistency: no-VTT (whisper windows) vs VTT assessment of the same labelled class. IDs and numbers only.

Usage: python scripts/compare_paths.py --tag final excellent/B472 weak/B482
Runs the pipeline once per class without the VTT (video + whisper; the windowed transcript is reused from the class
cache if present, so run scripts/bench_asr_windows.py with the same ASR_* settings first) and compares the label and
the per-criterion scores with the VTT runs stored by scripts/eval_labelled.py under .cache/eval/<tag>/<run>/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa import media  # noqa: E402
from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.pipeline import run  # noqa: E402
from cqa.schemas import Assessment  # noqa: E402


def safe(key: str) -> str:
    return key.replace("/", "__").replace(" ", "_")


def scores(a: Assessment) -> dict[str, str]:
    return {c.id: "-" if c.insufficient_evidence or c.score is None else str(c.score) for c in a.criteria}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="final")
    ap.add_argument("keys", nargs="+")
    args = ap.parse_args()
    s = get_settings()
    items = {i.key: i for i in discover(s.classes_dir, probe_duration=media.probe_duration)}
    eval_dir = s.cache_dir / "eval" / args.tag
    for key in args.keys:
        item = items[key]
        vtt_runs = [
            Assessment.model_validate_json(f.read_text(encoding="utf-8"))
            for f in sorted(eval_dir.glob(f"*/{safe(key)}.json"))
        ]
        t0 = time.perf_counter()
        a, _ = run(item.video, None)
        elapsed = time.perf_counter() - t0
        print(f"\n### {key} (no-VTT run {elapsed:.0f} s, transcript source {a.meta.get('transcript_source', '?')})")
        print(f"label: no-VTT {a.label} (confidence {a.confidence:.2f}) | VTT runs {[v.label for v in vtt_runs]}")
        ref = scores(vtt_runs[0]) if vtt_runs else {}
        print("| criterion | VTT run 0 | no-VTT |")
        print("|---|---|---|")
        for cid, sc in scores(a).items():
            print(f"| {cid} | {ref.get(cid, '?')} | {sc} |")
        kf, kv = a.key_facts, vtt_runs[0].key_facts if vtt_runs else None
        for fact in (
            "homework_assigned",
            "recap_present",
            "rapport_before_content",
            "each_child_asked_individually",
            "children_greeted_by_name",
        ):
            print(f"fact {fact}: VTT {getattr(kv, fact, '?')} | no-VTT {getattr(kf, fact, '?')}")
        print(f"warnings: {len(a.warnings)}; problems no-VTT {a.meta.get('problems')} vs VTT "
              f"{vtt_runs[0].meta.get('problems') if vtt_runs else '?'}")  # fmt: skip


if __name__ == "__main__":
    main()
