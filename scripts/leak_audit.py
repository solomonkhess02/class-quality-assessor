"""Privacy audit of the exact judge input, on both paths. No API calls, nothing transcribed. Prints counts only.

VTT path: every class with a VTT. Whisper (no-VTT) path: every class with a VTT and a cached whisper transcript for
the current window and language settings, audited against that class's VTT display names (D16 applies to both).

Usage: python scripts/leak_audit.py [--dump] [--path vtt|whisper|both]
  (a) capitalised tokens not in the allowlist, pseudonyms or fixed prompt template
  (b) tokens fuzzy-matching an original display-name token, split at 85: >= 85 must be 0; 70-84 is checked
      against the human labels in .cache/leak_review_labels.json (labelled names left must be 0; a labelled name
      that is also a dictionary word and occurs only in lowercase is accepted and counted separately, D16)
  (c) over-scrub: common English words replaced, as a share of tutor words (target <= 0.5% per class; a
      monitored utility metric, not part of the gate, DECISIONS D17)
Gate D16 is (b) only; unlabelled 70-84 matches leave it pending review.
  --dump  writes matched (b) tokens to .cache/leak_audit_hits.json for local review; contains real names,
          never print it, paste it or commit it.
demo/1 is listed but not gated: its transcript is English ASR of Malayalam speech (noise).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import dataclass  # noqa: E402

from cqa import media  # noqa: E402
from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.features.text_stats import compute_text_stats  # noqa: E402
from cqa.ingest import cache_dir_for, sha256_file  # noqa: E402
from cqa.judge.prompts import build_messages  # noqa: E402
from cqa.schemas import Transcript  # noqa: E402
from cqa.transcript.privacy import (  # noqa: E402
    COMMON_ZIPF,
    FUZZY_MIN,
    anonymise,
    display_name_tokens,
    is_common_word,
    leak_counts,
    over_scrub_share,
)
from cqa.transcript.roles import assign_roles  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402
from cqa.transcript.whisper import cache_name, configured_windows  # noqa: E402

NOT_GATED = {"demo/1"}
# D16 documented exceptions: class -> how many labelled names left are accepted, each only if the display-name token
# it matches is the tutor's. Tokens are deliberately not written here (real names stay out of the repo).
DOCUMENTED_EXCEPTIONS = {
    "excellent/B472": 1,  # ASR-mangled form of the tutor's name rendered as a different name, one occurrence (E24)
}
OVER_SCRUB_MAX = 0.005


def joined(messages: list[dict]) -> str:
    return "\n".join(m["content"] for m in messages)


HEADER = ["class", "turns", "(a) caps", "(a) distinct", "(b) >= 85", "(b) 70-84", "(b) labelled names left",
          "(b) accepted word-like names, lowercase only", "(b) documented exceptions", "(b) unlabelled 70-84",
          "(c) over-scrub"]  # fmt: skip


@dataclass
class JudgeInput:
    key: str  # class key, plus " (whisper)" for the no-VTT path
    label_keys: list[str]  # review-label lists that apply, most specific first
    raw: Transcript  # before anonymisation (real names, local only)
    transcript: Transcript  # what the judge sees
    display_names: list[str]  # the class's Zoom display names (from its VTT)
    vtt_path: bool


def judge_inputs(s, paths: set[str]) -> list[JudgeInput]:
    """The exact judge inputs: VTT path for every class with a VTT; whisper path for every class with a VTT and a
    cached whisper transcript for the current window and language settings (nothing is transcribed here)."""
    out = []
    for item in discover(s.classes_dir, probe_duration=media.probe_duration):
        if not item.vtt:
            continue
        vtt_raw = assign_roles(parse_vtt(item.vtt.read_text(encoding="utf-8")))
        names = sorted({t.speaker for t in vtt_raw.turns})
        if "vtt" in paths:
            out.append(JudgeInput(item.key, [item.key], vtt_raw, anonymise(vtt_raw)[0], names, True))
        if "whisper" in paths and item.video:
            # same steps as pipeline.run for a recording without a transcript
            duration = media.probe_duration(item.video)
            cached = cache_dir_for(sha256_file(item.video)) / cache_name(
                configured_windows(duration), s.whisper_language
            )
            if not cached.exists():
                continue
            raw = assign_roles(Transcript.model_validate_json(cached.read_text(encoding="utf-8")))
            transcript = anonymise(raw)[0]
            if not transcript.duration_s:
                transcript = transcript.model_copy(update={"duration_s": duration})
            labels = [f"whisper:{item.key}", item.key]  # a token reviewed on the VTT path keeps its label
            out.append(JudgeInput(f"{item.key} (whisper)", labels, raw, transcript, names, False))
    return out


def audit(paths: set[str] = frozenset({"vtt", "whisper"})) -> tuple[list[list[str]], str, dict]:
    """Returns (table rows, gate verdict line, matched tokens per input for the local dump)."""
    s = get_settings()
    labels_path = s.cache_dir / "leak_review_labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))["labels"] if labels_path.exists() else None
    rows, dump = [], {}
    high_total = names_left_total = unlabelled_total = over_scrub_failures = documented_total = 0
    win = (s.opening_window_min, s.closing_window_min)
    for ji in judge_inputs(s, set(paths)):
        raw, transcript = ji.raw, ji.transcript
        class_key = ji.label_keys[-1]
        stats = compute_text_stats(transcript, *win)
        text = joined(build_messages(transcript, stats, None, *win))
        template = joined(build_messages(transcript.model_copy(update={"turns": []}), stats, None, *win))
        lc = leak_counts(text, template, ji.display_names)
        high = [h for h in lc.hits if h[2] >= FUZZY_MIN]
        band = [h for h in lc.hits if h[2] < FUZZY_MIN]
        if labels is None:
            names_left = accepted = unlabelled = documented = "-"
        else:
            known = {w: v for k in reversed(ji.label_keys) for w, v in labels.get(k, [])}
            # D16 exception: a labelled name that is also an ordinary dictionary word and is left only in lowercase,
            # i.e. used as the word; capitalised forms are scrubbed mid-sentence and count as names left otherwise
            accepted = sum(
                known.get(w) == "name" and is_common_word(w, COMMON_ZIPF) and w not in lc.capitalised
                for w, _, _ in band
            )
            names_left = sum(known.get(w) == "name" for w, _, _ in band) - accepted
            documented = 0
            if ji.vtt_path:  # documented exceptions are per VTT judge input (E24)
                tutor_tokens = display_name_tokens(sorted({t.speaker for t in raw.turns if t.role == "tutor"}))
                tutor_hits = sum(known.get(w) == "name" and name in tutor_tokens for w, name, _ in band)
                documented = min(DOCUMENTED_EXCEPTIONS.get(class_key, 0), tutor_hits, names_left)
            names_left -= documented
            documented_total += documented
            unlabelled = sum(w not in known for w, _, _ in band)
            if class_key not in NOT_GATED:
                names_left_total += names_left
                unlabelled_total += unlabelled
        # over-scrub is a share of tutor words, so it needs speakers (VTT path only)
        over = over_scrub_share(raw, transcript) if ji.vtt_path else None
        if class_key not in NOT_GATED:
            high_total += len(high)
            over_scrub_failures += over is not None and over > OVER_SCRUB_MAX
        rows.append(
            [ji.key + (" (not gated)" if class_key in NOT_GATED else ""), str(len(transcript.turns)),
             str(lc.caps), str(lc.caps_distinct), str(len(high)), str(len(band)), str(names_left), str(accepted),
             str(documented), str(unlabelled), "-" if over is None else f"{over:.2%}"]
        )  # fmt: skip
        dump[ji.key] = lc.hits
    if labels is None:
        return rows, f"Gate D16: not checked ({labels_path.name} missing); hits >= 85: {high_total}", dump
    fail = high_total or names_left_total  # over-scrub is monitored, not gated (D17)
    verdict = "FAIL" if fail else "PENDING REVIEW" if unlabelled_total else "PASS"
    line = (
        f"Gate D16: {verdict} (hits >= 85: {high_total}, labelled names left: {names_left_total}, "
        f"unlabelled 70-84 to review: {unlabelled_total}, documented exceptions: {documented_total}); "
        "over-scrub, monitored not gated: "
        f"{over_scrub_failures} classes over {OVER_SCRUB_MAX:.1%}"
    )
    return rows, line, dump


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--path", choices=["vtt", "whisper", "both"], default="both")
    args = ap.parse_args()
    rows, verdict, dump = audit({"vtt", "whisper"} if args.path == "both" else {args.path})
    print("| " + " | ".join(HEADER) + " |")
    print("|" + "---|" * len(HEADER))
    for r in rows:
        print("| " + " | ".join(r) + " |")
    print("(b) columns count distinct tokens.")
    print(f"\n{verdict}")
    if args.dump:
        out = get_settings().cache_dir / "leak_audit_hits.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dump, indent=1), encoding="utf-8")
        print(f"\nWrote {out} (real names: local review only)")


if __name__ == "__main__":
    main()
