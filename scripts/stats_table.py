"""Print TextStats for every VTT in the dataset (class keys and numbers only)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.config import get_settings  # noqa: E402
from cqa.dataset import discover  # noqa: E402
from cqa.features.text_stats import compute_text_stats  # noqa: E402
from cqa.transcript.roles import assign_roles  # noqa: E402
from cqa.transcript.vtt import parse_vtt  # noqa: E402


def main() -> None:
    s = get_settings()
    cols = ["min", "tutor%", "stud", "st/min", "?/min", "sparse", "mono_s", "conn", "open", "close"]
    print(f"{'key':<20}{'label':<10}" + "".join(f"{c:>8}" for c in cols))
    for item in discover(s.classes_dir):
        if not item.vtt:
            continue
        t = assign_roles(parse_vtt(item.vtt.read_text(encoding="utf-8")))
        st = compute_text_stats(t, s.opening_window_min, s.closing_window_min)
        vals = [
            f"{st.duration_min:.0f}",
            f"{st.tutor_talk_share * 100:.0f}",
            st.distinct_students_spoke,
            f"{st.student_turns_per_min:.1f}",
            f"{st.tutor_questions_per_min:.1f}",
            "y" if st.sparse_punctuation else "n",
            f"{st.longest_tutor_monologue_s:.0f}",
            st.connectivity_mentions,
            st.opening_turns,
            st.closing_turns,
        ]
        print(f"{item.key:<20}{item.label or '-':<10}" + "".join(f"{v:>8}" for v in vals))


if __name__ == "__main__":
    main()
