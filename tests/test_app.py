"""Smoke tests for the Streamlit app with a synthetic assessment (no real data, no LLM calls)."""

import io
from pathlib import Path

from streamlit.testing.v1 import AppTest

from cqa.schemas import Assessment, CriterionResult, Evidence, FactEvidence, KeyFacts, VisualReport

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def synthetic_assessment() -> Assessment:
    return Assessment(
        label="Good",
        weighted_score=3.1,
        confidence=0.8,
        criteria=[
            CriterionResult(
                id="opening_rapport",
                score=3,
                observations=["TUTOR greets S1 and S2 by name at 00:00:40."],
                evidence=[Evidence(timestamp="00:00:40", speaker="TUTOR", quote="hello S1", verified=True)],
                improvement="Set the plan before the first puzzle.",
            ),
            CriterionResult(id="communication", score=None),
            CriterionResult(id="visual_presence", score=2, observations=["A face is visible in 55% of frames."]),
        ],
        key_facts=KeyFacts(
            homework_assigned=True,
            homework_what="5 puzzles",
            evidence=[FactEvidence(fact="homework_assigned", timestamp="00:50:00", speaker="TUTOR", quote="homework")],
        ),
        summary="Synthetic summary.",
        strengths=["Names children (00:00:40)."],
        improvements=["Assign homework."],
        warnings=["Synthetic warning."],
        visual=VisualReport(n_frames=10, pct_face=55, pct_dark=0, pct_black=0, pct_frozen=0, median_blur=100),
        meta={"problems": 1, "transcript_source": "vtt"},
    )


def test_initial_page_renders():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.button[0].label == "Run assessment"


def test_results_render():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["assessment"] = synthetic_assessment()
    at.run()
    assert not at.exception
    assert any("Synthetic warning." in w.value for w in at.warning)
    assert len(at.expander) >= 3
    assert any(m.label == "Confidence" and m.value == "80%" for m in at.metric)


def test_save_upload_dedupes_by_hash(monkeypatch, tmp_path):
    from cqa import config
    from cqa.ingest import save_upload

    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    try:
        a = save_upload(io.BytesIO(b"WEBVTT\n"), ".vtt")
        b = save_upload(io.BytesIO(b"WEBVTT\n"), ".vtt")
        assert a == b and a.read_bytes() == b"WEBVTT\n" and a.parent == tmp_path / "uploads"
        assert len(list((tmp_path / "uploads").iterdir())) == 1
    finally:
        config.get_settings.cache_clear()
