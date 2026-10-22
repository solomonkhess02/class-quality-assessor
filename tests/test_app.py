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
        meta={"problems": 1, "transcript_source": "vtt", "label_excluded": ["communication"]},
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
    headers = [h.value for h in at.subheader]
    for h in ("Key reasons", "What would move this class up a level", "Criteria", "Reliability notes",
              "What this assessment could not check"):  # fmt: skip
        assert h in headers
    assert headers.index("Key reasons") < headers.index("Criteria") < headers.index("Reliability notes")
    text = " ".join(m.value for m in at.markdown)
    assert "Synthetic warning." in text and "8x8 branded background" in text
    assert "Confidence" not in text and not any(m.label == "Confidence" for m in at.metric)
    assert any("not used for the label" in e.label for e in at.expander)
    assert len(at.expander) >= 3
    assert not [b for b in at.button if b.label.startswith("▶")]  # no recording kept, so nothing to play


def test_play_at_timestamp(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"synthetic")
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["assessment"] = synthetic_assessment()
    at.session_state["video_path"] = video
    at.run()
    play = [b for b in at.button if b.label == "▶ 00:00:40"]
    assert len(play) >= 2  # the quote appears under Key reasons and under its criterion
    assert not at.get("video")  # no player until a ▶ is pressed
    play[0].click().run()
    assert not at.exception
    assert at.session_state["seek_to"] == 40
    videos = at.get("video")
    assert len(videos) == 1 and videos[0].proto.start_time == 40  # one player for the whole page


def test_large_recording_gets_timestamps_only(tmp_path, monkeypatch):
    from cqa import report

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 2048)
    monkeypatch.setattr(report, "PLAYER_MAX_BYTES", 1024)  # stands in for a recording over 200 MB
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["assessment"] = synthetic_assessment()
    at.session_state["video_path"] = video
    at.run()
    assert not at.exception
    assert not [b for b in at.button if b.label.startswith("▶")] and not at.get("video")
    assert any("not played here" in c.value for c in at.caption)
    assert any("`00:00:40`" in m.value for m in at.markdown)  # the timestamp is still shown


def test_player_mode_size_switch(tmp_path, monkeypatch):
    from cqa import report

    video = tmp_path / "v.mp4"
    video.write_bytes(b"x" * 100)
    monkeypatch.setattr(report, "PLAYER_MAX_BYTES", 100)
    assert report.player_mode(video) == "player"
    video.write_bytes(b"x" * 101)
    assert report.player_mode(video) == "too_large"
    assert report.player_mode(tmp_path / "gone.mp4") == "none" and report.player_mode(None) == "none"


def test_hero_label_and_legend():
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["assessment"] = synthetic_assessment()
    at.run()
    assert not at.exception
    assert any("font-size:2.4rem" in m.value and ">Good<" in m.value for m in at.markdown)
    assert any(c.value.startswith("Children's names are replaced with S1, S2") for c in at.caption)


def test_transcript_caution_once_and_human_names():
    from cqa import report

    a = synthetic_assessment()
    a.meta["transcript_quality"] = {"usable": False, "words_per_min": 8.2, "non_english_share": 0.01}
    a.meta.update({"generic_observations": 2, "observations": 20})
    a.warnings = [
        "The Zoom transcript looks unusable (only 8 words per minute were transcribed) and there is no recording to "
        "fall back on.",
        "Transcript quality is low: only 8 words per minute were transcribed.",
        "technical_quality: no verified evidence, excluded from the score",
        "2 of 20 observations look generic",
    ]
    notes = report.reliability_notes(a, {})
    cautions = [n for n in notes if "words" in n and "min" in n]
    assert cautions == ["The transcript is sparse (8 words/min), so findings may miss parts of the class."]
    assert not any("unusable" in n or "generic" in n for n in notes)
    assert "Technical quality: no verified evidence, excluded from the score" in notes
    assert report.generic_note(a).startswith("2 of 20 judge observations look generic")
    assert not report.unusable_without_recording(a)  # sparse only: still assessed, with the caution


def test_unusable_transcript_without_recording_shows_no_label():
    a = synthetic_assessment()
    a.visual = None
    a.meta["transcript_quality"] = {"usable": False, "words_per_min": 30.0, "non_english_share": 0.4}
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["assessment"] = a
    at.run()
    assert not at.exception
    assert any("could not be assessed" in e.value for e in at.error)
    assert not any(">Good<" in m.value for m in at.markdown) and "Key reasons" not in [h.value for h in at.subheader]


def test_lighting_reason_in_plain_words():
    from cqa import report

    a = synthetic_assessment()
    a.visual = a.visual.model_copy(update={"median_face_brightness": 82.0, "pct_face_camera": 90.0})
    a.meta.update({"label_excluded": ["communication"], "critical": [], "moderate": ["lighting: face luma 82 (< 85)"]})
    reasons = report.key_reasons(a)
    assert reasons[0].text == (
        "Moderate problem. Lighting: the tutor's face is slightly underexposed (brightness 82, threshold 85)"
    )
    assert report.lighting_text(40).startswith("Lighting: the tutor's face is underexposed")


def test_report_sections():
    from cqa import report

    a = synthetic_assessment()
    a.meta.update({"label_excluded": ["communication"], "moderate": [], "critical": []})
    reasons = report.key_reasons(a)
    assert reasons and not reasons[0].problem and reasons[0].evidence.quote == "hello S1"
    assert report.next_level_actions(a)[0] == "Assign homework."
    assert any("not identified" in s for s in report.cannot_check(a, speakers_identified=False))
    notes = report.reliability_notes(a, {"transcript_format": "srt", "speakers_identified": False})
    assert notes[0] == "Transcript: SRT subtitles uploaded with the class; speakers not identified."
    assert report.first_sentence("One. Two.") == "One."


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
