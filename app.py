"""Streamlit UI: upload a class recording (or a Google Drive link) and an optional Zoom VTT, get an assessment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cqa.housekeeping import RUN_LOCK, delete_run_files, purge_old
from cqa.ingest import download_drive, save_upload
from cqa.judge.llm import JudgeUnavailable, judge_available
from cqa.judge.rubric import BY_ID
from cqa.models import ensure_models, missing_models
from cqa.pipeline import run
from cqa.schemas import Assessment

LABEL_COLOURS = {"Excellent": "green", "Good": "blue", "Average": "orange", "Poor": "red"}
JOINERS = {"invited": "yes", "not_invited": "no", "none": "no new joiners seen", "unknown": "unknown"}
UNAVAILABLE = (
    "Assessment service unavailable: the LLM judge is not configured or its account cannot be used right now. "
    "Please try again later."
)
BUSY = "Another assessment is running on this server. Please try again in a few minutes."

st.set_page_config(page_title="Class Quality Assessor", page_icon="♟️", layout="wide")
st.title("Class Quality Assessor")
st.caption(
    "Assesses a recorded 8x8 online chess class. Video and audio are processed on this server; only "
    "pseudonymised transcript text and numeric metrics are sent to the LLM judge."
)
if not judge_available():
    st.error(UNAVAILABLE)
if missing_models():  # first start on a fresh host (Community Cloud); local and Docker already have them
    try:
        with st.spinner("Downloading the face-detection and speech models (first start only, about 60 MB)"):
            ensure_models()
    except Exception as e:
        st.error(f"Could not download the models: {e}")


def get_inputs() -> tuple:
    """Widget values (video upload, Drive link, VTT upload) and whether Run was pressed."""
    src = st.radio("Recording", ["Upload mp4", "Google Drive link", "No recording (transcript only)"], horizontal=True)
    video = drive_url = None
    if src == "Upload mp4":
        video = st.file_uploader("Class recording", type=["mp4", "mov", "mkv", "webm"])
        st.caption("For recordings over about 200 MB, a Google Drive link is faster and more reliable than uploading.")
    elif src == "Google Drive link":
        drive_url = st.text_input("Google Drive share link (must be viewable by anyone with the link)")
    vtt = st.file_uploader("Zoom transcript (.vtt, optional but recommended)", type=["vtt"])
    if not vtt and src != "No recording (transcript only)":
        st.info(
            "No transcript given: the audio will be transcribed locally with whisper.cpp. This is much slower "
            "(roughly 10-30 minutes for a 50-minute class) and speakers are not identified, so engagement "
            "criteria are less reliable."
        )
    return video, drive_url, vtt, st.button("Run assessment", type="primary")


def save_inputs(video, drive_url: str | None, vtt) -> tuple[Path | None, Path | None]:
    video_path = vtt_path = None
    if video is not None:
        video_path = save_upload(video, Path(video.name).suffix.lower() or ".mp4")
    elif drive_url:
        with st.spinner("Downloading from Google Drive"):
            video_path = download_drive(drive_url.strip())
    if vtt is not None:
        vtt_path = save_upload(vtt, ".vtt")
    return video_path, vtt_path


def assess(video: Path | None, vtt: Path | None) -> Assessment | None:
    if not judge_available():  # fail before spending minutes on frames
        st.error(UNAVAILABLE)
        return None
    with st.status("Assessing the class...", expanded=True) as status:

        def progress(stage: str, message: str) -> None:
            status.update(label=message)
            status.write(f"**{stage}**: {message}")

        try:
            assessment, _ = run(video, vtt, progress=progress)
        except JudgeUnavailable:
            status.update(label="Assessment service unavailable", state="error")
            st.error(UNAVAILABLE)
            return None
        except Exception as e:  # show the failure in the UI instead of a stack trace
            status.update(label="Assessment failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            return None
        status.update(label="Assessment complete", state="complete", expanded=False)
    return assessment


def show(a: Assessment) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Label**")
        st.badge(a.label, color=LABEL_COLOURS[a.label])
    c2.metric("Confidence", f"{a.confidence:.0%}")
    c3.metric("Problems (criteria scored ≤ 2)", a.meta.get("problems", "-"))
    c4.metric("Weighted mean (1-4)", "-" if a.weighted_score is None else f"{a.weighted_score:.2f}")
    if a.summary:
        st.write(a.summary)
    for w in a.warnings:
        st.warning(w)

    st.subheader("Criteria")
    rows = []
    for c in a.criteria:
        crit = BY_ID.get(c.id)
        rows.append(
            {
                "criterion": crit.name if crit else c.id,
                "score (1-4)": "insufficient evidence" if c.insufficient_evidence or c.score is None else c.score,
                "problem": "yes" if (c.score is not None and not c.insufficient_evidence and c.score <= 2) else "",
                "weight": crit.weight if crit else None,
                "verified quotes": len(c.evidence),
                "scored by": "video metrics" if crit and crit.scored_by == "code" else "LLM judge",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    for c in a.criteria:
        crit = BY_ID.get(c.id)
        score = "insufficient evidence" if c.insufficient_evidence or c.score is None else f"{c.score}/4"
        with st.expander(f"{crit.name if crit else c.id}: {score}"):
            for o in c.observations:
                st.markdown(f"- {o}")
            if c.evidence:
                st.markdown("**Evidence (verified against the transcript)**")
                for e in c.evidence:
                    st.markdown(f"- `{e.timestamp}` **{e.speaker}**: “{e.quote}”")
            if c.improvement:
                st.markdown(f"**Improvement:** {c.improvement}")

    if a.key_facts:
        kf = a.key_facts
        st.subheader("Key facts")
        facts = pd.DataFrame(
            [
                {
                    "fact": "Homework assigned",
                    "value": f"yes: {kf.homework_what}" if kf.homework_assigned == "yes" else kf.homework_assigned,
                },
                {"fact": "New joiners invited to introduce themselves", "value": JOINERS[kf.new_joiners]},
                {"fact": "Children greeted by name before teaching", "value": str(kf.children_greeted_by_name)},
                {"fact": "Recap at the end", "value": kf.recap_present},
                {
                    "fact": "Warm-up before content (rapport)",
                    "value": "unknown (content start not found)"
                    if kf.warm_up_min is None
                    else f"{kf.warm_up_min:g} min, {kf.warm_up_students} children spoke: {kf.rapport_before_content}",
                },
            ]
        )
        st.dataframe(facts, hide_index=True, width="stretch")
        if kf.evidence:
            with st.expander("Key fact evidence"):
                for e in kf.evidence:
                    st.markdown(f"- {e.fact}: `{e.timestamp}` **{e.speaker}**: “{e.quote}”")

    col_s, col_i = st.columns(2)
    with col_s:
        st.subheader("Strengths")
        for s in a.strengths:
            st.markdown(f"- {s}")
    with col_i:
        st.subheader("Improvements")
        for s in a.improvements:
            st.markdown(f"- {s}")

    st.subheader("Measurements")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**Video (sampled frames)**")
        if a.visual:
            v = a.visual
            st.dataframe(
                pd.DataFrame(
                    [
                        {"metric": "frames analysed", "value": v.n_frames},
                        {"metric": "face visible (% of frames)", "value": v.pct_face},
                        {"metric": "screen share (% of frames)", "value": v.pct_screen_share},
                        {"metric": "face visible on camera frames (%)", "value": v.pct_face_camera},
                        {"metric": "dark frames (%)", "value": v.pct_dark},
                        {"metric": "black frames (%)", "value": v.pct_black},
                        {"metric": "frozen frames (%)", "value": v.pct_frozen},
                        {"metric": "median sharpness (Laplacian)", "value": v.median_blur},
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.write("Video not analysed.")
    with m2:
        st.markdown("**Transcript statistics**")
        if a.text_stats:
            st.dataframe(
                pd.DataFrame([{"metric": k, "value": v} for k, v in a.text_stats.model_dump().items()]),
                hide_index=True,
                width="stretch",
            )

    with st.expander("Run details"):
        st.json({k: v for k, v in a.meta.items() if k != "facts"})
    st.download_button(
        "Download assessment (JSON)",
        a.model_dump_json(indent=2),
        file_name="assessment.json",
        mime="application/json",
    )


def run_once(video, drive_url: str | None, vtt) -> None:
    """One assessment at a time per server (2 vCPUs, one API balance); inputs and media deleted afterwards."""
    if not RUN_LOCK.acquire(blocking=False):
        st.warning(BUSY)
        return
    video_path = vtt_path = None
    try:
        purge_old()
        video_path, vtt_path = save_inputs(video, drive_url, vtt)
        if video_path is None and vtt_path is None:
            st.error("Please provide a recording, a Drive link or a VTT transcript.")
            return
        st.session_state["assessment"] = assess(video_path, vtt_path)
    finally:
        delete_run_files(video_path, vtt_path)
        RUN_LOCK.release()


video_up, drive_url, vtt_up, go = get_inputs()
if go:
    run_once(video_up, drive_url, vtt_up)
if st.session_state.get("assessment"):
    show(st.session_state["assessment"])
