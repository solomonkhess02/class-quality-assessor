"""Streamlit UI: upload a class recording (or a Google Drive link) and an optional transcript, get an assessment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from cqa.housekeeping import RUN_LOCK, delete_run_files, purge_old
from cqa.ingest import VIDEO_TYPES, InputError, PreparedInputs, download_drive, prepare_inputs, save_upload
from cqa.judge.llm import JudgeUnavailable, judge_available
from cqa.judge.rubric import BY_ID
from cqa.models import ensure_models, missing_models
from cqa.pipeline import run
from cqa.report import (
    NEXT_LEVEL,
    PLAYER_MAX_BYTES,
    cannot_check,
    first_sentence,
    generic_note,
    key_reasons,
    next_level_actions,
    player_mode,
    reliability_notes,
    short,
    transcript_caution,
    unusable_without_recording,
)
from cqa.schemas import Assessment, Evidence, parse_ts
from cqa.transcript.formats import TRANSCRIPT_TYPES

LABEL_HEX = {"Excellent": "#1a7f37", "Good": "#0969da", "Average": "#b26a00", "Poor": "#cf222e"}
LEGEND = "Children's names are replaced with S1, S2… for privacy; TUTOR = the teacher."
JOINERS = {"invited": "yes", "not_invited": "no", "none": "no new joiners seen", "unknown": "unknown"}
UNAVAILABLE = (
    "Assessment service unavailable: the LLM judge is not configured or its account cannot be used right now. "
    "Please try again later."
)
BUSY = "Another assessment is running on this server. Please try again in a few minutes."
NOT_USED = "not used for the label"

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
    """Widget values (video upload, Drive link, transcript upload) and whether Run was pressed."""
    src = st.radio(
        "Recording", ["Upload recording", "Google Drive link", "No recording (transcript only)"], horizontal=True
    )
    video = drive_url = None
    if src == "Upload recording":
        video = st.file_uploader("Class recording", type=[t.lstrip(".") for t in VIDEO_TYPES])
        st.caption("For recordings over about 200 MB, a Google Drive link is faster and more reliable than uploading.")
    elif src == "Google Drive link":
        drive_url = st.text_input("Google Drive share link (must be viewable by anyone with the link)")
    transcript = st.file_uploader(
        "Transcript (optional but recommended): Zoom .vtt, or .srt, or .txt with 'Name: text' lines",
        type=list(TRANSCRIPT_TYPES),
    )
    st.selectbox(  # whisper language when no transcript is given (E25: no auto-detection)
        "Class language", ["en", "ml"], format_func={"en": "English", "ml": "Malayalam"}.get, key="class_language"
    )
    if not transcript and src != "No recording (transcript only)":
        st.info(
            "No transcript given: the audio will be transcribed locally with whisper.cpp. This is much slower "
            "(a few minutes: only the opening, closing and two short middle samples are transcribed) and "
            "speakers are not identified, so engagement criteria are less reliable."
        )
    return video, drive_url, transcript, st.button("Run assessment", type="primary")


def save_inputs(video, drive_url: str | None, transcript) -> tuple[Path | None, Path | None]:
    video_path = transcript_path = None
    if transcript is not None:
        transcript_path = save_upload(transcript, Path(transcript.name).suffix.lower() or ".vtt")
    if video is not None:
        video_path = save_upload(video, Path(video.name).suffix.lower() or ".mp4")
    elif drive_url and drive_url.strip():
        with st.spinner("Downloading from Google Drive"):
            video_path = download_drive(drive_url)
    return video_path, transcript_path


def assess(video: Path | None, vtt: Path | None) -> Assessment | None:
    if not judge_available():  # fail before spending minutes on frames
        st.error(UNAVAILABLE)
        return None
    with st.status("Assessing the class...", expanded=True) as status:

        def progress(stage: str, message: str) -> None:
            status.update(label=message)
            status.write(f"**{stage}**: {message}")

        try:
            assessment, _ = run(video, vtt, progress=progress, language=st.session_state.get("class_language"))
        except JudgeUnavailable:
            status.update(label="Assessment service unavailable", state="error")
            st.error(UNAVAILABLE)
            return None
        except Exception as e:  # show the failure in the UI instead of a stack trace
            status.update(label="Assessment failed", state="error")
            st.error(f"The assessment failed ({type(e).__name__}): {str(e)[:300]}")
            return None
        status.update(label="Assessment complete", state="complete", expanded=False)
    return assessment


# --- report ---


def kept_video() -> Path | None:
    p = st.session_state.get("video_path")
    return Path(p) if p is not None else None


def seek(seconds: int) -> None:
    st.session_state["seek_to"] = seconds


def show_player() -> None:
    """The page's single video player; the ▶ buttons only move its start time (seek_to)."""
    mode = player_mode(kept_video())
    if mode == "too_large":
        st.caption(
            f"The recording is over {PLAYER_MAX_BYTES // 2**20} MB, so it is not played here; open it in your own "
            "player at the timestamps shown."
        )
    elif mode == "player":
        if "seek_to" in st.session_state:
            st.video(str(kept_video()), start_time=st.session_state["seek_to"])
        else:
            st.caption("Press ▶ next to a quote to play the recording from that moment.")


def evidence_line(e: Evidence, key: str, prefix: str = "") -> None:
    """One verified quote, with a button that moves the player to its timestamp."""
    text = f"{prefix}`{e.timestamp}` **{e.speaker}**: “{short(e.quote)}”"
    if player_mode(kept_video()) != "player":
        st.markdown(f"- {text}")
        return
    c1, c2 = st.columns([6, 1])
    c1.markdown(f"- {text}")
    c2.button(
        f"▶ {e.timestamp}",
        key=f"play-{key}",
        help="Play the recording above from this moment",
        on_click=seek,  # callbacks run before the rerun, so the player above already sees the new time
        args=(parse_ts(e.timestamp),),
    )


def show_header(a: Assessment) -> None:
    """The label is the hero: a large colour-coded badge, the one-line summary under it."""
    colour = LABEL_HEX[a.label]
    st.markdown(
        f'<div style="display:inline-block;padding:0.35em 0.9em;border-radius:0.4em;background:{colour};'
        f'color:#fff;font-size:2.4rem;font-weight:700;line-height:1.2">{a.label}</div>',
        unsafe_allow_html=True,
    )
    if a.summary:
        st.markdown(f"**{first_sentence(a.summary)}**")


def show_reasons(a: Assessment) -> None:
    show_player()
    st.caption(LEGEND)
    st.subheader("Key reasons")
    reasons = key_reasons(a)
    if not reasons:
        st.write("No single finding stands out.")
    for i, r in enumerate(reasons):
        st.markdown(f"{i + 1}. {'⚠️' if r.problem else '✅'} {r.text}")
        if r.evidence is not None:
            evidence_line(r.evidence, f"reason-{i}")
    st.subheader("What would move this class up a level" if a.label in NEXT_LEVEL else "What would keep it there")
    if a.label in NEXT_LEVEL:
        st.caption(f"From {a.label} towards {NEXT_LEVEL[a.label]}.")
    for s in next_level_actions(a):
        st.markdown(f"- {s}")


def show_criteria(a: Assessment) -> None:
    st.subheader("Criteria")
    excluded = set(a.meta.get("label_excluded", []))
    rows = []
    for c in a.criteria:
        crit = BY_ID.get(c.id)
        rows.append(
            {
                "criterion": crit.name if crit else c.id,
                "score (1-4)": "insufficient evidence" if c.insufficient_evidence or c.score is None else c.score,
                "used for label": NOT_USED if c.id in excluded else "yes",
                "verified quotes": len(c.evidence),
                "scored by": "video metrics" if crit and crit.scored_by == "code" else "LLM judge",
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    for c in a.criteria:
        crit = BY_ID.get(c.id)
        score = "insufficient evidence" if c.insufficient_evidence or c.score is None else f"{c.score}/4"
        tag = f" ({NOT_USED})" if c.id in excluded else ""
        with st.expander(f"{crit.name if crit else c.id}: {score}{tag}"):
            if c.id in excluded:
                st.caption(f"Shown for information, {NOT_USED}: its score barely varies between classes (D15).")
            for o in c.observations:
                st.markdown(f"- {o}")
            if c.evidence:
                st.markdown("**Evidence (verified against the transcript)**")
                for i, e in enumerate(c.evidence):
                    evidence_line(e, f"{c.id}-{i}")
            if c.improvement:
                st.markdown(f"**Improvement:** {c.improvement}")


def show_key_facts(a: Assessment) -> None:
    if not a.key_facts:
        return
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
                "value": f"{kf.rapport_before_content} (judged from the transcript; speakers not identified)"
                if kf.rapport_method == "judge"
                else "unknown (content start not found)"
                if kf.warm_up_min is None
                else f"{kf.warm_up_min:g} min, {kf.warm_up_students} children spoke: {kf.rapport_before_content}",
            },
        ]
    )
    st.dataframe(facts, hide_index=True, width="stretch")
    if kf.evidence:
        with st.expander("Key fact evidence"):
            for i, e in enumerate(kf.evidence):
                evidence_line(e, f"fact-{i}", prefix=f"{e.fact}: ")


def show_limits(a: Assessment) -> None:
    inputs = a.meta.get("input", {})
    speakers = a.meta.get("transcript_source") != "whisper" and inputs.get("speakers_identified", True)
    col_c, col_r = st.columns(2)
    with col_c:
        st.subheader("What this assessment could not check")
        for s in cannot_check(a, speakers):
            st.markdown(f"- {s}")
    with col_r:
        st.subheader("Reliability notes")
        for s in reliability_notes(a, inputs):
            st.markdown(f"- {s}")


def show_measurements(a: Assessment) -> None:
    col_s, col_i = st.columns(2)
    with col_s:
        st.subheader("Strengths")
        for s in a.strengths:
            st.markdown(f"- {s}")
    with col_i:
        st.subheader("All improvements")
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
        if note := generic_note(a):
            st.markdown(note)
        st.json({k: v for k, v in a.meta.items() if k != "facts"})
    st.download_button(
        "Download assessment (JSON)",
        a.model_dump_json(indent=2, exclude={"confidence"}),
        file_name="assessment.json",
        mime="application/json",
    )


def show(a: Assessment) -> None:
    if unusable_without_recording(a):
        st.error(
            "This class could not be assessed: the transcript could not be read reliably and there is no recording "
            "to fall back on. " + (transcript_caution(a) or "")
        )
        return
    show_header(a)
    show_reasons(a)
    show_criteria(a)
    show_limits(a)
    show_key_facts(a)
    show_measurements(a)


def run_once(video, drive_url: str | None, transcript) -> None:
    """One assessment at a time per server (2 vCPUs, one API balance). Transcripts and derived media are deleted
    afterwards; the recording stays for playback until the cache purge."""
    if not RUN_LOCK.acquire(blocking=False):
        st.warning(BUSY)
        return
    video_path = transcript_path = None
    prep: PreparedInputs | None = None
    assessment = None
    try:
        purge_old()
        video_path, transcript_path = save_inputs(video, drive_url, transcript)
        if video_path is None and transcript_path is None:
            st.error("Please provide a recording, a Drive link or a transcript.")
            return
        prep = prepare_inputs(video_path, transcript_path)
        for note in prep.notes:
            st.info(note)
        assessment = assess(prep.video, prep.vtt)
        if assessment is not None:
            assessment.meta["input"] = {
                "transcript_format": prep.transcript_format,
                "speakers_identified": prep.speakers_identified,
                "duration_s": prep.duration_s,
                "notes": prep.notes,
            }
        st.session_state["assessment"] = assessment
        st.session_state["video_path"] = video_path if assessment is not None else None
        st.session_state.pop("seek_to", None)
    except InputError as e:
        st.error(str(e))
    finally:
        extra = [p for p in (transcript_path, *(prep.temp_files if prep else [])) if p is not None]
        delete_run_files(video_path, prep.vtt if prep else None, extra, keep_video=assessment is not None)
        RUN_LOCK.release()


video_up, drive_url, transcript_up, go = get_inputs()
if go:
    run_once(video_up, drive_url, transcript_up)
if st.session_state.get("assessment"):
    show(st.session_state["assessment"])
