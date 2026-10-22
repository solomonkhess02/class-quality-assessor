"""Transcript formats and input checks (synthetic files only)."""

import shutil
from pathlib import Path

import pytest

from cqa import config, ingest
from cqa.transcript.formats import (
    TranscriptFormatError,
    parse_transcript_bytes,
    restrict_to_windows,
    to_vtt,
)
from cqa.transcript.vtt import parse_vtt

VTT = (
    b"WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nTutor Name: Hello everyone\n\n"
    b"2\n00:00:05.000 --> 00:00:07.500\nKid A: Hi\n"
)
SRT = b"1\n00:00:01,000 --> 00:00:04,000\nTutor Name: Hello everyone\n\n2\n00:00:05,000 --> 00:00:07,500\nKid A: Hi\n"
SRT_NO_SPEAKERS = b"1\n00:00:01,000 --> 00:00:04,000\nHello everyone\n\n2\n00:00:05,000 --> 00:00:07,500\nhi there\n"
TXT_TIMED = b"[00:00:01] Tutor Name: Hello everyone\n[00:00:05] Kid A: Hi\n"
TXT_UNTIMED = b"Tutor Name: Hello everyone, today we learn forks\nKid A: Hi\nand a second line\n"


@pytest.mark.parametrize("data,suffix,fmt", [(VTT, ".vtt", "vtt"), (SRT, ".srt", "srt"), (TXT_TIMED, ".txt", "txt")])
def test_formats_with_speakers(data, suffix, fmt):
    p = parse_transcript_bytes(data, suffix)
    assert p.fmt == fmt and p.has_speakers and not p.notes
    assert [t.speaker for t in p.transcript.turns] == ["Tutor Name", "Kid A"]
    assert p.transcript.turns[1].start_s == 5.0


def test_srt_without_speakers():
    p = parse_transcript_bytes(SRT_NO_SPEAKERS, ".srt")
    assert p.fmt == "srt" and not p.has_speakers
    assert {t.speaker for t in p.transcript.turns} == {"UNKNOWN"}


def test_plain_text_without_times_is_estimated():
    p = parse_transcript_bytes(TXT_UNTIMED, ".txt")
    turns = p.transcript.turns
    assert p.has_speakers and p.notes and "estimated" in p.notes[0]
    assert turns[1].text == "Hi and a second line"
    assert turns[0].start_s == 0 and turns[1].start_s >= turns[0].end_s


def test_srt_detected_by_content_and_utf16():
    p = parse_transcript_bytes(SRT.decode().encode("utf-16"), ".srt")
    assert p.fmt == "srt" and len(p.transcript.turns) == 2


@pytest.mark.parametrize(
    "data,suffix,match",
    [
        (VTT, ".docx", "Unsupported transcript type"),
        (b"   ", ".vtt", "empty"),
        (b"WEBVTT\n\nno cues here\n", ".vtt", "no captions"),
        (b"just some words without a speaker\n", ".txt", "Name: text"),
        (b"WEBVTT\n\n00:xx --> 00:00:02.000\nA: b\n", ".vtt", "Could not read"),
    ],
)
def test_bad_transcripts_give_clear_messages(data, suffix, match):
    with pytest.raises(TranscriptFormatError, match=match):
        parse_transcript_bytes(data, suffix)


def test_to_vtt_round_trip():
    p = parse_transcript_bytes(SRT, ".srt")
    back = parse_vtt(to_vtt(p.transcript))
    assert [(t.start_s, t.end_s, t.speaker, t.text) for t in back.turns] == [
        (t.start_s, t.end_s, t.speaker, t.text) for t in p.transcript.turns
    ]
    unknown = parse_transcript_bytes(SRT_NO_SPEAKERS, ".srt").transcript
    assert "UNKNOWN" not in to_vtt(unknown)


def test_restrict_to_windows():
    t = parse_transcript_bytes(VTT, ".vtt").transcript
    assert [x.speaker for x in restrict_to_windows(t, [(4.0, 6.0)]).turns] == ["Kid A"]


# --- ingest: prepare_inputs, video and Drive checks ---


@pytest.fixture
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    config.get_settings.cache_clear()
    yield tmp_path
    config.get_settings.cache_clear()


def write(tmp: Path, name: str, data: bytes) -> Path:
    p = tmp / name
    p.write_bytes(data)
    return p


def test_prepare_zoom_vtt_passes_through(cache):
    vtt = write(cache, "a.vtt", VTT)
    p = ingest.prepare_inputs(None, vtt)
    assert p.vtt == vtt and p.speakers_identified and not p.temp_files


def test_prepare_srt_is_normalised(cache):
    srt = write(cache, "a.srt", SRT)
    p = ingest.prepare_inputs(None, srt)
    assert p.vtt is not None and p.vtt.suffix == ".vtt" and p.transcript_format == "srt"
    assert [t.speaker for t in parse_vtt(p.vtt.read_text(encoding="utf-8")).turns] == ["Tutor Name", "Kid A"]
    assert srt in p.temp_files


def test_prepare_no_speakers_with_video_uses_whisper_path(cache, monkeypatch):
    monkeypatch.setattr(ingest, "check_video", lambda path: 3000.0)
    video = write(cache, "v.mp4", b"synthetic")
    p = ingest.prepare_inputs(video, write(cache, "a.srt", SRT_NO_SPEAKERS))
    assert p.vtt is None and not p.speakers_identified
    assert any("not used" in n for n in p.notes)


def test_prepare_no_speakers_transcript_only(cache):
    p = ingest.prepare_inputs(None, write(cache, "a.srt", SRT_NO_SPEAKERS))
    assert p.vtt is not None and not p.speakers_identified


def test_prepare_long_recording_keeps_windows_only(cache, monkeypatch):
    monkeypatch.setattr(ingest, "check_video", lambda path: 3 * 3600.0)
    cues = "".join(
        f"\n{i}\n{i // 60:02d}:{i % 60:02d}:00.000 --> {i // 60:02d}:{i % 60:02d}:30.000\nTutor Name: minute {i}\n"
        for i in range(180)
    )
    vtt = write(cache, "long.vtt", ("WEBVTT\n" + cues).encode())
    p = ingest.prepare_inputs(write(cache, "v.mp4", b"synthetic"), vtt)
    assert p.vtt != vtt and any("3.0 h long" in n for n in p.notes)
    kept = parse_vtt(p.vtt.read_text(encoding="utf-8")).turns
    assert 10 < len(kept) < 60 and kept[0].start_s == 0 and kept[-1].start_s >= 170 * 60


def test_prepare_bad_transcript_is_input_error(cache):
    with pytest.raises(ingest.InputError, match="Unsupported transcript type"):
        ingest.prepare_inputs(None, write(cache, "a.pdf", b"%PDF"))


def test_unsupported_video_type(cache):
    with pytest.raises(ingest.InputError, match="Unsupported recording type"):
        ingest.check_video(write(cache, "a.avi", b"x"))


def test_corrupted_video(cache):
    ffprobe = config.get_settings().ffprobe_bin
    if not (Path(ffprobe).is_file() or shutil.which(ffprobe)):
        pytest.skip("ffprobe not available")
    with pytest.raises(ingest.InputError, match="corrupted"):
        ingest.check_video(write(cache, "broken.mp4", b"\x00\x00\x00\x18ftypmp42 not really a video" * 50))


@pytest.mark.parametrize(
    "url,match",
    [
        ("not a link", "does not look like a Google Drive file link"),
        ("https://example.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view", "does not look like"),
        ("https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz", "folder link"),
    ],
)
def test_invalid_drive_links(url, match):
    with pytest.raises(ingest.InputError, match=match):
        ingest.check_drive_url(url)


def test_private_drive_file_gives_clear_message(cache, monkeypatch):
    import gdown

    def refuse(*args, **kwargs):
        raise RuntimeError("Cannot retrieve the public link of the file.")  # stands in for gdown's own error

    monkeypatch.setattr(gdown, "download", refuse)
    with pytest.raises(ingest.InputError, match="Anyone with the link"):
        ingest.download_drive("https://drive.google.com/file/d/1AbCdEfGhIjKlMnOpQrStUvWxYz/view?usp=sharing")
