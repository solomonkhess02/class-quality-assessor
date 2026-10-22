"""Public deployment: service-unavailable handling, media cleanup, one run at a time (synthetic, no network)."""

import os
import time
from pathlib import Path

import httpx2  # openai >= 3 uses httpx2
import openai
import pytest
from pydantic import BaseModel
from streamlit.testing.v1 import AppTest

from cqa import config, housekeeping
from cqa.ingest import sha256_file
from cqa.judge import llm

APP = str(Path(__file__).resolve().parent.parent / "app.py")
KEY = "sk-test-not-a-real-key-0000"


@pytest.fixture
def settings(monkeypatch, tmp_path):
    """Isolated settings: cache in tmp_path; returns a setter for further env vars."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", KEY)

    def set_env(**values: str) -> config.Settings:
        for k, v in values.items():
            monkeypatch.setenv(k, v)
        config.get_settings.cache_clear()
        return config.get_settings()

    set_env()
    yield set_env
    config.get_settings.cache_clear()


class Out(BaseModel):
    ok: bool


MESSAGES = [{"role": "user", "content": "synthetic"}]


@pytest.mark.parametrize("key", ["", "your-deepseek-key-here"])
def test_missing_key_is_unavailable(settings, key):
    settings(DEEPSEEK_API_KEY=key)
    assert not llm.judge_available()
    with pytest.raises(llm.JudgeUnavailable):
        llm.call_json(MESSAGES, Out)


@pytest.mark.parametrize("status", [401, 402, 403])
def test_refused_account_is_unavailable(settings, monkeypatch, status):
    def refuse(*args, **kwargs):
        response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.example.invalid"))
        raise openai.APIStatusError("refused", response=response, body=None)

    monkeypatch.setattr(llm, "_create", refuse)
    with pytest.raises(llm.JudgeUnavailable) as err:
        llm.call_json(MESSAGES, Out)
    assert str(status) in str(err.value)
    assert KEY not in str(err.value)


def test_other_status_errors_still_raise(settings, monkeypatch):
    def bad_request(*args, **kwargs):
        response = httpx2.Response(400, request=httpx2.Request("POST", "https://api.example.invalid"))
        raise openai.BadRequestError("bad", response=response, body=None)

    monkeypatch.setattr(llm, "_create", bad_request)
    with pytest.raises(openai.BadRequestError):
        llm.call_json(MESSAGES, Out)


def test_app_shows_unavailable_without_key(settings):
    settings(DEEPSEEK_API_KEY="")
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Assessment service unavailable" in e.value for e in at.error)


def test_app_one_assessment_at_a_time(settings):
    assert housekeeping.RUN_LOCK.acquire(blocking=False)
    try:
        at = AppTest.from_file(APP, default_timeout=30).run()
        at.button[0].click().run()
        assert not at.exception
        assert any("Another assessment is running" in w.value for w in at.warning)
    finally:
        housekeeping.RUN_LOCK.release()


def make_class(cache: Path) -> tuple[Path, Path]:
    """A fake uploaded recording and its class cache dir with media, metrics and an assessment."""
    uploads = cache / "uploads"
    uploads.mkdir(parents=True)
    video = uploads / "abc.mp4"
    video.write_bytes(b"synthetic video bytes")
    cdir = cache / sha256_file(video)[:16]
    (cdir / "audio").mkdir(parents=True)
    (cdir / "audio" / "audio.wav").write_bytes(b"wav")
    (cdir / "frames").mkdir()
    (cdir / "frames" / "frame_000000.jpg").write_bytes(b"jpg")
    (cdir / "whisper.json").write_text("{}")
    (cdir / "assessment.json").write_text("{}")
    return video, cdir


def test_delete_run_files_keeps_only_assessment(settings):
    s = settings(DELETE_MEDIA_AFTER_RUN="true")
    video, cdir = make_class(s.cache_dir)
    housekeeping.delete_run_files(video, None)
    assert not video.exists()
    assert sorted(p.name for p in cdir.iterdir()) == ["assessment.json"]


def test_delete_run_files_off_by_default(settings):
    s = settings()
    video, cdir = make_class(s.cache_dir)
    housekeeping.delete_run_files(video, None)
    assert video.exists()
    assert (cdir / "audio" / "audio.wav").exists()


def test_purge_old_removes_only_old_entries(settings):
    s = settings(CACHE_MAX_AGE_H="2")
    video, cdir = make_class(s.cache_dir)
    (s.cache_dir / "llm").mkdir()
    old_llm, new_llm = s.cache_dir / "llm" / "old.json", s.cache_dir / "llm" / "new.json"
    old_llm.write_text("{}")
    new_llm.write_text("{}")
    three_hours_ago = time.time() - 3 * 3600
    for p in (video, cdir, old_llm):
        os.utime(p, (three_hours_ago, three_hours_ago))
    assert housekeeping.purge_old() == 3
    assert not video.exists() and not cdir.exists() and not old_llm.exists()
    assert new_llm.exists() and (s.cache_dir / "uploads").is_dir()


def test_purge_old_off_by_default(settings):
    s = settings()
    video, cdir = make_class(s.cache_dir)
    os.utime(cdir, (0, 0))
    assert housekeeping.purge_old() == 0
    assert cdir.exists()
