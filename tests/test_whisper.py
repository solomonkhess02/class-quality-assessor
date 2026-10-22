from cqa.transcript.whisper import parse_whisper_json

SAMPLE = {
    "result": {"language": "en"},
    "transcription": [
        {
            "timestamps": {"from": "00:00:00,000", "to": "00:00:02,500"},
            "offsets": {"from": 0, "to": 2500},
            "text": " Hello children.",
        },
        {"timestamps": {}, "offsets": {"from": 2500, "to": 3000}, "text": "   "},
        {
            "timestamps": {"from": "00:00:03,000", "to": "00:00:05,000"},
            "offsets": {"from": 3000, "to": 5000},
            "text": " What is a pin?",
        },
    ],
}


def test_parse_with_offset():
    turns, lang = parse_whisper_json(SAMPLE, offset_s=600)
    assert lang == "en"
    assert [(t.start_s, t.end_s, t.text) for t in turns] == [
        (600.0, 602.5, "Hello children."),
        (603.0, 605.0, "What is a pin?"),
    ]
    assert all(t.speaker == "UNKNOWN" and t.role == "unknown" for t in turns)


def test_parse_empty():
    assert parse_whisper_json({}) == ([], "unknown")


def test_invalid_utf8_from_whisper_is_replaced(tmp_path):
    from cqa.transcript.whisper import read_whisper_json

    # a Malayalam character cut in half by a token boundary: valid lead byte, missing continuation byte
    raw = b'{"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "ok \xe0\xb4 ok"}]}'
    path = tmp_path / "out.json"
    path.write_bytes(raw)
    data = read_whisper_json(path)
    assert data["transcription"][0]["text"].startswith("ok ") and "\ufffd" in data["transcription"][0]["text"]


def test_transcribe_wavs_detects_language_once_and_passes_prompt(monkeypatch):
    from pathlib import Path

    from cqa import config
    from cqa.transcript import whisper

    calls = []

    def fake(wav, offset, language=None, model=None, threads=None, prompt=""):
        calls.append((language, prompt))
        return [], "en" if offset == 0 else "hi"

    monkeypatch.setattr(whisper, "transcribe", fake)
    wavs = [(Path("a.wav"), 0.0), (Path("b.wav"), 600.0), (Path("c.wav"), 1200.0)]
    for once, prompt, expected in (
        ("true", "true", [("auto", whisper.CHESS_PROMPT), ("en", whisper.CHESS_PROMPT), ("en", whisper.CHESS_PROMPT)]),
        ("false", "false", [("auto", ""), ("auto", ""), ("auto", "")]),
    ):
        monkeypatch.setenv("WHISPER_LANGUAGE", "auto")
        monkeypatch.setenv("ASR_DETECT_LANGUAGE_ONCE", once)
        monkeypatch.setenv("ASR_CHESS_PROMPT", prompt)
        config.get_settings.cache_clear()
        calls.clear()
        try:
            whisper.transcribe_wavs(wavs)
        finally:
            config.get_settings.cache_clear()
        assert calls == expected
