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
