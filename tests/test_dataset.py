from pathlib import Path

from cqa.dataset import discover

VTT = "WEBVTT\n\n1\n00:00:00.000 --> {end}\nA: hi\n"


def make(folder: Path, name: str, content: str = "x") -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(content, encoding="utf-8")


def test_pairs_by_id_and_maps_labels(tmp_path: Path):
    make(tmp_path / "good", "GMT2024 B435 recording.mp4")
    make(tmp_path / "good", "B435 transcript.vtt", VTT.format(end="00:10:00.000"))
    make(tmp_path / "quality issue", "B392 rec.mp4")
    make(tmp_path / "quality issue", "B392.vtt", VTT.format(end="00:10:00.000"))
    items = {i.key: i for i in discover(tmp_path)}
    assert set(items) == {"good/B435", "quality issue/B392"}
    assert items["good/B435"].label == "Good"
    assert items["good/B435"].vtt.name == "B435 transcript.vtt"
    assert items["quality issue/B392"].label == "Poor"
    assert items["quality issue/B392"].technical_flag is True


def test_demo_paired_by_duration(tmp_path: Path):
    demo = tmp_path / "demo"
    make(demo, "a.mp4", "x" * 10)
    make(demo, "b.mp4", "x" * 20)
    make(demo, "long.vtt", VTT.format(end="00:50:00.000"))
    make(demo, "short.vtt", VTT.format(end="00:20:00.000"))
    durations = {"a.mp4": 3000.0, "b.mp4": 1200.0}
    items = discover(tmp_path, probe_duration=lambda p: durations[p.name])
    assert [(i.key, i.label, i.vtt.name) for i in items] == [
        ("demo/1", None, "long.vtt"),
        ("demo/2", None, "short.vtt"),
    ]


def test_load_expected_drops_nulls_and_notes(tmp_path: Path):
    import json

    from cqa.dataset import load_expected

    (tmp_path / "expected_criteria.json").write_text(
        json.dumps(
            {
                "_about": "meta",
                "good/B1": {"label": "Good", "opening_rapport": "good", "communication": "weak",
                            "closure_homework": None, "notes": "free text"},
            }
        )
    )  # fmt: skip
    assert load_expected(tmp_path) == {"good/B1": {"opening_rapport": "good", "communication": "weak"}}
    assert load_expected(tmp_path / "missing") == {}
