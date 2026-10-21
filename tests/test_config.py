from pathlib import Path

from cqa import config


def test_env_overrides_and_relative_paths(monkeypatch):
    monkeypatch.setenv("JUDGE_TEMPERATURE", "0.7")
    monkeypatch.setenv("CACHE_DIR", "some_cache")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-value")
    config.get_settings.cache_clear()
    try:
        s = config.get_settings()
        assert s.judge_temperature == 0.7
        assert s.cache_dir == config.REPO_ROOT / "some_cache"
        assert "secret-value" not in repr(s)
    finally:
        config.get_settings.cache_clear()


def test_absolute_path_kept(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    try:
        s = config.get_settings()
        assert s.data_dir == tmp_path
        assert s.classes_dir == tmp_path / "class recordings"
    finally:
        config.get_settings.cache_clear()
