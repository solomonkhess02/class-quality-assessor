"""Ablation switches (docs/EVALUATION.md, eval/ablations.md). Synthetic inputs only."""

import json

import pytest

from cqa import config
from cqa.judge import prompts


@pytest.fixture
def env(monkeypatch):
    """Set environment overrides for get_settings and clear its cache around the test."""

    def set_env(**values: str) -> None:
        for k, v in values.items():
            monkeypatch.setenv(k, v)
        config.get_settings.cache_clear()

    yield set_env
    config.get_settings.cache_clear()


def test_evidence_first_puts_score_last(env):
    env(JUDGE_EVIDENCE_FIRST="false")
    assert list(prompts.example()["criteria"][0])[1] == "score"  # baseline: score straight after the id
    env(JUDGE_EVIDENCE_FIRST="true")
    ex = prompts.example()
    for c in ex["criteria"]:
        keys = list(c)
        assert keys[-1] == "score"
        assert keys.index("observations") < keys.index("evidence") < keys.index("score")
    # same content, only the order changes
    assert json.dumps(ex, sort_keys=True) == json.dumps(prompts.EXAMPLE, sort_keys=True)
    sp = prompts.system_prompt()
    assert sp.index('"observations": [') < sp.index('"score": 3')
