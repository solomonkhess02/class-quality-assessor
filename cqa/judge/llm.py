"""The single entry point for LLM calls: DeepSeek via the openai SDK, JSON mode,
retries, re-ask on empty/invalid output, and a disk cache keyed by request hash."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from cqa.config import get_settings

T = TypeVar("T", bound=BaseModel)
MAX_REASKS = 2
TRANSIENT = (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError)


def _client() -> OpenAI:
    s = get_settings()
    return OpenAI(
        api_key=s.deepseek_api_key.get_secret_value(),
        base_url=s.deepseek_base_url,
        timeout=s.judge_timeout_s,
        max_retries=0,  # tenacity handles retries
    )


@retry(
    retry=retry_if_exception_type(TRANSIENT),
    stop=stop_after_attempt(4),
    wait=wait_exponential(min=2, max=30),
    reraise=True,
)
def _create(client: OpenAI, messages: list[dict]):
    s = get_settings()
    return client.chat.completions.create(
        model=s.judge_model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=s.judge_temperature,
        max_tokens=s.judge_max_tokens,
        extra_body={"thinking": {"type": s.judge_thinking}},
    )


def request_key(messages: list[dict]) -> str:
    s = get_settings()
    payload = {
        "model": s.judge_model,
        "temperature": s.judge_temperature,
        "thinking": s.judge_thinking,
        "messages": messages,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def call_json(messages: list[dict], schema: type[T]) -> tuple[T, dict]:
    """Return (validated output, meta). Cached on disk by hash of (model, settings, messages)."""
    s = get_settings()
    cache_dir = s.cache_dir / "llm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{request_key(messages)}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return schema.model_validate_json(cached["content"]), {**cached["meta"], "cached": True}

    client = _client()
    convo = list(messages)
    t0 = time.perf_counter()
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    last_error = ""
    for attempt in range(1, MAX_REASKS + 2):
        resp = _create(client, convo)
        if resp.usage:
            usage["prompt_tokens"] += resp.usage.prompt_tokens
            usage["completion_tokens"] += resp.usage.completion_tokens
        content = (resp.choices[0].message.content or "").strip()
        finish = resp.choices[0].finish_reason
        if not content:
            last_error = f"empty content (finish_reason={finish})"
            convo = [*messages, {"role": "user", "content": "Your reply was empty. Reply with the json object only."}]
            continue
        try:
            result = schema.model_validate_json(content)
        except ValidationError as e:
            last_error = f"invalid output (finish_reason={finish}): {e.error_count()} errors"
            convo = [
                *messages,
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": "That json did not match the required schema. Errors:\n"
                    f"{str(e)[:2000]}\nReturn the corrected json object only.",
                },
            ]
            continue
        meta = {
            "model": resp.model,
            "thinking": s.judge_thinking,
            "latency_s": round(time.perf_counter() - t0, 1),
            "attempts": attempt,
            **usage,
        }
        cache_file.write_text(json.dumps({"content": content, "meta": meta}, ensure_ascii=False), encoding="utf-8")
        return result, {**meta, "cached": False}
    raise RuntimeError(f"LLM judge failed after {MAX_REASKS + 1} attempts: {last_error}")
