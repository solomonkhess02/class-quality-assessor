"""Check binaries, models, data folder and the DeepSeek API. Never prints the API key."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cqa.config import get_settings  # noqa: E402

OK, BAD = "[ok]     ", "[FAIL]   "


def check_bin(name: str, cmd: list[str]) -> bool:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"{BAD}{name}: {type(e).__name__}")
        return False
    first = (r.stdout or r.stderr).strip().splitlines()[:1]
    print(f"{OK if r.returncode == 0 else BAD}{name}: rc={r.returncode} {first[0][:70] if first else ''}")
    return r.returncode == 0


def check_file(name: str, path: Path, min_bytes: int) -> bool:
    ok = path.is_file() and path.stat().st_size >= min_bytes
    size = f"{path.stat().st_size / 1e6:.1f} MB" if path.is_file() else "missing"
    print(f"{OK if ok else BAD}{name}: {path.name} ({size})")
    return ok


def check_data(classes_dir: Path) -> bool:
    if not classes_dir.is_dir():
        print(f"{BAD}DATA_DIR: '{classes_dir.name}' folder not found - extract the dataset zip into DATA_DIR")
        return False
    print(f"{OK}DATA_DIR: {classes_dir.name}")
    for d in sorted(p for p in classes_dir.iterdir() if p.is_dir()):
        n_mp4 = len(list(d.glob("*.mp4")))
        n_vtt = len(list(d.glob("*.vtt")))
        print(f"           {d.name:<14} mp4={n_mp4} vtt={n_vtt}")
    return True


def check_llm() -> bool:
    from openai import OpenAI

    s = get_settings()
    key = s.deepseek_api_key.get_secret_value()
    if not key or key.startswith("your-"):
        print(f"{BAD}DEEPSEEK_API_KEY not set")
        return False
    client = OpenAI(api_key=key, base_url=s.deepseek_base_url, timeout=60)
    try:
        ids = sorted(m.id for m in client.models.list().data)
        print(f"{OK}models endpoint: {ids}; JUDGE_MODEL listed: {s.judge_model in ids}")
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=s.judge_model,
            messages=[
                {"role": "system", "content": 'Reply in json like {"ok": true, "n": 2}.'},
                {"role": "user", "content": "What is 1+1? Answer as json."},
            ],
            response_format={"type": "json_object"},
            max_tokens=50,
            extra_body={"thinking": {"type": "disabled"}},
        )
        dt = time.perf_counter() - t0
        content = r.choices[0].message.content or ""
        parsed = json.loads(content)
        print(f"{OK}DeepSeek JSON mode: model={r.model} latency={dt:.2f}s reply={parsed}")
        return True
    except Exception as e:  # report and continue; this is a diagnostic script
        print(f"{BAD}DeepSeek: {type(e).__name__}: {str(e)[:200]}")
        return False


def main() -> int:
    s = get_settings()
    results = [
        check_bin("ffmpeg", [s.ffmpeg_bin, "-version"]),
        check_bin("ffprobe", [s.ffprobe_bin, "-version"]),
        check_bin("whisper-cli", [s.whisper_cli, "--version"]),
        check_file("whisper model", s.whisper_model, 10_000_000),
        check_file("face model", s.face_model, 100_000),
        check_data(s.classes_dir),
        check_llm(),
    ]
    print("\nAll checks passed." if all(results) else f"\n{results.count(False)} check(s) failed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
