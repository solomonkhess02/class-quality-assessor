# Class Quality Assessor: project context for Claude Code

## What we are building
A Streamlit app for the Humaniti AI build challenge. Input: a recording of an online kids' chess class (brand: 8x8) as an mp4 upload or Google Drive link, plus an optional Zoom .vtt transcript. Output: a quality label (Poor / Average / Good / Excellent) with specific, timestamped, evidence-backed observations and improvement suggestions.

Deadline: Fri 23 Oct 2026, 12:00 PM IST. Optimise for a working, explainable system over breadth.

Architecture: @docs/ARCHITECTURE.md
Decisions so far: @DECISIONS.md
Experiment log: EXPERIMENTS.md (append only; never rewrite past entries)

## Hard rules
- Never commit or publish recordings, transcripts, frames, feedbacks.pdf, or any derived file containing real names. Data lives in DATA_DIR (outside the repo). Cache lives in .cache/ (gitignored).
- Never commit secrets. Read config only through cqa/config.py, which loads .env. New settings go into .env.example with a placeholder.
- Privacy: only pseudonymised transcript text (tutor -> TUTOR, students -> S1, S2, ...) and numeric/aggregate features may be sent to the DeepSeek API. Never send audio, video or frames to any external API. Name mappings stay in process memory or the local cache.
- Do not print raw transcript text, real names or images into the chat. Show class IDs, counts, stats and short pseudonymised snippets only.
- Never delete or modify anything inside DATA_DIR.

## Tech
- Python 3.10+. Dev machine is Windows (PowerShell); deploy target is a Docker Hugging Face Space (Linux, 2 vCPU). Use pathlib, keep binaries configurable via .env.
- LLM judge: DeepSeek through the openai SDK (DEEPSEEK_BASE_URL, JUDGE_MODEL from .env), JSON output mode.
- ASR: whisper.cpp CLI via subprocess. A provided Zoom VTT is the fast path.
- Media: ffmpeg/ffprobe via subprocess. Vision: OpenCV, local only.
- UI: Streamlit. Contracts between stages: pydantic v2 models in cqa/schemas.py.

## Commands (PowerShell, from repo root)
- Setup: `./setup.ps1`
- Env check: `.venv\Scripts\python scripts\check_env.py`
- App: `.venv\Scripts\streamlit run app.py`
- Tests: `.venv\Scripts\python -m pytest -q`
- Lint: `.venv\Scripts\ruff check .`
- Eval on labelled set: `.venv\Scripts\python scripts\eval_labelled.py`

## How to work
- Plan before multi-file changes. Small steps, run tests after each milestone, one git commit per milestone.
- Tests use synthetic fixtures only (tests/fixtures/), never real data.
- Every LLM call goes through cqa/judge/llm.py (retries, empty-content retry, disk cache keyed by request hash).
- When a result changes what we believe, add an EXPERIMENTS.md entry: hypothesis, method, result, decision.
- Simple, readable code with type hints. No premature abstraction. If an external API detail is uncertain, check official docs or `--help` instead of guessing.
