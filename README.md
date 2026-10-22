---
title: Class Quality Assessor
emoji: ♟️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Rates recorded online chess classes with evidence
---

# Class Quality Assessor

Assesses an online kids' class recording and rates it Poor / Average / Good / Excellent, with timestamped evidence for every judgement.

Status: in development (Phase 1). Full README with run instructions, approach and limitations comes in Phase 3.

## Quick start (Windows dev)
1. `./setup.ps1`
2. Put your DeepSeek key in `.env`
3. `.venv\Scripts\streamlit run app.py`

## Docker / Hugging Face Space
The `Dockerfile` builds whisper.cpp (v1.9.5), downloads the whisper and face-detector models and serves the app on port 7860. The DeepSeek key comes from the environment (a Space secret named `DEEPSEEK_API_KEY`); without it the app shows "assessment service unavailable".
```
docker build -t cqa .
docker run -p 7860:7860 -e DEEPSEEK_API_KEY=... cqa
```
In the container, one assessment runs at a time; uploads, audio and frames are deleted after each assessment and cache entries older than 2 hours are purged (`DELETE_MEDIA_AFTER_RUN`, `CACHE_MAX_AGE_H`).

Recordings and transcripts are confidential and are never stored in this repository.
