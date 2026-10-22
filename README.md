# Class Quality Assessor

Assesses an online kids' class recording and rates it Poor / Average / Good / Excellent, with timestamped evidence for every judgement.

Status: in development (Phase 1). Full README with run instructions, approach and limitations comes in Phase 3.

## Quick start (Windows dev)
1. `./setup.ps1`
2. Put your DeepSeek key in `.env`
3. `.venv\Scripts\streamlit run app.py`

## Deployment (Streamlit Community Cloud)
Deployed from this repository: `app.py`, Python 3.11, `packages.txt` installs ffmpeg. The DeepSeek key goes in the app's secrets as `DEEPSEEK_API_KEY`; without it the app shows "assessment service unavailable". On Community Cloud the app automatically:
- transcribes with the pywhispercpp bindings (`WHISPER_BACKEND=py`), so no whisper.cpp build is needed;
- downloads the whisper and face-detector models at first start (sha256-checked);
- runs one assessment at a time, deletes uploads, audio and frames after each assessment and purges cache entries older than 2 hours (`DELETE_MEDIA_AFTER_RUN`, `CACHE_MAX_AGE_H`).

## Alternative: run with Docker
The `Dockerfile` builds whisper.cpp (v1.9.5), downloads the models and serves the app on port 7860 with the same privacy settings.
```
docker build -t cqa .
docker run -p 7860:7860 -e DEEPSEEK_API_KEY=... cqa
```

Recordings and transcripts are confidential and are never stored in this repository.

## Known limitations
- Rapport is measured as a warm-up between the tutor's first greeting and the first line of lesson content. A tutor who builds rapport by questioning each child individually at the start, without a separate chat, is marked as having no rapport (e.g. one Average class in the labelled set becomes Poor).
- The tutor-face check is a moderate problem, not a critical one: the screen-share tag agrees with hand labels on only 74% of sampled frames (target 85%), so a missing face cannot yet be told apart from an untagged screen share.
- The 8x8 branded background is not checked yet.
- Accuracy figures come from 10 labelled classes whose organiser notes also shaped the rule, so they are optimistic.
