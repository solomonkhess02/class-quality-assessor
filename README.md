# Class Quality Assessor

Rates a recorded 8x8 online kids' chess class as **Poor / Average / Good / Excellent**. Every finding cites a timestamp and a quote that code has checked against the transcript, and each finding comes with a concrete improvement.

**Live app:** `<LIVE_LINK>`

## How to use it
1. Give the recording: upload it (.mp4, .mkv, .webm or .mov), or paste a Google Drive link shared with "Anyone with the link". Use a Drive link for files over about 200 MB.
2. Add the transcript if you have one. **The Zoom .vtt is recommended**: it names the speakers and makes the assessment take seconds instead of minutes. A .srt file or a .txt file with `Name: text` lines also works. A transcript without speaker names is handled as if there were none.
3. Press **Run assessment**. The report shows:
   - the label and a one-line summary;
   - key reasons, each with a timestamp and a quote, plus a ▶ button that plays the recording from that moment;
   - what would move the class up a level;
   - per-criterion details;
   - what the tool could not check, and reliability notes.

Without a transcript, the audio is transcribed on the server: about 10-30 minutes for a 50-minute class, and speakers are not identified. Recordings longer than 2 hours are accepted; only the opening, three middle samples and the closing are assessed.

## How it works
```
mp4 / Drive link (+ optional .vtt/.srt/.txt)
 1. Ingest       save or download, check the file, sha256 cache key
 2. Media        ffprobe; 16 kHz WAV; 1 frame every 30 s
 3. Transcript   VTT/SRT/text parsed (speakers known)  |  else whisper.cpp on opening/middle/closing windows
                 -> tutor = most talk time -> pseudonymise (TUTOR, S1..Sn) -> scrub remaining names
 4. Features     talk share, student turns/min, warm-up length, connectivity incidents;
                 face %, lighting, frozen and black frames (OpenCV + YuNet, local)
 5. LLM judge    DeepSeek, JSON: score 1-4 per criterion, observations, timestamped quotes, yes/no facts
 6. Verify       each quote fuzzy-matched to the transcript within ±90 s; unverified quotes dropped
 7. Label        severity rule: any critical problem -> Poor; 2+ moderate -> Average; 1 -> Good;
                 none + positive evidence -> Excellent
```
Criteria (from the organiser's feedback on the sample classes): opening and rapport, individual engagement, energy and warmth, closure and homework, visual presence, technical quality. Code measures what it can measure reliably (statistics, frames, connectivity incidents). The LLM judges the rest against a fixed rubric. A fixed rule turns the findings into the label, so every label can be explained finding by finding. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [DECISIONS.md](DECISIONS.md).

## Evaluation
Measured on the 10 labelled sample classes, with repeated judge runs ([docs/EVALUATION.md](docs/EVALUATION.md), [eval/scorecard.md](eval/scorecard.md)).

| Metric | Result |
|---|---|
| Exact label | 5 / 10 (all 3 runs) |
| Within one level | 9.7 / 10 (mean of 3 runs, range 9-10) |
| Pairwise order agreement | 88% of 36 class pairs |
| Bias (mean predicted - true level) | -0.53 (harsh: both Average classes are rated Poor, both Excellent classes Good) |
| Same label across repeated runs | 9 / 10 classes |
| Criterion agreement with organiser notes | 73% balanced (96% of praised points, 50% of criticised points) |
| Privacy gate (no names sent to the judge) | 0 name matches at >= 85% similarity, 0 known names left; 4 borderline matches awaiting human review |
| No transcript given (whisper path) vs VTT, same class | 0 / 2 labels the same: Excellent B472 Good -> Average, weak B482 Poor -> Average |
| Whisper time per class (2 threads, 18 min of audio) | 2-6 min (133-357 s on 4 classes) |

Final configuration, 3 runs on the VTT path, 22 Oct ([eval/scorecard.md](eval/scorecard.md)). Trivial baselines: always "Poor" gets 4/10 exact and 50% pairwise. The label rule was written from the organiser's notes on these same 10 classes, so these numbers are optimistic.

## Privacy
- Audio, video and frames never leave the server. Transcription (whisper.cpp) and face detection (OpenCV) run locally.
- The judge only receives pseudonymised transcript text (TUTOR, S1, S2, ...; leftover names replaced by [NAME]) and numeric metrics. The name mapping stays in memory. A leak audit fails the release if any name gets through.
- On the public app, the uploaded transcript, the audio, the frames and the whisper transcript are deleted as soon as the assessment finishes. The recording is kept only so the ▶ buttons can play it, and is purged after 2 hours. Only one assessment runs at a time.
- No recordings, transcripts or derived files are stored in this repository.

## Limitations
- Voice energy and tone are not measured, only the words. The 8x8 branded background is not checked yet.
- The tutor-face check is indicative only. The screen-share tag agrees with hand labels on 74% of frames, so a missing face cannot yet be told apart from a screen share. It counts as a moderate problem, not a critical one.
- Rapport is measured as a warm-up between the tutor's first greeting and the first lesson content. A tutor who builds rapport by questioning each child at the start, with no separate chat, is marked as having no rapport.
- Without a speaker-labelled transcript (whisper path) the label is less reliable: on 2 tested classes it moved both towards Average. Speakers are not identified, so the warm-up and individual questioning cannot be measured and the "no rapport" problem can never fire. Only parts of the class are transcribed (first 8 min, last 6 min, two 2-min samples), so anything in between is missed: one class's homework, set 9 minutes before the end, was not seen. Whisper can also misdetect the language of a window (one class's opening was detected as Sinhala). Non-English classes (e.g. Malayalam) are not supported.
- The report shows no numeric confidence: on the labelled set it was as high on wrong labels as on right ones. Reliability notes take its place.

## Run locally (Windows)
```
./setup.ps1                                   # venv, dependencies, models, .env from .env.example
                                              # then set DEEPSEEK_API_KEY (and ffmpeg/whisper paths) in .env
.venv\Scripts\python scripts\check_env.py
.venv\Scripts\streamlit run app.py
.venv\Scripts\python -m pytest -q             # synthetic tests only
```

## Run with Docker
The image compiles whisper.cpp, includes the models and serves the app on port 7860 with the same privacy settings as the public app.
```
docker build -t cqa .
docker run -p 7860:7860 -e DEEPSEEK_API_KEY=... cqa
```
On Streamlit Community Cloud, the app uses the pywhispercpp bindings instead of a compiled whisper.cpp and downloads the models (sha256-checked) on first start. The key goes in the app's secrets as `DEEPSEEK_API_KEY`.

## Tech choices
- **Local whisper.cpp, not a speech API:** children's voices never leave the server. The base model runs at about 0.17x real time on 2 CPU threads. Transcribing only the opening, middle samples and closing keeps a class within minutes.
- **DeepSeek for the judge, text only:** inexpensive, OpenAI-compatible, with a JSON output mode. Its privacy policy allows training on inputs and stores data in China. That is why it only receives pseudonymised text and numbers, never images, even though the model accepts them.
- **No vector RAG:** a whole class transcript (~15-25k tokens) fits in one call, and with 10 labelled classes there is nothing to retrieve at scale.
- **A fixed rule, not a trained model:** 10 examples are too few to train on. Code-measured checks, a rubric and a written rule stay explainable and testable.
- **Streamlit:** a single Python app that handles large uploads and runs ffmpeg and whisper on the same machine.
