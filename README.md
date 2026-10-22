# Class Quality Assessor

Rates a recorded 8x8 online kids' chess class as **Poor / Average / Good / Excellent**. Every finding cites a timestamp and a quote that code has checked against the transcript, and each finding comes with a concrete improvement.

**Live app:** https://grdruygvyvrdqnjpappvzdu.streamlit.app/

## How to use it
1. Give the recording: upload it (.mp4, .mkv, .webm or .mov), or paste a Google Drive link shared with "Anyone with the link". Use a Drive link for files over about 200 MB.
2. Add the transcript if you have one. **The Zoom .vtt is recommended**: it names the speakers and makes the assessment take seconds instead of minutes. A .srt file or a .txt file with `Name: text` lines also works. A transcript without speaker names is handled as if there were none.
3. Press **Run assessment**. The report shows:
   - the label and a one-line summary;
   - key reasons, each with a timestamp and a quote, plus a ▶ button that plays the recording from that moment;
   - what would move the class up a level;
   - per-criterion details;
   - what the tool could not check, and reliability notes.

Without a transcript, only parts of the audio are transcribed on the server (the first 8 min, the last 6 min and two 2-min samples). That takes a few minutes per class, and speakers are not identified. Recordings longer than 2 hours are accepted, but the transcript is assessed from those same windows only.

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

Final configuration, VTT path with video, 3 judge runs (scorecard tag `final_v2`, 23 Oct). Values are the mean over the runs, with the range in brackets. `final_v2` replaces `final` (22 Oct, [eval/scorecard_final.md](eval/scorecard_final.md)) after a quote-verification bug credited some tutor lines to children (E26). The fix lowered the numbers: one Poor class (B441) is now rated Average. The rules were not re-tuned to win it back.

| Metric | Result | Best trivial baseline |
|---|---|---|
| Exact label | 4 / 10 [4-4] (was 5) | 4 / 10 (always "Poor") |
| Within one level | 9.7 / 10 [9-10] | 8 / 10 (always "Average") |
| Pairwise ordering (36 class pairs) | 83% [82-83] (was 88%) | 50% (any constant or random label) |
| Mean absolute error (levels) | 0.63 [0.60-0.70] (was 0.53) | 1.00 (always "Average") |
| Bias (predicted - true level; negative = harsh) | -0.43 [-0.50 to -0.40]: both Average classes rated Poor, both Excellent classes mostly Good | -0.20 (always "Average") |
| Weakness recall (organiser's criticised criteria caught) | 48% of 18 | - |
| Feedback recall, all organiser points | 65% of 47 | - |
| - points visible in the transcript | 87% of 28 | - |
| - points visible only on video | 24% of 12 | - |
| - points audible only in the voice | 39% of 6 | - |
| Evidence validity (quotes found in the transcript) | 99% [99-99] of ~193 quotes per run | - |
| Run agreement (same label in all 3 runs) | 9 / 10 classes | - |
| Privacy gate (no names sent to the judge) | Pass on both paths. VTT path: 0 name matches at >= 85% similarity, every borderline match human-reviewed, 1 documented exception (an ASR-mangled form of the tutor's name, E24). No-transcript path (3 classes with a cached whisper transcript): 0 name matches (E27) | - |

Trivial baselines in full ([eval/scorecard.md](eval/scorecard.md)):

| Baseline | Exact | Within one | MAE | Bias | Pairwise |
|---|---|---|---|---|---|
| Always "Poor" | 4/10 | 6/10 | 1.20 | -1.20 | 50% |
| Always "Average" | 2/10 | 8/10 | 1.00 | -0.20 | 50% |
| Random label (expected) | 2.5/10 | 6/10 | 1.30 | +0.30 | 50% |

**No-transcript path** (whisper instead of the Zoom VTT; EXPERIMENTS E22, E23, E25, E27):
- Same label as the VTT path on 1 of 3 tested classes. Excellent B472 matches (Good). Weak B482 (Poor) and good B552 (Good) both come out Average. The class language is set explicitly, and the closing window is 10 minutes like the judge's. Without speakers, rapport comes from the judge's quoted line instead of the warm-up measurement, and since a quote that can't be attributed is weak evidence, it counts only as a moderate problem (D22). That stops a single judged rapport finding from making a class Poor (B552 was rated Poor that way in a live test), but the path can now rarely give Poor. These are 3 classes, and the earlier changes were chosen after looking at two of them, so treat this as a sanity check, not an accuracy figure.

| Whisper time per class (2 threads, English) | Audio transcribed | Time |
|---|---|---|
| Excellent B472 (56 min class) | 22 min (8 + 10 + 2 x 2) | 282 s |
| Weak B482 (59 min class) | 22 min | 172 s |
| Four classes with the earlier 18-min windows and auto language (E22) | 18 min | 133-357 s |

**These numbers are optimistic.** The label rules were written from the organiser's notes on these same 10 classes, and with 2 classes per level a single class moves exact accuracy by 10 points. Treat them as a check that the method is sound, not as accuracy on unseen classes.

## Privacy
- Audio, video and frames never leave the server. Transcription (whisper.cpp) and face detection (OpenCV) run locally.
- The judge only receives pseudonymised transcript text (TUTOR, S1, S2, ...; leftover names replaced by [NAME]) and numeric metrics. The name mapping stays in memory. A leak audit fails the release if any name gets through.
- On the public app, the uploaded transcript, the audio, the frames and the whisper transcript are deleted as soon as the assessment finishes. The recording is kept only so the ▶ buttons can play it, and is purged after 2 hours. Only one assessment runs at a time.
- No recordings, transcripts or derived files are stored in this repository.

## Limitations
- Voice energy and tone are not measured, only the words. The 8x8 branded background is not checked yet.
- The tutor-face check is indicative only. The screen-share tag agrees with hand labels on 74% of frames, so a missing face cannot yet be told apart from a screen share. It counts as a moderate problem, not a critical one.
- Rapport is measured as a warm-up between the tutor's first greeting and the first lesson content. A tutor who builds rapport by questioning each child at the start, with no separate chat, is marked as having no rapport.
- Without a speaker-labelled transcript (whisper path) the label is checked on only 2 classes (both matched the VTT path). Speakers are not identified, so the warm-up is not measured: rapport is judged from a quoted line instead (the report says which method was used), and individual questioning cannot be attributed to children, so engagement tends to score lower. Only parts of the class are transcribed (first 8 min, last 10 min, two 2-min samples); anything in between is missed. The class language must be chosen in the app (English or Malayalam); auto-detection was dropped after it labelled an English opening as Sinhala. Malayalam is not usable with the current speech model (E21).
- Without a speaker-labelled transcript the app rarely gives Poor; upload the Zoom VTT for the most reliable label.
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
