# Phase 1: build the base pipeline

You are working with me (Solomon) as a senior AI engineer on the Humaniti AI build challenge. First read CLAUDE.md, docs/ARCHITECTURE.md, DECISIONS.md and EXPERIMENTS.md. They are the source of truth. If anything below conflicts with them, or a milestone turns out to be a bad idea, stop and tell me why before changing course.

## Goal of this phase
A working end-to-end base: given a class recording and its Zoom VTT, produce a validated Assessment (label, per-criterion scores, verified timestamped evidence, improvements) through the eval script and a minimal Streamlit page. The whisper module works on its own (tested on a short clip), but full no-VTT support (speaker role tagging, windowed transcription) is Phase 2. Visual signals in this phase are OpenCV metrics only.

## How I want you to work
1. Start in plan mode. Use the writing-plans skill to produce a plan for M0 to M8: files, key functions, tests, risks, and anything you think is wrong with this plan. Wait for my approval before writing code.
2. One milestone at a time. After each: run pytest and ruff, give me a short summary (what changed, test output, anything surprising), commit with a clear message, then continue to the next unless I say stop.
3. Use test-driven-development for pure functions (VTT parser, pseudonymiser, stats, scoring, evidence verification). Use systematic-debugging when something fails instead of guessing. Use verification-before-completion before saying a milestone is done: show the command and its output.
4. When a result changes what we believe (e.g. a signal does not separate classes), add an EXPERIMENTS.md entry and tell me.
5. If unsure about an external detail (DeepSeek parameters, whisper.cpp flags, gdown behaviour), check official docs or `--help` and tell me what you found.
6. Keep raw transcript text, real names and images out of our chat. Print class IDs, counts, stats and pseudonymised snippets only.

## Milestones

### M0: Environment check
- scripts/check_env.py: load settings via cqa/config.py; verify ffmpeg, ffprobe, whisper-cli and the whisper model exist and run; verify DATA_DIR exists and list the class folders found (counts only); send a tiny DeepSeek request in JSON mode and print model + latency. Never print the key.
- If the dataset zip is not extracted into DATA_DIR yet, tell me. Do not extract or move data without asking.

### M1: Config and contracts
- cqa/config.py: typed settings from .env (python-dotenv + pydantic), cross-platform defaults for binaries.
- cqa/schemas.py: the contracts in docs/ARCHITECTURE.md. Extend only if needed and update the doc in the same commit.

### M2: Transcript ingestion and privacy
- cqa/transcript/vtt.py: robust Zoom VTT parser (\r\n, multi-line cues, cue numbers, missing speaker prefix, "Name: text", curly apostrophes). Merge consecutive same-speaker cues less than 1.5 s apart into one turn.
- cqa/transcript/roles.py: tutor = speaker with most talk time; others are students.
- cqa/transcript/privacy.py: tutor -> TUTOR, students -> S1..Sn by first appearance; also replace their names inside the text (first token of display names, possessive device names like "x's iPhone", spelling variants via rapidfuzz >= 85 on capitalised tokens). Return pseudonymised transcript + mapping (mapping never sent anywhere).
- Tests with synthetic VTT fixtures in tests/fixtures/.

### M3: Text stats
- cqa/features/text_stats.py: duration, tutor talk share, student turns/min, distinct students who spoke, tutor questions/min (with a flag when punctuation is sparse), longest tutor monologue, connectivity phrases ("can you hear", "on mute", "network", "not visible"), opening and closing windows.
- Sanity check on all 12 VTTs: student turns/min should roughly match EXPERIMENTS.md E2. Print a table of class ID, label and stats only.

### M4: Media and visual metrics
- cqa/media.py: ffprobe duration; 16 kHz mono WAV (whole or a list of [start, end] windows); one frame every FRAME_INTERVAL_SEC into .cache/<hash>/frames at 640 px wide. Benchmark a single ffmpeg fps filter against per-timestamp fast seeks on one long file, keep the faster, log timings in EXPERIMENTS.md.
- cqa/features/visual.py: per frame brightness, Laplacian blur, near-black, frozen vs previous frame, face count with cv2.FaceDetectorYN using models/face_detection_yunet_2023mar.onnx (downloaded by setup.ps1; path FACE_MODEL in .env). Aggregate to VisualReport.
- Run on the 10 labelled recordings and print the aggregate table (no images). Record in EXPERIMENTS.md whether the two quality-issue classes stand out, and whether Zoom layout (gallery vs speaker view vs screen share) makes face % misleading.

### M5: Whisper wrapper
- cqa/transcript/whisper.py: whisper-cli via subprocess using .env settings (threads, model, language, -mc 0, JSON or SRT output); parse into Turns with speaker UNKNOWN; support a list of time windows with correct offsets. Check `whisper-cli -h` for exact flags in our build.
- Test on a 2-minute clip only. Report wall time and real-time factor.

### M6: Judge and scoring
- cqa/judge/rubric.py: six criteria, weights and 1-4 anchors from docs/ARCHITECTURE.md.
- cqa/judge/prompts.py: system prompt (experienced academic quality reviewer for kids' online chess classes; rubric; anchors; every observation must be specific and cite timestamped quotes; use insufficient_evidence instead of guessing; no generic praise) and user prompt (pseudonymised transcript as "[hh:mm:ss] SPEAKER: text", TextStats JSON, VisualReport JSON). Request JSON matching JudgeOutput and include an example (DeepSeek JSON mode needs the word "json" and an example).
- cqa/judge/llm.py: DeepSeek through the openai SDK, response_format json_object, timeout, tenacity retries, re-ask on empty content or invalid JSON (send back the validation error), disk cache keyed by hash of (model, messages). Check DeepSeek's docs on thinking vs non-thinking mode with JSON output and pick the reliable option; record it in DECISIONS.md.
- cqa/judge/judge.py: call, validate with pydantic, verify each evidence quote exists within +-90 s of its timestamp (rapidfuzz partial_ratio >= 85), drop unverified evidence and count it in warnings.
- cqa/scoring.py: visual_presence from VisualReport by code (thresholds in config, marked as initial guesses); other criteria from the judge; weighted mean over criteria with evidence; label thresholds and hard caps from ARCHITECTURE.md; confidence from evidence coverage, transcript source and warnings. Unit tests for edge cases.

### M7: Pipeline and evaluation
- cqa/pipeline.py: stages with a progress callback; cache each stage's output under .cache/<file-hash>/ so reruns are instant.
- scripts/eval_labelled.py: run on the labelled set using the VTTs (flag --skip-visual), map folders to labels (weak and quality issue -> Poor), write eval/results.md with class ID, true label, predicted label, weighted score and per-criterion scores only (no names, no quotes). Report exact accuracy and within-one-level accuracy. Also report the two demo classes.
- EXPERIMENTS.md entry with the results and the 2-3 biggest error patterns.

### M8: Minimal Streamlit app
- app.py: mp4 upload or Google Drive link (gdown), optional .vtt; Run button; st.status with stage progress; results: label badge, confidence, per-criterion table, expanders with observations, verified evidence (timestamps) and improvement; visual metrics; warnings; JSON download. Without a VTT, show a notice that the slower whisper path is used, and run it.

## Done when
- `pytest -q` and `ruff check .` pass.
- eval/results.md has the 10 labelled and 2 demo predictions.
- `streamlit run app.py` assesses one sample recording end to end with its VTT.
- EXPERIMENTS.md has entries for M3, M4, M5 and M7.

Then stop and give me a short report: what works, what does not, and the top 3 things to fix in Phase 2.
