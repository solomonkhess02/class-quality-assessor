# Architecture

## Pipeline
```
Input: mp4 upload | Google Drive link | (optional) Zoom .vtt
        |
 1. Ingest       save/download, sha256 -> .cache/<hash>/
 2. Media        ffprobe duration; 16 kHz mono WAV; 1 frame / FRAME_INTERVAL_SEC (640 px)
 3. Transcript   VTT given -> parse (exact speakers)
                 else      -> whisper.cpp (speaker UNKNOWN; Phase 2: LLM role tagging, windowed ASR)
                 -> tutor identification -> pseudonymise (TUTOR, S1..Sn)
 4. Features     text stats (talk share, student turns/min, distinct speakers, opening/closing windows,
                 connectivity phrases); visual metrics (face %, dark %, black %, frozen %, blur) - OpenCV + YuNet face detector, local
 5. LLM judge    DeepSeek, JSON mode: per-criterion score 1-4 + observations + timestamped quotes + improvement
 6. Verify       each quote fuzzy-matched to transcript within +-90 s; unverified evidence dropped
 7. Score        weighted mean -> label, hard caps, confidence
 8. UI / eval    Streamlit report; scripts/eval_labelled.py on the labelled set
```

## Rubric (derived from the organiser's feedback on the sample classes)
| id | Criterion | Looks good when | Main evidence | Weight | Scored by |
|---|---|---|---|---|---|
| opening_rapport | Opening and rapport | Warm greeting, chats with children before content, invites new joiners to introduce themselves, sets the plan | First ~10 min | 0.20 | LLM |
| individual_engagement | Individual engagement | Asks children questions by name, involves every child, children respond often | Whole transcript + stats | 0.25 | LLM |
| communication | Energy and warmth | Encouraging, praises effort, patient, lively, pleasant | Transcript tone | 0.15 | LLM |
| closure_homework | Closure and homework | Recap, explicit homework assigned, proper goodbye | Last ~10 min | 0.15 | LLM |
| visual_presence | Visual presence | Branded 8x8 background, tutor face visible, camera on, good lighting | Frames | 0.15 | Code (Phase 1: OpenCV metrics; Phase 2: background check) |
| technical_quality | Technical quality | Clear audio/video, no repeated connectivity problems | Frames + connectivity phrases | 0.10 | LLM with metrics as context |

Scale: 1 = absent or poor, 2 = present but weak, 3 = clearly good, 4 = exemplary. `insufficient_evidence = true` excludes a criterion from the weighted mean and lowers confidence.

## Label mapping and aggregation (DECISIONS D12; thresholds not fitted to the 10 samples)
- Dataset folders: excellent -> Excellent, good -> Good, average -> Average, weak -> Poor, quality issue -> Poor (+ technical flag), demo -> unlabelled.
- Shipped rule (AGGREGATION=problems): each criterion scored <= 2 is a problem; 0 -> Excellent, 1 -> Good, 2 -> Average, 3+ -> Poor. Fewer than 3 scored criteria -> placeholder label with a warning.
- Reported alongside (AGGREGATION=mean): weighted mean m; Excellent >= 3.4; Good >= 2.8; Average >= 2.2; else Poor.
- Face missing in > 40% of sampled frames -> warning only until screen-share frames are excluded reliably (E6). Transcript unusable (e.g. wrong-language ASR) -> confidence low, say so explicitly.
- Key facts (homework assigned and what, new joiners invited, children greeted by name, recap) come from the direct judge with verified quotes; shown in the report, not scored.

## Contracts (cqa/schemas.py)
- Turn: start_s, end_s, speaker, role ("tutor" | "student" | "unknown"), text
- Transcript: source ("vtt" | "whisper"), language, duration_s, turns
- TextStats: duration_min, tutor_talk_share, student_turns_per_min, distinct_students_spoke, tutor_questions_per_min, sparse_punctuation (questions/min unreliable when true), longest_tutor_monologue_s, connectivity_mentions, opening_turns, closing_turns
- VisualFrameMetrics: t_s, brightness, blur, is_black, is_frozen, face_count, flat_ratio, is_screen_share
- VisualReport: n_frames, pct_face, pct_dark, pct_black, pct_frozen, median_blur, pct_screen_share, pct_face_camera (face % over frames that are neither screen share nor black)
- Evidence: timestamp ("hh:mm:ss"), speaker, quote, verified
- CriterionResult: id, score (1-4 | null; null implies insufficient_evidence), insufficient_evidence, observations[], evidence[], improvement
- KeyFacts: homework_assigned, homework_what, new_joiners, children_greeted_by_name, recap_present, evidence[] (FactEvidence = Evidence + fact)
- JudgeOutput: criteria[], key_facts, summary, strengths[], improvements[], transcript_usable, transcript_note (Phase 1 stopgap for unusable ASR, e.g. wrong language)
- Assessment: label, weighted_score, confidence (0-1), criteria[], key_facts, summary, strengths[], improvements[], warnings[], text_stats, visual, meta (model, transcript source, timings)

## Module layout
```
app.py                         Streamlit entry
cqa/config.py                  settings from .env
cqa/schemas.py                 contracts above
cqa/ingest.py                  upload / Drive link (gdown) / hashing / cache dir
cqa/media.py                   ffprobe, audio extraction (windows), frame sampling
cqa/transcript/vtt.py          Zoom VTT parser
cqa/transcript/whisper.py      whisper.cpp wrapper
cqa/transcript/roles.py        tutor identification (Phase 2: LLM tagging for whisper)
cqa/transcript/privacy.py      pseudonymisation
cqa/features/text_stats.py
cqa/features/visual.py
cqa/judge/rubric.py | prompts.py | llm.py | judge.py
cqa/scoring.py
cqa/pipeline.py                orchestration + per-stage cache + progress callback
scripts/check_env.py | eval_labelled.py
tests/                         synthetic fixtures only
```

## Phases
1. Base (now): VTT path end to end, whisper wrapper, OpenCV visuals, DeepSeek judge, scoring, eval script, minimal Streamlit.
2. Robustness: whisper path with LLM role tagging and opening/closing/middle windows; 8x8 background check (reference-image matching, local); transcript-quality detection; optional audio energy; optional nearest-labelled-class examples.
3. Ship: Dockerfile (whisper.cpp compiled, model baked in), HF Space deploy, README, 300-word write-up, redacted AI logs.
