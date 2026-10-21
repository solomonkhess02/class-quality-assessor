# Experiment log
Append-only. Each entry: hypothesis, method, result, decision. No real names or raw transcript text.

## E1 (21 Oct): What do the organisers actually judge?
- Method: read the organiser's feedback notes for all 10 labelled classes.
- Result: recurring criteria are intro and rapport before teaching, time for new joiners to introduce themselves, asking each child questions individually, giving homework, a proper conclusion, energetic and pleasant communication, branded 8x8 background, tutor face visible / camera on, lighting, and video quality.
- Decision: about half of these are only visible on screen, so a transcript-only system is ruled out. Rubric in docs/ARCHITECTURE.md is built from these points.

## E2 (21 Oct): Do simple transcript stats separate quality levels?
- Method: parsed the 12 Zoom VTTs; tutor = speaker with most talk time.

| label | class | min | tutor talk % | other speakers | student turns/min | tutor "?"/min | homework mentions | students named by tutor |
|---|---|---|---|---|---|---|---|---|
| average | B435 | 38 | 57 | 4 | 2.5 | 2.9 | 0 | 1 |
| average | B516 | 44 | 77 | 5 | 3.4 | 3.6 | 7 | 2 |
| demo | english | 44 | 60 | 2 | 7.9 | 5.0 | 0 | 0 |
| demo | malayalam | 52 | 82 | 1 | 4.0 | 1.8 | 0 | 0 |
| excellent | B472 | 56 | 47 | 6 | 9.3 | 3.2 | 3 | 5 |
| excellent | B491 | 53 | 63 | 8 | 10.1 | 3.4 | 0 | 1 |
| good | B435 | 44 | 46 | 7 | 4.3 | 0.1 | 0 | 0 |
| good | B552 | 46 | 57 | 4 | 9.1 | 3.8 | 3 | 1 |
| quality issue | B392 | 60 | 74 | 5 | 4.3 | 2.8 | 0 | 5 |
| quality issue | B441 | 84 | 56 | 10 | 3.7 | 2.5 | 0 | 10 |
| weak | B482 | 59 | 66 | 5 | 2.3 | 1.0 | 0 | 3 |
| weak | B558 | 46 | 70 | 3 | 5.9 | 3.0 | 0 | 2 |

- Result: student turns/min is the strongest single signal (excellent ~9-10, average/weak mostly 2-4). Misleading signals: "?" counts depend on transcript punctuation (good B435 = 0.1); keyword counts for homework do not match whether homework was assigned (average B516 = 7, excellent B491 = 0); name mentions are highest in a quality-issue class. The quality-issue classes look normal on text; their problems were visual. The Malayalam demo's Zoom transcript is English ASR applied to Malayalam speech and is unusable.
- Decision: stats are evidence for the judge, not the classifier. Homework and engagement need LLM judgement. Visual module is required. Transcript quality must be detected.

## E3 (22 Oct): Is whisper.cpp viable?
- Method: whisper.cpp b5454 CPU build, ggml-large-v3-turbo-q5_0, 2-minute English clip (03:00-05:00 of average B435), 6 threads, -l en; compared with the Zoom VTT by word alignment.
- Result: 141 s for 120 s of audio (1.18x real time), peak RAM ~0.9 GB, encoder 92% of runtime. 84% word agreement with the VTT (both are machine transcripts, so this is agreement, not accuracy). Whisper was right on chess terms where the VTT was wrong.
- Decision: quality is good but turbo is too slow for a 2-vCPU host. Next: benchmark base-q5_1 and small-q5_1 at 2 threads; transcribe only opening/closing/middle windows if needed. Turbo is not trained for translation, so Malayalam needs -l ml (LLM reads it) or a non-turbo model with -tr.

## E4 (22 Oct): Text stats module vs the E2 hand analysis (M3)
- Method: cqa/features/text_stats.py on all 12 VTTs (scripts/stats_table.py); same-speaker cues < 1.5 s apart are merged into one turn.
- Result: tutor talk share and tutor "?"/min match E2 within 1 point. Student turns/min is 10-35% lower than E2 because merged cues count once (excellent 7.3 / 9.4, good B552 8.3, average 1.6 / 3.2, weak 2.2 / 5.3, quality issue 3.9 / 2.9); the ordering is unchanged. good B435 is the only class where "?" counts are unreliable: "?" is 5% of its sentence-ending marks (21-43% elsewhere) and it has only ~31 tutor words/min (others 40-70). Longest tutor monologue is highest in weak B558 (447 s; others 30-154 s).
- Decision: the sparse_punctuation flag now fires on low punctuation density or a "?" share < 10% of sentence ends; the judge is told to ignore "?"/min when it is set. Low words/min is a candidate transcript-quality signal for Phase 2.

## E5 (22 Oct): Frame sampling, one decode pass vs per-timestamp seeks (M4)
- Method: scripts/bench_frames.py on the longest recording (quality issue B441, 89.4 min), 1 frame / 30 s at 640 px, dev laptop. Strategy A: one ffmpeg pass with a select filter (the fps filter rounds to the nearest slot and dropped the last frame in a synthetic test). Strategy B: one fast input seek (-ss before -i) per timestamp.
- Result: seeks 55.7 s, single pass 94.9 s; both 179 frames.
- Decision: per-timestamp seeks are the default (media.sample_frames). Re-check on the 2-vCPU Space.

## E6 (22 Oct): Do OpenCV frame metrics separate the classes? (M4)
- Method: scripts/visual_table.py on all 12 recordings; YuNet face detector on 640 px frames every 30 s. face% = frames with >= 1 face; dark% = mean grey < 60; frozen% = unchanged from the previous sample.

| class | label | face% | multi-face% | dark% | black% | frozen% | median blur |
|---|---|---|---|---|---|---|---|
| average B435 | Average | 41 | 1 | 28 | 0 | 26 | 852 |
| average B516 | Average | 86 | 1 | 58 | 1 | 3 | 864 |
| excellent B472 | Excellent | 87 | 0 | 17 | 1 | 5 | 1874 |
| excellent B491 | Excellent | 79 | 1 | 27 | 1 | 9 | 2016 |
| good B435 | Good | 40 | 0 | 48 | 0 | 32 | 783 |
| good B552 | Good | 65 | 2 | 6 | 0 | 11 | 1918 |
| quality issue B392 | Poor | 62 | 3 | 45 | 0 | 20 | 1314 |
| quality issue B441 | Poor | 4 | 1 | 42 | 0 | 31 | 1355 |
| weak B482 | Poor | 85 | 0 | 29 | 0 | 12 | 2197 |
| weak B558 | Poor | 71 | 0 | 17 | 0 | 15 | 604 |
| demo 1 | - | 79 | 6 | 26 | 0 | 20 | 1636 |
| demo 2 | - | 98 | 9 | 18 | 0 | 4 | 1437 |

- Result: quality issue B441 stands out (face in 4% of frames). Quality issue B392 does not stand out on face% (62); its dark% (45) is high but average B516 is darker (58), so dark% alone does not separate. Both B435 classes (average and good, same batch) show face ~40% with ~30% frozen frames, which looks like long screen shares of a static board rather than a missing tutor. Multi-face frames are <= 9% everywhere, so gallery view is rare; layout effects come from screen share. Black frames are ~0 everywhere.
- Decision: the "face missing > 40% -> at most Average" cap would wrongly cap good B435, so it becomes a warning until screen-share frames are tagged and excluded (contact-sheet check by eye, M4 follow-up). Dark threshold and frozen% are not trusted as penalties until then.
