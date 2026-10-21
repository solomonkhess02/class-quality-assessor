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
