# Decisions

D1. No vector RAG. A full class transcript (~15-25k tokens) fits in one LLM call and there are only 10 labelled classes, so there is nothing to retrieve at scale. Possible Phase 2 extra: show the judge the nearest labelled classes (by stats) as calibration examples, evaluated leave-one-out.

D2. Hybrid pipeline: code measures what it can measure reliably (stats, frames), the LLM judges against a rubric derived from organiser feedback and must cite timestamped quotes, quotes are verified in code, and a fixed formula produces the label. Explainable and testable; no training on 10 samples.

D3. Transcription: a provided Zoom VTT is used directly (exact speakers, instant). Otherwise whisper.cpp runs locally, so audio never leaves the server. Model choice depends on 2-thread benchmark (pending).

D4. LLM judge: DeepSeek (deepseek-flash by default) via its OpenAI-compatible API. DeepSeek's privacy policy lists model training as a purpose and stores data in China, so only pseudonymised transcript text and numeric features are sent. Frames are never sent, even though deepseek-flash accepts images.

D5. UI: Streamlit, hosted as a Docker Hugging Face Space (large uploads, ffmpeg, compiled whisper.cpp, secrets).

D6. Labels: weak and quality issue both map to Poor; quality issue also raises a technical flag.

D7. Thresholds and weights start from the rubric and are checked on the labelled set, not fitted to it (2 samples per level would overfit).

D8. Face detection: OpenCV 5 removed CascadeClassifier, so we pin opencv-python-headless<5 and use FaceDetectorYN (YuNet ONNX, ~230 KB, downloaded at setup), which also handles small faces in Zoom gallery tiles better than Haar.

D9. DeepSeek thinking mode disabled for the judge (JUDGE_THINKING=disabled). The docs say thinking is on by default and that temperature is ignored in thinking mode, and they do not document JSON output together with thinking. Non-thinking mode keeps temperature control, keeps reasoning tokens out of the max_tokens budget and returned valid JSON first time on every Phase 1 call (~8 s for a 20k-token class). Revisit if judgement quality, not format, becomes the bottleneck.

D10. Judge design: direct per-criterion scoring (JUDGE_DESIGN=direct). An extract-then-score variant (LLM reports facts, code maps them to 1-4) separated an excellent and a weak class less well (E9): LLM-reported counts such as praise and negative remarks were unreliable, while yes/no facts (homework assigned, children greeted by name) did discriminate. The facts design stays available for Phase 2 (e.g. yes/no facts as evidence for the direct judge).

D11. Whisper model for the no-VTT path: ggml-base-q5_1 at 2 threads. On a 2-minute English clip it ran at 0.17x real time with 82% word agreement with the Zoom VTT, versus small-q5_1 0.50x / 83% and large-v3-turbo-q5_0 1.05x / 83% at 6 threads (E8). Agreement is with another machine transcript, on one clip; re-check chess terms and Indian-English speech before Phase 2 relies on it.

D12. Label aggregation: problem count (AGGREGATION=problems). Each criterion scored <= 2 is one problem; 0 = Excellent, 1 = Good, 2 = Average, 3+ = Poor; fewer than 3 scored criteria gives a placeholder label with a warning. Chosen because the organiser's labels read as "how many problems did this class have", the weighted mean could not produce Poor on the labelled set (E10), and a count is easy to explain ("Poor: problems with opening, closure and visual presence"). The weighted mean is still reported. The ARCHITECTURE face cap is a warning, not a cap, until screen-share frames are excluded reliably (E6).

D13. Aggregation is frozen; the problem-count rule (D12) stays. Every rule tried gives 3-5/10 exact labels, and dropping communication and technical_quality from the count gives 3/10, so the gaps between rules are run-to-run judge noise rather than one rule being better. Further tuning would only fit noise in 10 classes. The judge already agrees with the organiser on 74% of criterion expectations; what limits the label is signals we do not measure yet (voice energy for communication, the 8x8 background for visual presence), so Phase 2 effort goes there.

D14. technical_quality is scored by code from objective evidence, not by the LLM. The LLM gave it 2 for nearly every class, including both excellent ones (E10), so it added an almost constant problem to the count. Thresholds, fixed now before any results are seen: the class has a technical problem only if there are >= 3 verified connectivity incidents per hour of class, or frozen frames are >= 20% of camera (non-screen-share) frames. Otherwise no technical problem.

D15. A criterion whose scores barely vary across the labelled classes (standard deviation < 0.3) is still shown in the report with its observations, but is left out of the label. A near-constant score cannot tell classes apart; it only shifts every label the same way (communication in E10 is the example: 3 for 9-10 of 10 classes). Re-checked whenever the judge or a scorer changes.

D16. Privacy gate for anything sent to the judge (scripts/leak_audit.py, all classes except the Malayalam demo, whose transcript is ASR noise): zero tokens that fuzzy-match an original display-name token at >= 85, and zero human-labelled names left among the 70-84 matches. The 70-84 band is checked against a human review of the matches (.cache/leak_review_labels.json, local only), not failed automatically, because about half of the matches in that band are ordinary words (21 of 37 in the first review). New, unlabelled 70-84 matches must be reviewed before the gate counts as passed. Separate target for damage to the judge's input: over-scrub (common English words replaced, as a share of tutor words) <= 0.5%. A failing gate blocks release.
