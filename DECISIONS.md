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
