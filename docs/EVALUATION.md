# Evaluation plan

## Why this needs care
There are 10 labelled classes (2 per level, 4 Poor once weak and quality issue are merged). One class is 10 points of exact accuracy, and the same class can move a level between two judge runs. So:
- the headline metric is threshold-free and uses pairs, not 10 labels;
- every configuration is run 3 times and reported as mean [min-max];
- decisions and targets are written down before results (DECISIONS.md, this file);
- one change per ablation, compared with the baseline's noise band;
- nothing is fitted to the 10 labels; anything that uses labelled classes as examples is evaluated leave-one-out, also excluding the same batch (both B435 classes share a batch);
- the config is frozen before the final run reported in the README.

## Ground truth
- Labels: dataset folders (weak and quality issue -> Poor). Ordinal: Poor 1, Average 2, Good 3, Excellent 4.
- Criterion expectations: DATA_DIR/expected_criteria.json, 35 points (17 praised, 18 criticised), from the organiser's notes.
- Feedback points: the same file's `points`, 47 atomic points from the organiser's notes (e.g. "should give new joiners time to introduce themselves").
- This file and its notes contain no names and stay outside the repo.

## Metrics
| Group | Metric | Definition |
|---|---|---|
| **Primary** | Pairwise ordering accuracy | Over the 36 pairs of labelled classes with different true labels: share where the class with the higher label has the better severity score (fewer problems). Ties count 0.5. Threshold-free. |
| Label | Exact, within one, MAE, bias | Bias = mean(predicted - true); negative = harsh. Plus a 4x4 confusion matrix. |
| Criterion | Weakness recall | Of the 18 criticised points, share the app scores <= 2. Most important secondary: a quality tool must catch problems. |
| Criterion | Strength recall, balanced accuracy | Of the 17 praised points, share scored >= 3; balanced = mean of both recalls. Per criterion too. |
| Explanation | Feedback recall | Share of the 47 organiser points covered (also reported per source: transcript-observable, video-observable, voice-only, not in recording; recall on the points our inputs can observe is the fair number) by the app's observations and improvements. Graded by an LLM (yes / partial / no with the supporting line from the app output, temperature 0); Solomon spot-checks 3 classes and the agreement is reported. Partial = 0.5. |
| Explanation | Evidence validity, genericness | Share of quotes verified against the transcript; share of observations with no speaker, timestamp or quote. |
| Reliability | Run agreement | Classes whose label is identical in all 3 runs; mean range of criterion scores across runs. |
| Confidence | Separation | Mean confidence on correct vs wrong labels. Useful only if wrong < correct. |
| Path | VTT vs no-VTT | On 4 classes (one per level): label difference and criterion MAE between paths. Malayalam demo must get a real (non-placeholder) assessment. |
| Operational | Time, cost, failures | Wall time per class at 2 threads and on the Space; DeepSeek cost per class; pass rate on the failure-case suite (bad Drive link, .mkv/.webm, 90-min file, long silences, unusable VTT). |
| Gate | Privacy | D16: zero leak-audit (b) hits at >= 85 similarity, no human-labelled names left in the 70-84 band (documented exceptions only), and over-scrub (scrubbed common English words as a share of tutor words) <= 0.5% per class, excluding the ASR-noise demo. A failing gate blocks any release. |

## Trivial baselines (any real result must beat these)
| Baseline | Exact | Within one | MAE | Pairwise |
|---|---|---|---|---|
| Always "Poor" (majority class) | 4/10 | 6/10 | 1.20 | 50% (all ties) |
| Always "Average" | 2/10 | 8/10 | 1.00 | 50% |
| Random label | ~2.5/10 | - | - | ~50% |
Phase 1's 4/10 exact equals the always-Poor baseline; its pairwise 71% and within-one 8/10 are the parts that beat chance.

## Baseline (Phase 1, VTT + visual, single run)
| Metric | Value |
|---|---|
| Pairwise ordering | 25.5/36 = 71% |
| Exact / within one | 4/10 / 8/10 |
| MAE / bias | 0.80 / -0.60 (harsh) |
| Weakness recall | 11/18 = 61% |
| Strength recall | 15/17 = 88% |
| Balanced criterion accuracy | 75% |
| Evidence validity | 100% |
The judge confirms strengths well and misses problems; scores lean harsh at label level because of a near-constant technical_quality problem (E10).

## Targets (set before Phase 2 results; reported honestly whether met or not)
| Metric | Target |
|---|---|
| Pairwise ordering | >= 80% |
| Within one / MAE / bias | >= 9/10 / <= 0.6 / within +-0.3 |
| Weakness recall | >= 75% |
| Balanced criterion accuracy | >= 75% |
| Feedback recall | >= 60% |
| Evidence validity / genericness | >= 95% / <= 20% |
| Run agreement | >= 8/10 classes stable |
| Confidence | correct > wrong |
| VTT vs no-VTT | all 4 within one level |
| Time / cost | <= 10 min per class on the Space / <= $0.05 per class |
| Privacy gate | pass |

## Protocol
1. **Baseline v2**: current pipeline, 3 runs, all metrics. The run-to-run range of the primary metric is the noise band.
2. **Ablations**, each one change on top of the last kept one, 3 runs each:
   - A0: evidence before score (the judge's output lists evidence and observations first, score last)
   - A1: technical_quality by code (D14) + variance gate (D15). Expected on its own to lower exact accuracy: the constant tech problem was propping up the Poor predictions (E10 analysis).
   - A2: yes/no facts drive the criteria: rapport before content, new joiners invited, homework assigned, closing recap, each child asked individually, each with a verified quote; code maps facts to criterion scores, so a fact and its criterion can't contradict each other (e.g. homework given but closure 2).
   - A3: severity-tiered, two-sided label rule, written from the organiser's wording before this run. Critical problems (no rapport before content, tutor face/camera missing, no 8x8 background) -> Poor. Moderate problems (no homework, weak ending, low energy, lighting): 2+ -> Average, 1 -> Good. Excellent needs zero problems plus positive evidence (engagement 4 and rapport >= 3). Caveat: derived from the notes on these same 10 classes, so accuracy on them is optimistic.
   - A4: visual: screen-share-aware face metrics (after the contact-sheet check), longest camera-off stretch on camera frames, 8x8 background check.
   - A5: similar labelled classes as examples in the judge prompt (leave-one-out, same batch excluded).
   - A6: new signals (tutor speech rate, wait time after questions, audio clarity) as judge context.
   - A7 (only if time): comparative judging against labelled reference classes on profiles (facts, stats, observations), placing the new class by its wins; leave-one-out.
   All of these run on the 3-run median from baseline v2 onward.
3. **Keep rule**: keep a change if the primary metric improves by more than the noise band, or holds while weakness recall or bias clearly improves, and no gate fails. Otherwise revert and log it as a negative result. Negative results stay in EXPERIMENTS.md.
4. **Freeze** the config, run the final eval once (3 runs), and publish eval/scorecard.md. Only bug fixes after the freeze, with a re-run.
5. **Demo classes**: no labels. Solomon rates them with the rubric before seeing the app's output; the comparison is reported as a sanity check, not accuracy.

## Outputs
- eval/scorecard.md: metric | value mean [min-max] | target | n, generated by scripts/eval_labelled.py.
- eval/ablations.md: one row per configuration with the primary and key secondary metrics. Each row records the git commit and the judge-relevant settings (model, design, aggregation, temperature, prompt hash), so every number can be reproduced.
- Every run's full Assessment is kept under .cache/eval/<tag>/<run>/ (gitignored), so metrics can be recomputed without new API calls.
- Class IDs and numbers only. No names, no quotes.
