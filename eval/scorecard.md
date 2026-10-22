# Scorecard: a3b_measurement_fixes

2026-10-22, commit cd5dbfc, 3 runs, VTT + visual. Judge deepseek-flash, design direct, aggregation severity, temperature 0.2, thinking disabled, prompt hash faae0432756d.
Values are mean [min-max] across runs. Labels: weak and quality issue -> Poor. Criterion points: score <= 2 catches a criticised point, >= 3 a praised one; unscored counts as missed.

| metric | value | target | n |
|---|---|---|---|
| Pairwise ordering (problem count or severity, ties 0.5) | 75% [74%-78%] | >= 80% | 36 pairs |
| Exact label | 3.0/10 [3-3] | - | 10 |
| Within one level | 9.3/10 [9-10] | >= 9/10 | 10 |
| MAE (levels) | 0.77 [0.70-0.80] | <= 0.6 | 10 |
| Bias (predicted - true; negative = harsh) | +0.03 [+0.00 to +0.10] | within +-0.3 | 10 |
| Weakness recall | 46% [44%-50%] | >= 75% | 18 |
| Strength recall | 88% [82%-94%] | - | 17 |
| Balanced criterion accuracy | 67% [63%-69%] | >= 75% | 35 |
| Feedback recall (all points) | 61% [60%-62%] | >= 60% | 47 |
| Feedback recall, transcript-observable (T) | 80% [80%-80%] | - | 28 |
| Feedback recall, video-observable (V) | 18% [17%-21%] | - | 12 |
| Feedback recall, voice-only (A) | 47% [42%-58%] | - | 6 |
| Feedback recall, not in recording (N) | 100% [100%-100%] | - | 1 |
| Evidence validity (quotes verified) | 100% [100%-100%] | >= 95% | 193 quotes/run |
| Genericness (observations with no speaker, time or quote) | 12% [6%-17%] | <= 20% | 163 observations/run |
| Run agreement (label identical in all runs) | 9/10 | >= 8/10 | 3 runs |
| Mean criterion score range across runs | 0.17 | - | class x criterion |
| Confidence on correct labels | 0.83 [0.83-0.83] | correct > wrong | - |
| Confidence on wrong labels | 0.89 [0.89-0.89] |  | - |
| Confidence with run agreement, correct labels | 0.92 [0.92-0.92] | correct > wrong | mean of app confidence and run agreement |
| Confidence with run agreement, wrong labels | 0.92 [0.92-0.92] |  | - |
| Privacy gate (scripts/leak_audit.py) | PENDING REVIEW (hits >= 85: 0, labelled names left: 0, unlabelled 70-84 to review: 4); over-scrub, monitored not gated: 3 classes over 0.5% | pass | 11 classes |

Feedback grader: DeepSeek, temperature 0, sees only the report and the points. Verdicts whose cited line was not in the report were set to no: 0 across runs.

## Per criterion

| criterion | weakness recall | n | strength recall | n |
|---|---|---|---|---|
| opening_rapport | 67% [67%-67%] | 3 | 56% [33%-67%] | 3 |
| individual_engagement | - | 0 | 90% [86%-100%] | 7 |
| communication | 0% [0%-0%] | 5 | - | 0 |
| closure_homework | 83% [75%-100%] | 4 | 100% [100%-100%] | 3 |
| visual_presence | 50% [50%-50%] | 4 | 100% [100%-100%] | 4 |
| technical_quality | 50% [50%-50%] | 2 | - | 0 |

## Confusion matrix (summed over 3 runs; rows true, columns predicted)

| true \ predicted | Poor | Average | Good | Excellent |
|---|---|---|---|---|
| Poor | 0 | 12 | 0 | 0 |
| Average | 0 | 6 | 0 | 0 |
| Good | 0 | 3 | 3 | 0 |
| Excellent | 0 | 2 | 4 | 0 |

## Trivial baselines (any real result must beat these)

| baseline | exact | within one | MAE | bias | pairwise |
|---|---|---|---|---|---|
| Always "Poor" | 4/10 | 6/10 | 1.20 | -1.20 | 50% |
| Always "Average" | 2/10 | 8/10 | 1.00 | -0.20 | 50% |
| Random label (expected) | 2.5/10 | 6/10 | 1.30 | +0.30 | 50% |

## Per class

| class | true | label per run | score/100 per run | problems per run | problems found (run 0) | confidence per run |
|---|---|---|---|---|---|---|
| average/B435 | Average | Average / Average / Average | 39 / 33 / 49 | 3 / 3 / 3 | tutor face in only 38% of camera frames; no homework; lighting: face luma 25 (< 85) | 1.00 / 1.00 / 1.00 |
| average/B516 | Average | Average / Average / Average | 59 / 67 / 67 | 2 / 2 / 2 | no closing recap; lighting: face luma 75 (< 85) | 0.75 / 0.75 / 0.75 |
| excellent/B472 | Excellent | Good / Good / Good | 65 / 73 / 73 | 1 / 1 / 1 | no closing recap | 0.75 / 0.75 / 0.75 |
| excellent/B491 | Excellent | Average / Average / Good | 71 / 71 / 71 | 2 / 2 / 1 | no homework; no closing recap | 0.75 / 0.75 / 0.75 |
| good/B435 | Good | Average / Average / Average | 45 / 45 / 45 | 3 / 3 / 3 | tutor face in only 39% of camera frames; no homework; no closing recap | 1.00 / 1.00 / 1.00 |
| good/B552 | Good | Good / Good / Good | 71 / 71 / 71 | 1 / 1 / 1 | lighting: face luma 83 (< 85) | 0.75 / 0.75 / 0.75 |
| quality issue/B392 | Poor | Average / Average / Average | 57 / 49 / 49 | 3 / 3 / 3 | tutor face in only 58% of camera frames; no closing recap; lighting: face luma 78 (< 85) | 1.00 / 1.00 / 1.00 |
| quality issue/B441 | Poor | Average / Average / Average | 31 / 31 / 41 | 3 / 3 / 3 | tutor face in only 4% of camera frames; no homework; no closing recap | 1.00 / 1.00 / 1.00 |
| weak/B482 | Poor | Average / Average / Average | 53 / 53 / 53 | 2 / 2 / 2 | no homework; no closing recap | 0.75 / 0.75 / 0.75 |
| weak/B558 | Poor | Average / Average / Average | 57 / 63 / 63 | 3 / 3 / 3 | tutor face in only 50% of camera frames; no homework; no closing recap | 1.00 / 1.00 / 1.00 |
