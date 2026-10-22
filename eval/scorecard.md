# Scorecard: a3_severity_rule

2026-10-22, commit 5c41a28, 3 runs, VTT + visual. Judge deepseek-flash, design direct, aggregation severity, temperature 0.2, thinking disabled, prompt hash bcfc0831e6ec.
Values are mean [min-max] across runs. Labels: weak and quality issue -> Poor. Criterion points: score <= 2 catches a criticised point, >= 3 a praised one; unscored counts as missed.

| metric | value | target | n |
|---|---|---|---|
| Pairwise ordering (problem count or severity, ties 0.5) | 84% [82%-88%] | >= 80% | 36 pairs |
| Exact label | 5.0/10 [5-5] | - | 10 |
| Within one level | 7.0/10 [7-7] | >= 9/10 | 10 |
| MAE (levels) | 0.80 [0.80-0.80] | <= 0.6 | 10 |
| Bias (predicted - true; negative = harsh) | -0.80 [-0.80 to -0.80] | within +-0.3 | 10 |
| Weakness recall | 54% [50%-56%] | >= 75% | 18 |
| Strength recall | 82% [76%-88%] | - | 17 |
| Balanced criterion accuracy | 68% [63%-72%] | >= 75% | 35 |
| Feedback recall (all points) | 61% [60%-65%] | >= 60% | 47 |
| Feedback recall, transcript-observable (T) | 80% [77%-82%] | - | 28 |
| Feedback recall, video-observable (V) | 21% [21%-21%] | - | 12 |
| Feedback recall, voice-only (A) | 47% [25%-67%] | - | 6 |
| Feedback recall, not in recording (N) | 100% [100%-100%] | - | 1 |
| Evidence validity (quotes verified) | 100% [99%-100%] | >= 95% | 176 quotes/run |
| Genericness (observations with no speaker, time or quote) | 12% [11%-12%] | <= 20% | 165 observations/run |
| Run agreement (label identical in all runs) | 10/10 | >= 8/10 | 3 runs |
| Mean criterion score range across runs | 0.18 | - | class x criterion |
| Confidence on correct labels | 0.87 [0.85-0.90] | correct > wrong | - |
| Confidence on wrong labels | 0.82 [0.80-0.85] |  | - |
| Confidence with run agreement, correct labels | 0.93 [0.93-0.95] | correct > wrong | mean of app confidence and run agreement |
| Confidence with run agreement, wrong labels | 0.91 [0.90-0.93] |  | - |
| Privacy gate (scripts/leak_audit.py) | PENDING REVIEW (hits >= 85: 0, labelled names left: 0, unlabelled 70-84 to review: 4); over-scrub, monitored not gated: 3 classes over 0.5% | pass | 11 classes |

Feedback grader: DeepSeek, temperature 0, sees only the report and the points. Verdicts whose cited line was not in the report were set to no: 0 across runs.

## Per criterion

| criterion | weakness recall | n | strength recall | n |
|---|---|---|---|---|
| opening_rapport | 78% [67%-100%] | 3 | 67% [67%-67%] | 3 |
| individual_engagement | - | 0 | 71% [57%-86%] | 7 |
| communication | 7% [0%-20%] | 5 | - | 0 |
| closure_homework | 75% [75%-75%] | 4 | 100% [100%-100%] | 3 |
| visual_presence | 50% [50%-50%] | 4 | 100% [100%-100%] | 4 |
| technical_quality | 100% [100%-100%] | 2 | - | 0 |

## Confusion matrix (summed over 3 runs; rows true, columns predicted)

| true \ predicted | Poor | Average | Good | Excellent |
|---|---|---|---|---|
| Poor | 12 | 0 | 0 | 0 |
| Average | 3 | 3 | 0 | 0 |
| Good | 6 | 0 | 0 | 0 |
| Excellent | 0 | 3 | 3 | 0 |

## Trivial baselines (any real result must beat these)

| baseline | exact | within one | MAE | bias | pairwise |
|---|---|---|---|---|---|
| Always "Poor" | 4/10 | 6/10 | 1.20 | -1.20 | 50% |
| Always "Average" | 2/10 | 8/10 | 1.00 | -0.20 | 50% |
| Random label (expected) | 2.5/10 | 6/10 | 1.30 | +0.30 | 50% |

## Per class

| class | true | label per run | score/100 per run | problems per run | problems found (run 0) | confidence per run |
|---|---|---|---|---|---|---|
| average/B435 | Average | Poor / Poor / Poor | 49 / 49 / 49 | 22 / 22 / 22 | no rapport before content; tutor face in only 38% of camera frames; no homework; lighting: 28% dark frames | 1.00 / 1.00 / 1.00 |
| average/B516 | Average | Average / Average / Average | 53 / 53 / 63 | 2 / 2 / 2 | no closing recap; lighting: 58% dark frames | 0.75 / 0.75 / 0.75 |
| excellent/B472 | Excellent | Good / Good / Good | 69 / 69 / 69 | 1 / 1 / 1 | no closing recap | 0.75 / 0.75 / 0.75 |
| excellent/B491 | Excellent | Average / Average / Average | 67 / 57 / 67 | 3 / 2 / 2 | no homework; no closing recap; lighting: 27% dark frames | 1.00 / 0.75 / 0.75 |
| good/B435 | Good | Poor / Poor / Poor | 35 / 45 / 35 | 13 / 13 / 13 | tutor face in only 39% of camera frames; no homework; no closing recap; lighting: 48% dark frames | 0.75 / 0.75 / 0.75 |
| good/B552 | Good | Poor / Poor / Poor | 71 / 71 / 80 | 10 / 10 / 10 | no rapport before content | 0.75 / 0.75 / 0.75 |
| quality issue/B392 | Poor | Poor / Poor / Poor | 49 / 49 / 49 | 23 / 23 / 23 | no rapport before content; tutor face in only 58% of camera frames; no homework; no closing recap; lighting: 45% dark frames | 1.00 / 1.00 / 1.00 |
| quality issue/B441 | Poor | Poor / Poor / Poor | 31 / 31 / 22 | 23 / 23 / 23 | no rapport before content; tutor face in only 4% of camera frames; no homework; no closing recap; lighting: 42% dark frames | 1.00 / 1.00 / 1.00 |
| weak/B482 | Poor | Poor / Poor / Poor | 43 / 43 / 47 | 13 / 13 / 13 | no rapport before content; no homework; no closing recap; lighting: 29% dark frames | 0.75 / 0.75 / 0.75 |
| weak/B558 | Poor | Poor / Poor / Poor | 53 / 39 / 63 | 12 / 22 / 12 | tutor face in only 50% of camera frames; no homework; no closing recap | 0.75 / 1.00 / 0.75 |
