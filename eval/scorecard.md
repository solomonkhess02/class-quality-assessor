# Scorecard: baseline

2026-10-22, commit 179858d, 3 runs, VTT + visual. Judge deepseek-flash, design direct, aggregation problems, temperature 0.2, thinking disabled, prompt hash 100cb76c421f.
Values are mean [min-max] across runs. Labels: weak and quality issue -> Poor. Criterion points: score <= 2 catches a criticised point, >= 3 a praised one; unscored counts as missed.

| metric | value | target | n |
|---|---|---|---|
| Pairwise ordering (problem count, ties 0.5) | 66% [64%-68%] | >= 80% | 36 pairs |
| Exact label | 4.3/10 [4-5] | - | 10 |
| Within one level | 7.7/10 [7-8] | >= 9/10 | 10 |
| MAE (levels) | 0.80 [0.70-0.90] | <= 0.6 | 10 |
| Bias (predicted - true; negative = harsh) | -0.47 [-0.70 to -0.30] | within +-0.3 | 10 |
| Weakness recall | 61% [61%-61%] | >= 75% | 18 |
| Strength recall | 86% [82%-88%] | - | 17 |
| Balanced criterion accuracy | 74% [72%-75%] | >= 75% | 35 |
| Feedback recall (all points) | 66% [62%-68%] | >= 60% | 47 |
| Feedback recall, transcript-observable (T) | 85% [82%-88%] | - | 28 |
| Feedback recall, video-observable (V) | 26% [21%-29%] | - | 12 |
| Feedback recall, voice-only (A) | 53% [42%-67%] | - | 6 |
| Feedback recall, not in recording (N) | 100% [100%-100%] | - | 1 |
| Evidence validity (quotes verified) | 100% [100%-100%] | >= 95% | 167 quotes/run |
| Genericness (observations with no speaker, time or quote) | 12% [10%-13%] | <= 20% | 162 observations/run |
| Run agreement (label identical in all runs) | 8/10 | >= 8/10 | 3 runs |
| Mean criterion score range across runs | 0.08 | - | class x criterion |
| Confidence on correct labels | 1.00 [1.00-1.00] | correct > wrong | - |
| Confidence on wrong labels | 1.00 [1.00-1.00] |  | - |
| Privacy gate (scripts/leak_audit.py) | FAIL (hits >= 85: 0, labelled names left: 0, classes over 0.5% over-scrub: 3, unlabelled 70-84 to review: 4) | pass | 11 classes |

Feedback grader: DeepSeek, temperature 0, sees only the report and the points. Verdicts whose cited line was not in the report were set to no: 0 across runs.

## Per criterion

| criterion | weakness recall | n | strength recall | n |
|---|---|---|---|---|
| opening_rapport | 100% [100%-100%] | 3 | 56% [33%-67%] | 3 |
| individual_engagement | - | 0 | 100% [100%-100%] | 7 |
| communication | 0% [0%-0%] | 5 | - | 0 |
| closure_homework | 100% [100%-100%] | 4 | 67% [67%-67%] | 3 |
| visual_presence | 50% [50%-50%] | 4 | 100% [100%-100%] | 4 |
| technical_quality | 100% [100%-100%] | 2 | - | 0 |

## Confusion matrix (summed over 3 runs; rows true, columns predicted)

| true \ predicted | Poor | Average | Good | Excellent |
|---|---|---|---|---|
| Poor | 9 | 3 | 0 | 0 |
| Average | 3 | 3 | 0 | 0 |
| Good | 3 | 1 | 0 | 2 |
| Excellent | 0 | 4 | 1 | 1 |

## Trivial baselines (any real result must beat these)

| baseline | exact | within one | MAE | bias | pairwise |
|---|---|---|---|---|---|
| Always "Poor" | 4/10 | 6/10 | 1.20 | -1.20 | 50% |
| Always "Average" | 2/10 | 8/10 | 1.00 | -0.20 | 50% |
| Random label (expected) | 2.5/10 | 6/10 | 1.30 | +0.30 | 50% |

## Per class

| class | true | label per run | problems per run | confidence per run |
|---|---|---|---|---|
| average/B435 | Average | Poor / Poor / Poor | 4 / 4 / 4 | 1.00 / 1.00 / 1.00 |
| average/B516 | Average | Average / Average / Average | 2 / 2 / 2 | 1.00 / 1.00 / 1.00 |
| excellent/B472 | Excellent | Good / Excellent / Average | 1 / 0 / 2 | 1.00 / 1.00 / 1.00 |
| excellent/B491 | Excellent | Average / Average / Average | 2 / 2 / 2 | 1.00 / 1.00 / 1.00 |
| good/B435 | Good | Poor / Poor / Poor | 4 / 4 / 4 | 1.00 / 1.00 / 1.00 |
| good/B552 | Good | Excellent / Excellent / Average | 0 / 0 / 2 | 1.00 / 1.00 / 1.00 |
| quality issue/B392 | Poor | Average / Average / Average | 2 / 2 / 2 | 1.00 / 1.00 / 1.00 |
| quality issue/B441 | Poor | Poor / Poor / Poor | 4 / 4 / 4 | 1.00 / 1.00 / 1.00 |
| weak/B482 | Poor | Poor / Poor / Poor | 3 / 3 / 3 | 1.00 / 1.00 / 1.00 |
| weak/B558 | Poor | Poor / Poor / Poor | 3 / 3 / 3 | 1.00 / 1.00 / 1.00 |
