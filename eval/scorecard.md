# Scorecard: a1_tech_code_d15

2026-10-22, commit aea005d, 3 runs, VTT + visual. Judge deepseek-flash, design direct, aggregation problems, temperature 0.2, thinking disabled, prompt hash 8d133960672d.
Values are mean [min-max] across runs. Labels: weak and quality issue -> Poor. Criterion points: score <= 2 catches a criticised point, >= 3 a praised one; unscored counts as missed.

| metric | value | target | n |
|---|---|---|---|
| Pairwise ordering (problem count, ties 0.5) | 70% [68%-72%] | >= 80% | 36 pairs |
| Exact label | 3.3/10 [3-4] | - | 10 |
| Within one level | 8.3/10 [8-9] | >= 9/10 | 10 |
| MAE (levels) | 0.83 [0.80-0.90] | <= 0.6 | 10 |
| Bias (predicted - true; negative = harsh) | -0.30 [-0.40 to -0.20] | within +-0.3 | 10 |
| Weakness recall | 46% [44%-50%] | >= 75% | 18 |
| Strength recall | 90% [88%-94%] | - | 17 |
| Balanced criterion accuracy | 68% [66%-72%] | >= 75% | 35 |
| Feedback recall (all points) | 62% [60%-64%] | >= 60% | 47 |
| Feedback recall, transcript-observable (T) | 80% [77%-82%] | - | 28 |
| Feedback recall, video-observable (V) | 22% [21%-25%] | - | 12 |
| Feedback recall, voice-only (A) | 47% [42%-50%] | - | 6 |
| Feedback recall, not in recording (N) | 100% [100%-100%] | - | 1 |
| Evidence validity (quotes verified) | 100% [100%-100%] | >= 95% | 163 quotes/run |
| Genericness (observations with no speaker, time or quote) | 14% [13%-16%] | <= 20% | 166 observations/run |
| Run agreement (label identical in all runs) | 6/10 | >= 8/10 | 3 runs |
| Mean criterion score range across runs | 0.13 | - | class x criterion |
| Confidence on correct labels | 1.00 [1.00-1.00] | correct > wrong | - |
| Confidence on wrong labels | 1.00 [1.00-1.00] |  | - |
| Privacy gate (scripts/leak_audit.py) | PENDING REVIEW (hits >= 85: 0, labelled names left: 0, unlabelled 70-84 to review: 4); over-scrub, monitored not gated: 3 classes over 0.5% | pass | 11 classes |

Feedback grader: DeepSeek, temperature 0, sees only the report and the points. Verdicts whose cited line was not in the report were set to no: 0 across runs.

## Per criterion

| criterion | weakness recall | n | strength recall | n |
|---|---|---|---|---|
| opening_rapport | 56% [33%-67%] | 3 | 67% [67%-67%] | 3 |
| individual_engagement | - | 0 | 100% [100%-100%] | 7 |
| communication | 0% [0%-0%] | 5 | - | 0 |
| closure_homework | 67% [50%-75%] | 4 | 78% [67%-100%] | 3 |
| visual_presence | 50% [50%-50%] | 4 | 100% [100%-100%] | 4 |
| technical_quality | 100% [100%-100%] | 2 | - | 0 |

## Confusion matrix (summed over 3 runs; rows true, columns predicted)

| true \ predicted | Poor | Average | Good | Excellent |
|---|---|---|---|---|
| Poor | 5 | 7 | 0 | 0 |
| Average | 3 | 2 | 1 | 0 |
| Good | 3 | 0 | 3 | 0 |
| Excellent | 0 | 2 | 4 | 0 |

## Trivial baselines (any real result must beat these)

| baseline | exact | within one | MAE | bias | pairwise |
|---|---|---|---|---|---|
| Always "Poor" | 4/10 | 6/10 | 1.20 | -1.20 | 50% |
| Always "Average" | 2/10 | 8/10 | 1.00 | -0.20 | 50% |
| Random label (expected) | 2.5/10 | 6/10 | 1.30 | +0.30 | 50% |

## Per class

| class | true | label per run | problems per run | confidence per run |
|---|---|---|---|---|
| average/B435 | Average | Poor / Poor / Poor | 3 / 3 / 3 | 1.00 / 1.00 / 1.00 |
| average/B516 | Average | Average / Average / Good | 2 / 2 / 1 | 1.00 / 1.00 / 1.00 |
| excellent/B472 | Excellent | Good / Good / Good | 1 / 1 / 1 | 1.00 / 1.00 / 1.00 |
| excellent/B491 | Excellent | Average / Good / Average | 2 / 1 / 2 | 1.00 / 1.00 / 1.00 |
| good/B435 | Good | Poor / Poor / Poor | 3 / 3 / 3 | 1.00 / 1.00 / 1.00 |
| good/B552 | Good | Good / Good / Good | 1 / 1 / 1 | 1.00 / 1.00 / 1.00 |
| quality issue/B392 | Poor | Poor / Average / Average | 3 / 2 / 2 | 1.00 / 1.00 / 1.00 |
| quality issue/B441 | Poor | Poor / Poor / Poor | 3 / 4 / 4 | 1.00 / 1.00 / 1.00 |
| weak/B482 | Poor | Average / Average / Average | 2 / 2 / 2 | 1.00 / 1.00 / 1.00 |
| weak/B558 | Poor | Average / Average / Poor | 2 / 2 / 3 | 1.00 / 1.00 / 1.00 |
