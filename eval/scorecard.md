# Scorecard: final_v2

2026-10-22, commit 687a291-dirty, 3 runs, VTT + visual. Judge deepseek-flash, design direct, aggregation severity, temperature 0.2, thinking disabled, prompt hash b2eae1c92849.
Values are mean [min-max] across runs. Labels: weak and quality issue -> Poor. Criterion points: score <= 2 catches a criticised point, >= 3 a praised one; unscored counts as missed.

| metric | value | target | n |
|---|---|---|---|
| Pairwise ordering (problem count or severity, ties 0.5) | 83% [82%-83%] | >= 80% | 36 pairs |
| Exact label | 4.0/10 [4-4] | - | 10 |
| Within one level | 9.7/10 [9-10] | >= 9/10 | 10 |
| MAE (levels) | 0.63 [0.60-0.70] | <= 0.6 | 10 |
| Bias (predicted - true; negative = harsh) | -0.43 [-0.50 to -0.40] | within +-0.3 | 10 |
| Weakness recall | 48% [44%-50%] | >= 75% | 18 |
| Strength recall | 96% [94%-100%] | - | 17 |
| Balanced criterion accuracy | 72% [69%-75%] | >= 75% | 35 |
| Feedback recall (all points) | 65% [64%-66%] | >= 60% | 47 |
| Feedback recall, transcript-observable (T) | 87% [86%-88%] | - | 28 |
| Feedback recall, video-observable (V) | 24% [21%-25%] | - | 12 |
| Feedback recall, voice-only (A) | 39% [33%-42%] | - | 6 |
| Feedback recall, not in recording (N) | 100% [100%-100%] | - | 1 |
| Evidence validity (quotes verified) | 99% [99%-99%] | >= 95% | 193 quotes/run |
| Genericness (observations with no speaker, time or quote) | 16% [13%-21%] | <= 20% | 160 observations/run |
| Run agreement (label identical in all runs) | 9/10 | >= 8/10 | 3 runs |
| Mean criterion score range across runs | 0.10 | - | class x criterion |
| Confidence on correct labels | 0.75 [0.75-0.75] | correct > wrong | - |
| Confidence on wrong labels | 0.83 [0.83-0.83] |  | - |
| Confidence with run agreement, correct labels | 0.88 [0.88-0.88] | correct > wrong | mean of app confidence and run agreement |
| Confidence with run agreement, wrong labels | 0.89 [0.89-0.89] |  | - |
| Privacy gate (scripts/leak_audit.py) | PASS (hits >= 85: 0, labelled names left: 0, unlabelled 70-84 to review: 0, documented exceptions: 1); over-scrub, monitored not gated: 4 classes over 0.5% | pass | 11 classes |

Feedback grader: DeepSeek, temperature 0, sees only the report and the points. Verdicts whose cited line was not in the report were set to no: 0 across runs.

## Per criterion

| criterion | weakness recall | n | strength recall | n |
|---|---|---|---|---|
| opening_rapport | 89% [67%-100%] | 3 | 78% [67%-100%] | 3 |
| individual_engagement | - | 0 | 100% [100%-100%] | 7 |
| communication | 0% [0%-0%] | 5 | - | 0 |
| closure_homework | 75% [75%-75%] | 4 | 100% [100%-100%] | 3 |
| visual_presence | 50% [50%-50%] | 4 | 100% [100%-100%] | 4 |
| technical_quality | 50% [50%-50%] | 2 | - | 0 |

## Confusion matrix (summed over 3 runs; rows true, columns predicted)

| true \ predicted | Poor | Average | Good | Excellent |
|---|---|---|---|---|
| Poor | 9 | 3 | 0 | 0 |
| Average | 6 | 0 | 0 | 0 |
| Good | 0 | 3 | 3 | 0 |
| Excellent | 0 | 1 | 5 | 0 |

## Trivial baselines (any real result must beat these)

| baseline | exact | within one | MAE | bias | pairwise |
|---|---|---|---|---|---|
| Always "Poor" | 4/10 | 6/10 | 1.20 | -1.20 | 50% |
| Always "Average" | 2/10 | 8/10 | 1.00 | -0.20 | 50% |
| Random label (expected) | 2.5/10 | 6/10 | 1.30 | +0.30 | 50% |

## Per class

| class | true | label per run | score/100 per run | problems per run | problems found (run 0) | warm-up min / students per run | confidence per run |
|---|---|---|---|---|---|---|---|
| average/B435 | Average | Poor / Poor / Poor | 49 / 49 / 49 | 13 / 13 / 13 | no rapport before content; tutor face in only 38% of camera frames; no homework; lighting: face luma 25 (< 85) | 1.6/0 / 1.6/0 / 1.6/0 | 0.75 / 0.75 / 0.75 |
| average/B516 | Average | Poor / Poor / Poor | 59 / 59 / 59 | 12 / 12 / 12 | no rapport before content; no closing recap; lighting: face luma 75 (< 85) | 0.9/1 / 0.9/1 / 0.9/1 | 0.75 / 0.75 / 0.75 |
| excellent/B472 | Excellent | Good / Good / Good | 73 / 73 / 73 | 1 / 1 / 1 | no closing recap | 8.7/2 / 16/3 / 16/3 | 0.75 / 0.75 / 0.75 |
| excellent/B491 | Excellent | Good / Good / Average | 71 / 71 / 71 | 1 / 1 / 2 | no homework | 5.1/4 / 14.9/7 / 14.9/7 | 0.75 / 0.75 / 0.75 |
| good/B435 | Good | Average / Average / Average | 45 / 45 / 39 | 3 / 3 / 3 | tutor face in only 39% of camera frames; no homework; no closing recap | ? / ? / ? | 1.00 / 1.00 / 1.00 |
| good/B552 | Good | Good / Good / Good | 71 / 78 / 71 | 1 / 1 / 1 | lighting: face luma 83 (< 85) | 5.5/2 / 5.5/2 / 5.5/2 | 0.75 / 0.75 / 0.75 |
| quality issue/B392 | Poor | Poor / Poor / Poor | 49 / 49 / 49 | 13 / 13 / 13 | no rapport before content; tutor face in only 58% of camera frames; no closing recap; lighting: face luma 78 (< 85) | 0.6/1 / 0.6/1 / 0.6/1 | 0.75 / 0.75 / 0.75 |
| quality issue/B441 | Poor | Average / Average / Average | 31 / 31 / 39 | 3 / 3 / 3 | tutor face in only 4% of camera frames; no homework; no closing recap | 6.1/3 / 6.1/3 / 6.1/3 | 1.00 / 1.00 / 1.00 |
| weak/B482 | Poor | Poor / Poor / Poor | 53 / 53 / 53 | 12 / 12 / 12 | no rapport before content; no homework; no closing recap | 2.7/1 / 2.7/1 / 2.7/1 | 0.75 / 0.75 / 0.75 |
| weak/B558 | Poor | Poor / Poor / Poor | 55 / 45 / 39 | 13 / 13 / 13 | no rapport before content; tutor face in only 50% of camera frames; no homework; no closing recap | 1.9/1 / 1.9/1 / 1.9/1 | 0.75 / 0.75 / 0.75 |
