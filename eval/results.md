# Evaluation on the labelled set

Judge: deepseek-flash, design direct, thinking disabled, temperature 0.2. Labels: weak and quality issue -> Poor.
Criterion cells: score 1-4 ('-' = insufficient evidence); ✓ / ✗ = agrees / disagrees with the organiser's expectation (score >= 3 = good, <= 2 = weak), ? = organiser expectation but no score.
Mean rule: Excellent >= 3.4, Good >= 2.8, Average >= 2.2. Problem rule: criteria scored <= 2; 0 Excellent, 1 Good, 2 Average, 3+ Poor.
Shipped label uses AGGREGATION=problems; * = placeholder (fewer than 3 criteria scored).

## VTT + visual

- Weighted mean rule: exact 1/10, within one level 9/10
- Problem-count rule: exact 4/10, within one level 8/10

| criterion | agree | disagree | no score | agreement |
|---|---|---|---|---|
| opening_rapport | 6 | 0 | 0 | 100% |
| individual_engagement | 6 | 1 | 0 | 86% |
| communication | 0 | 5 | 0 | 0% |
| closure_homework | 6 | 1 | 0 | 86% |
| visual_presence | 6 | 2 | 0 | 75% |
| technical_quality | 2 | 0 | 0 | 100% |
| **all** | 26 | 9 | 0 | 74% |

### Labelled classes

| class | true | shipped | mean label | mean | problem label | problems | conf | open | engage | warmth | close | visual | tech | key facts | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| average/B435 | Average | Poor | Poor | 2.15 | Poor | 5 | 1.00 | 2 | 2 ✗ | 3 ✗ | 2 ✓ | 2 ✓ | 2 | hw=n join=not_invited greet=1 recap=y | 17/17 |
| average/B516 | Average | Poor | Average | 2.55 | Poor | 3 | 1.00 | 2 | 3 ✓ | 3 | 2 ✗ | 3 ✓ | 2 ✓ | hw=y join=none greet=3 recap=n | 19/19 |
| excellent/B472 | Excellent | Good | Good | 3.05 | Good | 1 | 1.00 | 3 ✓ | 3 ✓ | 3 | 3 ✓ | 4 | 2 | hw=y join=not_invited greet=2 recap=n | 18/18 |
| excellent/B491 | Excellent | Average | Good | 3.00 | Average | 2 | 1.00 | 3 | 4 ✓ | 3 | 2 ✓ | 3 ✓ | 2 | hw=n join=invited greet=6 recap=y | 22/22 |
| good/B435 | Good | Poor | Average | 2.30 | Poor | 3 | 1.00 | 3 ✓ | 3 ✓ | 3 | 1 ✓ | 1 | 2 | hw=n join=none greet=3 recap=n | 16/16 |
| good/B552 | Good | Good | Excellent | 3.45 | Good | 1 | 1.00 | 3 ✓ | 4 | 3 ✗ | 4 ✓ | 4 ✓ | 2 | hw=y join=none greet=4 recap=y | 18/18 |
| quality issue/B392 | Poor | Average | Good | 3.00 | Average | 2 | 1.00 | 3 | 4 ✓ | 3 ✗ | 2 | 3 ✗ | 2 ✓ | hw=n join=none greet=1 recap=n | 15/15 |
| quality issue/B441 | Poor | Poor | Average | 2.25 | Poor | 4 | 1.00 | 2 ✓ | 3 | 3 ✗ | 2 | 1 ✓ | 2 | hw=n join=not_invited greet=4 recap=n | 17/17 |
| weak/B482 | Poor | Poor | Average | 2.55 | Poor | 3 | 1.00 | 2 ✓ | 3 ✓ | 3 | 2 | 3 ✗ | 2 | hw=n join=invited greet=0 recap=n | 15/15 |
| weak/B558 | Poor | Poor | Average | 2.55 | Poor | 3 | 1.00 | 2 ✓ | 3 | 3 ✗ | 1 ✓ | 4 ✓ | 2 | hw=n join=none greet=2 recap=n | 15/15 |

### Demo classes (unlabelled)

| class | true | shipped | mean label | mean | problem label | problems | conf | open | engage | warmth | close | visual | tech | key facts | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| demo/1 | - | Average* | Average | 2.60 | Good | 1 | 0.15 | - | - | - | - | 3 | 2 | hw=n join=none greet=0 recap=n | 6/6 |
| demo/2 | - | Average | Average | 2.60 | Average | 2 | 1.00 | 3 | 2 | 3 | 1 | 4 | 3 | hw=n join=none greet=0 recap=n | 12/12 |

## VTT only

- Weighted mean rule: exact 3/10, within one level 8/10
- Problem-count rule: exact 5/10, within one level 7/10

| criterion | agree | disagree | no score | agreement |
|---|---|---|---|---|
| opening_rapport | 5 | 1 | 0 | 83% |
| individual_engagement | 7 | 0 | 0 | 100% |
| communication | 0 | 5 | 0 | 0% |
| closure_homework | 6 | 1 | 0 | 86% |
| visual_presence | 0 | 0 | 8 | - |
| technical_quality | 2 | 0 | 0 | 100% |
| **all** | 20 | 7 | 8 | 74% |

### Labelled classes

| class | true | shipped | mean label | mean | problem label | problems | conf | open | engage | warmth | close | visual | tech | key facts | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| average/B435 | Average | Average | Average | 2.59 | Average | 2 | 0.81 | 2 | 3 ✓ | 3 ✗ | 2 ✓ | - ? | 3 | hw=n join=not_invited greet=1 recap=y | 16/16 |
| average/B516 | Average | Poor | Average | 2.47 | Poor | 3 | 0.81 | 2 | 3 ✓ | 3 | 2 ✗ | - ? | 2 ✓ | hw=y join=none greet=4 recap=n | 17/17 |
| excellent/B472 | Excellent | Average | Average | 2.71 | Average | 2 | 0.81 | 3 ✓ | 3 ✓ | 2 | 3 ✓ | - | 2 | hw=y join=none greet=2 recap=n | 18/18 |
| excellent/B491 | Excellent | Average | Good | 3.00 | Average | 2 | 0.81 | 3 | 4 ✓ | 3 | 2 ✓ | - ? | 2 | hw=n join=invited greet=5 recap=y | 22/22 |
| good/B435 | Good | Poor | Average | 2.29 | Poor | 3 | 0.81 | 2 ✗ | 3 ✓ | 3 | 1 ✓ | - | 2 | hw=n join=none greet=3 recap=n | 16/16 |
| good/B552 | Good | Good | Good | 3.35 | Good | 1 | 0.81 | 3 ✓ | 4 | 3 ✗ | 4 ✓ | - ? | 2 | hw=y join=none greet=3 recap=y | 19/19 |
| quality issue/B392 | Poor | Average | Good | 3.00 | Average | 2 | 0.81 | 3 | 4 ✓ | 3 ✗ | 2 | - ? | 2 ✓ | hw=n join=none greet=1 recap=n | 15/15 |
| quality issue/B441 | Poor | Poor | Average | 2.47 | Poor | 3 | 0.81 | 2 ✓ | 3 | 3 ✗ | 2 | - ? | 2 | hw=n join=not_invited greet=2 recap=y | 17/17 |
| weak/B482 | Poor | Poor | Average | 2.47 | Poor | 3 | 0.81 | 2 ✓ | 3 ✓ | 3 | 2 | - ? | 2 | hw=n join=invited greet=0 recap=n | 15/15 |
| weak/B558 | Poor | Poor | Average | 2.47 | Poor | 3 | 0.81 | 2 ✓ | 3 | 3 ✗ | 2 ✓ | - ? | 2 | hw=n join=none greet=2 recap=n | 15/15 |

### Demo classes (unlabelled)

| class | true | shipped | mean label | mean | problem label | problems | conf | open | engage | warmth | close | visual | tech | key facts | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| demo/1 | - | Average* | Average | - | Excellent | 0 | 0.00 | - | - | - | - | - | - | hw=n join=none greet=0 recap=n | 14/14 |
| demo/2 | - | Average | Average | 2.35 | Average | 2 | 0.81 | 3 | 2 | 3 | 1 | - | 3 | hw=n join=none greet=0 recap=n | 12/12 |
