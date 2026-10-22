# Ablations

One row per configuration (docs/EVALUATION.md). Values are mean [min-max] across runs.

| tag | date | commit | mode | runs | model | design | aggregation | temp | thinking | prompt hash | pairwise | exact | within one | MAE | bias | weakness recall | balanced | feedback recall | settings |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 2026-10-22 | 179858d | VTT + visual | 3 | deepseek-flash | direct | problems | 0.2 | disabled | 100cb76c421f | 66% [64%-68%] | 4.3/10 [4-5] | 7.7/10 [7-8] | 0.80 [0.70-0.90] | -0.47 [-0.70 to -0.30] | 61% [61%-61%] | 74% [72%-75%] | 66% [62%-68%] | evidence_first=no, tech=llm, exclude=-, facts_drive=no |
| a0_evidence_first | 2026-10-22 | 190125d | VTT + visual | 3 | deepseek-flash | direct | problems | 0.2 | disabled | c3ab1df1e88a | 70% [67%-72%] | 5.0/10 [3-7] | 7.7/10 [7-8] | 0.73 [0.60-0.90] | -0.07 [-0.20 to +0.10] | 46% [44%-50%] | 68% [66%-72%] | 63% [63%-65%] | evidence_first=yes, tech=llm, exclude=-, facts_drive=no |
