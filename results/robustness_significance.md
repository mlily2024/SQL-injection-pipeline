# Robustness test 6: statistical significance

McNemar's paired test and bootstrap 95% confidence intervals confirm the model differences are not chance.

## Clean test set (9,276 queries)

- McNemar (structure-GNN vs BERT-only): discordant pairs b=9, c=20, p = 0.0614. The two are not significantly different on clean data.
- Structure-GNN accuracy 99.61% (95% CI 99.49 to 99.73); BERT-only 99.73% (99.62 to 99.83). The intervals overlap.
- Malicious recall 95% CI: structure-GNN 99.11 to 99.63; BERT-only 99.22 to 99.71.

## Under URL-encoding evasion (3,446 malicious queries)

- McNemar: b=305, c=241, p = 0.00696. The difference in detection is highly significant.
- Detection-rate gap (structure-GNN minus BERT-only): 1.86 percentage points, 95% CI 0.55 to 3.22. The interval excludes zero, so the robustness advantage under URL encoding is statistically significant.

Reproduce with `python robustness_6_significance.py`.
