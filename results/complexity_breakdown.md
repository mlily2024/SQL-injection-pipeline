# Accuracy by query complexity (structure-aware BERT-GNN vs BERT-only)

Complexity is the top quartile of the test set by query word count (the Query_Length feature of Section 4.2): complex = length strictly greater than the 75th percentile, which on this test set is 14 words. Predictions are the canonical committed predictions (structure-GNN 99.64%, Section 5.4; BERT-only MLP 99.73%, Table 3), aligned per query. For the all-malicious 'complex malicious' subset the figure is the detection rate (recall).

| Test subset | n | Structure-GNN (%) | BERT-only (%) |
|---|---|---|---|
| all | 9,276 | 99.64 | 99.73 |
| complex (top quartile, > 14 words) | 2,221 | 99.95 | 99.91 |
| complex malicious | 1,866 | 100.00 | 100.00 |

The two models are within run-to-run noise on every subset, including the most complex queries: there is no query-complexity regime in which the graph improves clean accuracy. Reproduce with `python complexity_breakdown.py`.
