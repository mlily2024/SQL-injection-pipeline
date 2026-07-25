# Seed-stability of structure-GNN URL-encode robustness

Robust config {'hid': 169, 'drop': 0.15579754426081674, 'lr': 7.52374288453485e-05, 'bs': 67} trained across 5 seeds (seeding before weight init). Recall on the 3,446 malicious test queries. BERT-only baseline is deterministic at 90.34%.

| Seed | Clean recall (%) | URL-encode recall (%) |
|---|---|---|
| 42 | 99.54 | 74.14 |
| 1 | 99.56 | 95.12 |
| 7 | 99.48 | 94.66 |
| 13 | 99.54 | 88.83 |
| 123 | 99.74 | 97.65 |

**URL-encode recall: mean 90.08% +/- 9.48** (range 74.14-97.65); BERT-only 90.34%. Clean recall mean 99.57% +/- 0.10.
