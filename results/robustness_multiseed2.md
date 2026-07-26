# Multi-seed adaptive-evasion + OOD: structure-GNN vs BERT-only

5 seeds, mean +/- std. Lower is better for evasion & FPR; higher for recall.

| Metric | Structure-GNN | BERT-only | GNN better & seed-stable? |
|---|---|---|---|
| adaptive_evasion | 10.15 +/- 9.26 | 8.0 | no (BERT-only better) |
| ood_fpr | 7.64 +/- 3.35 | 11.76 | no (within noise) |
| ood_recall | 100.00 +/- 0.00 | 100.0 | no (BERT-only better) |