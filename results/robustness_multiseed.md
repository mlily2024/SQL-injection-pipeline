# Multi-seed robustness: structure-GNN vs BERT-only

Structure-GNN (robust config) over 5 seeds vs the deterministic BERT-only baseline. mean +/- std. Lower is better for noise-flip and ECE; higher for recall/accuracy.

| Metric | Structure-GNN (mean +/- std) | BERT-only | GNN better & seed-stable? |
|---|---|---|---|
| clean_acc | 99.67 +/- 0.04 | 99.73 | no (BERT-only better) |
| url_recall | 90.08 +/- 9.48 | 90.34 | no (BERT-only better) |
| noise_flip_0.5 | 0.62 +/- 0.03 | 10.04 | YES |
| noise_flip_1.0 | 3.42 +/- 0.30 | 24.85 | YES |
| ece_clean | 0.00 +/- 0.00 | 0.0022 | no (within noise) |