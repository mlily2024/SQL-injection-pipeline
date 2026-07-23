# Full metric suite (beyond accuracy)

This is an imbalanced security task (about 63% benign, 37% malicious), so accuracy alone is misleading. The operationally critical metrics are the malicious-class recall (attack detection rate) and the false-positive rate (false-alarm rate). All values are on the identical 9,276-query test set (5,830 benign, 3,446 malicious).

## Clean test set

| Model | Accuracy | Mal. precision | Mal. recall (detection) | Mal. F1 | FPR (%) | FNR (%) | Balanced acc | MCC | FP/FN |
|---|---|---|---|---|---|---|---|---|---|
| Original BERT-GNN (leaky) | 99.48 | 99.56 | 99.04 | 99.30 | 0.257 | 0.96 | 99.39 | 0.9889 | 15/33 |
| Held-out BERT-GNN | 99.67 | 99.54 | 99.56 | 99.55 | 0.274 | 0.44 | 99.65 | 0.9928 | 16/15 |
| Structure-aware BERT-GNN | 99.64 | 99.51 | 99.54 | 99.52 | 0.292 | 0.46 | 99.62 | 0.9924 | 17/16 |
| BERT [CLS] + LogReg | 99.66 | 99.68 | 99.39 | 99.54 | 0.189 | 0.61 | 99.60 | 0.9926 | 11/21 |
| BERT [CLS] + MLP | 99.73 | 99.80 | 99.48 | 99.64 | 0.120 | 0.52 | 99.68 | 0.9942 | 7/18 |

Two points accuracy hides: (i) the leaky original has the lowest attack detection (99.04% recall, 33 missed), and the held-out protocol recovers this to 99.56% (15 missed); (ii) the BERT-only MLP has the lowest false-alarm rate (FPR 0.12%). On clean data the differences are within run-to-run noise.

## Under URL-encoding evasion (deployment scenario: benign clean, malicious obfuscated)

| Model | Accuracy | Precision | Recall (detection) | F1 | FPR (%) | MCC | Attacks missed (FN) |
|---|---|---|---|---|---|---|---|
| Structure-aware BERT-GNN | 99.14 | 99.50 | 98.17 | 98.83 | 0.292 | 0.9815 | 63 |
| BERT [CLS] + MLP | 96.33 | 99.78 | 90.34 | 94.82 | 0.120 | 0.9227 | 333 |

Precision stays high for both (benign traffic is not obfuscated), so the degradation is entirely in detection: under URL encoding the BERT-only model's F1 falls to 94.82% and it misses 333 malicious queries, against the structure-aware model's 98.83% F1 and 63 missed, a five-fold difference in attacks let through. This is the security-relevant failure mode.

Reproduce with `python metrics_summary.py` (source confusion matrices and recalls from `bert_only_ablation.py`, `corrected_bertgnn_retrain.py`, `structure_graph_gnn.py`, `obfuscation_robustness.py`).
