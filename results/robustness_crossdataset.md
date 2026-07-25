# Robustness test 5: out-of-distribution generalisation

Evaluation on a curated set that is deliberately different from the training dataset: 35 canonical SQL injection payloads (union-based, boolean and time-based blind, error-based, stacked, and classic WAF-bypass forms) and 34 legitimate SQL statements and benign text. This tests generalisation to a novel distribution.

| Model | Attack detection (%) | False-positive rate (%) | Accuracy (%) | Missed / false-alarms |
|---|---|---|---|---|
| Structure-aware BERT-GNN | 100.0 | 2.9 | 98.6 | 0 / 1 |
| BERT-only (MLP) | 100.0 | 11.8 | 94.2 | 0 / 4 |

Both models are trained only on the original dataset, so this measures how well the learned detector transfers to attack and benign patterns it was not trained on. Reproduce with `python robustness_5_crossdataset.py`.
