# Robustness test 7: calibration under attack

Expected calibration error (ECE, 15 bins): the gap between a model's confidence and its accuracy. Lower is better; a high ECE under attack means the model is confidently wrong.

| Setting | Structure-GNN ECE | BERT-only ECE |
|---|---|---|
| Clean test set | 0.0015 | 0.0022 |
| URL-encoded malicious | 0.0283 | 0.0592 |

Under URL encoding the BERT-only model not only misses more attacks (333 vs 63) but does so with high confidence (mean confidence 0.82 on the queries it misclassifies, against 0.75 for the structure-GNN). A detector that is confidently wrong under evasion is the more dangerous failure mode.

Reproduce with `python robustness_7_calibration.py`.
