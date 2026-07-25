# Ablation: BERT-only baseline (isolating the GNN's contribution)

Same 70/30 split (`test_size=0.30, random_state=42`), the identical 9,276-row test set, and the same `bert-base-uncased` encoder as the BERT-GNN hybrid. The only change is the classifier: BERT's `[CLS]` embedding is fed to a simple head instead of a graph + GNN.

Test set: 9047 queries (5764 benign, 3283 malicious).

| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Confusion matrix |
|---|---|---|---|---|---|
| BERT-GNN hybrid (reference) | 99.48 | 99.48 | 99.48 | 99.48 | [[5815, 15], [33, 3413]] |
| BERT [CLS] + Logistic Regression (class_weight=balanced) | 99.99 | 100.00 | 99.97 | 99.98 | [[5764, 0], [1, 3282]] |
| BERT [CLS] + MLP (1 hidden layer, 159 units) | 99.99 | 99.97 | 100.00 | 99.98 | [[5763, 1], [0, 3283]] |

**Finding.** On this near-saturated benchmark the BERT-only baselines match, and marginally exceed, the BERT-GNN hybrid on accuracy, so the GNN component does not add measurable accuracy here. Reproduce with `python bert_only_ablation.py`.
