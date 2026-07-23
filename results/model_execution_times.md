# Execution time across models (same machine)

Wall-clock times on a 12-core CPU, measured like-for-like (the original 23.13 s was on different hardware). Training time is to early stopping; inference time is over the 9,276-query test set. The BERT embedding is a shared cost common to every model and dominates deployment latency.

| Model | Training time (s) | Test-set inference (s) | Inference per query (ms) |
|---|---|---|---|
| BERT-GNN hybrid (chain graph) | 99.01 | 3.449 | 0.372 |
| Structure-aware BERT-GNN | 58.41 | 1.448 | 0.156 |
| BERT [CLS] + Logistic Regression | 1.0 | 0.0356 | 0.0038 |
| BERT [CLS] + MLP | 24.15 | 0.0982 | 0.0106 |

Shared BERT embedding: 98.83 ms per query (dominant deployment cost, common to all models). The head/graph inference above is on top of this shared embedding step. Reproduce with `python model_execution_times.py`.
