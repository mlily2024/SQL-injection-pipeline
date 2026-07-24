# BERT-GNN: held-out validation protocol

The original pipeline selected hyperparameters (Optuna, 50 trials) and applied early stopping using the **test** set, then reported on that same test set, so its 99.48% is optimistic. This run keeps the identical outer split (the same 9,276-row test set) but carves a **validation** set from the training portion, runs the hyperparameter search and early stopping on **validation**, and evaluates **once** on the untouched test set. Architecture, graph construction and class-weighted loss are unchanged.

Selected hyperparameters (by validation F1): `{'hidden_dim': 100, 'dropout_rate': 0.13906884560255356, 'lr': 0.0011290133559092666, 'batch_size': 74}`.

| Protocol | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Confusion matrix |
|---|---|---|---|---|---|
| Original (test used for selection) | 99.48 | 99.48 | 99.48 | 99.48 | [[5815, 15], [33, 3413]] |
| Held-out (validation) | 99.62 | 99.62 | 99.62 | 99.62 | [[5818, 12], [23, 3423]] |

### Confusion matrices

![Original BERT-GNN (test used for selection)](cm_original_bertgnn.png)
![Held-out BERT-GNN (validation protocol)](cm_corrected_bertgnn_heldout.png)

### Training vs validation loss, ROC and Precision-Recall (held-out model)

![Training vs validation loss](corrected_loss_curve.png)
![ROC (held-out test)](corrected_roc.png)
![Precision-Recall (held-out test)](corrected_pr.png)

Reproduce the evaluation and figures with `python corrected_bertgnn_retrain.py` (confusion-matrix and comparison figures: `python make_result_figures.py`).
