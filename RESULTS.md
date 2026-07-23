# Results — figures extracted from the trained pipelines

This page collates the key figures produced by the two pipelines so they can be viewed without running the notebooks. Each image is a faithful export of the corresponding cell output in the original notebook (see `results/` for all PNGs). For the full per-model numerical tables and discussion, see the MSc dissertation §4.

---

## DistilBERT-Stacked Ensemble pipeline

### Dataset exploration

Token-length distribution, query-length distribution by label, frequency of SQL keywords, target-variable distribution. See `results/distilbert_01_*.png` through `results/distilbert_11_*.png`.

![Correlation heatmap](results/distilbert_03_hybrid_of_distil_bert_machine_learning.png)
![Token-length distribution](results/distilbert_06_hybrid_of_distil_bert_machine_learning.png)
![SQL keyword frequency](results/distilbert_09_hybrid_of_distil_bert_machine_learning.png)

### Training and evaluation of base classifiers

Confusion matrices for the conventional ML / ensemble classifiers benchmarked under DistilBERT embeddings.

![Confusion matrices — base classifiers](results/distilbert_12_training_and_evaluating_machine_learning_and.png)

### Stacked-Ensemble meta-learner — headline results

Confusion matrix, ROC curve and Precision–Recall curve for the DistilBERT-Stacked Ensemble (Meta-Learner) — corresponds to dissertation Figures 4.2 / 4.4 / 4.5. Headline figures (accuracy / precision / recall / F1) all 99.81%.

![Stacked-Ensemble confusion matrix](results/distilbert_13_creating_a_distilbert_stacked_ensemble_meta.png)
![Stacked-Ensemble ROC / PR curves](results/distilbert_14_creating_a_distilbert_stacked_ensemble_meta.png)

### Hyperparameter optimisation (Optuna) + adversarial training (FGSM)

Optimisation traces, learning curve (dissertation Figure 4.3), and the post-optimisation evaluation including 99.77% adversarial accuracy reported in dissertation Table 4.2.

![Optuna optimisation trace](results/distilbert_15_hyperparameter_parameter_optimisation_using_optuna_for.png)
![Tuned learning curve](results/distilbert_16_hyperparameter_parameter_optimisation_using_optuna_for.png)
![Tuned ROC curve](results/distilbert_18_hyperparameter_parameter_optimisation_using_optuna_for.png)
![Tuned Precision–Recall](results/distilbert_19_hyperparameter_parameter_optimisation_using_optuna_for.png)

### Sensitivity analysis

Robustness assessment under perturbed inputs (dissertation Figure 4.6 / 4.7).

![Sensitivity analysis](results/distilbert_20_distilbert_stacked_ensemble_sensitivity_analysis.png)

---

## BERT-GNN pipeline

### Dataset exploration

Same dataset, parallel preprocessing. See `results/bertgnn_01_*.png` through `results/bertgnn_11_*.png`.

![Token-length distribution](results/bertgnn_06_loading_dataset_and_exploratory_data_analysis.png)
![SQL keyword frequency](results/bertgnn_09_loading_dataset_and_exploratory_data_analysis.png)

### Model training (with early stopping)

Training and validation loss over epochs (dissertation Figure 4.9), supporting plots from the training run.

![Training / validation loss](results/bertgnn_12_model_training_with_early_stopping.png)
![Training / validation loss (extended)](results/bertgnn_14_model_training_with_early_stopping.png)
![Optimisation traces](results/bertgnn_16_model_training_with_early_stopping.png)
![Optimisation traces (extended)](results/bertgnn_17_model_training_with_early_stopping.png)

### Headline evaluation — confusion matrix, ROC, Precision–Recall

Dissertation Figures 4.8 / 4.10 / 4.11. Headline figures (accuracy / precision / recall / F1) all 99.48%.

![BERT-GNN confusion matrix + ROC](results/bertgnn_18_model_training_with_early_stopping.png)
![BERT-GNN Precision–Recall](results/bertgnn_19_model_training_with_early_stopping.png)

### Sensitivity analysis

Robustness assessment (dissertation Figure 4.14). Sensitivity score reported in dissertation Table 4.3 as 0.0466.

![BERT-GNN sensitivity analysis](results/bertgnn_20_performing_a_sensitivity_analysis_for_bert.png)

### Ablation — BERT-only baseline (isolating the GNN)

A BERT-only baseline (BERT `[CLS]` embedding into a simple Logistic Regression / MLP head, no graph or GNN) trained on the identical 70/30 split and 9,276-row test set. On this near-saturated benchmark the BERT-only baselines match, and marginally exceed, the hybrid on accuracy, so the GNN adds no measurable accuracy here. Full table and reproduction: [`results/ablation_bert_only.md`](results/ablation_bert_only.md) (run `python bert_only_ablation.py`).

### Held-out validation evaluation

The original BERT-GNN selected hyperparameters (Optuna) and applied early stopping using the **test** set, so its 99.48% is optimistic. A held-out run keeps the identical 9,276-row test set pristine, carves a **validation** set from the training portion, runs the hyperparameter search and early stopping on **validation**, and evaluates once on the untouched test set. Result: **99.67%** (statistically indistinguishable from 99.48%), confirming the conclusion is robust to the model-selection protocol. Full table and reproduction: [`results/corrected_bertgnn_heldout.md`](results/corrected_bertgnn_heldout.md) (run `python corrected_bertgnn_retrain.py`).

![Test-set performance comparison](results/comparison_accuracy_f1.png)

Confusion matrices for all four models (`results/cm_*.png`) are generated by `python make_result_figures.py`.

### Comparison with conventional ML and ensemble classifiers

Per-model performance plots from the ML / ensemble benchmark (dissertation Table 4.4 — DistilBERT-Stacked Ensemble + BERT-GNN both outperform all conventional baselines).

![Per-model comparison](results/bertgnn_21_training_machine_learning_and_ensemble_classifiers.png)
![Per-model comparison (extended)](results/bertgnn_22_training_machine_learning_and_ensemble_classifiers.png)
![Per-model comparison (extended)](results/bertgnn_23_training_machine_learning_and_ensemble_classifiers.png)
![ML / ensemble classifier ROC](results/bertgnn_24_training_machine_learning_and_ensemble_classifiers.png)

---

*All figures are exact PNG exports of the cell outputs in the originally-submitted notebooks (commit history preserves the unstripped versions). Re-running either notebook end-to-end will regenerate these figures from the labelled `SQL_Injection_Dataset.csv`.*
