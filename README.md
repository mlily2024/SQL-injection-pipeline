# SQL Injection Detection — Hybrid ML Pipelines

**Source code for the MSc dissertation:** *Enhancing Web Application Firewall with Machine Learning for SQL Injection Detection*

| | |
|---|---|
| **Author** | Lilliane Linnet Musoke |
| **Institution** | University of Reading, Department of Computer Science |
| **Programme** | MSc Data Science and Advanced Computing |
| **Supervisor** | Professor Atta Badii |
| **Submitted** | 17 September 2024 |

---

## What this repository contains

Two novel Hybrid Machine-Learning pipelines, each implemented as a self-contained Jupyter notebook, for detecting SQL Injection (SQLi) attacks in web application traffic:

1. **`DistilBERT_Stacked_Ensemble_pipeline.ipynb`** — a DistilBERT-Stacked Ensemble (Meta-Learner) pipeline. Uses DistilBERT contextual embeddings as input to a stack of conventional ML and ensemble classifiers (Logistic Regression, XGBoost, SVM), combined under a neural-network meta-learner. Adversarial training is performed with the Fast Gradient Sign Method (FGSM); hyperparameters are tuned with Optuna.
2. **`BERT_GNN_pipeline_FINAL.ipynb`** — a BERT–Graph-Neural-Network hybrid pipeline. BERT generates contextual embeddings of SQL queries; a GNN models the graph-structured query representation to capture structural patterns. Hyperparameters tuned with Optuna.

Plus the dataset used to train and evaluate both pipelines:

3. **`SQL_Injection_Dataset.csv`** — labelled SQL queries (benign vs malicious).

## Headline results (from the dissertation)

| Pipeline | Accuracy | Adversarial accuracy (FGSM) | Notes |
|---|---|---|---|
| **DistilBERT-Stacked Ensemble** | **99.81%** | **99.77%** | Selected recommended approach; fast execution time |
| **BERT-GNN** | **99.48%** (99.67% held-out) | — | Superior structural understanding; longer execution time (23.13 s); held-out-validation retrain reaches 99.67%, see [`RESULTS.md`](RESULTS.md) |
| Best conventional baseline (Random Forest) | 94.47% | — | Benchmarked in the same study |

All four performance metrics (accuracy, precision, recall, F1-score) hit the same headline figure for both Hybrid pipelines. Full per-model tables, confusion matrices, ROC curves, learning curves and sensitivity analyses are in the notebooks and in the dissertation.

The DistilBERT-Stacked Ensemble adversarial accuracy of **99.77%** exceeds the comparable adversarial-testing result reported by Guan et al. (2023, *Future Internet* 15(4):133, DOI 10.3390/fi15040133) by **2.38%** — see the dissertation §4.4 for the full comparison.

## How to view the work

- **Notebooks** — the two `.ipynb` files in the repo root contain the full pipeline code (data loading, preprocessing, embedding, training, evaluation, adversarial robustness, sensitivity analysis). Outputs are stripped so the notebooks render fast and small on GitHub; running either notebook top-to-bottom regenerates every figure.
- **Results gallery** — [`RESULTS.md`](RESULTS.md) shows all the figures (confusion matrices, ROC curves, learning curves, sensitivity-analysis plots, per-model comparison) as a viewable gallery without needing to run anything.
- **All figures** — individual PNG exports of every result figure are in the `results/` folder, named by pipeline and section.

## How to run the notebooks locally

Tested under Python 3.10+ with a Jupyter environment. To install all dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

Then open either notebook in JupyterLab or VS Code and run cells top-to-bottom. Each pipeline is end-to-end self-contained: data loading and preprocessing → embedding extraction → model training → evaluation → adversarial-robustness check → sensitivity analysis. A GPU is recommended for the BERT-GNN training step but not required for inference or for the DistilBERT-Stacked Ensemble.

**Resource requirement.** The DistilBERT (and BERT) embedding-extraction step holds the language model and its full-dataset embeddings in memory simultaneously. End-to-end execution needs approximately **6–8 GB of free RAM** for the DistilBERT-Stacked Ensemble pipeline and **8–12 GB** for BERT-GNN. The notebooks were originally developed on Google Colab (which provides 12–16 GB and a free GPU). If running locally on a machine with 8 GB total RAM, close other applications first or run in Colab via the badge links at the top of each notebook.

## Reproducing the revision results

The scripts in the repository root regenerate the extended analysis reported in the revised BERT-GNN paper (ablation, corrected held-out evaluation, structure-aware graph, obfuscation robustness, the seven-test robustness suite, full metrics, and cross-model execution time). They read the committed `SQL_Injection_Dataset.csv`, write their outputs to `results/`, and cache intermediate artifacts (BERT embeddings, graphs, trained models) under `.structure_work/` and `.corrected_work/`. Those cache directories are intentionally not tracked; each script rebuilds them from the dataset and is resumable, so a run interrupted on a CPU-only machine can simply be started again and will continue.

The scripts form a producer→consumer chain through those caches, so run them in this order (each step only needs the dataset plus the caches written by earlier steps):

```bash
pip install -r requirements.txt

python structure_graph_gnn.py        # structure graphs + best.json (Optuna) + structure-GNN result
python corrected_bertgnn_retrain.py  # held-out-validation retrain (chain graph) -> corrected result
python extract_train_cls.py          # train-split BERT [CLS] embeddings (train_cls.npy)
python bert_only_ablation.py         # BERT-only heads on the identical test set
python obfuscation_robustness.py     # trains + caches gnn_model.pt / mlp_model.pkl; evasion recall
python robustness_1_sensitivity.py   # seven-test robustness suite, one script each
python robustness_2_adversarial.py
python robustness_3_obfuscation_extended.py
python robustness_4_adaptive.py
python robustness_5_crossdataset.py
python robustness_6_significance.py
python robustness_7_calibration.py
python metrics_summary.py            # full per-class metric tables
python complexity_breakdown.py       # accuracy by query complexity (Table 4)
python make_result_figures.py        # confusion matrices + comparison charts
python model_execution_times.py      # training + inference time across all models
```

All scripts resolve their paths relative to the repository, use a fixed seed (`random_state=42`), and require no arguments. Running on CPU is fully supported (a GPU only speeds up the embedding and GNN-training steps).

**Expected precision of reproduction.** Numbers reproduce to roughly ±0.1–0.2 percentage points rather than bit-for-bit. Minor variation comes from CPU versus GPU execution, PyTorch non-determinism, and the early-stopping epoch differing by one or two between runs. The reported conclusions (the graph adds no clean-accuracy gain but improves robustness to URL-encoding evasion, and the associated statistical tests) are stable well within that margin.

## Acknowledgements

Supervisor: Professor Atta Badii (University of Reading). Acknowledgement also to PhD candidate Ahmed Ashlam for advice during the project's execution. Both acknowledged in the dissertation.

## Citation

If you reference this work:

> Musoke, L. L. (2024). *Enhancing Web Application Firewall with Machine Learning for SQL Injection Detection.* MSc dissertation, University of Reading.

## Licence

Released under the [MIT Licence](LICENSE).

## Companion repository

This GitHub repository mirrors the original submission at the University of Reading's institutional Gitlab: `https://csgitlab.reading.ac.uk/qz820024/sql-injection-pipeline-project`.

---

*Original work declaration (from the dissertation):* "I, Lilliane Linnet Musoke, from the University of Reading's Department of Computer Science, attest that this is my original work, except for those instances where I have explicitly acknowledged the contributions of other authors."
