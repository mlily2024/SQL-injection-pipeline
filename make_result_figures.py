#!/usr/bin/env python3
"""
Generate the confusion-matrix and comparison figures for the BERT-GNN
ablation and the corrected held-out evaluation, in the same style as the
existing repo figures (seaborn 'Blues' heatmap, Benign/Malicious labels).

Confusion matrices are the exact values produced by:
  * bert_only_ablation.py        -> results/ablation_bert_only.md
  * corrected_bertgnn_retrain.py -> results/corrected_bertgnn_heldout.md
and the original hybrid (dissertation, 9,276-row test set).

Run:  python make_result_figures.py   ->  writes PNGs into results/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
LABELS = ["Benign", "Malicious"]

# --- confusion matrices on the identical 9,276-row test set -----------------
MODELS = {
    "Original BERT-GNN (test used for selection)": {
        "cm": [[5815, 15], [33, 3413]], "acc": 99.48, "f1": 99.48,
        "file": "cm_original_bertgnn.png"},
    "BERT-GNN (held-out validation)": {
        "cm": [[5814, 16], [15, 3431]], "acc": 99.67, "f1": 99.67,
        "file": "cm_corrected_bertgnn_heldout.png"},
    "BERT [CLS] + Logistic Regression": {
        "cm": [[5819, 11], [21, 3425]], "acc": 99.66, "f1": 99.54,
        "file": "cm_bert_only_logreg.png"},
    "BERT [CLS] + MLP (159)": {
        "cm": [[5823, 7], [18, 3428]], "acc": 99.73, "f1": 99.64,
        "file": "cm_bert_only_mlp.png"},
}


def plot_cm(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=ax,
                annot_kws={"size": 13})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


def plot_comparison(path):
    names = list(MODELS)
    short = ["Original\nBERT-GNN", "Held-out\nBERT-GNN",
             "BERT+LogReg", "BERT+MLP"]
    acc = [MODELS[n]["acc"] for n in names]
    f1 = [MODELS[n]["f1"] for n in names]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    b1 = ax.bar(x - w/2, acc, w, label="Accuracy", color="#3b6ea5")
    b2 = ax.bar(x + w/2, f1, w, label="F1 (weighted)", color="#a5c8e1")
    ax.set_ylim(98.5, 100.0)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=9)
    ax.set_title("Test-set performance on the identical 9,276-query split", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width()/2, b.get_height()),
                        ha="center", va="bottom", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, HERE))


def main():
    for name, m in MODELS.items():
        plot_cm(m["cm"], f"{name}\nAcc {m['acc']:.2f}%  |  F1 {m['f1']:.2f}%",
                os.path.join(OUT, m["file"]))
    plot_comparison(os.path.join(OUT, "comparison_accuracy_f1.png"))


if __name__ == "__main__":
    main()
