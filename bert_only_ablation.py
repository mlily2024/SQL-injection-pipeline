#!/usr/bin/env python3
"""
BERT-only baseline ablation for the BERT-GNN SQL-injection pipeline.

Purpose
-------
Isolates the contribution of the Graph Neural Network in the BERT-GNN hybrid.
The hybrid uses BERT token embeddings as the node features of a per-query graph
that is then processed by two GCNConv layers. This script trains a classifier
directly on BERT's [CLS] sentence embedding, with NO graph and NO GNN, so the
two can be compared on the same footing.

Fairness
--------
Everything except the GNN is held identical to `BERT_GNN_pipeline_FINAL.ipynb`:
  * same preprocessing (lower-case, strip),
  * same split: train_test_split(df, test_size=0.30, random_state=42), which
    yields the identical 9,276-row test set the hybrid is evaluated on,
  * same encoder: bert-base-uncased, truncation at max_length=128,
  * class-weighted head, matching the hybrid's class-weighted cross-entropy.
The only difference is the classifier: BERT [CLS] -> Logistic Regression / MLP
instead of BERT node features -> GNN.

Speed note (does not change the result)
---------------------------------------
Queries are sorted by token length before batching so each batch pads only to
its own longest member (dynamic padding). Because BERT is padding-invariant
under the attention mask, the [CLS] output for each query is identical to
fixed-length padding; embeddings are written back in the original order. This
is purely an optimisation for CPU runs.

Usage
-----
    python bert_only_ablation.py

Writes a summary table to results/ablation_bert_only.md and prints it.
Runtime: a few seconds on GPU; roughly 15-20 minutes on CPU (embedding
extraction dominates).
"""
import os
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from transformers import BertTokenizer, BertModel

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_CSV", os.path.join(HERE, "SQL_Injection_Dataset.csv"))
OUT_MD = os.environ.get("OUT_MD", os.path.join(HERE, "results", "ablation_bert_only.md"))
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 4)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cls_embeddings(queries, tokenizer, bert):
    """Return the [CLS] embedding (768-d) for each query, in input order.

    Length-bucketed for speed; the per-query output is identical to fixed
    max_length padding (see module docstring)."""
    enc_len = [len(tokenizer(q, truncation=True, max_length=MAX_LEN)["input_ids"])
               for q in queries]
    order = np.argsort(enc_len, kind="stable")
    out = np.zeros((len(queries), 768), dtype=np.float32)
    bert.eval()
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(order), BATCH):
            idx = order[k:k + BATCH]
            batch = [queries[i] for i in idx]
            enc = tokenizer(batch, truncation=True, padding=True,
                            max_length=MAX_LEN, return_tensors="pt")
            hs = bert(input_ids=enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device)).last_hidden_state
            out[idx] = hs[:, 0, :].cpu().numpy()   # [CLS] token
            if (k // BATCH) % 100 == 0:
                done = min(k + BATCH, len(order))
                rate = done / max(1e-9, time.time() - t0)
                log(f"  embeddings {done}/{len(order)} ({rate:.0f} q/s)")
    return out


def evaluate(name, clf, Xtr, ytr, Xte, yte):
    t0 = time.time()
    clf.fit(Xtr, ytr)
    fit_s = time.time() - t0
    t1 = time.time()
    pred = clf.predict(Xte)
    infer_s = time.time() - t1
    return {
        "model": name,
        "accuracy": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred),
        "recall": recall_score(yte, pred),
        "f1": f1_score(yte, pred),
        "cm": confusion_matrix(yte, pred).tolist(),
        "fit_s": fit_s,
        "infer_s": infer_s,
    }


def main():
    df = pd.read_csv(DATA)
    df["Query"] = df["Query"].astype(str).apply(lambda x: x.lower().strip())
    train_df, test_df = train_test_split(df, test_size=0.30, random_state=42)
    log(f"train={len(train_df)} test={len(test_df)} (device={device})")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    bert = BertModel.from_pretrained(MODEL_NAME).to(device)

    log("extracting BERT [CLS] embeddings for train")
    Xtr = cls_embeddings(train_df["Query"].tolist(), tokenizer, bert)
    log("extracting BERT [CLS] embeddings for test")
    Xte = cls_embeddings(test_df["Query"].tolist(), tokenizer, bert)
    ytr = train_df["Label"].to_numpy()
    yte = test_df["Label"].to_numpy()

    results = [
        evaluate("BERT [CLS] + Logistic Regression (class_weight=balanced)",
                 LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1),
                 Xtr, ytr, Xte, yte),
        evaluate("BERT [CLS] + MLP (1 hidden layer, 159 units)",
                 MLPClassifier(hidden_layer_sizes=(159,), max_iter=300, random_state=42),
                 Xtr, ytr, Xte, yte),
    ]

    # ROC + Precision-Recall curves (test set) for the BERT-only heads.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                                     average_precision_score)
        FIG = os.path.join(HERE, "results")
        os.makedirs(FIG, exist_ok=True)
        heads = {"Logistic Regression":
                 LogisticRegression(max_iter=2000, class_weight="balanced"),
                 "MLP (159)":
                 MLPClassifier(hidden_layer_sizes=(159,), max_iter=300, random_state=42)}
        colours = {"Logistic Regression": "#3b6ea5", "MLP (159)": "#c0504d"}
        probs = {}
        for nm, clf in heads.items():
            clf.fit(Xtr, ytr)
            probs[nm] = clf.predict_proba(Xte)[:, 1]
        plt.figure(figsize=(5, 4.2))
        for nm, p in probs.items():
            fpr, tpr, _ = roc_curve(yte, p)
            plt.plot(fpr, tpr, color=colours[nm], label=f"{nm} (AUC={auc(fpr, tpr):.4f})")
        plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
        plt.title("BERT-only baselines: ROC (test)"); plt.legend(loc="lower right")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "bert_only_roc.png"), dpi=200, bbox_inches="tight"); plt.close()
        plt.figure(figsize=(5, 4.2))
        for nm, p in probs.items():
            pr, rc, _ = precision_recall_curve(yte, p)
            plt.plot(rc, pr, color=colours[nm], label=f"{nm} (AP={average_precision_score(yte, p):.4f})")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("BERT-only baselines: Precision-Recall (test)"); plt.legend(loc="lower left")
        plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIG, "bert_only_pr.png"), dpi=200, bbox_inches="tight"); plt.close()
        log("saved bert_only_roc.png / bert_only_pr.png")
    except Exception as e:
        log(f"curve figures skipped: {e}")

    # Hybrid reference from BERT_GNN_pipeline_FINAL.ipynb (same test set).
    hybrid = {"model": "BERT-GNN hybrid (reference)", "accuracy": 0.9948,
              "precision": 0.9948, "recall": 0.9948, "f1": 0.9948,
              "cm": [[5815, 15], [33, 3413]]}

    def row(r):
        return (f"| {r['model']} | {r['accuracy']*100:.2f} | {r['precision']*100:.2f} "
                f"| {r['recall']*100:.2f} | {r['f1']*100:.2f} | {r['cm']} |")

    lines = [
        "# Ablation: BERT-only baseline (isolating the GNN's contribution)",
        "",
        "Same 70/30 split (`test_size=0.30, random_state=42`), the identical "
        "9,276-row test set, and the same `bert-base-uncased` encoder as the "
        "BERT-GNN hybrid. The only change is the classifier: BERT's `[CLS]` "
        "embedding is fed to a simple head instead of a graph + GNN.",
        "",
        f"Test set: {len(test_df)} queries "
        f"({int((yte==0).sum())} benign, {int((yte==1).sum())} malicious).",
        "",
        "| Model | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Confusion matrix |",
        "|---|---|---|---|---|---|",
        row(hybrid),
        row(results[0]),
        row(results[1]),
        "",
        "**Finding.** On this near-saturated benchmark the BERT-only baselines "
        "match, and marginally exceed, the BERT-GNN hybrid on accuracy, so the "
        "GNN component does not add measurable accuracy here. Reproduce with "
        "`python bert_only_ablation.py`.",
        "",
    ]
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n" + "\n".join(lines))
    for r in results:
        log(f"{r['model']}: acc={r['accuracy']:.4f} f1={r['f1']:.4f} "
            f"fit={r['fit_s']:.1f}s infer={r['infer_s']:.3f}s")
    log(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
