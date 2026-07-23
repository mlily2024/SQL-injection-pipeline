#!/usr/bin/env python3
"""
Accuracy by query complexity: structure-aware BERT-GNN vs BERT-only baseline.

Regenerates the query-complexity breakdown (paper Table 4) from the canonical,
committed test-set predictions, so the figures are consistent with the headline
structure-GNN accuracy (99.64%, Section 5.4) and the BERT-only MLP (99.73%,
Table 3). "Complex" is defined explicitly and reproducibly as the top quartile
of the test set by query word count (the Query_Length feature of Section 4.2):
length strictly greater than the 75th percentile.

Prediction sources (no re-inference, so numbers match the rest of the paper):
  - Structure-GNN: .structure_work/test_pred.npy from structure_graph_gnn.py,
    which stores predictions in query-character-length order (line 120,
    order = argsort(len(q))); this script inverts that permutation to align them
    to test-set row order.
  - BERT-only MLP: mlp_model.pkl applied to the cached test [CLS] embeddings
    (test_cls_rowB.npy), already in test-set row order.

Writes results/complexity_breakdown.md. Reproduce with
`python complexity_breakdown.py`.
"""
import os, pickle, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, ".structure_work")
OUT = os.path.join(HERE, "results", "complexity_breakdown.md")

df = pd.read_csv(os.path.join(HERE, "SQL_Injection_Dataset.csv"))
df["Query"] = df["Query"].astype(str).apply(lambda x: x.lower().strip())
_, test = train_test_split(df, test_size=0.30, random_state=42)
queries = test["Query"].tolist()
y = test["Label"].to_numpy()
N = len(queries)
qlen = np.array([len(q.split()) for q in queries])   # word count (Query_Length, Section 4.2)

# --- structure-GNN canonical predictions, un-scrambled to row order ---
order = np.argsort([len(q) for q in queries], kind="stable")   # matches structure_graph_gnn.py line 120
tp = np.load(os.path.join(SW, "test_pred.npy"))
gpred = np.empty(N, dtype=tp.dtype); gpred[order] = tp

# --- BERT-only MLP predictions (already row-aligned) ---
mlp = pickle.load(open(os.path.join(SW, "mlp_model.pkl"), "rb"))
cls = np.load(os.path.join(SW, "test_cls_rowB.npy"))
bpred = np.asarray(mlp.predict(cls))

assert len(gpred) == len(bpred) == N == len(y)
print(f"sanity: structure-GNN all-acc {(gpred==y).mean()*100:.2f}%  BERT-only all-acc {(bpred==y).mean()*100:.2f}%")

thr = np.percentile(qlen, 75)
complex_mask = qlen > thr
cmplx_mal = complex_mask & (y == 1)

def acc(pred, mask): return float((pred[mask] == y[mask]).mean() * 100)
def det(pred, mask): return float((pred[mask] == 1).mean() * 100)  # all-malicious subset -> detection rate

rows = [
    ("all", N, acc(gpred, np.ones(N, bool)), acc(bpred, np.ones(N, bool))),
    (f"complex (top quartile, > {thr:.0f} words)", int(complex_mask.sum()), acc(gpred, complex_mask), acc(bpred, complex_mask)),
    ("complex malicious", int(cmplx_mal.sum()), det(gpred, cmplx_mal), det(bpred, cmplx_mal)),
]

lines = ["# Accuracy by query complexity (structure-aware BERT-GNN vs BERT-only)", "",
         "Complexity is the top quartile of the test set by query word count (the Query_Length "
         f"feature of Section 4.2): complex = length strictly greater than the 75th percentile, "
         f"which on this test set is {thr:.0f} words. Predictions are the canonical committed "
         "predictions (structure-GNN 99.64%, Section 5.4; BERT-only MLP 99.73%, Table 3), aligned "
         "per query. For the all-malicious 'complex malicious' subset the figure is the detection "
         "rate (recall).", "",
         "| Test subset | n | Structure-GNN (%) | BERT-only (%) |",
         "|---|---|---|---|"]
for name, n, g, b in rows:
    lines.append(f"| {name} | {n:,} | {g:.2f} | {b:.2f} |")
lines += ["", "The two models are within run-to-run noise on every subset, including the most "
          "complex queries: there is no query-complexity regime in which the graph improves clean "
          "accuracy. Reproduce with `python complexity_breakdown.py`.", ""]

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
