#!/usr/bin/env python3
"""Error-agreement + ensemble check. The CKA result showed BERT-only ([CLS]) and
the structure-GNN (mean-token) live in near-orthogonal representation spaces, so
they may err on different queries. Here we quantify their error overlap on the
9,276-query test set and test whether a simple probability-average ensemble beats
either model alone. Complementary errors + an ensemble gain would be a genuine
(if modest) positive contribution."""
import numpy as np, robustness_common as rc

gnn, mlp, mlp_t, bp = rc.load_models()
graphs, cls, y = rc.test_features()
y = np.asarray(y)
gpred, gprob = rc.gnn_predict(gnn, graphs, want_prob=True)   # gprob = P(malicious)
gpred = np.asarray(gpred); gprob = np.asarray(gprob)
bprob = mlp.predict_proba(cls)[:, 1]; bpred = (bprob >= 0.5).astype(int)

n = len(y)
gc = gpred == y; bc = bpred == y
both_c = int((gc & bc).sum()); both_w = int((~gc & ~bc).sum())
gnn_only = int((gc & ~bc).sum()); bert_only = int((~gc & bc).sum())

def acc(p): return float((p == y).mean() * 100)
acc_g, acc_b = acc(gpred), acc(bpred)
# ensembles
ens_avg = ((gprob + bprob) / 2 >= 0.5).astype(int)
ens_or  = ((gpred == 1) | (bpred == 1)).astype(int)   # flag if either flags (recall-max)
ens_and = ((gpred == 1) & (bpred == 1)).astype(int)   # flag only if both (precision-max)
acc_ens, acc_or, acc_and = acc(ens_avg), acc(ens_or), acc(ens_and)

# malicious recall (detection) for each, and false-positive rate
mal = y == 1; ben = y == 0
def rec(p): return float((p[mal] == 1).mean() * 100)
def fpr(p): return float((p[ben] == 1).mean() * 100)

lines = ["# Error agreement + ensemble (BERT-only [CLS] vs structure-GNN mean-token)", "",
 f"Test set n={n}. Do the two models err on the *same* queries? If errors are complementary, "
 "an ensemble can beat either alone.", "",
 "## Error overlap",
 "| | count |", "|---|---|",
 f"| Both correct | {both_c} |",
 f"| Both wrong | {both_w} |",
 f"| GNN correct, BERT-only wrong | {gnn_only} |",
 f"| BERT-only correct, GNN wrong | {bert_only} |", "",
 f"Of the {both_w+gnn_only+bert_only} queries at least one model gets wrong, only {both_w} are missed "
 f"by BOTH; {gnn_only+bert_only} are 'saved' by one model. Errors are "
 + ("largely complementary." if both_w < (gnn_only+bert_only) else "largely shared."), "",
 "## Accuracy / detection",
 "| Model | Accuracy (%) | Mal. recall (%) | FPR (%) |", "|---|---|---|---|",
 f"| BERT-only ([CLS]+MLP) | {acc_b:.2f} | {rec(bpred):.2f} | {fpr(bpred):.3f} |",
 f"| Structure-GNN | {acc_g:.2f} | {rec(gpred):.2f} | {fpr(gpred):.3f} |",
 f"| Ensemble (prob average) | {acc_ens:.2f} | {rec(ens_avg):.2f} | {fpr(ens_avg):.3f} |",
 f"| Ensemble (OR / flag-if-either) | {acc_or:.2f} | {rec(ens_or):.2f} | {fpr(ens_or):.3f} |",
 f"| Ensemble (AND / flag-if-both) | {acc_and:.2f} | {rec(ens_and):.2f} | {fpr(ens_and):.3f} |", "",
 f"Best single = {max(acc_b,acc_g):.2f}%; best ensemble (avg) = {acc_ens:.2f}%. "
 + ("Ensemble beats the best single model." if acc_ens > max(acc_b,acc_g)+1e-9 else
    "Ensemble does NOT beat the best single model.")
 + " NOTE: the GNN is seed-noisy; treat a single-run ensemble gain as provisional pending multi-seed.", ""]
open("results/error_agreement.md","w",encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
