#!/usr/bin/env python3
"""
Full metric suite for every model, from the committed confusion matrices.

Accuracy alone is misleading on this imbalanced security task (about 63% benign,
37% malicious). What matters operationally is the malicious-class recall (the
attack detection rate) and the false-positive rate (the false-alarm rate). This
script reports precision, recall, F1 (per class + macro), false-positive and
false-negative rates, balanced accuracy and MCC, on the clean test set and under
URL-encoding evasion.

The confusion matrices and the URL-encode detection rates are the values produced
by bert_only_ablation.py, corrected_bertgnn_retrain.py, structure_graph_gnn.py
and obfuscation_robustness.py. Run: python metrics_summary.py -> results/full_metrics.md
"""
import os, math

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "full_metrics.md")

# cm rows = true [benign, malicious], cols = pred; TN=cm00 FP=cm01 FN=cm10 TP=cm11
CLEAN = {
    "Original BERT-GNN (leaky)":     [[5815, 15], [33, 3413]],
    "Held-out BERT-GNN": [[5814, 16], [15, 3431]],
    "Structure-aware BERT-GNN":      [[5813, 17], [16, 3430]],
    "BERT [CLS] + LogReg":           [[5819, 11], [21, 3425]],
    "BERT [CLS] + MLP":              [[5823,  7], [18, 3428]],
}
# URL-encode deployment scenario: benign clean + malicious URL-encoded
BENIGN_CLEAN = {"Structure-aware BERT-GNN": (5813, 17), "BERT [CLS] + MLP": (5823, 7)}
URLENC_RECALL = {"Structure-aware BERT-GNN": 0.9817, "BERT [CLS] + MLP": 0.9034}
N_MAL = 3446

def m(cm):
    (TN, FP), (FN, TP) = cm
    acc = (TP + TN) / (TP + TN + FP + FN)
    mp = TP / (TP + FP); mr = TP / (TP + FN); mf1 = 2 * mp * mr / (mp + mr)
    bp = TN / (TN + FN); br = TN / (TN + FP); bf1 = 2 * bp * br / (bp + br)
    fpr = FP / (FP + TN); fnr = FN / (FN + TP)
    mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return dict(acc=acc, mp=mp, mr=mr, mf1=mf1, fpr=fpr, fnr=fnr,
                macro_f1=(mf1 + bf1) / 2, bal_acc=(mr + br) / 2, mcc=mcc, FP=FP, FN=FN)

def row(name, cm):
    x = m(cm)
    return (f"| {name} | {x['acc']*100:.2f} | {x['mp']*100:.2f} | {x['mr']*100:.2f} "
            f"| {x['mf1']*100:.2f} | {x['fpr']*100:.3f} | {x['fnr']*100:.2f} "
            f"| {x['bal_acc']*100:.2f} | {x['mcc']:.4f} | {x['FP']}/{x['FN']} |")

lines = [
    "# Full metric suite (beyond accuracy)",
    "",
    "This is an imbalanced security task (about 63% benign, 37% malicious), so accuracy "
    "alone is misleading. The operationally critical metrics are the malicious-class recall "
    "(attack detection rate) and the false-positive rate (false-alarm rate). All values are "
    "on the identical 9,276-query test set (5,830 benign, 3,446 malicious).",
    "",
    "## Clean test set",
    "",
    "| Model | Accuracy | Mal. precision | Mal. recall (detection) | Mal. F1 | FPR (%) | FNR (%) | Balanced acc | MCC | FP/FN |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for name, cm in CLEAN.items():
    lines.append(row(name, cm))
lines += [
    "",
    "Two points accuracy hides: (i) the leaky original has the lowest attack detection (99.04% "
    "recall, 33 missed), and the held-out protocol recovers this to 99.56% (15 missed); (ii) the "
    "BERT-only MLP has the lowest false-alarm rate (FPR 0.12%). On clean data the differences are "
    "within run-to-run noise.",
    "",
    "## Under URL-encoding evasion (deployment scenario: benign clean, malicious obfuscated)",
    "",
    "| Model | Accuracy | Precision | Recall (detection) | F1 | FPR (%) | MCC | Attacks missed (FN) |",
    "|---|---|---|---|---|---|---|---|",
]
for name in ("Structure-aware BERT-GNN", "BERT [CLS] + MLP"):
    TN, FP = BENIGN_CLEAN[name]; TP = round(URLENC_RECALL[name] * N_MAL); FN = N_MAL - TP
    acc = (TP + TN) / (TP + TN + FP + FN); P = TP / (TP + FP); R = TP / (TP + FN); F1 = 2 * P * R / (P + R)
    fpr = FP / (FP + TN); mcc = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    lines.append(f"| {name} | {acc*100:.2f} | {P*100:.2f} | {R*100:.2f} | {F1*100:.2f} | {fpr*100:.3f} | {mcc:.4f} | {FN} |")
lines += [
    "",
    "Precision stays high for both (benign traffic is not obfuscated), so the degradation is "
    "entirely in detection: under URL encoding the BERT-only model's F1 falls to 94.82% and it "
    "misses 333 malicious queries, against the structure-aware model's 98.83% F1 and 63 missed, "
    "a five-fold difference in attacks let through. This is the security-relevant failure mode.",
    "",
    "Reproduce with `python metrics_summary.py` (source confusion matrices and recalls from "
    "`bert_only_ablation.py`, `corrected_bertgnn_retrain.py`, `structure_graph_gnn.py`, "
    "`obfuscation_robustness.py`).",
    "",
]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
