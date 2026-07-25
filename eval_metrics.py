#!/usr/bin/env python3
"""Imbalance-robust evaluation used across the battery. Because the SQLi task is
class-imbalanced (~64% benign / 36% malicious), raw accuracy is misleading, so
every model is judged on: malicious recall (detection rate), false-positive rate,
malicious precision/F1, balanced accuracy, and MCC (Matthews correlation, robust
to imbalance). Import full_metrics() wherever predictions are scored."""
import numpy as np
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, balanced_accuracy_score,
                             f1_score, precision_score, recall_score, accuracy_score)

def full_metrics(y_true, y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "accuracy":     round(accuracy_score(y_true,y_pred)*100, 2),
        "mal_recall":   round(recall_score(y_true,y_pred,pos_label=1,zero_division=0)*100, 2),   # detection rate
        "mal_precision":round(precision_score(y_true,y_pred,pos_label=1,zero_division=0)*100, 2),
        "mal_f1":       round(f1_score(y_true,y_pred,pos_label=1,zero_division=0)*100, 2),
        "fpr":          round(fp/(fp+tn)*100, 3) if (fp+tn) else 0.0,                            # false-alarm rate
        "balanced_acc": round(balanced_accuracy_score(y_true,y_pred)*100, 2),
        "mcc":          round(float(matthews_corrcoef(y_true,y_pred)), 4),
        "fp": int(fp), "fn": int(fn),
    }

COLS=["accuracy","mal_recall","mal_precision","mal_f1","fpr","balanced_acc","mcc","fp","fn"]
HEADER=("| Model | Accuracy | Mal. recall | Mal. prec | Mal. F1 | FPR | Bal. acc | MCC | FP | FN |\n"
        "|---|---|---|---|---|---|---|---|---|---|")

def row(name, m): return "| "+name+" | "+" | ".join(str(m[c]) for c in COLS)+" |"

def table(named_metrics):
    """named_metrics: list of (name, metrics_dict) -> markdown table string."""
    return "\n".join([HEADER]+[row(n,m) for n,m in named_metrics])

if __name__=="__main__":
    import numpy as np
    y=np.array([0]*90+[1]*10); p=np.array([0]*88+[1]*2+[0]*3+[1]*7)
    print(table([("demo", full_metrics(y,p))]))
