#!/usr/bin/env python3
"""
BERT-GNN training with a held-out validation set (leakage-free protocol).

Why
---
The original BERT_GNN_pipeline_FINAL.ipynb selected hyperparameters (Optuna,
50 trials, cell 23) and applied early stopping (cell 27) using the TEST set,
and reported the final metrics on that same test set. There is no validation
split, so the reported 99.48% is optimistic (the test set influenced model
selection).

This script fixes the protocol WITHOUT changing the architecture:
  * identical outer split: train_test_split(df, test_size=0.30, random_state=42)
    -> the SAME pristine 9,276-row TEST set, never used for selection;
  * a VALIDATION set is carved from the training portion (stratified);
  * a validation-based hyperparameter search (Optuna over the original space);
  * early stopping on VALIDATION loss;
  * a SINGLE final evaluation on the untouched TEST set.
The graph construction, node features (BERT last_hidden_state, first-N tokens),
undirected chain edges, GCNConv x2 + mean-pool + 2 FC, and class-weighted loss
are all replicated from the notebook.

Resumable
---------
Runs in stages with a per-invocation time budget (MAX_SECONDS) and on-disk
checkpoints, so an external kill just resumes. Re-invoke until it prints DONE.
Runtime on CPU: roughly 1-1.5 hours total across several invocations.
"""
import os, time, json, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "SQL_Injection_Dataset.csv")
WORK = os.path.join(HERE, ".corrected_work")      # caches/checkpoints (gitignored)
os.makedirs(WORK, exist_ok=True)
OUT_MD = os.path.join(HERE, "results", "corrected_bertgnn_heldout.md")
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "500"))
SEED = 42
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 4)
_T0 = time.time()

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def budget_hit():
    if time.time() - _T0 > MAX_SECONDS:
        log(f"BUDGET HIT ({MAX_SECONDS:.0f}s); RESUME NEEDED"); raise SystemExit(3)

# ---------------------------------------------------------------- splits
df = pd.read_csv(DATA)
df["Query"] = df["Query"].astype(str).apply(lambda x: x.lower().strip())
train_df, test_df = train_test_split(df, test_size=0.30, random_state=SEED)          # SAME as notebook
tr_df, val_df = train_test_split(train_df, test_size=0.15, random_state=SEED,
                                 stratify=train_df["Label"])                          # NEW held-out val
log(f"train={len(tr_df)} val={len(val_df)} test={len(test_df)} (test is the pristine 9276)")

# ---------------------------------------------------------------- Stage A: node features -> graphs (cached)
def graphs_path(tag): return os.path.join(WORK, f"graphs_{tag}.pkl")

def build_graphs(tag, frame):
    """Replicates the notebook: node i (i<num_words) = BERT last_hidden_state[i];
    undirected chain edges; drop empty graphs. Cached to disk."""
    p = graphs_path(tag)
    if os.path.exists(p):
        return
    from transformers import BertTokenizer, BertModel
    tok = BertTokenizer.from_pretrained(MODEL_NAME)
    bert = BertModel.from_pretrained(MODEL_NAME).to(device).eval()
    queries = frame["Query"].tolist(); labels = frame["Label"].tolist()
    # length-bucket for speed (identical per-token output under attention mask)
    nwords = [max(1, len(q.split())) for q in queries]
    order = np.argsort([len(tok(q, truncation=True, max_length=MAX_LEN)["input_ids"])
                        for q in queries], kind="stable")
    graphs = [None] * len(queries)
    B = 32
    with torch.no_grad():
        for k in range(0, len(order), B):
            budget_hit()
            idx = order[k:k+B]
            batch = [queries[i] for i in idx]
            enc = tok(batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
            hs = bert(input_ids=enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device)).last_hidden_state.cpu()
            for b, i in enumerate(idx):
                n = min(nwords[i], hs.shape[1])          # first-N token positions as nodes
                x = hs[b, :n, :].clone().float()
                if n <= 0:
                    continue
                ei = (torch.tensor([[j, j+1] for j in range(n-1)], dtype=torch.long).t().contiguous()
                      if n > 1 else torch.empty((2, 0), dtype=torch.long))
                graphs[i] = Data(x=x, edge_index=ei, y=torch.tensor([labels[i]], dtype=torch.long))
            if (k // B) % 100 == 0:
                log(f"  [{tag}] {min(k+B,len(order))}/{len(order)} graphs")
    graphs = [g for g in graphs if g is not None]        # drop empties
    with open(p, "wb") as f:
        pickle.dump(graphs, f)
    log(f"  [{tag}] cached {len(graphs)} graphs -> {p}")

for tag, fr in [("train", tr_df), ("val", val_df), ("test", test_df)]:
    build_graphs(tag, fr)

def load_graphs(tag):
    with open(graphs_path(tag), "rb") as f: return pickle.load(f)

train_graphs, val_graphs, test_graphs = load_graphs("train"), load_graphs("val"), load_graphs("test")
log(f"graphs loaded: train={len(train_graphs)} val={len(val_graphs)} test={len(test_graphs)}")

cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                          y=np.array([int(g.y.item()) for g in train_graphs]))
class_weights = torch.tensor(cw, dtype=torch.float).to(device)

# ---------------------------------------------------------------- model (replicates cell 25)
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def run_epoch(model, loader, opt=None, want_probs=False):
    crit = nn.CrossEntropyLoss(weight=class_weights)
    train = opt is not None
    model.train() if train else model.eval()
    tot, preds, labs, probs = 0.0, [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for data in loader:
            data = data.to(device)
            if train: opt.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = crit(out, data.y)
            if train:
                loss.backward(); opt.step()
            tot += loss.item()
            preds.append(out.argmax(1).cpu()); labs.append(data.y.cpu())
            if want_probs:
                probs.append(F.softmax(out, dim=1)[:, 1].cpu())
    preds = torch.cat(preds).numpy(); labs = torch.cat(labs).numpy()
    if want_probs:
        return tot/len(loader), preds, labs, torch.cat(probs).numpy()
    return tot/len(loader), preds, labs

def train_with_earlystop(params, max_epochs, patience, tag):
    torch.manual_seed(SEED)
    model = GNNModel(768, params["hidden_dim"], 2, params["dropout_rate"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
    tl = GeoLoader(train_graphs, batch_size=params["batch_size"], shuffle=True)
    vl = GeoLoader(val_graphs, batch_size=params["batch_size"])
    best_vl, best_state, bad = float("inf"), None, 0
    history = []   # (train_loss, val_loss) per epoch
    for ep in range(max_epochs):
        budget_hit()
        tloss, _, _ = run_epoch(model, tl, opt)
        vloss, vpred, vlab = run_epoch(model, vl)
        vf1 = f1_score(vlab, vpred, average="weighted")
        history.append((tloss, vloss))
        if vloss < best_vl - 1e-4:
            best_vl, best_state, bad = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        log(f"    [{tag}] epoch {ep+1} train_loss={tloss:.4f} val_loss={vloss:.4f} val_f1={vf1:.4f} (best_vl={best_vl:.4f} bad={bad})")
        if bad >= patience: break
    model.load_state_dict(best_state)
    return model, best_vl, history

# ---------------------------------------------------------------- Stage B: val-based HP search (Optuna, resumable)
HP_JSON = os.path.join(WORK, "hp_results.json")
best_json = os.path.join(WORK, "best_params.json")
if not os.path.exists(best_json):
    import optuna
    done = json.load(open(HP_JSON)) if os.path.exists(HP_JSON) else []
    N_TRIALS = 8
    def objective(trial):
        params = {"hidden_dim": trial.suggest_int("hidden_dim", 32, 256),
                  "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
                  "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                  "batch_size": trial.suggest_int("batch_size", 32, 128)}
        model, _, _ = train_with_earlystop(params, max_epochs=12, patience=3, tag=f"hp{len(done)}")
        _, vpred, vlab = run_epoch(model, GeoLoader(val_graphs, batch_size=params["batch_size"]))
        vf1 = f1_score(vlab, vpred, average="weighted")
        done.append({"params": params, "val_f1": vf1})
        json.dump(done, open(HP_JSON, "w"), indent=2)
        log(f"  HP trial {len(done)}/{N_TRIALS} val_f1={vf1:.4f} params={params}")
        return vf1
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    if done:  # replay completed trials so Optuna resumes intelligently
        for d in done: study.add_trial(optuna.trial.create_trial(
            params=d["params"],
            distributions={"hidden_dim": optuna.distributions.IntDistribution(32,256),
                           "dropout_rate": optuna.distributions.FloatDistribution(0.1,0.5),
                           "lr": optuna.distributions.FloatDistribution(1e-5,1e-2,log=True),
                           "batch_size": optuna.distributions.IntDistribution(32,128)},
            value=d["val_f1"]))
    remaining = N_TRIALS - len(done)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    best = max(json.load(open(HP_JSON)), key=lambda d: d["val_f1"])
    json.dump(best["params"], open(best_json, "w"), indent=2)
    log(f"best params (by val F1): {best}")

best_params = json.load(open(best_json))

# ---------------------------------------------------------------- Stage C: final train + single test eval
log(f"final training with best params {best_params} (early stop on VALIDATION)")
model, best_vl, history = train_with_earlystop(best_params, max_epochs=50, patience=5, tag="final")
_, tpred, tlab, tprob = run_epoch(model, GeoLoader(test_graphs, batch_size=best_params["batch_size"]),
                                  want_probs=True)

# --- figures: loss curve, ROC, PR (held-out model) ---------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    FIG = os.path.join(HERE, "results")
    os.makedirs(FIG, exist_ok=True)

    tr_l = [h[0] for h in history]; va_l = [h[1] for h in history]
    ep = range(1, len(history) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ep, tr_l, marker="o", ms=3, label="Training loss")
    plt.plot(ep, va_l, marker="s", ms=3, label="Validation loss")
    plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss")
    plt.title("Held-out BERT-GNN: training vs validation loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "corrected_loss_curve.png"), dpi=200, bbox_inches="tight"); plt.close()

    fpr, tpr, _ = roc_curve(tlab, tprob); roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4.2))
    plt.plot(fpr, tpr, color="#3b6ea5", label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Held-out BERT-GNN: ROC (held-out test)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "corrected_roc.png"), dpi=200, bbox_inches="tight"); plt.close()

    prec, rec, _ = precision_recall_curve(tlab, tprob); ap = average_precision_score(tlab, tprob)
    plt.figure(figsize=(5, 4.2))
    plt.plot(rec, prec, color="#3b6ea5", label=f"PR (AP = {ap:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Held-out BERT-GNN: Precision-Recall (held-out test)")
    plt.legend(loc="lower left"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(FIG, "corrected_pr.png"), dpi=200, bbox_inches="tight"); plt.close()
    log("saved corrected_loss_curve.png / corrected_roc.png / corrected_pr.png")
except Exception as e:
    log(f"figure generation skipped: {e}")
res = {
    "protocol": "held-out validation: HP search + early stopping on VAL, single eval on the pristine test set",
    "best_params": best_params,
    "test_rows": int(len(test_graphs)),
    "held_out_test": {
        "accuracy": round(float(accuracy_score(tlab, tpred)), 6),
        "precision": round(float(precision_score(tlab, tpred, average="weighted")), 6),
        "recall": round(float(recall_score(tlab, tpred, average="weighted")), 6),
        "f1": round(float(f1_score(tlab, tpred, average="weighted")), 6),
        "confusion_matrix": confusion_matrix(tlab, tpred).tolist(),
    },
    "leaky_reference": {"accuracy": 0.9948, "note": "original, test used for HP + early stopping"},
}
json.dump(res, open(os.path.join(WORK, "corrected_result.json"), "w"), indent=2)

c = res["held_out_test"]
lines = [
    "# BERT-GNN: held-out validation protocol",
    "",
    "The original pipeline selected hyperparameters (Optuna, 50 trials) and applied "
    "early stopping using the **test** set, then reported on that same test set, so "
    "its 99.48% is optimistic. This run keeps the identical outer split (the same "
    "9,276-row test set) but carves a **validation** set from the training portion, "
    "runs the hyperparameter search and early stopping on **validation**, and "
    "evaluates **once** on the untouched test set. Architecture, graph construction "
    "and class-weighted loss are unchanged.",
    "",
    f"Selected hyperparameters (by validation F1): `{best_params}`.",
    "",
    "| Protocol | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Confusion matrix |",
    "|---|---|---|---|---|---|",
    f"| Original (test used for selection) | 99.48 | 99.48 | 99.48 | 99.48 | [[5815, 15], [33, 3413]] |",
    f"| Held-out (validation) | {c['accuracy']*100:.2f} | {c['precision']*100:.2f} "
    f"| {c['recall']*100:.2f} | {c['f1']*100:.2f} | {c['confusion_matrix']} |",
    "",
    "### Confusion matrices",
    "",
    "![Original BERT-GNN (test used for selection)](cm_original_bertgnn.png)",
    "![Held-out BERT-GNN (validation protocol)](cm_corrected_bertgnn_heldout.png)",
    "",
    "### Training vs validation loss, ROC and Precision-Recall (held-out model)",
    "",
    "![Training vs validation loss](corrected_loss_curve.png)",
    "![ROC (held-out test)](corrected_roc.png)",
    "![Precision-Recall (held-out test)](corrected_pr.png)",
    "",
    "Reproduce the evaluation and figures with `python corrected_bertgnn_retrain.py` "
    "(confusion-matrix and comparison figures: `python make_result_figures.py`).",
    "",
]
os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
log(f"DONE -> {OUT_MD}")
