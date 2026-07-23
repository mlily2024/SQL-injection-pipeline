#!/usr/bin/env python3
"""
Execution-time comparison across all models, measured consistently on one machine.

Reports, for the BERT-GNN hybrid, the structure-aware BERT-GNN and the two
BERT-only heads: training time (to early stopping), test-set inference time, and
per-query inference latency. The shared cost of extracting BERT embeddings is
reported separately because it dominates deployment latency and is common to all
models. Times are wall-clock on the same CPU; they replace the single 23.13 s
figure (measured on different hardware) with a like-for-like comparison.
Resumable: caches per-model timings.
"""
import os, json, time, pickle
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import pandas as pd
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as GL

HERE = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(HERE, ".structure_work"); CW = os.path.join(HERE, ".corrected_work")
OUT_MD = os.path.join(HERE, "results", "model_execution_times.md")
RES = os.path.join(SW, "exec_times.json")
SEED = 42; torch.set_num_threads(os.cpu_count() or 4)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

df = pd.read_csv(os.path.join(HERE, "SQL_Injection_Dataset.csv"))
df["Query"] = df["Query"].astype(str).apply(lambda x: x.lower().strip())
train_df, test_df = train_test_split(df, test_size=0.30, random_state=SEED)

# ---- features (lazy: only loaded when timing the BERT-only heads) ----
ytr = train_df["Label"].to_numpy(); yte = test_df["Label"].to_numpy()
def load_cls():
    tp = os.path.join(SW, "train_cls.npy")
    if not os.path.exists(tp): raise SystemExit("train_cls.npy missing; run extract_train_cls.py first")
    return np.load(tp), np.load(os.path.join(SW, "test_cls_rowB.npy"))

def load_split(prefix, tag):
    """Load one split (train|val|test) of graphs, on demand, to keep peak memory low."""
    if prefix == "chain":
        with open(os.path.join(CW, f"graphs_{tag}.pkl"), "rb") as f: return pickle.load(f)
    g = []; c = 0
    while os.path.exists(os.path.join(SW, f"sg_{tag}_{c:03d}.pkl")):
        with open(os.path.join(SW, f"sg_{tag}_{c:03d}.pkl"), "rb") as f: g += pickle.load(f)
        c += 1
    return g

class GNNModel(nn.Module):
    def __init__(s, hid, drop):
        super().__init__(); s.c1=GCNConv(768,hid); s.c2=GCNConv(hid,hid)
        s.f1=nn.Linear(hid,hid); s.f2=nn.Linear(hid,2); s.dp=nn.Dropout(drop)
    def forward(s,x,ei,b):
        x=F.relu(s.c1(x,ei)); x=F.relu(s.c2(x,ei)); x=global_mean_pool(x,b)
        return s.f2(s.dp(F.relu(s.f1(x))))

def time_gnn(name, prefix):
    import gc
    bp = json.load(open(os.path.join(SW, "best.json")))
    tr = load_split(prefix, "train"); va = load_split(prefix, "val")
    cw = compute_class_weight("balanced", classes=np.array([0,1]), y=np.array([int(g.y.item()) for g in tr]))
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float))
    torch.manual_seed(SEED); m = GNNModel(bp["hid"], bp["drop"]); opt = torch.optim.Adam(m.parameters(), lr=bp["lr"])
    tl = GL(tr, batch_size=bp["bs"], shuffle=True); vl = GL(va, batch_size=bp["bs"])
    # --- training time (to early stop) ---
    t0 = time.time(); best, bad = 1e9, 0
    for ep in range(50):
        m.train()
        for d in tl:
            opt.zero_grad(); loss = crit(m(d.x, d.edge_index, d.batch), d.y); loss.backward(); opt.step()
        m.eval(); vlo = 0.0
        with torch.no_grad():
            for d in vl: vlo += crit(m(d.x, d.edge_index, d.batch), d.y).item()
        vlo /= len(vl)
        if vlo < best - 1e-4: best, bad = vlo, 0
        else: bad += 1
        if bad >= 5: break
    train_s = time.time() - t0
    del tr, va, tl, vl; gc.collect()
    # --- inference time (test set) ---
    te = load_split(prefix, "test"); tel = GL(te, batch_size=128); m.eval()
    t1 = time.time()
    with torch.no_grad():
        for d in tel: m(d.x, d.edge_index, d.batch).argmax(1)
    infer_s = time.time() - t1
    n = len(te); del te, tel; gc.collect()
    return {"train_s": round(train_s,2), "infer_s": round(infer_s,3), "n_test": n,
            "infer_ms_per_query": round(infer_s/n*1000, 3)}

def time_head(name, clf):
    Xtr, Xte = load_cls()
    t0 = time.time(); clf.fit(Xtr, ytr); train_s = time.time()-t0
    t1 = time.time(); clf.predict(Xte); infer_s = time.time()-t1
    return {"train_s": round(train_s,2), "infer_s": round(infer_s,4), "n_test": len(Xte),
            "infer_ms_per_query": round(infer_s/len(Xte)*1000, 4)}

res = json.load(open(RES)) if os.path.exists(RES) else {}
if "chain_gnn" not in res:
    log("timing chain-graph BERT-GNN"); res["chain_gnn"] = time_gnn("chain", "chain"); json.dump(res, open(RES,"w"), indent=2)
    log(f"  {res['chain_gnn']}")
if "structure_gnn" not in res:
    log("timing structure-aware BERT-GNN"); res["structure_gnn"] = time_gnn("structure", "structure"); json.dump(res, open(RES,"w"), indent=2)
    log(f"  {res['structure_gnn']}")
if "bert_logreg" not in res:
    log("timing BERT-only LogReg"); res["bert_logreg"] = time_head("lr", LogisticRegression(max_iter=2000, class_weight="balanced")); json.dump(res, open(RES,"w"), indent=2)
if "bert_mlp" not in res:
    log("timing BERT-only MLP"); res["bert_mlp"] = time_head("mlp", MLPClassifier(hidden_layer_sizes=(159,), max_iter=300, random_state=SEED)); json.dump(res, open(RES,"w"), indent=2)

# shared BERT embedding latency (sample)
if "bert_embed" not in res:
    from transformers import BertTokenizerFast, BertModel
    tok = BertTokenizerFast.from_pretrained("bert-base-uncased"); bert = BertModel.from_pretrained("bert-base-uncased").eval()
    sample = test_df["Query"].tolist()[:200]
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(sample), 32):
            enc = tok(sample[i:i+32], truncation=True, max_length=128, padding=True, return_tensors="pt")
            bert(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    emb_s = time.time() - t0
    res["bert_embed"] = {"sample": len(sample), "total_s": round(emb_s,2), "ms_per_query": round(emb_s/len(sample)*1000,2)}
    json.dump(res, open(RES,"w"), indent=2); log(f"  BERT embed: {res['bert_embed']}")

names = {"chain_gnn":"BERT-GNN hybrid (chain graph)", "structure_gnn":"Structure-aware BERT-GNN",
         "bert_logreg":"BERT [CLS] + Logistic Regression", "bert_mlp":"BERT [CLS] + MLP"}
lines = ["# Execution time across models (same machine)", "",
         f"Wall-clock times on a {os.cpu_count()}-core CPU, measured like-for-like (the original "
         "23.13 s was on different hardware). Training time is to early stopping; inference time is "
         "over the 9,276-query test set. The BERT embedding is a shared cost common to every model "
         "and dominates deployment latency.", "",
         "| Model | Training time (s) | Test-set inference (s) | Inference per query (ms) |",
         "|---|---|---|---|"]
for k in ["chain_gnn","structure_gnn","bert_logreg","bert_mlp"]:
    r = res[k]; lines.append(f"| {names[k]} | {r['train_s']} | {r['infer_s']} | {r['infer_ms_per_query']} |")
be = res["bert_embed"]
lines += ["", f"Shared BERT embedding: {be['ms_per_query']} ms per query (dominant deployment cost, "
          "common to all models). The head/graph inference above is on top of this shared embedding "
          "step. Reproduce with `python model_execution_times.py`.", ""]
os.makedirs(os.path.dirname(OUT_MD), exist_ok=True); open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines)); log(f"wrote {OUT_MD}")
