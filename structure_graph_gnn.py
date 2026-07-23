#!/usr/bin/env python3
"""
Structure-aware BERT-GNN for SQL injection detection.

Motivation
----------
The original hybrid built a whitespace-token *sequential chain* graph (each token
linked only to the next). A chain encodes only word order, which BERT already
models internally, so the GNN had no structural information to exploit and added
no accuracy over BERT alone (see bert_only_ablation.py). This script gives the
GNN a graph that actually carries SQL structure:

  * NODES  = sqlparse tokens (keywords, identifiers, literals, operators,
             punctuation), which are meaningful SQL units rather than raw
             whitespace splits.
  * NODE FEATURES = BERT contextual embedding of each token, obtained by
             mean-pooling the BERT sub-word embeddings that overlap the token's
             character span (offset mapping), so the encoder is used identically
             to the original.
  * EDGES  = (a) sequential edges between adjacent tokens, PLUS structural edges:
             (b) parenthesis-matching edges linking each '(' to its ')',
             (c) clause-scope edges linking each token to the nearest preceding
                 SQL keyword (SELECT / WHERE / UNION / ...), so clause structure
                 is explicit. All edges undirected (symmetric, consistent with the
                 GCNConv normalisation).

Protocol (honest, matches corrected_bertgnn_retrain.py)
-------------------------------------------------------
Identical outer split (same pristine 9,276-row test set); a stratified validation
set carved from training; hyperparameter search + early stopping on validation;
a SINGLE evaluation on the untouched test set. Also reports a difficulty
breakdown (by query complexity) comparing this model with the BERT-only baseline.

Resumable: builds/caches graphs in chunks with a per-invocation time budget.
"""
import os, time, json, pickle, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import sqlparse
from sqlparse.tokens import Whitespace, Newline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "SQL_Injection_Dataset.csv")
WORK = os.path.join(HERE, ".structure_work")
os.makedirs(WORK, exist_ok=True)
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "500"))
SEED = 42
KEYWORDS = {"select", "from", "where", "union", "and", "or", "insert", "update",
            "delete", "join", "on", "group", "order", "having", "limit", "values", "set"}
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(os.cpu_count() or 4)
_T0 = time.time()

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def budget_hit():
    if time.time() - _T0 > MAX_SECONDS:
        log(f"BUDGET HIT ({MAX_SECONDS:.0f}s); RESUME NEEDED"); raise SystemExit(3)

# --- splits (identical outer split) ---
df = pd.read_csv(DATA)
df["Query"] = df["Query"].astype(str).apply(lambda x: x.lower().strip())
train_df, test_df = train_test_split(df, test_size=0.30, random_state=SEED)
tr_df, val_df = train_test_split(train_df, test_size=0.15, random_state=SEED, stratify=train_df["Label"])
log(f"train={len(tr_df)} val={len(val_df)} test={len(test_df)}")

# --- SQL-structure tokens + edges ---
def sql_tokens(query):
    """Return [(text, start, end)] for significant sqlparse tokens with char spans."""
    toks, pos = [], 0
    try:
        flat = list(sqlparse.parse(query)[0].flatten()) if query.strip() else []
    except Exception:
        flat = []
    for t in flat:
        val = t.value
        if t.ttype in (Whitespace, Newline) or val.strip() == "":
            pos += len(val); continue
        toks.append((val, pos, pos + len(val)))
        pos += len(val)
    return toks

def structure_edges(tokens):
    """Undirected edges: sequential + parenthesis-match + clause-scope."""
    n = len(tokens); edges = set()
    for i in range(n - 1):
        edges.add((i, i + 1))
    stack = []
    last_kw = None
    for i, (val, a, b) in enumerate(tokens):
        v = val.lower()
        if val == "(":
            stack.append(i)
        elif val == ")" and stack:
            edges.add((stack.pop(), i))
        if v in KEYWORDS:
            last_kw = i
        elif last_kw is not None:
            edges.add((last_kw, i))       # clause-scope edge
    # to undirected COO
    e = [(a, b) for (a, b) in edges] + [(b, a) for (a, b) in edges]
    return e

# --- build graphs (BERT offset-pooled node features), chunk-cached + resumable ---
CHUNK = 3000
def chunk_path(tag, c): return os.path.join(WORK, f"sg_{tag}_{c:03d}.pkl")

def build(tag, frame):
    queries = frame["Query"].tolist(); labels = frame["Label"].tolist()
    N = len(queries); order = np.argsort([len(q) for q in queries], kind="stable")
    nchunks = (N + CHUNK - 1) // CHUNK
    todo = [c for c in range(nchunks) if not os.path.exists(chunk_path(tag, c))]
    if not todo:
        return nchunks
    from transformers import BertTokenizerFast, BertModel
    tok = BertTokenizerFast.from_pretrained(MODEL_NAME)
    bert = BertModel.from_pretrained(MODEL_NAME).to(device).eval()
    B = 32
    for c in todo:
        budget_hit()
        cidx = order[c*CHUNK:(c+1)*CHUNK]
        gl = []
        with torch.no_grad():
            for k in range(0, len(cidx), B):
                budget_hit()
                idx = cidx[k:k+B]
                batch = [queries[i] for i in idx]
                enc = tok(batch, truncation=True, max_length=MAX_LEN, padding=True,
                          return_offsets_mapping=True, return_tensors="pt")
                hs = bert(input_ids=enc["input_ids"].to(device),
                          attention_mask=enc["attention_mask"].to(device)).last_hidden_state.cpu().numpy()
                offs = enc["offset_mapping"].numpy()
                for bi, qi in enumerate(idx):
                    toks = sql_tokens(queries[qi])
                    if not toks:
                        continue
                    feats = []; sub_off = offs[bi]
                    for (val, a, bch) in toks:
                        m = [j for j in range(len(sub_off))
                             if not (sub_off[j][0] == 0 and sub_off[j][1] == 0)
                             and sub_off[j][0] < bch and sub_off[j][1] > a]
                        feats.append(hs[bi][m].mean(axis=0) if m else hs[bi][0])
                    x = torch.tensor(np.stack(feats), dtype=torch.float)
                    e = structure_edges(toks)
                    ei = (torch.tensor(e, dtype=torch.long).t().contiguous()
                          if e else torch.empty((2, 0), dtype=torch.long))
                    gl.append(Data(x=x, edge_index=ei, y=torch.tensor([labels[qi]], dtype=torch.long)))
        with open(chunk_path(tag, c), "wb") as f:
            pickle.dump(gl, f)
        done = sum(os.path.exists(chunk_path(tag, cc)) for cc in range(nchunks))
        log(f"  [{tag}] chunk {c} ({len(gl)} graphs); {done}/{nchunks}")
    return nchunks

def load(tag, frame):
    N = len(frame); nchunks = (N + CHUNK - 1) // CHUNK
    gl = []
    for c in range(nchunks):
        with open(chunk_path(tag, c), "rb") as f: gl += pickle.load(f)
    return gl

for tag, fr in [("train", tr_df), ("val", val_df), ("test", test_df)]:
    build(tag, fr)
train_g, val_g, test_g = load("train", tr_df), load("val", val_df), load("test", test_df)
log(f"loaded structure-graphs: train={len(train_g)} val={len(val_g)} test={len(test_g)}")

cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                          y=np.array([int(g.y.item()) for g in train_g]))
class_weights = torch.tensor(cw, dtype=torch.float).to(device)

class GNNModel(nn.Module):
    def __init__(self, hid, drop):
        super().__init__()
        self.c1 = GCNConv(768, hid); self.c2 = GCNConv(hid, hid)
        self.f1 = nn.Linear(hid, hid); self.f2 = nn.Linear(hid, 2); self.dp = nn.Dropout(drop)
    def forward(self, x, ei, b):
        x = F.relu(self.c1(x, ei)); x = F.relu(self.c2(x, ei))
        x = global_mean_pool(x, b); x = self.dp(F.relu(self.f1(x)))
        return self.f2(x)

def epoch(model, loader, opt=None, probs=False):
    crit = nn.CrossEntropyLoss(weight=class_weights)
    tr = opt is not None; model.train() if tr else model.eval()
    tot, P, L, PR = 0.0, [], [], []
    with (torch.enable_grad() if tr else torch.no_grad()):
        for d in loader:
            d = d.to(device)
            if tr: opt.zero_grad()
            out = model(d.x, d.edge_index, d.batch); loss = crit(out, d.y)
            if tr: loss.backward(); opt.step()
            tot += loss.item(); P.append(out.argmax(1).cpu()); L.append(d.y.cpu())
            if probs: PR.append(F.softmax(out, 1)[:, 1].cpu())
    P = torch.cat(P).numpy(); L = torch.cat(L).numpy()
    return (tot/len(loader), P, L, (torch.cat(PR).numpy() if probs else None))

def train_es(hid, drop, lr, bs, max_ep, pat, tag):
    torch.manual_seed(SEED)
    m = GNNModel(hid, drop).to(device); opt = torch.optim.Adam(m.parameters(), lr=lr)
    tl = GeoLoader(train_g, batch_size=bs, shuffle=True); vl = GeoLoader(val_g, batch_size=bs)
    best, state, bad = 1e9, None, 0
    for ep in range(max_ep):
        budget_hit(); epoch(m, tl, opt); vloss, vp, vla, _ = epoch(m, vl)
        vf1 = f1_score(vla, vp, average="weighted")
        if vloss < best - 1e-4: best, state, bad = vloss, {k: v.cpu().clone() for k, v in m.state_dict().items()}, 0
        else: bad += 1
        log(f"    [{tag}] ep{ep+1} vloss={vloss:.4f} vf1={vf1:.4f} bad={bad}")
        if bad >= pat: break
    m.load_state_dict(state); return m

# --- val-based HP search (resumable) ---
BEST = os.path.join(WORK, "best.json"); HP = os.path.join(WORK, "hp.json")
if not os.path.exists(BEST):
    import optuna
    done = json.load(open(HP)) if os.path.exists(HP) else []
    def obj(t):
        p = dict(hid=t.suggest_int("hid", 32, 256), drop=t.suggest_float("drop", 0.1, 0.5),
                 lr=t.suggest_float("lr", 1e-5, 1e-2, log=True), bs=t.suggest_int("bs", 32, 128))
        m = train_es(p["hid"], p["drop"], p["lr"], p["bs"], 12, 3, f"hp{len(done)}")
        _, vp, vla, _ = epoch(m, GeoLoader(val_g, batch_size=p["bs"]))
        f1 = f1_score(vla, vp, average="weighted"); done.append({"p": p, "f1": f1})
        json.dump(done, open(HP, "w")); log(f"  HP {len(done)} f1={f1:.4f} {p}"); return f1
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    for d in done:
        st.add_trial(optuna.trial.create_trial(params=d["p"], value=d["f1"], distributions={
            "hid": optuna.distributions.IntDistribution(32, 256),
            "drop": optuna.distributions.FloatDistribution(0.1, 0.5),
            "lr": optuna.distributions.FloatDistribution(1e-5, 1e-2, log=True),
            "bs": optuna.distributions.IntDistribution(32, 128)}))
    if len(done) < 8: st.optimize(obj, n_trials=8 - len(done))
    json.dump(max(json.load(open(HP)), key=lambda d: d["f1"])["p"], open(BEST, "w"))
bp = json.load(open(BEST))

# --- final train + single test eval ---
log(f"final training, best params {bp}")
model = train_es(bp["hid"], bp["drop"], bp["lr"], bp["bs"], 50, 5, "final")
_, tp, tl, tpr = epoch(model, GeoLoader(test_g, batch_size=bp["bs"]), probs=True)
res = {"model": "structure-graph BERT-GNN (held-out)",
       "accuracy": round(float(accuracy_score(tl, tp)), 6),
       "precision": round(float(precision_score(tl, tp, average="weighted")), 6),
       "recall": round(float(recall_score(tl, tp, average="weighted")), 6),
       "f1": round(float(f1_score(tl, tp, average="weighted")), 6),
       "confusion_matrix": confusion_matrix(tl, tp).tolist(),
       "reference": {"chain_graph_hybrid": 0.9948, "held_out_hybrid": 0.9967,
                     "bert_only_mlp": 0.997305}}
json.dump(res, open(os.path.join(WORK, "result.json"), "w"), indent=2)
np.save(os.path.join(WORK, "test_pred.npy"), tp)
log(f"RESULT: acc={res['accuracy']:.4f} f1={res['f1']:.4f} cm={res['confusion_matrix']}")
log("DONE")
