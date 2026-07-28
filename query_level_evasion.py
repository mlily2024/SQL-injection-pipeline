#!/usr/bin/env python3
"""Query-level evasion robustness (Atta id73): instead of perturbing the internal
meta-feature vector (the FGSM setup), apply realistic WAF-bypass obfuscations to
the MALICIOUS test queries, re-embed with DistilBERT, and measure detection recall
for BOTH the stacked ensemble and the SVM baseline. Reuses the cached pipeline
(.distil_work/: embeddings, base models, meta-features). Resumable per transform."""
import os, time, json, re, numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
HERE=os.path.dirname(os.path.abspath(__file__)); W=os.path.join(HERE,".distil_work")
MAX_SECONDS=float(os.environ.get("MAX_SECONDS","500")); T0=time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
torch.set_num_threads(os.cpu_count() or 4)

df=pd.read_csv(os.path.join(HERE,"SQL_Injection_Dataset.csv")); df["Query"]=df["Query"].astype(str)
y=LabelEncoder().fit_transform(df["Label"])
_,te_idx=train_test_split(np.arange(len(df)),test_size=0.2,random_state=42)
MAL=1 if np.bincount(y)[1]<np.bincount(y)[0] else 0   # malicious = minority class
mal_mask=(y[te_idx]==MAL); mal_q=df["Query"].values[te_idx][mal_mask]
log(f"malicious test queries: {len(mal_q)} (label {MAL})")

# ---- obfuscation transforms (surface-form WAF-bypass rewrites) ----
def url_encode(q): return q.replace(" ","%20").replace("'","%27").replace("=","%3d").replace("\"","%22")
def inline_comment(q): return q.replace(" ","/**/")
def case_random(q):
    rng=np.random.RandomState(abs(hash(q))%(2**31))
    return "".join(c.upper() if (c.isalpha() and rng.rand()<0.5) else c.lower() for c in q)
def tab_ws(q): return re.sub(r" +","\t",q)
def combined(q): return case_random(url_encode(q))
TRANSFORMS={"clean":lambda q:q,"url_encode":url_encode,"inline_comment":inline_comment,
            "case_random":case_random,"tab_whitespace":tab_ws,"combined_url_case":combined}

# ---- DistilBERT embedder (cached per transform) ----
def embed(tag, queries):
    path=os.path.join(W,f"obf_{tag}.npy")
    if os.path.exists(path): return np.load(path)
    from transformers import DistilBertTokenizerFast, DistilBertModel
    tok=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bert=DistilBertModel.from_pretrained("distilbert-base-uncased").eval()
    out=np.zeros((len(queries),768),np.float32); B=16
    with torch.no_grad():
        for k in range(0,len(queries),B):
            if time.time()-T0>MAX_SECONDS:
                log(f"  [{tag}] budget hit @ {k}; resume (no partial save)"); raise SystemExit(3)
            enc=tok(list(queries[k:k+B]),truncation=True,padding=True,max_length=128,return_tensors="pt")
            out[k:k+B]=bert(**enc).last_hidden_state[:,0,:].numpy()
            if (k//B)%100==0: log(f"  [{tag}] {k}/{len(queries)}")
    np.save(path,out); log(f"  [{tag}] embedded+cached"); return out

# ---- models: base (cached) + meta-learner (retrain, same cfg/seed as McNemar) ----
base={n:joblib.load(os.path.join(W,f"model_{n}.joblib")) for n in ("lr","xgb","svm")}
mtr=np.load(os.path.join(W,"meta_train.npy"))
_,tr_idx0=None,None
tr_idx,_=train_test_split(np.arange(len(df)),test_size=0.2,random_state=42); ytr=y[tr_idx]
class Meta(nn.Module):
    def __init__(s,d,h,dr): super().__init__(); s.f1=nn.Linear(d,h); s.dp=nn.Dropout(dr); s.f2=nn.Linear(h,2)
    def forward(s,x): return s.f2(s.dp(torch.relu(s.f1(x))))
def train_meta():
    p=os.path.join(W,"meta_learner.pt")
    m=Meta(mtr.shape[1],170,0.421)
    if os.path.exists(p): m.load_state_dict(torch.load(p)); m.eval(); return m
    torch.manual_seed(42); opt=torch.optim.Adam(m.parameters(),lr=8.3e-3); crit=nn.CrossEntropyLoss()
    Xt=torch.tensor(mtr,dtype=torch.float32,requires_grad=True); Yt=torch.tensor(ytr,dtype=torch.long); eps=0.10
    for ep in range(100):
        m.train()
        if Xt.grad is not None: Xt.grad.zero_()
        opt.zero_grad(); crit(m(Xt),Yt).backward()            # clean loss -> grads to params + input
        Xadv=(Xt+eps*Xt.grad.sign()).detach()                 # FGSM on the meta-feature input
        opt.step()                                            # step on clean loss
        opt.zero_grad(); crit(m(Xadv),Yt).backward(); opt.step()  # step on adversarial loss
    torch.save(m.state_dict(),p); m.eval(); return m
meta=train_meta()

def preds(emb):
    metaf=np.column_stack([base[n].predict_proba(emb) for n in ("lr","xgb","svm")])
    with torch.no_grad(): ens=meta(torch.tensor(metaf,dtype=torch.float32)).argmax(1).numpy()
    svm=base["svm"].predict(emb)
    return ens, svm

# ---- run each transform, measure recall (fraction of malicious still detected) ----
RES=os.path.join(W,"evasion_recall.json"); res=json.load(open(RES)) if os.path.exists(RES) else {}
for name,fn in TRANSFORMS.items():
    if name in res: log(f"{name} cached: {res[name]}"); continue
    emb=embed(name, np.array([fn(q) for q in mal_q]))
    ens,svm=preds(emb)
    res[name]={"ensemble_recall":round(float((ens==MAL).mean())*100,2),
               "svm_recall":round(float((svm==MAL).mean())*100,2)}
    json.dump(res,open(RES,"w"),indent=2); log(f"{name}: {res[name]}")

if len(res)==len(TRANSFORMS):
    lines=["# Query-level evasion robustness (DistilBERT-Stacked Ensemble vs SVM)","",
      f"Recall (% of {len(mal_q)} malicious test queries still detected) after applying WAF-bypass "
      "obfuscation transforms to the raw query, then re-embedding with DistilBERT. Unlike the meta-feature "
      "FGSM, this measures resistance to realistic query-level evasion.","",
      "| Transform | Ensemble recall (%) | SVM recall (%) |","|---|---|---|"]
    for n in TRANSFORMS:
        lines.append(f"| {n} | {res[n]['ensemble_recall']} | {res[n]['svm_recall']} |")
    open(os.path.join(HERE,"results","evasion_query_level.md"),"w",encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines)); log("DONE")
