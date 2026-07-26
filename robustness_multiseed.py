#!/usr/bin/env python3
"""Comprehensive MULTI-SEED robustness: is any structure-GNN advantage real or a
seed artefact? Trains the robust config across 5 seeds and measures, per seed,
metrics where the GNN looked better in single runs: feature-noise flip rate (the
14x gap), URL-encode recall, and calibration ECE - versus the deterministic
BERT-only baseline. Reports mean +/- std so we can state which advantages survive.
Resumable per seed. Reuses cached obfuscated graphs from robustness_seeds."""
import os, json, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
import sqlparse
from sqlparse.tokens import Whitespace, Newline
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from transformers import BertTokenizerFast, BertModel

HERE=os.path.dirname(os.path.abspath(__file__)); SW=os.path.join(HERE,".structure_work")
SEEDS=[42,1,7,13,123]; MODEL_NAME="bert-base-uncased"; MAX_LEN=128
KEYWORDS={"select","from","where","union","and","or","insert","update","delete","join","on","group","order","having","limit","values","set"}
torch.set_num_threads(os.cpu_count() or 4)
def log(m): print(f"[{__import__('time').strftime('%H:%M:%S')}] {m}",flush=True)
def url_encode(q): return q.replace(" ","%20").replace("'","%27").replace("=","%3d")

df=pd.read_csv(os.path.join(HERE,"SQL_Injection_Dataset.csv")); df["Query"]=df["Query"].astype(str).apply(lambda x:x.lower().strip())
train_df,test_df=train_test_split(df,test_size=0.30,random_state=42)
tr_df,val_df=train_test_split(train_df,test_size=0.15,random_state=42,stratify=train_df["Label"])
tok=BertTokenizerFast.from_pretrained(MODEL_NAME); bert=BertModel.from_pretrained(MODEL_NAME).eval()

def sql_tokens(q):
    toks,pos=[],0; flat=list(sqlparse.parse(q)[0].flatten()) if q.strip() else []
    for t in flat:
        if t.ttype in (Whitespace,Newline) or t.value.strip()=="": pos+=len(t.value); continue
        toks.append((t.value,pos,pos+len(t.value))); pos+=len(t.value)
    return toks
def structure_edges(tokens):
    n=len(tokens); edges=set(); stack=[]; last_kw=None
    for i in range(n-1): edges.add((i,i+1))
    for i,(val,a,b) in enumerate(tokens):
        v=val.lower()
        if val=="(": stack.append(i)
        elif val==")" and stack: edges.add((stack.pop(),i))
        if v in KEYWORDS: last_kw=i
        elif last_kw is not None: edges.add((last_kw,i))
    return [(a,b) for a,b in edges]+[(b,a) for a,b in edges]
def embed(queries, labels):
    order=np.argsort([len(q) for q in queries],kind="stable"); cls=np.zeros((len(queries),768),np.float32); graphs=[None]*len(queries)
    with torch.no_grad():
        for k in range(0,len(order),32):
            idx=order[k:k+32]; enc=tok([queries[i] for i in idx],truncation=True,max_length=MAX_LEN,padding=True,return_offsets_mapping=True,return_tensors="pt")
            hs=bert(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"]).last_hidden_state.numpy(); offs=enc["offset_mapping"].numpy()
            for bi,qi in enumerate(idx):
                cls[qi]=hs[bi][0]; toks=sql_tokens(queries[qi])
                if not toks: graphs[qi]=Data(x=torch.tensor(hs[bi][:1]),edge_index=torch.empty((2,0),dtype=torch.long),y=torch.tensor([labels[qi]])); continue
                feats=[]; so=offs[bi]
                for (val,a,bch) in toks:
                    m=[j for j in range(len(so)) if not(so[j][0]==0 and so[j][1]==0) and so[j][0]<bch and so[j][1]>a]
                    feats.append(hs[bi][m].mean(axis=0) if m else hs[bi][0])
                e=structure_edges(toks)
                graphs[qi]=Data(x=torch.tensor(np.stack(feats),dtype=torch.float),edge_index=(torch.tensor(e,dtype=torch.long).t().contiguous() if e else torch.empty((2,0),dtype=torch.long)),y=torch.tensor([labels[qi]]))
    return cls,graphs

def cache(name, queries, labels):
    p=os.path.join(SW,f"ms_{name}.pkl")
    if os.path.exists(p): return pickle.load(open(p,"rb"))
    log(f"embedding {name} ({len(queries)})..."); r=embed(queries,labels); pickle.dump(r,open(p,"wb")); return r
# full test (clean) + url-encoded malicious
te_cls, te_g = cache("test", test_df["Query"].tolist(), test_df["Label"].tolist())
yte=test_df["Label"].to_numpy()
mal_idx=np.where(yte==1)[0]; mal_q=[test_df["Query"].tolist()[i] for i in mal_idx]
url_cls, url_g = cache("urlmal", [url_encode(q) for q in mal_q], [1]*len(mal_q))

def load_g(tag,frame):
    C=3000; n=(len(frame)+C-1)//C; gl=[]
    for c in range(n):
        with open(os.path.join(SW,f"sg_{tag}_{c:03d}.pkl"),"rb") as f: gl+=pickle.load(f)
    return gl
tg,vg=load_g("train",tr_df),load_g("val",val_df)
bp=json.load(open(os.path.join(SW,"best.json")))
cw=compute_class_weight("balanced",classes=np.array([0,1]),y=np.array([int(g.y.item()) for g in tg]))
crit=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float))
import pickle as pk
mlp=pk.load(open(os.path.join(SW,"mlp_model.pkl"),"rb"))

class GNNModel(nn.Module):
    def __init__(s,hid,drop):
        super().__init__(); s.c1=GCNConv(768,hid); s.c2=GCNConv(hid,hid); s.f1=nn.Linear(hid,hid); s.f2=nn.Linear(hid,2); s.dp=nn.Dropout(drop)
    def forward(s,x,ei,b):
        x=F.relu(s.c1(x,ei)); x=F.relu(s.c2(x,ei)); x=global_mean_pool(x,b); return s.f2(s.dp(F.relu(s.f1(x))))
def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=GNNModel(bp["hid"],bp["drop"]); opt=torch.optim.Adam(m.parameters(),lr=bp["lr"])
    tl=GeoLoader(tg,batch_size=bp["bs"],shuffle=True); vl=GeoLoader(vg,batch_size=bp["bs"]); best,state,bad=1e9,None,0
    for ep in range(50):
        m.train()
        for d in tl: opt.zero_grad(); loss=crit(m(d.x,d.edge_index,d.batch),d.y); loss.backward(); opt.step()
        m.eval(); vlo=0.0
        with torch.no_grad():
            for d in vl: vlo+=crit(m(d.x,d.edge_index,d.batch),d.y).item()
        vlo/=len(vl)
        if vlo<best-1e-4: best,state,bad=vlo,{k:v.clone() for k,v in m.state_dict().items()},0
        else: bad+=1
        if bad>=5: break
    m.load_state_dict(state); m.eval(); return m
def gnn_probs(m,graphs):
    P=[]
    with torch.no_grad():
        for d in GeoLoader(graphs,batch_size=128): P.append(F.softmax(m(d.x,d.edge_index,d.batch),1)[:,1])
    return torch.cat(P).numpy()
def ece(prob,y,bins=15):
    conf=np.where(prob>=0.5,prob,1-prob); pred=(prob>=0.5).astype(int); correct=(pred==y).astype(float)
    e=0.0
    for i in range(bins):
        lo,hi=i/bins,(i+1)/bins; m=(conf>lo)&(conf<=hi)
        if m.sum(): e+=m.mean()*abs(correct[m].mean()-conf[m].mean())
    return e
def noise_flip_gnn(m,graphs,sigma,seed):
    torch.manual_seed(seed)
    base=(gnn_probs(m,graphs)>=0.5).astype(int)
    ng=[Data(x=g.x+torch.randn_like(g.x)*sigma,edge_index=g.edge_index,y=g.y) for g in graphs]
    noisy=(gnn_probs(m,ng)>=0.5).astype(int)
    return float((base!=noisy).mean()*100)

# BERT-only (deterministic) references
b_prob=mlp.predict_proba(te_cls)[:,1]; b_pred=(b_prob>=0.5).astype(int)
b_url=(mlp.predict(url_cls)==1).mean()*100
def noise_flip_mlp(sigma,seed):
    rng=np.random.RandomState(seed); base=mlp.predict(te_cls); noisy=mlp.predict(te_cls+rng.randn(*te_cls.shape).astype(np.float32)*sigma)
    return float((base!=noisy).mean()*100)
BERT={"clean_acc":round((b_pred==yte).mean()*100,2),"url_recall":round(b_url,2),
      "noise_flip_0.5":round(noise_flip_mlp(0.5,0),2),"noise_flip_1.0":round(noise_flip_mlp(1.0,0),2),
      "ece_clean":round(ece(b_prob,yte),4)}
log(f"BERT-only (fixed): {BERT}")

RES=os.path.join(SW,"multiseed_robust.json"); res=json.load(open(RES)) if os.path.exists(RES) else {}
for s in SEEDS:
    if str(s) in res: log(f"seed {s} cached"); continue
    m=train_seed(s); pg=gnn_probs(m,te_g); pred=(pg>=0.5).astype(int)
    r={"clean_acc":round((pred==yte).mean()*100,2),
       "url_recall":round((gnn_probs(m,url_g)>=0.5).mean()*100,2),
       "noise_flip_0.5":round(noise_flip_gnn(m,te_g,0.5,s),2),
       "noise_flip_1.0":round(noise_flip_gnn(m,te_g,1.0,s),2),
       "ece_clean":round(ece(pg,yte),4)}
    res[str(s)]=r; json.dump(res,open(RES,"w"),indent=2); log(f"seed {s}: {r}")

if len(res)==len(SEEDS):
    def ms(k): v=np.array([res[str(s)][k] for s in SEEDS]); return v.mean(),v.std(ddof=1)
    lines=["# Multi-seed robustness: structure-GNN vs BERT-only","",
      "Structure-GNN (robust config) over 5 seeds vs the deterministic BERT-only baseline. "
      "mean +/- std. Lower is better for noise-flip and ECE; higher for recall/accuracy.","",
      "| Metric | Structure-GNN (mean +/- std) | BERT-only | GNN better & seed-stable? |","|---|---|---|---|"]
    def verdict(k,lower_better):
        gm,gs=ms(k); b=BERT[k]
        better = (gm<b) if lower_better else (gm>b)
        stable = abs(gm-b) > 2*gs   # advantage exceeds ~2 std
        return f"| {k} | {gm:.2f} +/- {gs:.2f} | {b} | "+("YES" if (better and stable) else ("no (within noise)" if better else "no (BERT-only better)"))+" |"
    for k,lb in [("clean_acc",False),("url_recall",False),("noise_flip_0.5",True),("noise_flip_1.0",True),("ece_clean",True)]:
        lines.append(verdict(k,lb))
    open(os.path.join(HERE,"results","robustness_multiseed.md"),"w",encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
