#!/usr/bin/env python3
"""Edge ablation: does the graph STRUCTURE contribute, or only the (BERT) node
features? Retrain the structure-GNN with (a) original structural edges, (b) no
edges, (c) random edges of the same count, across 3 seeds each, and compare clean
test accuracy. If none/random == original, the graph topology is inert (the 'G'
in GNN adds nothing). Resumable per (condition, seed)."""
import os, json, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader

HERE=os.path.dirname(os.path.abspath(__file__)); SW=os.path.join(HERE,".structure_work")
SEEDS=[42,1,7]; CONDS=["original","none","random"]
torch.set_num_threads(os.cpu_count() or 4)
def log(m): print(f"[{__import__('time').strftime('%H:%M:%S')}] {m}", flush=True)

df=pd.read_csv(os.path.join(HERE,"SQL_Injection_Dataset.csv"))
df["Query"]=df["Query"].astype(str).apply(lambda x:x.lower().strip())
train_df,test_df=train_test_split(df,test_size=0.30,random_state=42)
tr_df,val_df=train_test_split(train_df,test_size=0.15,random_state=42,stratify=train_df["Label"])
def load_g(tag,frame):
    C=3000; n=(len(frame)+C-1)//C; gl=[]
    for c in range(n):
        with open(os.path.join(SW,f"sg_{tag}_{c:03d}.pkl"),"rb") as f: gl+=pickle.load(f)
    return gl
train_g,val_g,test_g=load_g("train",tr_df),load_g("val",val_df),load_g("test",test_df)
log(f"graphs: train={len(train_g)} val={len(val_g)} test={len(test_g)}")
bp=json.load(open(os.path.join(SW,"best.json"))); log(f"config: {bp}")
cw=compute_class_weight("balanced",classes=np.array([0,1]),y=np.array([int(g.y.item()) for g in train_g]))
crit=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float))

def transform(graphs, cond, seed):
    if cond=="original": return graphs
    rng=np.random.RandomState(seed); out=[]
    for g in graphs:
        n=g.x.shape[0]
        if cond=="none" or n<2 or g.edge_index.shape[1]==0:
            ei=torch.empty((2,0),dtype=torch.long)
        else:  # random edges, same count, no self-loops
            m=g.edge_index.shape[1]; src=rng.randint(0,n,size=m); dst=rng.randint(0,n,size=m)
            keep=src!=dst; ei=torch.tensor(np.stack([src[keep],dst[keep]]),dtype=torch.long)
        out.append(Data(x=g.x,edge_index=ei,y=g.y))
    return out

class GNNModel(nn.Module):
    def __init__(s,hid,drop):
        super().__init__(); s.c1=GCNConv(768,hid); s.c2=GCNConv(hid,hid); s.f1=nn.Linear(hid,hid); s.f2=nn.Linear(hid,2); s.dp=nn.Dropout(drop)
    def forward(s,x,ei,b):
        x=F.relu(s.c1(x,ei)); x=F.relu(s.c2(x,ei)); x=global_mean_pool(x,b); return s.f2(s.dp(F.relu(s.f1(x))))

def train_eval(tg,vg,te,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m=GNNModel(bp["hid"],bp["drop"]); opt=torch.optim.Adam(m.parameters(),lr=bp["lr"])
    tl=GeoLoader(tg,batch_size=bp["bs"],shuffle=True); vl=GeoLoader(vg,batch_size=bp["bs"])
    best,state,bad=1e9,None,0
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
    m.load_state_dict(state); m.eval()
    tl2=GeoLoader(te,batch_size=128); pred=[]; lab=[]
    with torch.no_grad():
        for d in tl2: pred.append(m(d.x,d.edge_index,d.batch).argmax(1)); lab.append(d.y)
    return accuracy_score(torch.cat(lab).numpy(),torch.cat(pred).numpy())*100

RES=os.path.join(SW,"edge_ablation.json"); res=json.load(open(RES)) if os.path.exists(RES) else {}
for cond in CONDS:
    tg=transform(train_g,cond,0); vg=transform(val_g,cond,0); te=transform(test_g,cond,0)
    for seed in SEEDS:
        key=f"{cond}_{seed}"
        if key in res: log(f"{key} cached: {res[key]:.2f}"); continue
        acc=train_eval(tg if cond=="original" else transform(train_g,cond,seed),
                       vg if cond=="original" else transform(val_g,cond,seed),
                       te if cond=="original" else transform(test_g,cond,seed), seed)
        res[key]=round(acc,2); json.dump(res,open(RES,"w"),indent=2); log(f"{key}: acc={acc:.2f}")

lines=["# Edge ablation: does graph structure contribute?","",
       f"Structure-GNN (config {bp}) retrained with original / no / random edges, 3 seeds each. "
       "Clean test accuracy (%). If none and random match original, the graph topology is inert.","",
       "| Edge condition | Mean acc (%) | Per-seed |","|---|---|---|"]
for cond in CONDS:
    vals=[res[f"{cond}_{s}"] for s in SEEDS if f"{cond}_{s}" in res]
    if vals: lines.append(f"| {cond} | {np.mean(vals):.2f} +/- {np.std(vals,ddof=1):.2f} | {vals} |")
open(os.path.join(HERE,"results","edge_ablation.md"),"w",encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
