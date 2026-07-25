#!/usr/bin/env python3
"""Low-data learning curves: does the structure graph help as an inductive bias
when training data is scarce? Train structure-GNN and BERT-only(MLP) on 1%..100%
of the training set (stratified, 3 seeds each), evaluate on the full 9,276-query
test set. If the GNN beats BERT-only at low fractions (a crossing), structure buys
sample efficiency = a genuine positive. Resumable per (fraction, seed, model)."""
import os, json, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as GeoLoader

HERE=os.path.dirname(os.path.abspath(__file__)); SW=os.path.join(HERE,".structure_work")
FRACS=[0.01,0.02,0.05,0.10,0.25,0.50,1.0]; SEEDS=[42,1,7]
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
yg_train=np.array([int(g.y.item()) for g in train_g]); yg_test=np.array([int(g.y.item()) for g in test_g])
# BERT-only features: train_cls is over train_df (21,643); align to tr_df (same pool as the GNN)
Xtr_full=np.load(os.path.join(SW,"train_cls.npy")); Xte=np.load(os.path.join(SW,"test_cls_rowB.npy"))
pos=train_df.index.get_indexer(tr_df.index)          # tr_df rows within train_df order
assert (pos>=0).all() and len(pos)==len(tr_df)
Xtr=Xtr_full[pos]; ytr_cls=tr_df["Label"].to_numpy(); yte_cls=test_df["Label"].to_numpy()
log(f"graphs train={len(train_g)} | Xtr={Xtr.shape} ytr_cls={len(ytr_cls)} (aligned to tr_df)")
bp=json.load(open(os.path.join(SW,"best.json")))

class GNNModel(nn.Module):
    def __init__(s,hid,drop):
        super().__init__(); s.c1=GCNConv(768,hid); s.c2=GCNConv(hid,hid); s.f1=nn.Linear(hid,hid); s.f2=nn.Linear(hid,2); s.dp=nn.Dropout(drop)
    def forward(s,x,ei,b):
        x=F.relu(s.c1(x,ei)); x=F.relu(s.c2(x,ei)); x=global_mean_pool(x,b); return s.f2(s.dp(F.relu(s.f1(x))))

def strat_idx(y, frac, seed):
    rng=np.random.RandomState(seed); idx=[]
    for c in (0,1):
        ci=np.where(y==c)[0]; rng.shuffle(ci); idx+=list(ci[:max(1,int(len(ci)*frac))])
    return np.array(idx)

def gnn_fit_eval(frac,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    idx=strat_idx(yg_train,frac,seed); sub=[train_g[i] for i in idx]
    cw=compute_class_weight("balanced",classes=np.array([0,1]),y=yg_train[idx])
    crit=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float))
    m=GNNModel(bp["hid"],bp["drop"]); opt=torch.optim.Adam(m.parameters(),lr=bp["lr"])
    tl=GeoLoader(sub,batch_size=min(bp["bs"],max(2,len(sub))),shuffle=True); vl=GeoLoader(val_g,batch_size=128)
    best,state,bad=1e9,None,0
    for ep in range(60):
        m.train()
        for d in tl: opt.zero_grad(); loss=crit(m(d.x,d.edge_index,d.batch),d.y); loss.backward(); opt.step()
        m.eval(); vlo=0.0
        with torch.no_grad():
            for d in vl: vlo+=crit(m(d.x,d.edge_index,d.batch),d.y).item()
        vlo/=len(vl)
        if vlo<best-1e-4: best,state,bad=vlo,{k:v.clone() for k,v in m.state_dict().items()},0
        else: bad+=1
        if bad>=6: break
    m.load_state_dict(state); m.eval(); pred=[]
    with torch.no_grad():
        for d in GeoLoader(test_g,batch_size=128): pred.append(m(d.x,d.edge_index,d.batch).argmax(1))
    p=torch.cat(pred).numpy()
    return accuracy_score(yg_test,p)*100, recall_score(yg_test,p,pos_label=1)*100

def mlp_fit_eval(frac,seed):
    idx=strat_idx(ytr_cls,frac,seed)
    clf=MLPClassifier(hidden_layer_sizes=(159,),max_iter=300,random_state=seed).fit(Xtr[idx],ytr_cls[idx])
    p=clf.predict(Xte)
    return accuracy_score(yte_cls,p)*100, recall_score(yte_cls,p,pos_label=1)*100

RES=os.path.join(SW,"low_data.json"); res=json.load(open(RES)) if os.path.exists(RES) else {}
for frac in FRACS:
    for seed in SEEDS:
        for name,fn in (("gnn",gnn_fit_eval),("mlp",mlp_fit_eval)):
            k=f"{name}_{frac}_{seed}"
            if k in res: continue
            a,r=fn(frac,seed); res[k]={"acc":round(a,2),"rec":round(r,2)}
            json.dump(res,open(RES,"w"),indent=2); log(f"{k}: acc={a:.2f} rec={r:.2f}")

lines=["# Low-data learning curves: structure-GNN vs BERT-only (MLP)","",
 "Trained on stratified fractions of the training set (3 seeds each), tested on the full "
 "9,276-query test set. If the GNN exceeds BERT-only at small fractions, structure buys sample "
 "efficiency.","","| Train fraction | GNN acc (%) | BERT-only acc (%) | GNN recall (%) | BERT-only recall (%) |","|---|---|---|---|---|"]
for frac in FRACS:
    g=[res[f"gnn_{frac}_{s}"] for s in SEEDS if f"gnn_{frac}_{s}" in res]
    b=[res[f"mlp_{frac}_{s}"] for s in SEEDS if f"mlp_{frac}_{s}" in res]
    if len(g)==len(SEEDS)==len(b):
        ga=np.mean([x["acc"] for x in g]); gr=np.mean([x["rec"] for x in g])
        ba=np.mean([x["acc"] for x in b]); br=np.mean([x["rec"] for x in b])
        lines.append(f"| {int(frac*100)}% | {ga:.2f} | {ba:.2f} | {gr:.2f} | {br:.2f} |")
open(os.path.join(HERE,"results","low_data_curve.md"),"w",encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
