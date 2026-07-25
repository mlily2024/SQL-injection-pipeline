#!/usr/bin/env python3
"""Representation redundancy via linear CKA. If the structure-GNN's graph-level
representation is highly aligned with BERT's [CLS] embedding, the GNN is
re-encoding what BERT already produced (redundant by construction). Reports
linear CKA(GNN penultimate rep, BERT [CLS]) on the 9,276-query test set, with
reference points: CKA vs mean BERT token embedding, and vs a random baseline.
Also a linear-probe R^2 (how well [CLS] linearly reconstructs the GNN rep)."""
import os, json, pickle, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as GeoLoader

HERE=os.path.dirname(os.path.abspath(__file__)); SW=os.path.join(HERE,".structure_work")
torch.set_num_threads(os.cpu_count() or 4)
def log(m): print(f"[{__import__('time').strftime('%H:%M:%S')}] {m}", flush=True)

df=pd.read_csv(os.path.join(HERE,"SQL_Injection_Dataset.csv"))
df["Query"]=df["Query"].astype(str).apply(lambda x:x.lower().strip())
train_df,test_df=train_test_split(df,test_size=0.30,random_state=42)
def load_g(tag,frame):
    C=3000; n=(len(frame)+C-1)//C; gl=[]
    for c in range(n):
        with open(os.path.join(SW,f"sg_{tag}_{c:03d}.pkl"),"rb") as f: gl+=pickle.load(f)
    return gl
test_g=load_g("test",test_df); log(f"test graphs: {len(test_g)}")
bp=json.load(open(os.path.join(SW,"best.json")))

class GNNModel(nn.Module):
    def __init__(s,hid,drop):
        super().__init__(); s.c1=GCNConv(768,hid); s.c2=GCNConv(hid,hid); s.f1=nn.Linear(hid,hid); s.f2=nn.Linear(hid,2); s.dp=nn.Dropout(drop)
    def rep(s,x,ei,b):                                   # penultimate graph representation
        x=F.relu(s.c1(x,ei)); x=F.relu(s.c2(x,ei)); pooled=global_mean_pool(x,b)
        return pooled, F.relu(s.f1(pooled))
gnn=GNNModel(bp["hid"],bp["drop"]); gnn.load_state_dict(torch.load(os.path.join(SW,"gnn_model.pt"))); gnn.eval()

# GNN reps (pooled + penultimate) + mean node feature, aligned to test row order
pooled_all=[]; pen_all=[]; meanfeat_all=[]
ld=GeoLoader(test_g,batch_size=128)
with torch.no_grad():
    for d in ld:
        pooled,pen=gnn.rep(d.x,d.edge_index,d.batch)
        mf=global_mean_pool(d.x,d.batch)                # mean raw BERT token feature
        pooled_all.append(pooled); pen_all.append(pen); meanfeat_all.append(mf)
POOL=torch.cat(pooled_all).numpy(); PEN=torch.cat(pen_all).numpy(); MEANF=torch.cat(meanfeat_all).numpy()
CLS=np.load(os.path.join(SW,"test_cls_rowB.npy"))       # BERT [CLS], test row order
log(f"reps: pooled{POOL.shape} pen{PEN.shape} cls{CLS.shape}")

def linear_cka(X,Y):
    X=X-X.mean(0); Y=Y-Y.mean(0)
    return (np.linalg.norm(Y.T@X)**2)/(np.linalg.norm(X.T@X)*np.linalg.norm(Y.T@Y))

rng=np.random.RandomState(0); RAND=rng.randn(*CLS.shape).astype(np.float32)
results={
 "CKA(GNN penultimate, BERT[CLS])": float(linear_cka(PEN,CLS)),
 "CKA(GNN pooled,      BERT[CLS])": float(linear_cka(POOL,CLS)),
 "CKA(GNN pooled, mean BERT token)": float(linear_cka(POOL,MEANF)),
 "CKA(BERT[CLS], mean BERT token)": float(linear_cka(CLS,MEANF)),
 "CKA(GNN pooled, RANDOM) [floor]": float(linear_cka(POOL,RAND)),
}
# linear probe: how well does [CLS] linearly reconstruct the GNN penultimate rep?
r=Ridge(alpha=1.0).fit(CLS,PEN); probe_r2=r2_score(PEN, r.predict(CLS))
results["Linear probe R^2: [CLS] -> GNN rep"]=float(probe_r2)

lines=["# Representation redundancy (linear CKA)","",
 "How much is the structure-GNN's representation a re-encoding of BERT's [CLS]? "
 "CKA in [0,1]; 1 = identical geometry. A high value + a high linear-probe R^2 means "
 "the GNN adds no representation BERT did not already provide.","",
 "| Comparison | Value |","|---|---|"]
for k,v in results.items(): lines.append(f"| {k} | {v:.3f} |")
lines+=["", f"Interpretation: CKA(GNN, [CLS]) = {results['CKA(GNN penultimate, BERT[CLS])']:.3f} "
 f"(random floor {results['CKA(GNN pooled, RANDOM) [floor]']:.3f}); [CLS] linearly reconstructs the "
 f"GNN representation with R^2 = {probe_r2:.3f}.",""]
open(os.path.join(HERE,"results","cka_redundancy.md"),"w",encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
