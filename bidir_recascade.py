#!/usr/bin/env python3
"""Bidirectional held-out model: rebuild graphs undirected (edges both ways,
matching Eq 6), retrain (SEED=42, cached best_params), regenerate the four
figures the paper uses (same filenames) and compute every scalar."""
import os,json,pickle,time,numpy as np,torch,torch.nn as nn,torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
HERE=os.path.dirname(os.path.abspath(__file__)); WORK=os.path.join(HERE,".corrected_work"); RES=os.path.join(HERE,"results")
SEED=42; torch.set_num_threads(os.cpu_count() or 4)
bp=json.load(open(f"{WORK}/best_params.json"))
def load(t): return pickle.load(open(f"{WORK}/graphs_{t}.pkl","rb"))
def bidir(gs):
    out=[]
    for g in gs:
        ei=g.edge_index; ei2=torch.cat([ei,ei.flip(0)],1) if ei.numel() else ei
        out.append(Data(x=g.x,edge_index=ei2,y=g.y))
    return out
tr,va,te=bidir(load("train")),bidir(load("val")),bidir(load("test"))
cw=compute_class_weight("balanced",classes=np.array([0,1]),y=np.array([int(g.y.item()) for g in tr]))
crit=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float))
class GNN(nn.Module):
    def __init__(s,i,h,o,d):
        super().__init__(); s.conv1=GCNConv(i,h); s.conv2=GCNConv(h,h); s.fc1=nn.Linear(h,h); s.fc2=nn.Linear(h,o); s.dropout=nn.Dropout(d)
    def forward(s,x,ei,b):
        x=F.relu(s.conv1(x,ei)); x=F.relu(s.conv2(x,ei)); x=global_mean_pool(x,b); return s.fc2(s.dropout(F.relu(s.fc1(x))))
def run(m,loader,opt=None,probs=False):
    trn=opt is not None; m.train() if trn else m.eval(); tot=0; P=[]; L=[]; PR=[]
    ctx=torch.enable_grad() if trn else torch.no_grad()
    with ctx:
        for d in loader:
            if trn: opt.zero_grad()
            o=m(d.x,d.edge_index,d.batch); l=crit(o,d.y)
            if trn: l.backward(); opt.step()
            tot+=l.item(); P.append(o.argmax(1)); L.append(d.y)
            if probs: PR.append(F.softmax(o,1)[:,1])
    return (tot/len(loader),torch.cat(P).numpy(),torch.cat(L).numpy(),(torch.cat(PR).numpy() if probs else None))
torch.manual_seed(SEED)
m=GNN(768,bp["hidden_dim"],2,bp["dropout_rate"]); opt=torch.optim.Adam(m.parameters(),lr=bp["lr"])
tl=GeoLoader(tr,batch_size=bp["batch_size"],shuffle=True); vl=GeoLoader(va,batch_size=bp["batch_size"])
best,state,bad,hist=1e9,None,0,[]; t0=time.time()
for ep in range(50):
    tloss,_,_,_=run(m,tl,opt); vloss,_,_,_=run(m,vl); hist.append((tloss,vloss))
    if vloss<best-1e-4: best,state,bad=vloss,{k:v.clone() for k,v in m.state_dict().items()},0
    else: bad+=1
    if bad>=5: break
train_secs=time.time()-t0; m.load_state_dict(state); m.eval()
torch.save(state,f"{WORK}/heldout_bidir_model.pt")
_,tp,tlab,tprob=run(m,GeoLoader(te,batch_size=bp["batch_size"]),probs=True)
acc=accuracy_score(tlab,tp); pr=precision_score(tlab,tp,average="weighted"); rc=recall_score(tlab,tp,average="weighted")
f1=f1_score(tlab,tp,average="weighted"); cm=confusion_matrix(tlab,tp)
fpr,tpr,_=roc_curve(tlab,tprob); AUC=auc(fpr,tpr)
print(f"BIDIR acc={acc*100:.2f} p={pr*100:.2f} r={rc*100:.2f} f1={f1*100:.2f} cm={cm.tolist()} auc={AUC:.4f} secs={train_secs:.1f} eps={ep+1}")

# ---- figures (same filenames the docx references) ----
fig,ax=plt.subplots(figsize=(5,4))
sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",cbar=False,xticklabels=["Benign","Malicious"],
            yticklabels=["Benign","Malicious"],ax=ax,annot_kws={"size":13})
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title("BERT-GNN",fontsize=10)
fig.tight_layout(); fig.savefig(f"{RES}/cm_corrected_bertgnn_heldout.png",dpi=200,bbox_inches="tight"); plt.close(fig)

tr_l=[h[0] for h in hist]; va_l=[h[1] for h in hist]; ep_r=range(1,len(hist)+1)
plt.figure(figsize=(6,4)); plt.plot(ep_r,tr_l,marker="o",ms=3,label="Training loss"); plt.plot(ep_r,va_l,marker="s",ms=3,label="Validation loss")
plt.xlabel("Epoch"); plt.ylabel("Cross-entropy loss"); plt.title("Held-out BERT-GNN: training vs validation loss")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{RES}/corrected_loss_curve.png",dpi=200,bbox_inches="tight"); plt.close()

plt.figure(figsize=(5,4.2)); plt.plot(fpr,tpr,color="#3b6ea5",label=f"ROC (AUC = {AUC:.4f})"); plt.plot([0,1],[0,1],"--",color="grey",lw=1)
plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("Held-out BERT-GNN: ROC (held-out test)")
plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{RES}/corrected_roc.png",dpi=200,bbox_inches="tight"); plt.close()

# ---- sensitivity (prob-change, 99 malicious graphs, bidirectional rebuild) ----
def prob_mal(x,ei):
    with torch.no_grad():
        return float(F.softmax(m(x,ei,torch.zeros(x.size(0),dtype=torch.long)),1)[0,1])
def perturb(g,rng):
    n=g.x.size(0)
    if n<3: return None
    k=max(1,int(round(0.10*n))); drop=set(rng.choice(n,k,replace=False).tolist())
    keep=[i for i in range(n) if i not in drop]; mm=len(keep)
    fwd=[[j,j+1] for j in range(mm-1)]
    ei=(torch.tensor(fwd+[[b,a] for a,b in fwd],dtype=torch.long).t().contiguous() if mm>1 else torch.empty((2,0),dtype=torch.long))
    return g.x[keep], ei
rng=np.random.RandomState(SEED)
mal=[g for g in te if int(g.y.item())==1 and g.x.size(0)>=5][:99]
scores=[]
for g in mal:
    base=prob_mal(g.x,g.edge_index); ds=[]
    for _ in range(20):
        p=perturb(g,rng)
        if p is None: continue
        ds.append(abs(prob_mal(p[0],p[1])-base))
    scores.append(float(np.mean(ds)) if ds else 0.0)
scores=np.array(scores); ms=float(scores.mean()); order=np.argsort(scores)
hi=[(int(i+1),round(float(scores[i]),3)) for i in order[::-1][:3]]
frac=float((scores<0.01).mean())
print(f"SENS mean={ms:.4f} median={np.median(scores):.4f} frac<0.01={frac:.2f} highest={hi}")
plt.figure(figsize=(6.6,3.6)); plt.plot(range(1,len(scores)+1),scores,marker="o",ms=3,ls="-",color="#3b6ea5")
plt.axhline(ms,color="#a5322d",ls="--",lw=1,label=f"Mean = {ms:.4f}")
plt.title("Sensitivity scores - held-out BERT-GNN"); plt.xlabel("Attack-pattern graph index"); plt.ylabel("Change in malicious-class probability")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(f"{RES}/sensitivity_heldout.png",dpi=200,bbox_inches="tight"); plt.close()

json.dump({"acc":round(acc*100,2),"prec":round(pr*100,2),"rec":round(rc*100,2),"f1":round(f1*100,2),
           "cm":cm.tolist(),"auc":round(float(AUC),4),"train_secs":round(train_secs,1),"epochs":ep+1,
           "loss_tr_first":round(tr_l[0],3),"loss_tr_last":round(tr_l[-1],3),
           "loss_va_min":round(min(va_l),3),"loss_va_max":round(max(va_l),3),
           "sens_mean":round(ms,4),"sens_median":round(float(np.median(scores)),4),
           "sens_frac_lt01":round(frac,2),"sens_highest":hi},
          open(f"{WORK}/bidir_result.json","w"),indent=2)
print("DONE -> figures overwritten + .corrected_work/bidir_result.json")
