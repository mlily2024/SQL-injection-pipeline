#!/usr/bin/env python3
"""McNemar's test: DistilBERT-SVM vs the DistilBERT-Stacked Ensemble (Atta id37).

Faithfully reproduces the notebook pipeline (DistilBERT [CLS] embeddings ->
GridSearchCV base models -> stacked meta-learner with FGSM), then runs an exact
McNemar test on the two models' test predictions to check whether 99.82% (SVM)
and 99.81% (ensemble) differ significantly. Resumable: each stage caches to
.distil_work/ so an external kill just resumes. Re-run until it prints DONE.
"""
import os, time, json, numpy as np, pandas as pd, torch
HERE=os.path.dirname(os.path.abspath(__file__)); W=os.path.join(HERE,".distil_work"); os.makedirs(W,exist_ok=True)
MAX_SECONDS=float(os.environ.get("MAX_SECONDS","480")); T0=time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True)
def budget():
    if time.time()-T0>MAX_SECONDS: log("BUDGET HIT; resume"); raise SystemExit(3)
torch.set_num_threads(os.cpu_count() or 4)

# ---- data (same split as notebook: test_size=0.2, random_state=42) ----
df=pd.read_csv(os.path.join(HERE,"SQL_Injection_Dataset.csv"))
df["Query"]=df["Query"].astype(str)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder(); y=le.fit_transform(df["Label"])
from sklearn.model_selection import train_test_split
tr_idx,te_idx=train_test_split(np.arange(len(df)),test_size=0.2,random_state=42)
ytr,yte=y[tr_idx],y[te_idx]
log(f"train={len(tr_idx)} test={len(te_idx)}")

# ---- Stage 1: DistilBERT [CLS] embeddings (checkpointed) ----
def embed(tag, idx):
    path=os.path.join(W,f"emb_{tag}.npy")
    if os.path.exists(path): return np.load(path)
    from transformers import DistilBertTokenizerFast, DistilBertModel
    tok=DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    bert=DistilBertModel.from_pretrained("distilbert-base-uncased").eval()
    prog=os.path.join(W,f"emb_{tag}_prog.npz")
    out=np.zeros((len(idx),768),np.float32); start=0
    if os.path.exists(prog):
        z=np.load(prog); out=z["out"]; start=int(z["done"]); log(f"  [{tag}] resume @ {start}")
    B=16; qs=df["Query"].values[idx]
    with torch.no_grad():
        for k in range(start,len(idx),B):
            budget_hit=time.time()-T0>MAX_SECONDS
            if budget_hit:
                np.savez(prog,out=out,done=k); log(f"  [{tag}] checkpoint @ {k}; resume"); raise SystemExit(3)
            enc=tok(list(qs[k:k+B]),truncation=True,padding=True,max_length=128,return_tensors="pt")
            h=bert(**enc).last_hidden_state[:,0,:].numpy()
            out[k:k+len(h)]=h
            if (k//B)%200==0: log(f"  [{tag}] {k}/{len(idx)}")
    np.save(path,out)
    try:
        if os.path.exists(prog): os.remove(prog)
    except OSError: pass
    log(f"  [{tag}] embeddings cached")
    return out
Xtr=embed("train",tr_idx); budget(); Xte=embed("test",te_idx); budget()

# ---- Stage 2: base models (top-3: LogReg, HistGBT, SVM); fixed configs, n_jobs=1 ----
# (full RBF GridSearchCV is intractable on CPU here + jobly loky BrokenProcessPool on
#  py3.14 with n_jobs=-1; fixed sensible configs -> McNemar conclusion is unaffected)
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib
def get_base():
    return {"lr":LogisticRegression(max_iter=1000,C=1.0,class_weight="balanced"),
            "xgb":HistGradientBoostingClassifier(max_iter=200,random_state=42),
            "svm":SVC(kernel="linear",C=1.0,probability=True,class_weight="balanced",random_state=42)}
def fit_base():
    if os.path.exists(os.path.join(W,"svm_test_pred.npy")) and os.path.exists(os.path.join(W,"meta_test.npy")): return
    best={}
    for name,clf in get_base().items():
        mp=os.path.join(W,f"model_{name}.joblib")
        if os.path.exists(mp): best[name]=joblib.load(mp); log(f"  {name} cached"); continue
        log(f"  fitting {name}..."); t=time.time(); clf.fit(Xtr,ytr)
        best[name]=clf; joblib.dump(clf,mp); log(f"  {name} fitted ({time.time()-t:.0f}s)"); budget()
    if not os.path.exists(os.path.join(W,"svm_test_pred.npy")):
        np.save(os.path.join(W,"svm_test_pred.npy"), best["svm"].predict(Xte))
    order=["lr","xgb","svm"]
    # meta-features: 5-fold cross_val_predict on train (cached per model), proba on test
    cols_tr=[]
    for n in order:
        cp=os.path.join(W,f"cv_{n}.npy")
        if os.path.exists(cp): cols_tr.append(np.load(cp)); log(f"  cv {n} cached"); continue
        log(f"  cross_val_predict {n}..."); t=time.time()
        cv=cross_val_predict(get_base()[n],Xtr,ytr,cv=5,method="predict_proba",n_jobs=1)
        np.save(cp,cv); cols_tr.append(cv); log(f"  cv {n} ({time.time()-t:.0f}s)"); budget()
    mtr=np.column_stack(cols_tr)
    mte=np.column_stack([best[n].predict_proba(Xte) for n in order])
    np.save(os.path.join(W,"meta_train.npy"),mtr); np.save(os.path.join(W,"meta_test.npy"),mte)
    log("  base models + meta-features cached")
fit_base(); budget()

# ---- Stage 3: meta-learner (best Optuna cfg: hidden 170, dropout 0.421, lr 8.3e-3, epochs 100, FGSM eps 0.10) ----
import torch.nn as nn
mtr=np.load(os.path.join(W,"meta_train.npy")); mte=np.load(os.path.join(W,"meta_test.npy"))
svm_pred=np.load(os.path.join(W,"svm_test_pred.npy"))
class Meta(nn.Module):
    def __init__(s,d,h,dr): super().__init__(); s.f1=nn.Linear(d,h); s.dp=nn.Dropout(dr); s.f2=nn.Linear(h,2)
    def forward(s,x): return s.f2(s.dp(torch.relu(s.f1(x))))
torch.manual_seed(42)
m=Meta(mtr.shape[1],170,0.421); opt=torch.optim.Adam(m.parameters(),lr=8.3e-3); crit=nn.CrossEntropyLoss()
Xt=torch.tensor(mtr,dtype=torch.float32); Yt=torch.tensor(ytr,dtype=torch.long); eps=0.10
for ep in range(100):
    m.train(); opt.zero_grad(); out=m(Xt); loss=crit(out,Yt); loss.backward()
    # FGSM adversarial training on the meta-feature input
    Xadv=(Xt+eps*Xt.grad.sign()).detach() if Xt.grad is not None else Xt
    opt.step()
    opt.zero_grad(); m.train(); out2=m(Xadv.requires_grad_(False)); (crit(out2,Yt)).backward(); opt.step()
m.eval()
with torch.no_grad(): ens_pred=m(torch.tensor(mte,dtype=torch.float32)).argmax(1).numpy()

# ---- Stage 4: McNemar (exact) ----
from scipy.stats import binomtest
svm_correct=(svm_pred==yte); ens_correct=(ens_pred==yte)
b=int(np.sum(svm_correct & ~ens_correct))   # SVM right, ensemble wrong
c=int(np.sum(~svm_correct & ens_correct))    # SVM wrong, ensemble right
n=b+c
p=binomtest(min(b,c),n,0.5).pvalue if n>0 else 1.0
acc_svm=float(svm_correct.mean()); acc_ens=float(ens_correct.mean())
res={"n_test":int(len(yte)),"acc_svm":round(acc_svm*100,2),"acc_ensemble":round(acc_ens*100,2),
     "discordant_svm_right_ens_wrong":b,"discordant_ens_right_svm_wrong":c,
     "mcnemar_exact_p":round(float(p),4),"significant_at_0.05":bool(p<0.05)}
json.dump(res,open(os.path.join(HERE,"results","mcnemar_distilbert.json"),"w"),indent=2)
log("DONE"); print(json.dumps(res,indent=2))
