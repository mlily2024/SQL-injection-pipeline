#!/usr/bin/env python3
"""Proper resumable held-out Optuna HP search (40 unique trials, SQLite-backed so
resume never duplicates trials). Selects on validation F1, writes the winner to
.corrected_work/best_params.json. Replicates corrected_bertgnn_retrain.py's model
and training (chain graphs, class-weighted CE, early stop on validation).
Loads only train+val graphs (test not needed for the search) to keep memory low.
Re-run to continue until it prints SEARCH COMPLETE."""
import os, time, json, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader as GeoLoader
import optuna

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.join(HERE, ".corrected_work")
SEED = 42; TARGET = 40
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "1800"))
STORAGE = "sqlite:///" + os.path.join(WORK, "optuna_heldout.db").replace("\\", "/")
STUDY = "heldout_bertgnn_40"
torch.set_num_threads(os.cpu_count() or 4); device = "cpu"; _T0 = time.time()
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

train_graphs = pickle.load(open(os.path.join(WORK, "graphs_train.pkl"), "rb"))
val_graphs   = pickle.load(open(os.path.join(WORK, "graphs_val.pkl"), "rb"))
log(f"graphs: train={len(train_graphs)} val={len(val_graphs)}")
cw = compute_class_weight("balanced", classes=np.array([0, 1]),
                          y=np.array([int(g.y.item()) for g in train_graphs]))
class_weights = torch.tensor(cw, dtype=torch.float)

class GNNModel(nn.Module):
    def __init__(s, input_dim, hidden_dim, output_dim, dropout_rate):
        super().__init__(); s.conv1=GCNConv(input_dim,hidden_dim); s.conv2=GCNConv(hidden_dim,hidden_dim)
        s.fc1=nn.Linear(hidden_dim,hidden_dim); s.fc2=nn.Linear(hidden_dim,output_dim); s.dropout=nn.Dropout(dropout_rate)
    def forward(s,x,ei,b):
        x=F.relu(s.conv1(x,ei)); x=F.relu(s.conv2(x,ei)); x=global_mean_pool(x,b)
        return s.fc2(s.dropout(F.relu(s.fc1(x))))

def run_epoch(model, loader, opt=None):
    crit=nn.CrossEntropyLoss(weight=class_weights); tr=opt is not None
    model.train() if tr else model.eval(); tot=0.0; preds=[]; labs=[]
    ctx=torch.enable_grad() if tr else torch.no_grad()
    with ctx:
        for d in loader:
            if tr: opt.zero_grad()
            out=model(d.x,d.edge_index,d.batch); loss=crit(out,d.y)
            if tr: loss.backward(); opt.step()
            tot+=loss.item(); preds.append(out.argmax(1)); labs.append(d.y)
    return tot/len(loader), torch.cat(preds).numpy(), torch.cat(labs).numpy()

def train_with_earlystop(params, max_epochs=12, patience=3):
    torch.manual_seed(SEED)
    m=GNNModel(768, params["hidden_dim"], 2, params["dropout_rate"])
    opt=torch.optim.Adam(m.parameters(), lr=params["lr"])
    tl=GeoLoader(train_graphs, batch_size=params["batch_size"], shuffle=True)
    vl=GeoLoader(val_graphs, batch_size=params["batch_size"])
    best=1e9; bad=0; best_state=None
    for ep in range(max_epochs):
        run_epoch(m, tl, opt); vloss,_,_=run_epoch(m, vl)
        if vloss < best-1e-4: best=vloss; bad=0; best_state={k:v.clone() for k,v in m.state_dict().items()}
        else: bad+=1
        if bad>=patience: break
    if best_state: m.load_state_dict(best_state)
    return m

def objective(trial):
    params={"hidden_dim":trial.suggest_int("hidden_dim",32,256),
            "dropout_rate":trial.suggest_float("dropout_rate",0.1,0.5),
            "lr":trial.suggest_float("lr",1e-5,1e-2,log=True),
            "batch_size":trial.suggest_int("batch_size",32,128)}
    m=train_with_earlystop(params)
    _,vp,vl=run_epoch(m, GeoLoader(val_graphs, batch_size=params["batch_size"]))
    vf1=f1_score(vl, vp, average="weighted")
    log(f"  trial done val_f1={vf1:.4f} {params}")
    return vf1

def stop_cb(study, trial):
    if time.time()-_T0 > MAX_SECONDS:
        log("BUDGET HIT; stopping (resume to continue)"); study.stop()

study=optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                          storage=STORAGE, study_name=STUDY, load_if_exists=True)
done=len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE])
log(f"resuming: {done}/{TARGET} trials complete")
if done < TARGET:
    study.optimize(objective, n_trials=TARGET-done, callbacks=[stop_cb])
done=len([t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE])
if done >= TARGET:
    json.dump(study.best_params, open(os.path.join(WORK,"best_params.json"),"w"), indent=2)
    json.dump({"best_params":study.best_params,"best_val_f1":study.best_value,"n_trials":done},
              open(os.path.join(WORK,"hp_search_v2_summary.json"),"w"), indent=2)
    log(f"SEARCH COMPLETE ({done} trials). best={study.best_params} val_f1={study.best_value:.5f}")
else:
    log(f"progress {done}/{TARGET}; re-run to continue")
