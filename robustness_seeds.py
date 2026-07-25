#!/usr/bin/env python3
"""Seed-stability of the structure-GNN's URL-encode robustness. Trains the robust
config across 5 seeds (seeding BEFORE weight init, fixing the non-determinism that
made the metric swing 92-98%) and reports mean +/- std of URL-encode recall on the
3,446 malicious test queries. BERT-only recall is deterministic (90.34%). Obfuscated
structure-graphs are embedded once and cached; resumable per seed."""
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
MODEL_NAME="bert-base-uncased"; MAX_LEN=128; SEEDS=[42,1,7,13,123]
KEYWORDS={"select","from","where","union","and","or","insert","update","delete",
          "join","on","group","order","having","limit","values","set"}
torch.set_num_threads(os.cpu_count() or 4)
def log(m): print(f"[{__import__('time').strftime('%H:%M:%S')}] {m}", flush=True)
def url_encode(q): return q.replace(" ","%20").replace("'","%27").replace("=","%3d")

df=pd.read_csv(os.path.join(HERE,"SQL_Injection_Dataset.csv"))
df["Query"]=df["Query"].astype(str).apply(lambda x:x.lower().strip())
train_df,test_df=train_test_split(df,test_size=0.30,random_state=42)
tr_df,val_df=train_test_split(train_df,test_size=0.15,random_state=42,stratify=train_df["Label"])
mal_test=test_df[test_df["Label"]==1]["Query"].tolist()
log(f"malicious test queries: {len(mal_test)}")

tok=BertTokenizerFast.from_pretrained(MODEL_NAME); bert=BertModel.from_pretrained(MODEL_NAME).eval()
def sql_tokens(q):
    toks,pos=[],0
    flat=list(sqlparse.parse(q)[0].flatten()) if q.strip() else []
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
def embed_and_graph(queries):
    order=np.argsort([len(q) for q in queries],kind="stable"); graphs=[None]*len(queries); B=32
    with torch.no_grad():
        for k in range(0,len(order),B):
            idx=order[k:k+B]; batch=[queries[i] for i in idx]
            enc=tok(batch,truncation=True,max_length=MAX_LEN,padding=True,return_offsets_mapping=True,return_tensors="pt")
            hs=bert(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"]).last_hidden_state.numpy()
            offs=enc["offset_mapping"].numpy()
            for bi,qi in enumerate(idx):
                toks=sql_tokens(queries[qi])
                if not toks:
                    graphs[qi]=Data(x=torch.tensor(hs[bi][:1]),edge_index=torch.empty((2,0),dtype=torch.long),y=torch.tensor([1])); continue
                feats=[]; so=offs[bi]
                for (val,a,bch) in toks:
                    m=[j for j in range(len(so)) if not(so[j][0]==0 and so[j][1]==0) and so[j][0]<bch and so[j][1]>a]
                    feats.append(hs[bi][m].mean(axis=0) if m else hs[bi][0])
                e=structure_edges(toks)
                graphs[qi]=Data(x=torch.tensor(np.stack(feats),dtype=torch.float),
                    edge_index=(torch.tensor(e,dtype=torch.long).t().contiguous() if e else torch.empty((2,0),dtype=torch.long)),
                    y=torch.tensor([1]))
    return graphs

# obfuscated + clean malicious-test graphs (cached once)
def cached_graphs(name, queries):
    p=os.path.join(SW,f"seedstab_{name}.pkl")
    if os.path.exists(p): return pickle.load(open(p,"rb"))
    log(f"embedding {name} graphs ({len(queries)})..."); g=embed_and_graph(queries); pickle.dump(g,open(p,"wb")); return g
url_g=cached_graphs("url",[url_encode(q) for q in mal_test])
clean_g=cached_graphs("clean",mal_test)

class GNNModel(nn.Module):
    def __init__(s,hid,drop):
        super().__init__(); s.c1=GCNConv(768,hid); s.c2=GCNConv(hid,hid); s.f1=nn.Linear(hid,hid); s.f2=nn.Linear(hid,2); s.dp=nn.Dropout(drop)
    def forward(s,x,ei,b):
        x=F.relu(s.c1(x,ei)); x=F.relu(s.c2(x,ei)); x=global_mean_pool(x,b); return s.f2(s.dp(F.relu(s.f1(x))))
def load_g(tag,frame):
    C=3000; n=(len(frame)+C-1)//C; gl=[]
    for c in range(n):
        with open(os.path.join(SW,f"sg_{tag}_{c:03d}.pkl"),"rb") as f: gl+=pickle.load(f)
    return gl
bp=json.load(open(os.path.join(SW,"best.json"))); log(f"config: {bp}")
tg,vg=load_g("train",tr_df),load_g("val",val_df)
cw=compute_class_weight("balanced",classes=np.array([0,1]),y=np.array([int(g.y.item()) for g in tg]))
crit=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float))

def train_seed(seed):
    torch.manual_seed(seed); np.random.seed(seed)      # BEFORE init (the fix)
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
    m.load_state_dict(state); m.eval(); return m
def recall(m,graphs):
    ld=GeoLoader(graphs,batch_size=64); pred=[]
    with torch.no_grad():
        for d in ld: pred.append(m(d.x,d.edge_index,d.batch).argmax(1))
    p=torch.cat(pred).numpy(); return float((p==1).mean()*100)

RES=os.path.join(SW,"seedstab_results.json"); res=json.load(open(RES)) if os.path.exists(RES) else {}
for s in SEEDS:
    if str(s) in res: log(f"seed {s} cached: {res[str(s)]}"); continue
    m=train_seed(s); u=recall(m,url_g); c=recall(m,clean_g)
    res[str(s)]={"url_encode_recall":round(u,2),"clean_recall":round(c,2)}
    json.dump(res,open(RES,"w"),indent=2); log(f"seed {s}: url_encode={u:.2f} clean={c:.2f}")

urls=np.array([res[str(s)]["url_encode_recall"] for s in SEEDS if str(s) in res])
cleans=np.array([res[str(s)]["clean_recall"] for s in SEEDS if str(s) in res])
if len(urls)==len(SEEDS):
    lines=["# Seed-stability of structure-GNN URL-encode robustness","",
        f"Robust config {bp} trained across {len(SEEDS)} seeds (seeding before weight init). "
        "Recall on the 3,446 malicious test queries. BERT-only baseline is deterministic at 90.34%.","",
        "| Seed | Clean recall (%) | URL-encode recall (%) |","|---|---|---|"]
    for s in SEEDS: lines.append(f"| {s} | {res[str(s)]['clean_recall']} | {res[str(s)]['url_encode_recall']} |")
    lines+=["",f"**URL-encode recall: mean {urls.mean():.2f}% +/- {urls.std(ddof=1):.2f}** (range {urls.min():.2f}-{urls.max():.2f}); "
        f"BERT-only 90.34%. Clean recall mean {cleans.mean():.2f}% +/- {cleans.std(ddof=1):.2f}.",""]
    open(os.path.join(HERE,"results","robustness_seed_stability.md"),"w",encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
