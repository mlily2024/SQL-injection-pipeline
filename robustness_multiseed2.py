#!/usr/bin/env python3
"""Multi-seed adaptive-evasion + OOD robustness (the two remaining metrics the GNN
led on in single runs). Trains the robust config across 5 seeds; measures adaptive
best-of-7 evasion rate on 400 malicious test queries and false-positive/detection
on the out-of-distribution set, vs the deterministic BERT-only baseline. Reports
mean +/- std with a seed-stability verdict. Resumable per seed."""
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
KW=["select","union","from","where","insert","update","delete","or","and"]
torch.set_num_threads(os.cpu_count() or 4)
def log(m): print(f"[{__import__('time').strftime('%H:%M:%S')}] {m}",flush=True)
def url_encode(q): return q.replace(" ","%20").replace("'","%27").replace("=","%3d")
def inline(q): return q.replace(" ","/**/")
def case_mix(q): return "".join(c.upper() if (i%2==0 and c.isalpha()) else c for i,c in enumerate(q))
def kw_split(q):
    for k in KW: q=q.replace(k,k[:2]+"/**/"+k[2:])
    return q
def mysql(q):
    for k in KW: q=q.replace(k,f"/*!50000{k}*/")
    return q
def double_url(q): return q.replace("%","%25").replace(" ","%2520").replace("'","%2527")
def combined(q): return case_mix(inline(q))
LIB=[url_encode,inline,case_mix,kw_split,mysql,double_url,combined]

MALICIOUS=["' or 1=1--","' or '1'='1","admin'--","' or 1=1#","') or ('1'='1","' union select null,username,password from users--","1 union all select 1,2,3,4,5,6,7,8,9,10--","' union select @@version--","-1' union select database(),user()--","1' and 1=convert(int,(select @@version))--","' and extractvalue(1,concat(0x7e,version()))--","' and updatexml(1,concat(0x7e,(select version())),1)--","1' waitfor delay '0:0:5'--","' or sleep(5)#","1 and (select sleep(5))","'; drop table users;--","1'; exec xp_cmdshell('whoami')--","1); drop table logs;--","' or username like '%admin%'--","1 procedure analyse(extractvalue(1,concat(0x3a,version())),1)","' union select 1,2,load_file('/etc/passwd')--","' or 1=1 limit 1 offset 0--","1' group by columnnames having 1=1--","' unioN sELeCt 1,2,3--","1/**/union/**/select/**/1,2,3--","%27%20or%201%3d1--","' or 0x50=0x50--","1' or 'a'='a","'||(select user())||'","'+(select top 1 name from sysobjects)+'","1' and substring(@@version,1,1)='5'--","' or exists(select * from users)--","'; begin declare @x int; end--","1 or 1=1","' having 1=1--"]
BENIGN=["select * from products where id = 42","SELECT name, price FROM catalogue WHERE category = 'books'","update users set last_login = now() where user_id = 7","insert into orders (customer, total) values ('alice', 29.99)","delete from cart where session = 'abc123'","select count(*) from visits where day = '2024-05-01'","john.doe@example.com","my order number is 12345","search: blue running shoes size 10","the quick brown fox jumps over the lazy dog","please reset my password","2024-05-14T10:30:00","SELECT first_name, last_name FROM employees ORDER BY hire_date DESC","select p.name, c.title from products p join categories c on p.cat_id = c.id","update inventory set qty = qty - 1 where sku = 'ABC-123'","hello world","invoice #INV-2024-0042","contact us at support@shop.example.com","London, United Kingdom","temperature is 21 degrees","select * from bookings where checkin between '2024-06-01' and '2024-06-07'","add 3 items to the basket","the meeting is at 3pm on tuesday","insert into feedback (rating, comment) values (5, 'great service')","user profile updated successfully","select avg(price) from listings where region = 'north'","my name is o'brien","select id, email from subscribers where active = true","product code: XZ-9981","order confirmed, thank you for shopping","grant select on reports to analyst_role","show me my recent transactions","select title from articles where published = 1 and author = 'smith'","café latte and a croissant"]

df=pd.read_csv(os.path.join(HERE,"SQL_Injection_Dataset.csv")); df["Query"]=df["Query"].astype(str).apply(lambda x:x.lower().strip())
train_df,test_df=train_test_split(df,test_size=0.30,random_state=42)
tr_df,val_df=train_test_split(train_df,test_size=0.15,random_state=42,stratify=train_df["Label"])
rng=np.random.RandomState(0); mal=[q for q,l in zip(test_df["Query"],test_df["Label"]) if l==1]
samp=[mal[i] for i in rng.choice(len(mal),400,replace=False)]
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
def embed(queries):
    order=np.argsort([len(q) for q in queries],kind="stable"); cls=np.zeros((len(queries),768),np.float32); graphs=[None]*len(queries)
    with torch.no_grad():
        for k in range(0,len(order),32):
            idx=order[k:k+32]; enc=tok([queries[i] for i in idx],truncation=True,max_length=MAX_LEN,padding=True,return_offsets_mapping=True,return_tensors="pt")
            hs=bert(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"]).last_hidden_state.numpy(); offs=enc["offset_mapping"].numpy()
            for bi,qi in enumerate(idx):
                cls[qi]=hs[bi][0]; toks=sql_tokens(queries[qi])
                if not toks: graphs[qi]=Data(x=torch.tensor(hs[bi][:1]),edge_index=torch.empty((2,0),dtype=torch.long),y=torch.tensor([0])); continue
                feats=[]; so=offs[bi]
                for (val,a,bch) in toks:
                    m=[j for j in range(len(so)) if not(so[j][0]==0 and so[j][1]==0) and so[j][0]<bch and so[j][1]>a]
                    feats.append(hs[bi][m].mean(axis=0) if m else hs[bi][0])
                e=structure_edges(toks)
                graphs[qi]=Data(x=torch.tensor(np.stack(feats),dtype=torch.float),edge_index=(torch.tensor(e,dtype=torch.long).t().contiguous() if e else torch.empty((2,0),dtype=torch.long)),y=torch.tensor([0]))
    return cls,graphs
def cache(name,queries):
    p=os.path.join(SW,f"ms2_{name}.pkl")
    if os.path.exists(p): return pickle.load(open(p,"rb"))
    log(f"embed {name} ({len(queries)})"); r=embed(queries); pickle.dump(r,open(p,"wb")); return r
# adaptive: 400 queries x 7 transforms (flattened, row-major: query*7+t)
adv_q=[LIB[t](q) for q in samp for t in range(7)]
adv_cls,adv_g=cache("adv",adv_q)
ood_cls,ood_g=cache("ood",MALICIOUS+BENIGN)
ood_y=np.array([1]*len(MALICIOUS)+[0]*len(BENIGN))

def load_g(tag,frame):
    C=3000; n=(len(frame)+C-1)//C; gl=[]
    for c in range(n):
        with open(os.path.join(SW,f"sg_{tag}_{c:03d}.pkl"),"rb") as f: gl+=pickle.load(f)
    return gl
tg,vg=load_g("train",tr_df),load_g("val",val_df); bp=json.load(open(os.path.join(SW,"best.json")))
cw=compute_class_weight("balanced",classes=np.array([0,1]),y=np.array([int(g.y.item()) for g in tg]))
crit=nn.CrossEntropyLoss(weight=torch.tensor(cw,dtype=torch.float))
mlp=pickle.load(open(os.path.join(SW,"mlp_model.pkl"),"rb"))
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
def gnn_pred(m,graphs):
    P=[]
    with torch.no_grad():
        for d in GeoLoader(graphs,batch_size=128): P.append(m(d.x,d.edge_index,d.batch).argmax(1))
    return torch.cat(P).numpy()
def adaptive_evasion(pred):  # pred over 400*7, evaded if any transform -> benign(0)
    p=pred.reshape(400,7); return float((p==0).any(1).mean()*100)

# BERT-only fixed
b_adv=adaptive_evasion(mlp.predict(adv_cls)); b_oodp=mlp.predict(ood_cls)
BERT={"adaptive_evasion":round(b_adv,2),"ood_fpr":round((b_oodp[ood_y==0]==1).mean()*100,2),"ood_recall":round((b_oodp[ood_y==1]==1).mean()*100,2)}
log(f"BERT-only: {BERT}")
RES=os.path.join(SW,"multiseed2.json"); res=json.load(open(RES)) if os.path.exists(RES) else {}
for s in SEEDS:
    if str(s) in res: continue
    m=train_seed(s); ap=gnn_pred(m,adv_g); op=gnn_pred(m,ood_g)
    r={"adaptive_evasion":round(adaptive_evasion(ap),2),"ood_fpr":round((op[ood_y==0]==1).mean()*100,2),"ood_recall":round((op[ood_y==1]==1).mean()*100,2)}
    res[str(s)]=r; json.dump(res,open(RES,"w"),indent=2); log(f"seed {s}: {r}")
if len(res)==len(SEEDS):
    def ms(k): v=np.array([res[str(s)][k] for s in SEEDS]); return v.mean(),v.std(ddof=1)
    lines=["# Multi-seed adaptive-evasion + OOD: structure-GNN vs BERT-only","",
      "5 seeds, mean +/- std. Lower is better for evasion & FPR; higher for recall.","",
      "| Metric | Structure-GNN | BERT-only | GNN better & seed-stable? |","|---|---|---|---|"]
    for k,lb in [("adaptive_evasion",True),("ood_fpr",True),("ood_recall",False)]:
        gm,gs=ms(k); b=BERT[k]; better=(gm<b) if lb else (gm>b); stable=abs(gm-b)>2*gs
        lines.append(f"| {k} | {gm:.2f} +/- {gs:.2f} | {b} | "+("YES" if (better and stable) else ("no (within noise)" if better else "no (BERT-only better)"))+" |")
    open(os.path.join(HERE,"results","robustness_multiseed2.md"),"w",encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))
