#!/usr/bin/env python3
"""HttpParams dataset (Morzeux) - Step 1: cleaning, feature engineering, EDA.
Builds a clean SQLi-vs-benign dataset from payload_full.csv (attack_type in
{sqli, norm}), matching the preprocessing used on the Kaggle set, and writes an
EDA report + figures. Output: httpparams_sqli.csv (Query, Label)."""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"results")
os.makedirs(RES,exist_ok=True)
raw=pd.read_csv(os.path.join(HERE,"payload_full.csv"))
report=[]
def R(s): report.append(s); print(s)

R(f"# HttpParams (Morzeux) SQLi dataset - cleaning, feature engineering, EDA\n")
R(f"Source `payload_full.csv`: {len(raw):,} rows, columns {list(raw.columns)}.")
R(f"attack_type counts:\n\n```\n{raw['attack_type'].value_counts().to_string()}\n```\n")

# ---- filter to the SQLi task (sqli vs norm) ----
df=raw[raw["attack_type"].isin(["sqli","norm"])].copy()
R(f"**Task filter** attack_type in {{sqli, norm}}: {len(df):,} rows "
  f"({(df['attack_type']=='sqli').sum():,} sqli, {(df['attack_type']=='norm').sum():,} norm).")

# ---- cleaning ----
df["Query"]=df["payload"].astype(str)
before=len(df)
n_missing=df["Query"].isin(["", "nan", "NaN"]).sum() + df["payload"].isna().sum()
df=df[df["payload"].notna()]
df["Query"]=df["Query"].str.lower().str.strip()                    # match Kaggle preprocessing
df=df[df["Query"].str.len()>0]
R(f"\n## Cleaning")
R(f"- Lowercased + stripped whitespace (identical to the Kaggle-set preprocessing).")
R(f"- Removed {before-len(df):,} empty/missing payloads.")
# label + duplicate handling
df["Label"]=(df["attack_type"]=="sqli").astype(int)
# label-conflicting duplicates: same Query appearing as both sqli and norm
conf=df.groupby("Query")["Label"].nunique()
conflicting=set(conf[conf>1].index)
R(f"- Label-conflicting duplicates (same string labelled both sqli and norm): "
  f"{len(conflicting):,} distinct strings -> dropped (ambiguous).")
df=df[~df["Query"].isin(conflicting)]
dups=df.duplicated(subset=["Query"]).sum()
df=df.drop_duplicates(subset=["Query"]).reset_index(drop=True)
R(f"- Exact duplicate strings removed: {dups:,}.")
R(f"- **Clean dataset: {len(df):,} rows** ({(df['Label']==1).sum():,} SQLi, {(df['Label']==0).sum():,} benign; "
  f"{(df['Label']==1).mean()*100:.1f}% malicious).")

# ---- feature engineering ----
df["Query_Length"]=df["Query"].str.split().apply(len)              # word count (as in Kaggle pipeline)
df["char_len"]=df["Query"].str.len()
for ch,name in [("'","quote"),("=","eq"),("(","paren"),("%","pct"),("-","dash")]:
    df[f"n_{name}"]=df["Query"].str.count(re.escape(ch))  # literal char count
R(f"\n## Feature engineering")
R(f"- `Query_Length` (word count) and `char_len` (character count).")
R(f"- Special-character counts (quote, =, parenthesis, %, dash) as EDA descriptors.")

# ---- EDA ----
R(f"\n## Exploratory data analysis")
q=df.groupby("Label")["Query_Length"].describe()[["mean","50%","max"]]
R(f"Query length (words) by class:\n\n```\n{q.to_string()}\n```")
R(f"99th-percentile query length: {df['Query_Length'].quantile(0.99):.0f} words; "
  f"max {df['Query_Length'].max()} words, {df['char_len'].max()} chars.")
R(f"Mean special chars (SQLi vs benign): "
  f"quote {df[df.Label==1].n_quote.mean():.2f} vs {df[df.Label==0].n_quote.mean():.2f}; "
  f"= {df[df.Label==1].n_eq.mean():.2f} vs {df[df.Label==0].n_eq.mean():.2f}; "
  f"( {df[df.Label==1].n_paren.mean():.2f} vs {df[df.Label==0].n_paren.mean():.2f}.")

# figures
plt.figure(figsize=(4,3)); df["Label"].map({0:"benign",1:"SQLi"}).value_counts().plot.bar(color=["#3b6ea5","#a5322d"])
plt.title("Class distribution (HttpParams)"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(os.path.join(RES,"httpparams_01_class_distribution.png"),dpi=200); plt.close()
plt.figure(figsize=(5,3))
for lb,c,name in [(0,"#3b6ea5","benign"),(1,"#a5322d","SQLi")]:
    plt.hist(df[df.Label==lb]["Query_Length"].clip(upper=60),bins=40,alpha=0.6,color=c,label=name,density=True)
plt.title("Query length by class"); plt.xlabel("words"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RES,"httpparams_02_length_by_class.png"),dpi=200); plt.close()
plt.figure(figsize=(5,3))
for lb,c,name in [(0,"#3b6ea5","benign"),(1,"#a5322d","SQLi")]:
    plt.hist(df[df.Label==lb]["char_len"].clip(upper=200),bins=40,alpha=0.6,color=c,label=name,density=True)
plt.title("Character length by class"); plt.xlabel("chars"); plt.ylabel("density"); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(RES,"httpparams_03_charlen_by_class.png"),dpi=200); plt.close()
R(f"\nFigures: `httpparams_01_class_distribution.png`, `httpparams_02_length_by_class.png`, "
  f"`httpparams_03_charlen_by_class.png`.")

# ---- class-imbalance handling (technique + justification) ----
mal=(df["Label"]==1).mean()*100
R(f"\n## Class-imbalance handling")
R(f"The set is {100-mal:.1f}% benign / {mal:.1f}% malicious (mild imbalance). We handle it with "
  f"**cost-sensitive learning (balanced class weights in the loss)**, not resampling, and we judge "
  f"models on **imbalance-robust metrics** (malicious recall, false-positive rate, F1, MCC, balanced "
  f"accuracy) rather than raw accuracy.")
R(f"- *Technique:* `compute_class_weight('balanced')` weights the minority (SQLi) class inversely to "
  f"its frequency inside the GNN's class-weighted cross-entropy and the logistic-regression head; the "
  f"MLP head, which sklearn cannot class-weight directly, is handled at the evaluation level.")
R(f"- *Why not resampling:* SMOTE interpolates in feature space, but there is no meaningful "
  f"interpolation between two SQL query strings, and a synthetic BERT embedding between two queries "
  f"corresponds to no real query - it injects artefacts, not signal. Random oversampling merely "
  f"duplicates rows (overfitting risk); undersampling would discard about 40% of real benign data for "
  f"a mild imbalance. Class weighting keeps every real example and fixes the loss asymmetry directly.")
R(f"- *Why these metrics:* at {100-mal:.0f}/{mal:.0f} a trivial all-benign classifier already scores "
  f"~{100-mal:.0f}% accuracy, so accuracy is misleading; malicious recall (attack detection), FPR "
  f"(false alarms), F1, MCC and balanced accuracy are where imbalance actually bites, and they are "
  f"the fair basis for comparing the GNN against BERT-only.")

# ---- save clean dataset ----
out=df[["Query","Label"]].copy()
out.to_csv(os.path.join(HERE,"httpparams_sqli.csv"),index=False)
R(f"\n## Output\nClean modelling dataset written to `httpparams_sqli.csv` ({len(out):,} rows, columns Query/Label).")
open(os.path.join(RES,"httpparams_eda.md"),"w",encoding="utf-8").write("\n".join(report))
print("\n[wrote results/httpparams_eda.md + httpparams_sqli.csv]")
