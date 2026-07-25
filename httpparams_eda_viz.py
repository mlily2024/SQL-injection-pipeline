#!/usr/bin/env python3
"""HttpParams EDA - paper-quality visualisations. Produces figures for the paper:
class balance, query-length distributions, SQL-syntax marker frequencies, top SQL
keywords by class, and word clouds (SQLi vs benign). Reads httpparams_sqli.csv."""
import os, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"results")
BEN,MAL="#3b6ea5","#a5322d"
d=pd.read_csv(os.path.join(HERE,"httpparams_sqli.csv"))
d["Query"]=d["Query"].astype(str)
d["wlen"]=d["Query"].str.split().apply(len); d["clen"]=d["Query"].str.len()
def save(name): plt.tight_layout(); plt.savefig(os.path.join(RES,name),dpi=200,bbox_inches="tight"); plt.close(); print("wrote",name)

# 1) class balance
plt.figure(figsize=(4,3.2)); vc=d["Label"].value_counts().sort_index()
bars=plt.bar(["Benign","SQLi"],vc.values,color=[BEN,MAL])
for b,v in zip(bars,vc.values): plt.text(b.get_x()+b.get_width()/2,v,f"{v:,}\n({v/len(d)*100:.1f}%)",ha="center",va="bottom",fontsize=9)
plt.title("HttpParams: class distribution"); plt.ylabel("queries"); plt.ylim(0,vc.max()*1.18); save("httpparams_eda_class.png")

# 2) query length: box + hist
fig,ax=plt.subplots(1,2,figsize=(9,3.2))
ax[0].boxplot([d[d.Label==0].wlen,d[d.Label==1].wlen],labels=["Benign","SQLi"],showfliers=False,
              patch_artist=True,boxprops=dict(facecolor="#dfe8f3"))
ax[0].set_title("Query length (words)"); ax[0].set_ylabel("words")
for lb,c,n in [(0,BEN,"benign"),(1,MAL,"SQLi")]:
    ax[1].hist(d[d.Label==lb].wlen.clip(upper=30),bins=30,alpha=0.6,color=c,label=n,density=True)
ax[1].set_title("Length distribution"); ax[1].set_xlabel("words"); ax[1].set_ylabel("density"); ax[1].legend()
save("httpparams_eda_length.png")

# 3) SQL-syntax markers by class
marks={"quote '":r"'","equals =":r"=","paren (":r"\(","semicolon ;":r";",
       "comment --":r"--","comment /*":r"/\*","url % ":r"%","keyword OR":r"\bor\b","keyword UNION":r"\bunion\b"}
rows=[]
for name,pat in marks.items():
    rows.append((name, d[d.Label==0].Query.str.count(pat).mean(), d[d.Label==1].Query.str.count(pat).mean()))
mk=pd.DataFrame(rows,columns=["marker","benign","sqli"])
plt.figure(figsize=(7,3.4)); x=np.arange(len(mk)); w=0.4
plt.bar(x-w/2,mk.benign,w,label="Benign",color=BEN); plt.bar(x+w/2,mk.sqli,w,label="SQLi",color=MAL)
plt.xticks(x,mk.marker,rotation=35,ha="right",fontsize=8); plt.ylabel("mean count / query")
plt.title("SQL-syntax marker frequency by class"); plt.legend(); save("httpparams_eda_markers.png")

# 4) top SQL keywords by class
KW=["select","union","or","and","from","where","insert","update","delete","drop","null","exec","concat","chr","count","case","when","having"]
def kwfreq(sub):
    txt=" ".join(sub.Query.tolist()); return {k:len(re.findall(rf"\b{k}\b",txt)) for k in KW}
fs=kwfreq(d[d.Label==1]); fb=kwfreq(d[d.Label==0])
top=sorted(KW,key=lambda k:-fs[k])[:10]
plt.figure(figsize=(7,3.4)); x=np.arange(len(top)); w=0.4
plt.bar(x-w/2,[fb[k] for k in top],w,label="Benign",color=BEN); plt.bar(x+w/2,[fs[k] for k in top],w,label="SQLi",color=MAL)
plt.xticks(x,top,rotation=35,ha="right"); plt.ylabel("occurrences"); plt.title("Top SQL keywords by class"); plt.legend()
save("httpparams_eda_keywords.png")

# 5) word clouds
try:
    from wordcloud import WordCloud
    fig,ax=plt.subplots(1,2,figsize=(9,3.6))
    for i,(lb,n) in enumerate([(1,"SQLi"),(0,"Benign")]):
        wc=WordCloud(width=500,height=300,background_color="white",colormap="Reds" if lb else "Blues",
                     max_words=80).generate(" ".join(d[d.Label==lb].Query.tolist())[:400000])
        ax[i].imshow(wc); ax[i].axis("off"); ax[i].set_title(f"{n} payloads")
    save("httpparams_eda_wordcloud.png")
except Exception as e:
    print("wordcloud skipped:",e)

print("EDA visualisations complete.")
