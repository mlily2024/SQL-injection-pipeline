#!/usr/bin/env python3
"""Extract train-split BERT [CLS] embeddings, chunk-cached and resumable.
Needed to measure BERT-only head training time in model_execution_times.py."""
import os, time, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertModel

HERE = os.path.dirname(os.path.abspath(__file__)); SW = os.environ.get("WORK_DIR", os.path.join(HERE, ".structure_work"))
os.makedirs(SW, exist_ok=True)
os.makedirs(SW, exist_ok=True)
MAX_SECONDS = 3000; CHUNK = 2000; SEED = 42
torch.set_num_threads(os.cpu_count() or 4)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

df = pd.read_csv(os.environ.get("DATA_CSV", os.path.join(HERE, "SQL_Injection_Dataset.csv")))
df["Query"] = df["Query"].astype(str).apply(lambda x: x.lower().strip())
train_df, _ = train_test_split(df, test_size=0.30, random_state=SEED)
queries = train_df["Query"].tolist(); N = len(queries)
n_chunks = (N + CHUNK - 1) // CHUNK
log(f"train queries: {N} -> {n_chunks} chunks")

tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").eval()
t_start = time.time()
for ci in range(n_chunks):
    cf = os.path.join(SW, f"traincls_{ci:03d}.npy")
    if os.path.exists(cf): continue
    if time.time() - t_start > MAX_SECONDS:
        log(f"time budget hit at chunk {ci}/{n_chunks}; rerun to continue"); raise SystemExit(0)
    seg = queries[ci*CHUNK:(ci+1)*CHUNK]; out = []
    with torch.no_grad():
        for i in range(0, len(seg), 32):
            enc = tok(seg[i:i+32], truncation=True, max_length=128, padding=True, return_tensors="pt")
            cls = bert(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state[:,0,:]
            out.append(cls.numpy())
    np.save(cf, np.vstack(out)); log(f"chunk {ci+1}/{n_chunks} done")

parts = [np.load(os.path.join(SW, f"traincls_{ci:03d}.npy")) for ci in range(n_chunks)]
X = np.vstack(parts); np.save(os.path.join(SW, "train_cls.npy"), X)
log(f"assembled train_cls.npy shape={X.shape}")
