# utils.py
import numpy as np
import os, glob
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

class FallbackModel:
    def predict_proba(self, X):
        # simple neutral probability ~0.6 for demo
        p = np.full((X.shape[0], 2), 0.0)
        p[:,1] = 0.6
        p[:,0] = 0.4
        return p

def load_baseline(path="models/baseline.pkl"):
    p = Path(path)
    if p.exists():
        bundle = joblib.load(p)
        return bundle["model"], bundle["feats"], bundle.get("val_auc", None), bundle.get("val_acc", None)
    return FallbackModel(), ["seq_index","prev_correct_user","rolling_acc_user","prev_correct_skill","rolling_acc_skill","delta_t"], None, None

def load_corpus(folder="content"):
    files = sorted(glob.glob(os.path.join(folder, "*.txt")))
    docs = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            txt = f.read().strip()
            if txt:
                docs.append({"id": os.path.basename(fp), "text": txt})
    return docs

class Retriever:
    def __init__(self, docs):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.docs = docs
        if docs:
            self.mat = np.vstack([self.embedder.encode(d["text"], normalize_embeddings=True) for d in docs])
        else:
            self.mat = np.zeros((0,384))

    def query(self, q, k=3):
        if self.mat.shape[0] == 0:
            return []
        qv = self.embedder.encode(q, normalize_embeddings=True)
        sims = self.mat @ qv
        idx = sims.argsort()[-k:][::-1]
        return [self.docs[i] for i in idx]
