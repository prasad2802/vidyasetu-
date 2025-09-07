# scripts/train_baseline.py
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

CLEAN = Path("data/assistments_clean.csv")
SPLITS = Path("data/splits.pkl")
OUT = Path("models/baseline.pkl")

FEATS = ["seq_index","prev_correct_user","rolling_acc_user",
         "prev_correct_skill","rolling_acc_skill","delta_t"]

def train():
    df = pd.read_csv(CLEAN)

    # >>> IMPORTANT: ensure consistent types for slicing
    df["user_id"] = df["user_id"].astype(str)          # <--- add
    df["skill_id"] = df["skill_id"].astype(str)        # (safe)
    df["item_id"]  = df["item_id"].astype(str)         # (safe)
    # <<<

    splits = pd.read_pickle(SPLITS)
    train_users = set(map(str, splits["train_users"]))  # also enforce str
    val_users   = set(map(str, splits["val_users"]))

    def slice_users(users):
        return df[df["user_id"].isin(users)].copy()

    train_df = slice_users(train_users)
    val_df   = slice_users(val_users)

    # Guardrail: fail early if empty
    if len(train_df) == 0:
        raise RuntimeError("Training split has 0 rows after slicing. "
                           "Check user_id types or that splits.pkl matches the CSV.")

    Xtr, ytr = train_df[FEATS].values, train_df["correct"].values
    Xv,  yv  = val_df[FEATS].values,  val_df["correct"].values

    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(Xtr, ytr)

    pv = clf.predict_proba(Xv)[:,1]
    auc = roc_auc_score(yv, pv)
    acc = accuracy_score(yv, (pv>=0.5).astype(int))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feats": FEATS, "val_auc": auc, "val_acc": acc}, OUT)
    print(f"Saved baseline to {OUT}. Val AUC={auc:.3f}, Acc={acc:.3f}")

if __name__ == "__main__":
    if not CLEAN.exists() or not SPLITS.exists():
        raise SystemExit("Run preprocessing first: python scripts/prep_assistments.py")
    train()
