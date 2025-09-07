# scripts/prep_assistments.py
import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/assistments.csv")
OUT_CLEAN = Path("data/assistments_clean.csv")
OUT_SPLITS = Path("data/splits.pkl")

def load_raw(path: Path) -> pd.DataFrame:
    # Load
    df = pd.read_csv(path)

    # ---- Standardize core IDs / names if present ----
    # Prefer existing if already present
    if "user_id" not in df.columns:
        for c in ["Anon Student Id", "student_id"]:
            if c in df.columns:
                df["user_id"] = df[c].astype(str)
                break

    # item_id from any of these
    if "item_id" not in df.columns:
        for c in ["problem_id", "assistment_id", "Problem Name", "question_id"]:
            if c in df.columns:
                df["item_id"] = df[c].astype(str)
                break

    # correct -> 0/1
    if "correct" in df.columns:
        df["correct"] = df["correct"].apply(
            lambda x: 1 if str(x).strip().lower() in {"1","true","t","yes","y"} or x==1 else 0
        )
    else:
        raise ValueError("No 'correct' column found.")

    # ---- Map skill_id from list columns or single name ----
    if "skill_id" not in df.columns:
        if "list_skill_ids" in df.columns:
            df["skill_id"] = df["list_skill_ids"].astype(str).apply(
                lambda x: (x.split(",")[0].strip() if isinstance(x, str) and x.strip() else "unknown")
            )
        elif "list_skills" in df.columns:
            df["skill_id"] = df["list_skills"].astype(str).apply(
                lambda x: (x.split(",")[0].strip() if isinstance(x, str) and x.strip() else "unknown")
            )
        elif "KC(Default)" in df.columns:
            df["skill_id"] = df["KC(Default)"].astype(str)
        else:
            # fall back to a single global skill if nothing exists
            df["skill_id"] = "unknown"

    # ---- Create a usable timestamp if missing ----
    if "timestamp" not in df.columns:
        # We’ll order events per user by best-available sequence key,
        # then synthesize a monotonic timestamp.
        order_keys = []
        for c in ["timestamp", "sequence_id", "order_id", "position", "base_sequence_id"]:
            if c in df.columns:
                order_keys.append(c)

        if not order_keys:
            # Use the file order as a last resort
            df["_row_idx"] = np.arange(len(df))
            order_keys = ["_row_idx"]

        # Ensure keys are numeric where needed
        for k in order_keys:
            if k != "timestamp":
                df[k] = pd.to_numeric(df[k], errors="coerce")

        # Sort per user
        df["user_id"] = df["user_id"].astype(str)
        df = df.sort_values(["user_id"] + order_keys).reset_index(drop=True)

        # If we have response time, accumulate; else just 1 sec increments
        if "ms_first_response_time" in df.columns:
            # per user cumulative sum of ms, convert to seconds
            df["__dt"] = df.groupby("user_id")["ms_first_response_time"].apply(
                lambda s: s.fillna(0).astype(float).cumsum() / 1000.0
            ).values
        else:
            # per-user 1-second steps
            df["__dt"] = df.groupby("user_id").cumcount().astype(float)

        # Create a pseudo timestamp (epoch + delta seconds)
        df["timestamp"] = pd.to_datetime(1577836800 + df["__dt"], unit="s")  # 2020-01-01 base
        df.drop(columns=[c for c in ["_row_idx","__dt"] if c in df.columns], inplace=True)
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ---- Final sanity on required columns ----
    needed = ["user_id","item_id","skill_id","correct","timestamp"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after mapping: {missing}. Present: {df.columns.tolist()}")

    # Types
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df["skill_id"] = df["skill_id"].astype(str)

    # Sort per user by time
    df = df.sort_values(["user_id","timestamp"]).reset_index(drop=True)

    # Keep users with at least 10 interactions (tunable)
    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= 10].index
    df = df[df["user_id"].isin(keep_users)].copy()

    # Rolling features
    df["seq_index"] = df.groupby("user_id").cumcount()
    df["prev_correct_user"] = df.groupby("user_id")["correct"].shift(1).fillna(0)
    df["rolling_acc_user"] = (
        df.groupby("user_id")["correct"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0.5)
    )

    df["prev_correct_skill"] = df.groupby(["user_id","skill_id"])["correct"].shift(1).fillna(0)
    df["rolling_acc_skill"] = (
        df.groupby(["user_id","skill_id"])["correct"]
        .transform(lambda s: s.shift(1).expanding().mean())
        .fillna(0.5)
    )

    # Time since last interaction (cap to 6h)
    dt = df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
    df["delta_t"] = np.clip(dt.fillna(0), 0, 6*3600)

    # ---- Train/val/test split by user (70/15/15) ----
    users = df["user_id"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(users)
    n = len(users)
    train_u = set(users[: int(0.7*n)])
    val_u   = set(users[int(0.7*n): int(0.85*n)])
    test_u  = set(users[int(0.85*n):])

    df.to_csv(OUT_CLEAN, index=False)
    splits = {"train_users": list(train_u), "val_users": list(val_u), "test_users": list(test_u)}
    pd.to_pickle(splits, OUT_SPLITS)

    print(f"Saved cleaned data to {OUT_CLEAN} and splits to {OUT_SPLITS}. "
          f"Users: train={len(train_u)}, val={len(val_u)}, test={len(test_u)}")
    return df

if __name__ == "__main__":
    if not RAW.exists():
        raise SystemExit(f"Put your raw ASSISTments CSV at {RAW.resolve()}")
    load_raw(RAW)
