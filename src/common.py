import numpy as np
import pandas as pd

CATS = ["payment", "delivery", "refund", "tech_issue", "other"]


def to_bool(s):
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def load_merged(golden_path, results_path):
    gold = pd.read_csv(golden_path)
    res = pd.read_csv(results_path)
    df = gold.merge(res, on="id", how="inner", suffixes=("", "_res"))
    df["gold_escalate"] = to_bool(df["escalate"])
    df["pred_escalate_b"] = to_bool(df["pred_escalate"].fillna(False))  # невалидный ответ -> без эскалации (пропуск)
    df["pred_category"] = df["pred_category"].fillna("INVALID")
    return df


def accuracy(y, p):
    return float(np.mean(y == p))


def precision(y, p):
    tp = np.sum((y == 1) & (p == 1)); fp = np.sum((y == 0) & (p == 1))
    return float(tp / (tp + fp)) if (tp + fp) else float("nan")


def recall(y, p):
    tp = np.sum((y == 1) & (p == 1)); fn = np.sum((y == 1) & (p == 0))
    return float(tp / (tp + fn)) if (tp + fn) else float("nan")


def boot_ci(y, p, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(fn(y[i], p[i]))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
