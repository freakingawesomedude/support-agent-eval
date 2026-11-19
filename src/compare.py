"""Парный бутстрап: действительно ли версия B лучше версии A на том же golden set?"""
import argparse
import numpy as np
from common import load_merged

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--golden", default="data/golden.csv")
ap.add_argument("--n", type=int, default=5000)
args = ap.parse_args()

A = load_merged(args.golden, args.a).set_index("id")
B = load_merged(args.golden, args.b).set_index("id")
ids = A.index.intersection(B.index)
A, B = A.loc[ids], B.loc[ids]
rng = np.random.default_rng(0)


def report(label, ya, yb):
    d = []
    for _ in range(args.n):
        i = rng.integers(0, len(ya), len(ya))
        d.append(yb[i].mean() - ya[i].mean())
    lo, hi = np.percentile(d, [2.5, 97.5])
    sig = "ЗНАЧИМО" if lo > 0 or hi < 0 else "не значимо (может быть шумом)"
    print(f"{label}: A={ya.mean():.3f} B={yb.mean():.3f} diff={yb.mean()-ya.mean():+.3f} "
          f"95% ДИ [{lo:+.3f}, {hi:+.3f}] -> {sig}")


print(f"n={len(ids)}")
report("category accuracy", (A["category"] == A["pred_category"]).values.astype(float),
       (B["category"] == B["pred_category"]).values.astype(float))
pos = A["gold_escalate"].values
report("escalation recall", A["pred_escalate_b"].values[pos].astype(float),
       B["pred_escalate_b"].values[pos].astype(float))
report("escalation correct (all rows)", (A["gold_escalate"] == A["pred_escalate_b"]).values.astype(float),
       (B["gold_escalate"] == B["pred_escalate_b"]).values.astype(float))
