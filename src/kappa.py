"""Согласованность аннотаторов: сравнение первой и повторной разметки одних и тех же строк."""
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ap = argparse.ArgumentParser()
ap.add_argument("first"); ap.add_argument("second")
a = ap.parse_args()
x, y = pd.read_csv(a.first), pd.read_csv(a.second)
m = x.merge(y, on="id", suffixes=("_1", "_2"))
for col in ["category", "escalate"]:
    k = cohen_kappa_score(m[f"{col}_1"].astype(str), m[f"{col}_2"].astype(str))
    agree = (m[f"{col}_1"].astype(str) == m[f"{col}_2"].astype(str)).mean()
    print(f"{col}: kappa={k:.3f}, raw agreement={agree:.3f}, n={len(m)}")
    print(m[m[f"{col}_1"].astype(str) != m[f"{col}_2"].astype(str)][["id", f"{col}_1", f"{col}_2"]].to_string(index=False))
print("\nСправочник по каппе: <0.4 - слабое, 0.4-0.6 - умеренное, 0.6-0.8 - хорошее, >0.8 - очень хорошее. Низкая каппа - править гайдлайн.")
