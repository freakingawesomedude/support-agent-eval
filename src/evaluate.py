"""Метрики одного прогона и таблица ошибок для ручного поиска причины."""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from common import CATS, load_merged, accuracy, precision, recall, boot_ci

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--golden", default="data/golden.csv")
ap.add_argument("--price-in", type=float, default=None, help="$ per 1M input tokens")
ap.add_argument("--price-out", type=float, default=None, help="$ per 1M output tokens")
a = ap.parse_args()

df = load_merged(a.golden, a.results)
name = os.path.splitext(os.path.basename(a.results))[0]
n = len(df)
print(f"\n=== {name} | n={n} ===")

# Категории
y, p = df["category"].values, df["pred_category"].values
lo, hi = boot_ci(y, p, accuracy)
print(f"\nТочность категорий: {accuracy(y, p):.3f}  (95% ДИ {lo:.3f}-{hi:.3f})")
print(classification_report(y, p, labels=CATS, zero_division=0))
print("Матрица ошибок (строки=эталон, столбцы=предсказание):")
labs = CATS + ["INVALID"]
print(pd.DataFrame(confusion_matrix(y, p, labels=labs), index=labs, columns=labs).iloc[:5])

# Эскалация
ye, pe = df["gold_escalate"].astype(int).values, df["pred_escalate_b"].astype(int).values
print("\nЭскалация (escalate = true):")
for nm, fn in [("recall", recall), ("precision", precision)]:
    lo, hi = boot_ci(ye, pe, fn)
    print(f"  {nm}: {fn(ye, pe):.3f}  (95% ДИ {lo:.3f}-{hi:.3f})")
tp = int(((ye == 1) & (pe == 1)).sum()); fn_ = int(((ye == 1) & (pe == 0)).sum())
fp = int(((ye == 0) & (pe == 1)).sum()); tn = int(((ye == 0) & (pe == 0)).sum())
print(f"  TP={tp} FN={fn_} FP={fp} TN={tn}")

# Операционные метрики
print("\nОперационные метрики:")
print(f"  доля невалидных ответов: {(~df['valid'].astype(bool)).mean():.3f}")
print(f"  задержка p50/p95: {df['latency_s'].quantile(.5):.2f}s / {df['latency_s'].quantile(.95):.2f}s")
pt, ct = df["prompt_tokens"].sum(), df["completion_tokens"].sum()
print(f"  токены in/out: {pt:.0f} / {ct:.0f}")
if a.price_in is not None and a.price_out is not None:
    cost = pt / 1e6 * a.price_in + ct / 1e6 * a.price_out
    print(f"  стоимость: ${cost:.4f} всего, ${cost / n:.5f} на запрос")

# Ошибки и причины для разбора вручную
bad = df[(df["category"] != df["pred_category"]) | (df["gold_escalate"] != df["pred_escalate_b"])].copy()
bad["error_type"] = np.where(bad["category"] != bad["pred_category"], "category",
                      np.where(bad["gold_escalate"], "escalate_FN", "escalate_FP"))
bad["cause"] = ""   # заполнить вручную: данные / промпт / гайдлайн / модель
if "comment" not in bad.columns:
    bad["comment"] = ""
cols = ["id", "text", "category", "pred_category", "gold_escalate", "pred_escalate_b", "error_type", "cause", "comment"]
out = f"results/errors_{name}.csv"
bad[cols].to_csv(out, index=False)
print(f"\nОшибок: {len(bad)} -> {out} (нужно заполнить колонку 'cause')")
