"""Скачать датасет Bitext и сэмплировать стратифицированные кандидаты для golden set.
Результат: data/candidates.csv (разметка вручную, затем сохранить как data/golden.csv).
"""
import argparse
import pandas as pd

URL = ("https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/"
       "resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")

# ТОЛЬКО ПРЕДЛОЖЕНИЕ. Окончательные метки ставит человек по docs/labeling_guidelines.md и по собственной оценке.
MAP = {"ACCOUNT": "tech_issue", "CANCEL": "refund", "CONTACT": "other", "DELIVERY": "delivery",
       "FEEDBACK": "other", "INVOICE": "payment", "ORDER": "delivery", "PAYMENT": "payment",
       "REFUND": "refund", "SHIPPING": "delivery", "SUBSCRIPTION": "other"}

ap = argparse.ArgumentParser()
ap.add_argument("--per-category", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = pd.read_csv(URL)
print("Колонки:", list(df.columns), "строк:", len(df))
parts = []
for cat, g in df.groupby("category"):
    noisy = g[g["flags"].astype(str).str.contains("Z")]       # тексты с опечатками
    clean = g[~g.index.isin(noisy.index)]
    k_noisy = min(len(noisy), max(1, a.per_category // 4))
    parts.append(noisy.sample(k_noisy, random_state=a.seed))
    parts.append(clean.sample(min(len(clean), a.per_category - k_noisy), random_state=a.seed))
out = pd.concat(parts).sample(frac=1, random_state=a.seed).reset_index(drop=True)
res = pd.DataFrame({
    "id": [f"b{i:03d}" for i in range(len(out))],
    "text": out["instruction"],
    "source": "bitext",
    "bitext_category": out["category"],
    "bitext_intent": out["intent"],
    "suggested_category": out["category"].map(MAP).fillna("other"),
    "category": "",      # <- заполнить вручную
    "escalate": "",      # <- заполнить вручную (true/false)
    "type": "",          # typical / rare / edge / negative / noisy
    "comment": "",
})
res.to_csv("data/candidates.csv", index=False)
print(f"Сохранено data/candidates.csv: {len(res)} строк")
