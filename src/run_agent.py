"""Запуск агента (промпт + LLM через OpenRouter) на golden set."""
import argparse, hashlib, json, os, re, time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from common import CATS

load_dotenv()


def parse_output(text):
    """Возвращает (category, escalate, valid) — кортеж со значениями категории, эскалации и флага валидности."""
    if not text:
        return None, None, False
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        d = json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            d = None
    if not isinstance(d, dict):
        return None, None, False
    cat, esc = d.get("category"), d.get("escalate")
    valid = cat in CATS and isinstance(esc, bool)
    return (cat if cat in CATS else None), (esc if isinstance(esc, bool) else None), valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.5")  # проверить слаг модели на openrouter.ai/models
    ap.add_argument("--data", default="data/golden.csv")
    ap.add_argument("--name", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rps", type=float, default=0.33, help="запросов в секунду (новые аккаунты ограничены)")
    a = ap.parse_args()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    system = open(a.prompt, encoding="utf-8").read()
    df = pd.read_csv(a.data)
    if a.limit:
        df = df.head(a.limit)
    name = a.name or f"{os.path.splitext(os.path.basename(a.prompt))[0]}__{a.model.replace('/', '_')}"

    rows = []
    last_call = 0.0
    for i, r in df.iterrows():
        # соблюдать лимит запросов между вызовами (новому аккаунту OpenRouter: 20 req/min)
        wait = 1.0 / a.rps - (time.time() - last_call)
        if wait > 0:
            time.sleep(wait)
        text = "" if pd.isna(r["text"]) else str(r["text"])
        raw, usage, t0 = None, None, time.time()
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model=a.model, temperature=0, max_tokens=100,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": f'Message: "{text}"'}])
                raw, usage = resp.choices[0].message.content, resp.usage
                break
            except Exception as e:
                print("повторная попытка", r["id"], e); time.sleep(2 * (attempt + 1))
        cat, esc, valid = parse_output(raw)
        last_call = time.time()
        rows.append({"id": r["id"], "pred_category": cat, "pred_escalate": esc, "valid": valid,
                     "latency_s": round(time.time() - t0, 3),
                     "prompt_tokens": getattr(usage, "prompt_tokens", None),
                     "completion_tokens": getattr(usage, "completion_tokens", None),
                     "raw": raw})
        print(f"{i+1}/{len(df)} {r['id']} -> {cat} {esc}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(f"results/{name}.csv", index=False)
    meta = {"name": name, "model": a.model, "prompt_file": a.prompt,
            "prompt_sha": hashlib.sha1(system.encode()).hexdigest()[:10], "n": len(rows)}
    json.dump(meta, open(f"results/{name}.meta.json", "w"), indent=2)
    print("Сохранено", f"results/{name}.csv")


if __name__ == "__main__":
    main()
