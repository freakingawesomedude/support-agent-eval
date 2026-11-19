# Eval AI-агента разбора обращений клиентов

Агент получает текст обращения и возвращает JSON: `category` (payment, delivery, refund, tech_issue, other) и `escalate` (передать человеку).
Проект: golden set, две версии промпта, метрики, разбор ошибок.

## Структура
```
data/            golden.csv (эталон), candidates.csv (кандидаты)
docs/            labeling_guidelines.md, claude_code_prompts.md
prompts/         prompt_v1.txt, prompt_v2.txt, ...
src/             make_candidates, run_agent, evaluate, compare, kappa
results/         выходы прогонов и таблицы ошибок
PROMPT_LOG.md    журнал версий промптов и метрик
CLAUDE.md        правила для Claude Code
```

## Быстрый старт
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # вписать OPENROUTER_API_KEY

python src/make_candidates.py --per-category 10   # data/candidates.csv (~110 строк из Bitext)
# 1) разметить вручную category и escalate по docs/labeling_guidelines.md
# 2) добавить ~50 сложных случаев (docs/claude_code_prompts.md, промпт 3), разметить их тоже
# 3) сохранить итог как data/golden.csv (колонки: id,text,source,category,escalate,type,comment)

python src/run_agent.py --prompt prompts/prompt_v1.txt
python src/run_agent.py --prompt prompts/prompt_v2.txt
python src/evaluate.py --results results/prompt_v1__anthropic_claude-sonnet-4.5.csv
python src/compare.py results/prompt_v1__anthropic_claude-sonnet-4.5.csv results/prompt_v2__anthropic_claude-sonnet-4.5.csv
```
Slug модели проверьте на openrouter.ai/models. Для оценки цены передайте `--price-in/--price-out` ($ за 1M токенов).

## Целевые пороги (гипотеза, уточняйте)
- Escalation recall >= 0.90 (пропустить мошенничество дороже, чем лишняя эскалация)
- Escalation precision >= 0.70
- Category accuracy >= 0.85
- Invalid JSON <= 1%

## Датасет и лицензия
Тексты берутся из Bitext Customer Support (Hugging Face: bitext/Bitext-customer-support-llm-chatbot-training-dataset, лицензия CDLA-Sharing-1.0). Он синтетический, поэтому метрики на нём оптимистичнее реальных; именно для этого в golden set добавляются ручные и сложные случаи.

## Правила чистоты оценки
- Метки в golden.csv ставит человек.
- Примеры из golden set не попадают в промпты.
- Версии промптов не перезаписываются.
