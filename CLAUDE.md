# Проект: eval AI-агента разбора обращений клиентов

## Цель
Агент: текст обращения -> JSON {"category": ..., "escalate": ...}. Проект строит golden set, прогоняет версии промпта и считает метрики.

## Структура
- data/golden.csv: эталон (человеческая разметка). ЕГО МЕТКИ (category, escalate) НЕ МЕНЯТЬ без явной просьбы.
- data/candidates.csv: кандидаты для разметки
- docs/labeling_guidelines.md: источник правды по правилам разметки
- prompts/: версии промптов (prompt_vN.txt). Старые версии не перезаписывать, создавать новую.
- src/: скрипты (make_candidates, run_agent, evaluate, compare, kappa)
- results/: выходы прогонов
- PROMPT_LOG.md: журнал; после каждого прогона добавляй строку

## Жёсткие правила
1. Никогда не копируй тексты из data/golden.csv в prompts/ (утечка golden set).
2. Не придумывай метки "за человека": метки в golden.csv ставит человек. Свои предложения клади в отдельные колонки (pre_label_*) или отдельные файлы.
3. Не печатай и не коммить API-ключ. Ключ только в .env (он в .gitignore).
4. Перед правкой промпта: скажи гипотезу в одной строке. После прогона: покажи метрики до/после и доверительный интервал (src/compare.py).
5. Метки и тексты, сгенерированные LLM, помечай source=llm_draft.

## Команды
python src/run_agent.py --prompt prompts/prompt_v1.txt --model anthropic/claude-sonnet-4.5
python src/evaluate.py --results results/<run>.csv
python src/compare.py results/<runA>.csv results/<runB>.csv
