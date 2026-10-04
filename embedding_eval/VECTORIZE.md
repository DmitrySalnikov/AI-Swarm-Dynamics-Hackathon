# vectorize.py — эмбеддинги рассуждений и чатов для δ(t)

Один файл, зависимости объявлены в шапке (`uv run` ставит их сам). Схема выбрана и проверена в отдельной оценке
(репозиторий `MadExplorer/AI-Village`, `embedding_eval/REPORT.md`, `results/improve_eval.md`); параметры не менять
без перепроверки.

## Схема

1. Текст режется на окна **2048** токенов DeepSeek-V3 с шагом **1024**; последнее окно выровнено по концу.
2. Окно → **Qwen3-Embedding-4B** через OpenRouter (2560 чисел), инструкция фиксирована.
3. Векторы Qwen **центрируются** по всем окнам эксперимента (вычесть среднее, длина 1).
4. **TF-IDF char 3–5** по тексту с замаскированными числами, обучается на всех окнах эксперимента.
5. Гибрид: `sim = 0.5·cos(Qwen) + 0.5·cos(TF-IDF)`.

На `set_1`/`set_2` (тестовая половина задач) эта схема против окна 1024 без центрирования:
d′ 3.28 → 4.03, AUC 0.972 → 0.979, однотипные задачи («близнецы») 0.615 → 0.742.

## Что эмбеддится

| Источник | Фрагмент | `log_id` в индексе |
|---|---|---|
| одиночные решения `**/<задача>.reasoning.txt` (`gen_runs.py`) | окна рассуждения | `solo/<run_k>/<set>/<domain>` |
| ход агента в чате `agent_<i>.jsonl` (`chat_runs.py`) | **собственные** `reasoning` + `message` этого агента в этом ходе | `chat/<set>/<kind>/run_<k>/agent_<i>/turn_<NNN>` |
| решение в merge `solve/agent_<i>.json` | `reasoning`, если пустой — `response` | `chat/<set>/merge/run_<k>/agent_<i>/solve` |

Чужие сообщения и общий текст чата не берутся: иначе у всех агентов одинаковые векторы по построению.
Прогоны `*_invalid_*` пропускаются.

## Запуск

Ключ OpenRouter — один раз в терминале (ввод скрыт, ключ попадёт в связку ключей macOS):
```
security add-generic-password -U -s openrouter-api -a "$USER" -w
```
или переменная `OPENROUTER_API_KEY`.

```
# оценка объёма и стоимости
uv run embedding_eval/vectorize.py --logs-dir embedding_eval/runs/deepseek-v4-flash --chat-dir embedding_eval/runs/chat --dry-run
# всё сразу: одиночные решения + чаты → одна общая TF-IDF и одно центрирование
uv run embedding_eval/vectorize.py --logs-dir embedding_eval/runs/deepseek-v4-flash \
    --chat-dir embedding_eval/runs/chat --out embedding_eval/vectors --budget-usd 1
```
Выход: `vectors/qwen.npy` (окна × 2560, без центрирования), `vectors/tfidf.npz`, `vectors/index.jsonl`.
В индексе для каждого окна: `log_id`, `source` (solo / chat / solve), `idx`, `tok_start`, `tok_end`, `n_tok`,
`rel_center`; для чатов ещё `run`, `kind`, `bench`, `agent`, `turn`, `round`; для solve — `task_id`, `solve_text`.
Для 80 решений и 40 прогонов чатов: ≈ 4.5 млн токенов, ≈ $0.10.

Из кода:
```python
import vectorize as V
logs, extra = V.load_chat_logs("embedding_eval/runs/chat")      # или свой {log_id: текст}
Q, meta = V.embed_logs(logs, budget_usd=1.0)                    # кэш по (лог, номер окна)
T, _ = V.tfidf_fit_transform([m["text"] for m in meta])
S = V.hybrid_sim(Q, T)          # центрирование Qwen внутри; передавайте ВСЕ окна эксперимента сразу
delta = 1 - S[np.ix_(ids, ids)].mean()   # δ для набора окон ids (векторы единичной длины)
```

## Важно

- **TF-IDF и центрирование — один раз на всех окнах эксперимента** (одиночные решения + чаты вместе).
- **Одна модель-генератор во всём эксперименте.** Одиночные решения и чаты должны быть от одной и той же модели,
  иначе δ(t) скачет на стыках из-за смены модели.
- **Однотипные задачи** (одна задача с другими данными) различаются хуже разнотипных (AUC ≈ 0.74 против 0.98).
  Не ставить в один прогон biology или music из разных наборов, а также computer_science из `set_1` и `set_3`.
- Кэш векторов — `embedding_eval/cache/` (путь меняет `VECTORIZE_CACHE`); его и `embedding_eval/vectors/`
  в git не коммитить. Лимит расходов жёсткий, журнал — `cache/spend.jsonl`.
