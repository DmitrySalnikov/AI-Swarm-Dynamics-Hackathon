# AI-Swarm-Dynamics-Hackathon

https://swarmchasing.com/

Бенчмарк для экспериментов с роем агентов: четыре набора по 10 вопросов.
в среднем 10–20 тыс. токенов), `set_3` лёгкий (2–5 тыс. токенов). `set_4` целиком математический:
10 разных разделов, 1–5 тыс. токенов. Все вопросы решаемы моделью среднего уровня, ответ короткий и проверяется автоматически.

> **Не публиковать.** Часть вопросов взята из GPQA, HLE и LAB-Bench, авторы которых просят
> не выкладывать их открытым текстом и не использовать в обучении. В таких вопросах сохранено
> поле `canary`. Репозиторий должен оставаться приватным.

## Структура

```
bench/set_{1,2,3,4}.jsonl                вопросы
results/<модель>/<set>/<domain>.json     ответ модели, токены, вердикт
results/<модель>/<set>/<domain>.reasoning.txt   рассуждение модели
results/<модель>/summary.md              сводная таблица
scripts/run.py                           прогон модели на наборах
```

## Наборы

`linguistics`, `physics`, `biology`, `chemistry`, `classics`, `economics`, `music`.

|---|---|---|---|
| mathematics | вероятность, 12 букв в пары (AIME 2025 #7) | четырёхугольник из центров описанных окружностей (HMMT 2025 #26) | числа из цифр 1–8, делящиеся на 22 (AIME 2025 #5) |
| finance | пенсионный аннуитет (TheoremQA) | PE-коэффициент (TheoremQA) | NPV проекта (собственный) |
| linguistics | формы глаголов шинка (Linguini) | сопоставление слов зуни (Linguini) | числительные выдуманного языка (собственный) |
| physics | масса нити лампы накаливания (OlympiadBench) | олимпиадная задача #900 (OlympiadBench) | парашютист с квадратичным сопротивлением (TheoremQA) |
| music | чистый строй, «Hänschen klein» (собственный) | чистый строй, вторая мелодия (собственный) | чистый строй, 13 нот (собственный) |

Вопросы GPQA заданы открыто, без вариантов ответа: ответ — число, угадать нельзя.
Музыкальные вопросы составлены нами (мелодия и таблица интервалов даны в условии), эталон
трудные вопросы из GPQA и Linguini модель не успевала решить за 5000 токенов.

### set_4: математика

| Раздел | Задача | Источник |
|---|---|---|
| geometry | расстояние между центром описанной окружности и ортоцентром | собственный |
| linear_algebra | определитель целочисленной матрицы 5×5 | собственный |
| calculus | несобственный интеграл ∫₀^∞ x²e⁻ˣ sin x dx | собственный |
| differential_equations | y″ + 2y′ + 5y = 10 cos x, найти y(π) | собственный |
| series | двойной ряд | TheoremQA |
| number_theory | делители 9!, оканчивающиеся на 1 | HMMT Feb 2025 #1 |
| combinatorics | раскраска отрезков сетки 2×2 | AIME 2025 #18 |
| probability | случайное подмножество делителей 2025 | AIME 2025 #22 |
| algebra | произведение логарифмов | AIME 2025 #19 |
| complex_numbers | система с модулями комплексных чисел | AIME 2025 #8 |

Эталоны собственных задач проверены независимо: подстановкой в уравнение, численным
интегрированием, второй формулой.

### Источники

| Вопрос | Бенчмарк | ID в источнике |
|---|---|---|
| `set_3/computer_science` | собственный | — |
| `set_3/mathematics` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | задача 5 |
| `set_3/finance` | собственный | — |
| `set_3/linguistics` | собственный | — |
| `set_3/physics` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | вопрос «A parachutist with mass m=80 kg…» |
| `set_3/biology` | собственный | — |
| `set_3/chemistry` | собственный | — |
| `set_3/classics` | собственный | — |
| `set_3/economics` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | вопрос «An investor has utility function…» |
| `set_3/music` | собственный | — |
| `set_4/geometry` | собственный | — |
| `set_4/linear_algebra` | собственный | — |
| `set_4/calculus` | собственный | — |
| `set_4/differential_equations` | собственный | — |
| `set_4/series` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | вопрос «Sum the series …» |
| `set_4/number_theory` | [HMMT February 2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) | задача 1 |
| `set_4/combinatorics` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | задача 18 |
| `set_4/probability` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | задача 22 |
| `set_4/algebra` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | задача 19 |
| `set_4/complex_numbers` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | задача 8 |

Linguini собран из задач Международной олимпиады по лингвистике (IOL), LAB-Bench SeqQA — из
вопросов по работе с последовательностями ДНК, OlympiadBench — из олимпиадных задач по физике.

### Формат вопроса

```json
{
  "domain": "computer_science",
  "title": "Взлом Диффи–Хеллмана, p=1009",
  "prompt": "… finish with a final line of the form `ANSWER: <your answer>`.",
  "answer": "760",
  "check": {"type": "exact"},
  "canary": "…"
}
```

`prompt` — готовый текст для модели; ответ ожидается последней строкой `ANSWER: …`.
Тип проверки `check.type`:
- `exact` — совпадение строки без учёта регистра и пробелов;
- `list` — совпадение списка через запятую по порядку;
- `num` — число с относительным допуском `rel_tol`.

### Как отбирали вопросы

Ориентир — «уровень Диффи–Хеллмана»: метод решения понятен, но требует длинной аккуратной
работы (вычисления, перебор, применение правил), а ответ однозначен. Кандидатов прогоняли на
рассуждения; слишком тяжёлые (модель уходила за 50–100 тыс. токенов или не отвечала) заменяли
более простыми. В `set_3` целились в 2–3 тыс., в `set_4` — в 1–5 тыс., при жёстком лимите 5000:
кандидат, не уложившийся в лимит, отбрасывали, слишком лёгкий заменяли более трудным. Ошибки модели допустимы: в наборах есть нерешённые вопросы.

## Модель

**DeepSeek V4 Flash** (`deepseek-ai/DeepSeek-V4-Flash-0731`) через Together AI, `reasoning_effort=high`,
токенов — `reasoning_tokens` из `usage`.

[`results/deepseek-v4-flash-0731/summary.md`](results/deepseek-v4-flash-0731/summary.md).

Это одна выборка на вопрос. Чтобы надёжно оценить решаемость, нужно 3–5 прогонов.

## Запуск

Нужны [uv](https://docs.astral.sh/uv/) и ключ Together AI:

```bash
export TOGETHER_API_KEY=...
uv run scripts/run.py --set set_3 --set set_4 --max-tokens 5000   # лёгкие наборы, как при отборе
uv run scripts/run.py --base-url https://другой-провайдер/v1 --model ...
```

Подходит любой OpenAI-совместимый API; ключ берётся из `TOGETHER_API_KEY`. Результаты пишутся
в `results/<модель>/`, уже посчитанные вопросы пропускаются. Чтобы пересчитать вопрос, удалите
его `.json`. Прогоны идут по 5 параллельно. Один вопрос считается от минуты до ~20 минут.

Провайдер иногда обрывает длинный поток. Тогда в `.json` будет `finish_reason: null` и пустой
ответ: удалите файл и запустите снова.
