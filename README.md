# AI-Swarm-Dynamics-Hackathon

https://swarmchasing.com/

Бенчмарк для экспериментов с роем агентов: четыре набора по 10 вопросов.
`set_1`–`set_3` охватывают 10 разных областей: `set_1` и `set_2` трудные (длинное рассуждение,
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

В `set_1`–`set_3` по одному вопросу на область: `computer_science`, `mathematics`, `finance`,
`linguistics`, `physics`, `biology`, `chemistry`, `classics`, `economics`, `music`.

| Область | set_1 | set_2 | set_3 (лёгкий) |
|---|---|---|---|
| computer_science | взлом Диффи–Хеллмана, p=1009 (HLE) | вывод Scheme-программы с `call/cc` (HLE) | Диффи–Хеллман, p=227 (собственный) |
| mathematics | вероятность, 12 букв в пары (AIME 2025 #7) | четырёхугольник из центров описанных окружностей (HMMT 2025 #26) | числа из цифр 1–8, делящиеся на 22 (AIME 2025 #5) |
| finance | пенсионный аннуитет (TheoremQA) | PE-коэффициент (TheoremQA) | NPV проекта (собственный) |
| linguistics | формы глаголов шинка (Linguini) | сопоставление слов зуни (Linguini) | числительные выдуманного языка (собственный) |
| physics | масса нити лампы накаливания (OlympiadBench) | олимпиадная задача #900 (OlympiadBench) | парашютист с квадратичным сопротивлением (TheoremQA) |
| biology | GC-состав последовательности (LAB-Bench) | GC-состав последовательности (LAB-Bench) | GC-состав 150 нуклеотидов (собственный) |
| chemistry | число стереоизомеров (GPQA Diamond) | 4-стадийный синтез, типы H (GPQA Diamond) | смесь NaHCO₃ и MgCO₃ по потере массы (собственный) |
| classics | схема латинского гекзаметра (HLE) | схема строки Плавта (HLE) | 6 дат римского календаря (собственный) |
| economics | благосостояние при равновесии (HLE) | поиск работы и пособие (HLE) | детерминированный эквивалент (TheoremQA) |
| music | чистый строй, «Hänschen klein» (собственный) | чистый строй, вторая мелодия (собственный) | чистый строй, 13 нот (собственный) |

Вопросы GPQA заданы открыто, без вариантов ответа: ответ — число, угадать нельзя.
Музыкальные вопросы составлены нами (мелодия и таблица интервалов даны в условии), эталон
вычислен кодом. Для первой мелодии он совпал с эталоном исходного вопроса HLE 66f57e3ddc7259d8b5bb0b46.
В `set_3` 7 из 10 вопросов собственные, по образцу задач из `set_1`/`set_2` с меньшими параметрами:
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
| `set_1/computer_science` | [HLE](https://huggingface.co/datasets/cais/hle) | `67192b9472c6fd14e759e369` |
| `set_1/mathematics` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | задача 7 |
| `set_1/finance` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | вопрос «Aisha graduates college…» |
| `set_1/linguistics` | [Linguini](https://huggingface.co/datasets/facebook/linguini) | `012023010100` |
| `set_1/physics` | [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | `OE_TO_physics_en_COMP`, id 908 |
| `set_1/biology` | [LAB-Bench SeqQA](https://huggingface.co/datasets/futurehouse/lab-bench) | `97e6e1a0-f207-46d0-9201-f0081f40e19a` |
| `set_1/chemistry` | [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | `rectXfsCM1dj4Kv2c` |
| `set_1/classics` | [HLE](https://huggingface.co/datasets/cais/hle) | `66fc698fd90ebe461bfd0cc4` |
| `set_1/economics` | [HLE](https://huggingface.co/datasets/cais/hle) | `66fc23cfa7be4edbe85cf177` |
| `set_1/music` | собственный, по мотивам HLE `66f57e3ddc7259d8b5bb0b46` | — |
| `set_2/computer_science` | [HLE](https://huggingface.co/datasets/cais/hle) | `66f4aa5df382ae9214c8dc9b` |
| `set_2/mathematics` | [HMMT February 2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) | задача 26 |
| `set_2/finance` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | вопрос «Estimate the PE ratio…» |
| `set_2/linguistics` | [Linguini](https://huggingface.co/datasets/facebook/linguini) | `012021020100` |
| `set_2/physics` | [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | `OE_TO_physics_en_COMP`, id 900 |
| `set_2/biology` | [LAB-Bench SeqQA](https://huggingface.co/datasets/futurehouse/lab-bench) | `64c5e31b-5b98-494a-92a7-1019723204d6` |
| `set_2/chemistry` | [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | `recJZ3QEfRKjYw9a7` |
| `set_2/classics` | [HLE](https://huggingface.co/datasets/cais/hle) | `67015a7f6a2b21f149f3aaba` |
| `set_2/economics` | [HLE](https://huggingface.co/datasets/cais/hle) | `6711e5e05e64a53ed09449fd` |
| `set_2/music` | собственный | — |
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
  "id": "set_1/computer_science",
  "set": "set_1",
  "domain": "computer_science",
  "title": "Взлом Диффи–Хеллмана, p=1009",
  "source": {"dataset": "HLE", "id": "67192b9472c6fd14e759e369", "url": "https://huggingface.co/datasets/cais/hle"},
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
целевой модели. В `set_1`/`set_2` брали вопросы, на которые она тратит порядка 5–30 тыс. токенов
рассуждения; слишком тяжёлые (модель уходила за 50–100 тыс. токенов или не отвечала) заменяли
более простыми. В `set_3` целились в 2–3 тыс., в `set_4` — в 1–5 тыс., при жёстком лимите 5000:
кандидат, не уложившийся в лимит, отбрасывали, слишком лёгкий заменяли более трудным. Ошибки модели допустимы: в наборах есть нерешённые вопросы.

## Модель

**DeepSeek V4 Flash** (`deepseek-ai/DeepSeek-V4-Flash-0731`) через Together AI, `reasoning_effort=high`,
один прогон на вопрос; `max_tokens=100000` для `set_1`/`set_2` и 5000 для `set_3`/`set_4`. Модель отдаёт рассуждение отдельным полем; его число
токенов — `reasoning_tokens` из `usage`.

Результат: **set_1 — 8/10, set_2 — 7/10, set_3 — 10/10, set_4 — 10/10**, подробности в
[`results/deepseek-v4-flash-0731/summary.md`](results/deepseek-v4-flash-0731/summary.md).

Это одна выборка на вопрос. Чтобы надёжно оценить решаемость, нужно 3–5 прогонов.

## Запуск

Нужны [uv](https://docs.astral.sh/uv/) и ключ Together AI:

```bash
export TOGETHER_API_KEY=...
uv run scripts/run.py --set set_1                       # DeepSeek V4 Flash, reasoning high
uv run scripts/run.py --set set_2 --model Qwen/Qwen3.7-Plus
uv run scripts/run.py --set set_3 --set set_4 --max-tokens 5000   # лёгкие наборы, как при отборе
uv run scripts/run.py --base-url https://другой-провайдер/v1 --model ...
```

Подходит любой OpenAI-совместимый API; ключ берётся из `TOGETHER_API_KEY`. Результаты пишутся
в `results/<модель>/`, уже посчитанные вопросы пропускаются. Чтобы пересчитать вопрос, удалите
его `.json`. Прогоны идут по 5 параллельно. Один вопрос считается от минуты до ~20 минут.

Провайдер иногда обрывает длинный поток. Тогда в `.json` будет `finish_reason: null` и пустой
ответ: удалите файл и запустите снова.
