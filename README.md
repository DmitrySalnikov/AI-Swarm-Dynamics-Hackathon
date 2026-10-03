# AI-Swarm-Dynamics-Hackathon

https://swarmchasing.com/

Бенчмарк для экспериментов с роем агентов: два набора по 10 трудных вопросов из 10 разных областей.
Каждый вопрос требует длинного рассуждения (в среднем 10–20 тыс. токенов), но решаем моделью
среднего уровня. Ответ короткий и проверяется автоматически.

> **Не публиковать.** Часть вопросов взята из GPQA, HLE и LAB-Bench, авторы которых просят
> не выкладывать их открытым текстом и не использовать в обучении. В таких вопросах сохранено
> поле `canary`. Репозиторий должен оставаться приватным.

## Структура

```
bench/set_1.jsonl, bench/set_2.jsonl     вопросы
results/<модель>/<set>/<domain>.json     ответ модели, токены, вердикт
results/<модель>/<set>/<domain>.reasoning.txt   рассуждение модели
results/<модель>/summary.md              сводная таблица
scripts/run.py                           прогон модели на наборах
```

## Наборы

В каждом наборе по одному вопросу на область: `computer_science`, `mathematics`, `finance`,
`linguistics`, `physics`, `biology`, `chemistry`, `classics`, `economics`, `music`.

| Область | set_1 | set_2 |
|---|---|---|
| computer_science | взлом Диффи–Хеллмана, p=1009 (HLE) | вывод Scheme-программы с `call/cc` (HLE) |
| mathematics | вероятность, 12 букв в пары (AIME 2025 #7) | четырёхугольник из центров описанных окружностей (HMMT 2025 #26) |
| finance | пенсионный аннуитет (TheoremQA) | PE-коэффициент (TheoremQA) |
| linguistics | формы глаголов шинка (Linguini) | сопоставление слов зуни (Linguini) |
| physics | масса нити лампы накаливания (OlympiadBench) | олимпиадная задача #900 (OlympiadBench) |
| biology | GC-состав последовательности (LAB-Bench) | GC-состав последовательности (LAB-Bench) |
| chemistry | число стереоизомеров (GPQA Diamond) | 4-стадийный синтез, типы H (GPQA Diamond) |
| classics | схема латинского гекзаметра (HLE) | схема строки Плавта (HLE) |
| economics | благосостояние при равновесии (HLE) | поиск работы и пособие (HLE) |
| music | чистый строй, «Hänschen klein» (собственный) | чистый строй, вторая мелодия (собственный) |

Вопросы GPQA заданы открыто, без вариантов ответа: ответ — число, угадать нельзя.
Музыкальные вопросы составлены нами (мелодия и таблица интервалов даны в условии), эталон
вычислен кодом. Для первой мелодии он совпал с эталоном исходного вопроса HLE 66f57e3ddc7259d8b5bb0b46.

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
целевой модели. В наборы брали вопросы, на которые она тратит порядка 5–30 тыс. токенов
рассуждения. Слишком тяжёлые (модель уходила за 50–100 тыс. токенов или не отвечала) заменяли
более простыми. Ошибки модели допустимы: в наборах есть нерешённые вопросы.

## Модель

**DeepSeek V4 Flash** (`deepseek-ai/DeepSeek-V4-Flash-0731`) через Together AI, `reasoning_effort=high`,
`max_tokens=100000`, один прогон на вопрос. Модель отдаёт рассуждение отдельным полем; его число
токенов — `reasoning_tokens` из `usage`.

Результат: **set_1 — 8/10, set_2 — 7/10**, подробности в
[`results/deepseek-v4-flash-0731/summary.md`](results/deepseek-v4-flash-0731/summary.md).

Это одна выборка на вопрос. Чтобы надёжно оценить решаемость, нужно 3–5 прогонов.

## Запуск

Нужны [uv](https://docs.astral.sh/uv/) и ключ Together AI:

```bash
export TOGETHER_API_KEY=...
uv run scripts/run.py --set set_1                       # DeepSeek V4 Flash, reasoning high
uv run scripts/run.py --set set_2 --model Qwen/Qwen3.7-Plus
uv run scripts/run.py --base-url https://другой-провайдер/v1 --model ...
```

Подходит любой OpenAI-совместимый API; ключ берётся из `TOGETHER_API_KEY`. Результаты пишутся
в `results/<модель>/`, уже посчитанные вопросы пропускаются. Чтобы пересчитать вопрос, удалите
его `.json`. Прогоны идут по 5 параллельно. Один вопрос считается от минуты до ~20 минут.

Провайдер иногда обрывает длинный поток. Тогда в `.json` будет `finish_reason: null` и пустой
ответ: удалите файл и запустите снова.
