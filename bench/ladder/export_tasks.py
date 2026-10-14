"""Human-readable view of the final ladder: bench/ladder/tasks/<rung folder>/<NN_domain>.md + tasks/README.md.

  python bench/ladder/export_tasks.py

Each task file: title, source and licence, generator (parameter, seed), the full prompt, the reference answer and
how it is checked, and the selection statistics (median length, CV, correct runs). Built from rung_{1..4}.jsonl and
the run files, so it can be regenerated at any time.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "bench" / "ladder"
sys.path[:0] = [str(LADDER), str(LADDER / "generators")]
import select_tasks as S  # noqa: E402

RUNGS = {
    1: ("rung_1_different_fields", "Ступень 1 — далёкие: разные области знаний"),
    2: ("rung_2_physics_sections", "Ступень 2 — средние: физика, разные разделы"),
    3: ("rung_3_linear_algebra_types", "Ступень 3 — близкие: линейная алгебра, разные типы задач"),
    4: ("rung_4_one_template", "Ступень 4 — один шаблон: определитель 6×6, разные матрицы"),
}
CHECK = {"exact": "строка совпадает", "num": "число", "list": "список через запятую"}


def check_text(c):
    t = CHECK.get(c["type"], c["type"])
    if c.get("rel_tol") == 0:
        return t + ", точное совпадение"
    tol = [f"{k} {c[k]}" for k in ("rel_tol", "abs_tol") if k in c]
    return t + (f" ({', '.join(tol)})" if tol else "")


def main():
    cands = {(it["gen"]["name"], it["gen"]["value"], it["gen"]["seed"]): it for it in S.items().values()}
    out = LADDER / "tasks"
    readme = ["# Лестница семантической близости — задачи", "",
              "По 10 задач на ступень; в каждой папке — файл на задачу (условие, эталон, проверка, статистика отбора).",
              "Длина — токены решения (рассуждение + ответ) у DeepSeek-V4-Flash (cloud.ru, апрельская версия), медиана "
              "по 5–8 прогонам.", "", "```", "ladder"]
    tables = []
    for k, (folder, head) in RUNGS.items():
        tasks = [json.loads(l) for l in (LADDER / f"rung_{k}.jsonl").read_text().splitlines()]
        d = out / folder
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.md"):
            f.unlink()
        last = k == 4
        readme.append(f"{'└──' if last else '├──'} {folder}/   {head}")
        rows = [f"### {head}", "", "| № | Файл | Задача | Параметр | Медиана, ток. | CV | Верно |", "|---|---|---|---|---|---|---|"]
        for i, t in enumerate(tasks, 1):
            g = t["gen"]
            s = S.stats_of(cands[(g["name"], g["value"], g["seed"])])
            name = f"{i:02d}_{t['domain']}.md"
            near = " (принята как максимально близкая: CV выше 0,25)" if s.get("near") else ""
            src = t["source"]
            body = [f"# {t['title']}", "",
                    f"- id: `{t['id']}`",
                    f"- источник: {src['dataset']}" + (f", {src['id']}" if src.get("id") else "") + f"; лицензия: {src.get('license', '—')}",
                    f"- генератор: `{g['name']}`, {g['param']} = {g['value']}, сид {g['seed']}",
                    f"- отбор: медиана {s['median']:.0f} ток., CV {s['cv']:.2f}, верно {s['correct']}/{s['n']}{near}",
                    "", "## Условие", "", "```text", t["prompt"], "```", "",
                    "## Эталон", "", f"`{t['answer']}` — проверка: {check_text(t['check'])}", ""]
            (d / name).write_text("\n".join(body))
            pre = "    " if last else "│   "
            readme.append(f"{pre}{'└──' if i == len(tasks) else '├──'} {name:42s} {t['title']}")
            rows.append(f"| {i} | `{name}` | {t['title']} | {g['param']} = {g['value']} | {s['median']:.0f} | "
                        f"{s['cv']:.2f} | {s['correct']}/{s['n']}{' ⚠' if s.get('near') else ''} |")
        tables += rows + [""]
    readme += ["```", "", *tables, "⚠ — принята как максимально близкая (CV чуть выше 0,25), см. REPORT_ladder.md.", ""]
    (out / "README.md").write_text("\n".join(readme))
    print("\n".join(readme[6:readme.index("```", 6)]))


if __name__ == "__main__":
    main()
