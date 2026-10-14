"""Does the model already know a ladder task (was it in pre-training)? Signs looked for in the chain of thought (CoT)
of every run of the 40 final tasks:
- recall phrases: the model says it recognises the problem or remembers its answer;
- early answer: where the reference answer first appears in the CoT, as a fraction of the CoT length (0 = at the very
  start). A model that knows the answer tends to state it before deriving it. Only answers that cannot appear by
  chance are searched: the whole list for list answers, numbers with >= 4 significant characters otherwise.

  python bench/ladder/cot_check.py [--md FILE]
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "bench" / "ladder"
sys.path[:0] = [str(LADDER), str(LADDER / "generators")]
import select_tasks as S  # noqa: E402

RECALL = re.compile(
    r"\b(I (?:recall|remember)|(?:well|widely)[- ]known (?:problem|puzzle|result|answer)|this is a (?:classic|famous|known)"
    r" (?:problem|puzzle|exercise)|I(?:'ve| have) seen (?:this|it) before|from (?:the )?(?:TheoremQA|textbook|dataset|"
    r"benchmark|competition)|the (?:known|standard|textbook) answer is|answer (?:key|is known))", re.I)


def answer_patterns(t):
    """Regexes for the reference answer that are unlikely to match by chance (None = too short to search)."""
    ans = t["answer"]
    if t["check"]["type"] == "list":
        parts = [re.escape(p.strip()) for p in ans.split(",")]
        return re.compile(r"[\s,;]*".join(parts)) if len(parts) >= 2 else None
    core = ans.strip().lstrip("-")
    if len(re.sub(r"\D", "", core)) < 4 and not re.search(r"[A-Za-z]{4,}", core):
        return None
    return re.compile(re.escape(core).replace(r"\.", r"\.?"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md")
    ap.add_argument("--all", action="store_true", help="every candidate of the pool, not only rung_{1..4}.jsonl")
    args = ap.parse_args()
    cands = {(it["gen"]["name"], it["gen"]["value"], it["gen"]["seed"]): it for it in S.items().values()}
    lines = ["| Ступень | Задача | Источник | Прогонов | Фразы узнавания | Ответ впервые в CoT, доля длины (медиана; мин) |",
             "|---|---|---|---|---|---|"]
    hits_total = 0
    if args.all:
        tasks = sorted(cands.values(), key=lambda t: (t["set"], t["domain"]))
    else:
        tasks = [json.loads(l) for k in range(1, 5) for l in (LADDER / f"rung_{k}.jsonl").read_text().splitlines()]
    for t in tasks:
        g = t["gen"]
        cand = cands[(g["name"], g["value"], g["seed"])]
        pat = answer_patterns(t)
        recall, first = [], []
        for r in S.runs_of(cand):
            f = Path(r["file"])
            stem = f.name.replace(".failed.json", "").replace(".json", "")
            rf = f.with_name(stem + (".failed" if f.name.endswith(".failed.json") else "") + ".reasoning.txt")
            cot = rf.read_text() if rf.exists() else ""
            recall += [m.group(0) for m in RECALL.finditer(cot)]
            if pat and cot:
                m = pat.search(cot)
                first.append(m.start() / len(cot) if m else None)
        hits_total += len(recall)
        found = [x for x in first if x is not None]
        pos = "не ищется (ответ короткий)" if pat is None else (
            f"{st.median(found):.2f}; {min(found):.2f} (найден в {len(found)}/{len(first)})" if found else "не найден")
        src = "датасет" if not t["source"]["dataset"].startswith("own") else "генератор"
        lines.append(f"| {t['set'][-1]} | {t['domain']} | {src} | {len(first) or len(S.runs_of(cand))} | "
                     f"{len(recall)}{': ' + '; '.join(sorted(set(recall)))[:120] if recall else ''} | {pos} |")
    text = "\n".join(lines) + f"\n\nВсего фраз узнавания: {hits_total}.\n"
    print(text)
    if args.md:
        Path(args.md).write_text(text)


if __name__ == "__main__":
    main()
