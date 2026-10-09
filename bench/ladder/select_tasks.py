"""Main selection of ladder tasks (steps 3–4).

  python bench/ladder/select_tasks.py stats [--md FILE]   # per-candidate statistics, filters, borderline cases
  python bench/ladder/select_tasks.py borderline          # -> candidates/borderline.jsonl (runs 6..8 for borderline cases)
  python bench/ladder/select_tasks.py embed [--budget-usd 1]   # memorisation flag (Qwen3-Embedding-4B via vectorize.py)
  python bench/ladder/select_tasks.py final [--md FILE]   # 10 tasks per rung -> bench/ladder/rung_{1..4}.jsonl

Runs: embedding_eval/runs/ladder/deepseek-v4-flash/main/run_K/ladder_rung_k/<task>.json (.failed.json = no visible
answer / error). Per candidate:
- length = usage.completion_tokens (reasoning + visible answer); median and CV = sample std (ddof=1) / mean;
- a run longer than MAX_LEN (cloud.ru does not enforce max_tokens) counts as cut off and wrong;
- correctness is re-graded from the stored response with the current grader (gen_runs.grade, fallback to the last
  line when there is no ANSWER line); runs that ended with a transport error are not counted (they are redone).
Filters: LEN_LO <= median <= LEN_HI; CV <= CV_MAX; wrong <= 20 % of runs (5 runs: at most 1 wrong; 8 runs: at most 1).
Borderline (extra runs up to 8): passes the length filter and has 3 correct of 5, or CV within ±0.05 of CV_MAX.
"""
import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bench" / "ladder" / "generators"), str(ROOT / "embedding_eval")]
from gen_runs import extract, grade  # noqa: E402

RUNS = ROOT / "embedding_eval" / "runs" / "ladder" / "deepseek-v4-flash" / "main"
CAND = ROOT / "bench" / "ladder" / "candidates"
LADDER = ROOT / "bench" / "ladder"
MEMO = LADDER / "memorisation.json"
MAX_LEN, LEN_LO, LEN_HI, TARGET = 60000, 10000, 25000, 20000
CV_MAX, WRONG_SHARE, MAX_RUNS = 0.25, 0.2, 8
# rung 2 had only 7 passing tasks; the user chose (2026-10-09, option A) to take the three nearest misses
# ("максимально близкое"): all other filters pass, CV exceeds 0.25 by at most 0.0022
NEAR_ACCEPT = {
    "ladder_rung_2/r2_pipeline_bernoulli_v2": "CV чуть выше порога",
    "ladder_rung_2/r2_ladder_mesh": "CV чуть выше порога",
    "ladder_rung_2/r2_composite_wall": "CV чуть выше порога",
}


def items():
    out = {}
    for f in ("main.jsonl", "rung2_extra.jsonl", "borderline.jsonl"):
        p = CAND / f
        if p.exists():
            for l in p.read_text().splitlines():
                it = json.loads(l)
                out.setdefault(it["id"], it)
    return out


def runs_of(it):
    """List of run dicts for one task."""
    res = []
    for d in sorted(RUNS.glob(f"run_*/{it['set']}")):
        f = d / f"{it['domain']}.json"
        failed = not f.exists()
        f = f if f.exists() else d / f"{it['domain']}.failed.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        if r.get("error"):  # transport error: incomplete, redone on the next launch
            continue
        L = r.get("completion_tokens") or (r.get("usage") or {}).get("completion_tokens")
        cut = r["finish_reason"] == "length" or (L or 0) > MAX_LEN
        got = extract(r["response"], fallback=True)
        ok = (not failed) and (not cut) and grade(it, got)
        res.append(dict(run=int(d.parent.name.split("_")[1]), length=L, cut=cut, correct=ok,
                        format_ok=extract(r["response"]) is not None, file=str(f)))
    return res


def stats_of(it):
    rs = runs_of(it)
    Ls = [r["length"] for r in rs if r["length"]]
    n = len(rs)
    s = dict(id=it["id"], set=it["set"], name=it["gen"]["name"], value=it["gen"]["value"], n=n,
             median=st.median(Ls) if Ls else None, cv=(st.stdev(Ls) / st.mean(Ls)) if len(Ls) > 1 else None,
             correct=sum(r["correct"] for r in rs), cut=sum(r["cut"] for r in rs),
             no_answer_line=sum(not r["format_ok"] for r in rs), lengths=Ls)
    why = []
    if n < 5:
        why.append(f"мало прогонов ({n})")
    if s["median"] is not None and not LEN_LO <= s["median"] <= LEN_HI:
        why.append(f"медиана {s['median']:.0f} вне [{LEN_LO}, {LEN_HI}]")
    if s["cv"] is not None and s["cv"] > CV_MAX:
        why.append(f"CV {s['cv']:.2f} > {CV_MAX}")
    if n and (n - s["correct"]) > WRONG_SHARE * n:
        why.append(f"верно {s['correct']}/{n}")
    s["why"] = why
    s["pass"] = n >= 5 and not why
    s["near"] = NEAR_ACCEPT.get(it["id"]) if not s["pass"] else None
    len_ok = s["median"] is not None and LEN_LO <= s["median"] <= LEN_HI
    s["borderline"] = n < MAX_RUNS and n >= 5 and len_ok and (
        (s["correct"] == n - 2 and (s["cv"] or 0) <= CV_MAX + 0.05) or
        (s["cv"] is not None and abs(s["cv"] - CV_MAX) <= 0.05 and n - s["correct"] <= WRONG_SHARE * MAX_RUNS))
    return s


def all_stats():
    return [stats_of(it) for it in items().values()]


def fmt_row(s, memo):
    m = memo.get(s["id"], {})
    flag = "⚑" if m.get("flag") else ""
    sim = f"{m['sim']:.3f}{flag}" if "sim" in m else "—"
    med = f"{s['median']:.0f}" if s["median"] is not None else "—"
    cv = f"{s['cv']:.2f}" if s["cv"] is not None else "—"
    status = ("прошёл" if s["pass"] else
              f"принят как максимально близкий ({'; '.join(s['why'])})" if s["near"] else
              "пограничный" if s["borderline"] else "нет: " + "; ".join(s["why"]))
    return (f"| {s['set'][-1]} | {s['name']} | {s['value']} | {s['n']} | {med} | {cv} | {s['correct']}/{s['n']} | "
            f"{s['cut']} | {s['no_answer_line']} | {sim} | {status} |")


HEADER = ["| Ступень | Кандидат | p | Прогонов | Медиана, ток. | CV | Верно | >60 тыс. | Без ANSWER | Схожесть | Итог |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]


def cmd_stats(args):
    memo = json.loads(MEMO.read_text()) if MEMO.exists() else {}
    S = sorted(all_stats(), key=lambda s: (s["set"], s["name"], s["id"]))
    lines = HEADER + [fmt_row(s, memo) for s in S]
    text = "\n".join(lines)
    print(text)
    for k in sorted({s["set"] for s in S}):
        ss = [s for s in S if s["set"] == k]
        print(f"{k}: {len(ss)} candidates, passed {sum(s['pass'] for s in ss)}, borderline "
              f"{sum(s['borderline'] for s in ss)}")
    if args.md:
        Path(args.md).write_text(text + "\n")


def cmd_borderline(args):
    its = items()
    b = [its[s["id"]] for s in all_stats() if s["borderline"]]
    (CAND / "borderline.jsonl").write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in b))
    print(f"{len(b)} borderline candidates -> candidates/borderline.jsonl (run ids 6 7 8)")


def cmd_embed(args):
    """Mean pairwise cosine between independent solutions of a task: solution vector = mean of the centred Qwen
    window vectors (vectorize.py scheme, centring over all windows of all candidates); flag = above the rung median +
    3·MAD (scaled MAD, 1.4826) of that rung."""
    import numpy as np
    import vectorize as V
    its, logs, owner = items(), {}, {}
    for it in its.values():
        for r in runs_of(it):
            f = Path(r["file"])
            stem = f.name.replace(".failed.json", "").replace(".json", "")
            rf = f.with_name(stem + (".failed" if f.name.endswith(".failed.json") else "") + ".reasoning.txt")
            resp = json.loads(f.read_text())["response"]
            lid = f"ladder/{it['set']}/{it['domain']}/run_{r['run']}"
            logs[lid] = (rf.read_text() if rf.exists() else "") + "\n\n" + resp
            owner[lid] = it["id"]
    Q, meta = V.embed_logs(logs, budget_usd=args.budget_usd)
    Z = V.center(Q)
    vec = {}
    for m, z in zip(meta, Z):
        vec.setdefault(m["log_id"], []).append(z)
    sol = {lid: (lambda v: v / np.linalg.norm(v))(np.mean(vs, 0)) for lid, vs in vec.items()}
    by_task = {}
    for lid, v in sol.items():
        by_task.setdefault(owner[lid], []).append(v)
    out = {}
    for tid, vs in by_task.items():
        if len(vs) < 2:
            continue
        M = np.stack(vs) @ np.stack(vs).T
        iu = np.triu_indices(len(vs), 1)
        out[tid] = {"sim": float(M[iu].mean()), "n": len(vs)}
    for k in {its[t]["set"] for t in out}:
        sims = [out[t]["sim"] for t in out if its[t]["set"] == k]
        med = st.median(sims)
        mad = 1.4826 * st.median([abs(x - med) for x in sims])
        for t in out:
            if its[t]["set"] == k:
                out[t].update(rung_median=med, rung_mad=mad, flag=out[t]["sim"] > med + 3 * mad)
    MEMO.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{len(out)} tasks; flagged {sum(v['flag'] for v in out.values())} -> {MEMO.relative_to(ROOT)}")


def pick(rows, k, target):
    """10 passing tasks of rung k closest to target (log scale); rung 1: at most one per area."""
    chosen, areas = [], set()
    for s in sorted(rows, key=lambda s: abs(math.log(s["median"] / target))):
        if k == 1 and s["area"] in areas:
            continue
        chosen.append(s)
        areas.add(s["area"])
        if len(chosen) == 10:
            break
    return chosen


def cmd_final(args):
    import importlib
    gens = {}
    for p in sorted((LADDER / "generators").glob("*.py")):
        if p.stem in {"common", "selftest", "fetch_sources"}:
            continue
        for g in getattr(importlib.import_module(p.stem), "GENS", []):
            gens[g.name] = g
    its = items()
    S = [dict(s, area=gens[s["name"]].area) for s in all_stats() if s["pass"] or s["near"]]
    rungs = sorted({s["set"] for s in all_stats()})
    best = None
    for T in range(LEN_LO, LEN_HI + 1, 250):  # common target that makes the rung medians closest to each other
        sel = {k: pick([s for s in S if s["set"] == k], int(k[-1]), T) for k in rungs}
        if any(len(v) < 10 for v in sel.values()):
            meds = None
        else:
            meds = [st.median([s["median"] for s in v]) for v in sel.values()]
        key = (meds is None, (max(meds) - min(meds)) if meds else 0, abs(T - TARGET))
        if best is None or key < best[0]:
            best = (key, T, sel)
    _, T, sel = best
    lines = [f"Общая цель T = {T} ток. (медианы ступеней как можно ближе друг к другу)", ""]
    for k, chosen in sel.items():
        n = int(k[-1])
        lines.append(f"### Ступень {n}: выбрано {len(chosen)} из {sum(s['set'] == k for s in S)} допущенных; "
                     f"медиана медиан {st.median([s['median'] for s in chosen]) if chosen else '—'}")
        out = []
        for i, s in enumerate(sorted(chosen, key=lambda s: s["name"])):
            it = dict(its[s["id"]])
            g = gens[s["name"]]
            dom = g.area if n == 1 else (f"determinant_{i + 1:02d}" if n == 4 else g.domain)
            it.update(id=f"ladder_rung_{n}/{dom}", set=f"ladder_rung_{n}", domain=dom)
            it.pop("data", None)
            out.append(it)
            near = f" — принят как максимально близкий: CV {s['cv']:.4f} > {CV_MAX}" if s["near"] else ""
            lines.append(f"- {dom}: {s['name']} p={s['value']}, медиана {s['median']:.0f}, CV {s['cv']:.2f}, "
                         f"верно {s['correct']}/{s['n']}{near}")
        (LADDER / f"rung_{n}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("stats")
    a.add_argument("--md")
    sub.add_parser("borderline")
    e = sub.add_parser("embed")
    e.add_argument("--budget-usd", type=float, default=1.0)
    f = sub.add_parser("final")
    f.add_argument("--md")
    args = ap.parse_args()
    {"stats": cmd_stats, "borderline": cmd_borderline, "embed": cmd_embed, "final": cmd_final}[args.cmd](args)


if __name__ == "__main__":
    main()
