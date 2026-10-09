"""Length calibration of the ladder generators (step 2).

  python bench/ladder/calibrate.py items --round 1            # probe values of every generator -> candidates/calib_round1.jsonl
  python bench/ladder/calibrate.py items --round 2            # value fitted to the target length -> candidates/calib_round2.jsonl
  python bench/ladder/calibrate.py table [--md FILE]          # parameter -> length, correctness

Runs are made by embedding_eval/gen_runs.py (--items candidates/calib_roundK.jsonl --run-id K); results are read from
embedding_eval/runs/ladder/deepseek-v4-flash/calib/run_K/ladder_calib/<generator>@<value>.json (.failed.json = cut off
or error). Length = usage.completion_tokens (reasoning + visible answer); a run cut at max_tokens counts as censored.
Fit: L = c * p^k by least squares in log-log over all uncensored points of the generator (k = 1 if only one point);
the chosen value is the grid value whose predicted length is closest to the target in log scale.
"""
import argparse
import importlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = ROOT / "bench" / "ladder" / "generators"
CAND = ROOT / "bench" / "ladder" / "candidates"
RUNS = ROOT / "embedding_eval" / "runs" / "ladder" / "deepseek-v4-flash" / "calib"
SET = "ladder_calib"
TARGET = 20000
MAX_LEN = 60000  # counted as cut off by length
sys.path[:0] = [str(GEN_DIR)]
from common import build  # noqa: E402

sys.path.insert(0, str(ROOT / "embedding_eval"))
from gen_runs import extract, grade  # noqa: E402


def generators():
    skip = {"common", "selftest", "fetch_sources"}
    out = {}
    for p in sorted(GEN_DIR.glob("*.py")):
        if p.stem in skip:
            continue
        for g in getattr(importlib.import_module(p.stem), "GENS", []):
            out[g.name] = g
    return out


_ITEMS = {}


def regrade(r, gens):
    """Correctness by the current grader from the stored response: the task is rebuilt from (generator, value, seed)."""
    g = r["gen"]
    key = (g["name"], g["value"], g["seed"])
    if key not in _ITEMS:
        _ITEMS[key] = build(gens[g["name"]], g["value"], g["seed"], SET)
    item = _ITEMS[key]
    assert item["answer"] == r["answer"], f"rebuilt task differs from the run: {key}"
    got = extract(r["response"], fallback=True)
    return grade(item, got), extract(r["response"]) is not None


def results(gens=None):
    """(generator, value) -> list of dicts(run, length, censored, correct, format_ok, finish, error).
    censored = cut off at max_tokens or longer than MAX_LEN (cloud.ru does not enforce max_tokens); censored runs
    count as wrong."""
    gens = gens or generators()
    res = {}
    for f in sorted(RUNS.glob(f"run_*/{SET}/*.json")):
        failed = f.name.endswith(".failed.json")
        if failed and f.with_name(f.name.replace(".failed.json", ".json")).exists():
            continue
        r = json.loads(f.read_text())
        L = r.get("completion_tokens") or (r.get("usage") or {}).get("completion_tokens")
        censored = r["finish_reason"] == "length" or (L or 0) > MAX_LEN
        correct, fmt = regrade(r, gens) if not failed else (False, False)
        res.setdefault((r["gen"]["name"], r["gen"]["value"]), []).append(dict(
            run=int(f.parent.parent.name.split("_")[1]), length=L, censored=censored,
            correct=correct and not censored, format_ok=fmt, finish=r["finish_reason"], error=r.get("error")))
    return res


def predict(points):
    """points [(p, L)] -> (p*, model): the parameter value expected to give TARGET tokens.
    linear L = a + b·p (fixed overhead + per-item cost) when b > 0 and a >= 0; power law L = c·p^k when the growth
    is superlinear (a < 0); when lengths do not grow with p (noise), scale proportionally from the point closest to
    the target."""
    ps = sorted({p for p, _ in points})
    if len(ps) >= 2:
        n = len(points)
        mp, mL = sum(p for p, _ in points) / n, sum(L for _, L in points) / n
        b = sum((p - mp) * (L - mL) for p, L in points) / sum((p - mp) ** 2 for p, _ in points)
        a = mL - b * mp
        if b > 0 and a >= 0:
            return (TARGET - a) / b, f"lin a={a:.0f} b={b:.0f}"
        if b > 0:
            xs, ys = [math.log(p) for p, _ in points], [math.log(L) for _, L in points]
            mx, my = sum(xs) / n, sum(ys) / n
            k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
            return math.exp((math.log(TARGET) - (my - k * mx)) / k), f"pow k={k:.2f}"
    p0, L0 = min(points, key=lambda t: abs(math.log(t[1] / TARGET)))
    return p0 * TARGET / L0, "prop"


def choose(gen, res):
    """-> (grid value, p*, model, note)."""
    rs = [(v, r) for (n, v), lst in res.items() if n == gen.name for r in lst]
    # cloud.ru returns the full length even beyond MAX_LEN, so such runs still count as length points;
    # only runs really cut at max_tokens (finish_reason=length) are censored for the fit
    pts = [(v, r["length"]) for v, r in rs if r["length"] and r["finish"] != "length" and not r["error"]]
    cens = [v for v, r in rs if r["finish"] == "length"]
    if not pts:
        if not cens:
            return None, None, None, ""
        pstar, model = min(cens) / 3, "cut"  # longer than MAX_LEN = 3x the target
    else:
        pstar, model = predict(pts)
        if cens and pstar >= min(cens):
            pstar = min(cens) / 3
    best = min(gen.grid, key=lambda v: abs(math.log(max(v, 1e-9) / max(pstar, 1e-9))))
    note = "above grid" if pstar > gen.grid[-1] * 1.15 else "below grid" if pstar < gen.grid[0] / 1.15 else ""
    return best, pstar, model, note


def cmd_items(args):
    gens = generators()
    if args.only:
        gens = {k: v for k, v in gens.items() if k in set(args.only)}
    res = results()
    items = []
    for g in gens.values():
        if args.round == 1:
            values = list(g.probe)
        else:
            v = choose(g, res)[0]
            values = [v] if v is not None else []
        for v in values:
            if (g.name, v) in res and args.round == 1:
                continue
            it = build(g, v, seed=args.round, set_name=SET, task_id=f"{g.name}@{v}")
            items.append(it)
    CAND.mkdir(parents=True, exist_ok=True)
    out = CAND / f"calib_round{args.round}.jsonl"
    out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    print(f"{len(items)} items -> {out.relative_to(ROOT)}")


def cmd_table(args):
    """One row per generator: points 'value → length' with marks ✓ correct, ✗ wrong, ✂ cut off (> MAX_LEN),
    ° no ANSWER line (bare last line used); the fitted value p* and the chosen grid value."""
    gens = generators()
    res = results(gens)
    lines = ["| Генератор | Ступени | Параметр | Точки: значение → длина | Модель | p* | Выбор | Замечание |",
             "|---|---|---|---|---|---|---|---|"]
    for g in gens.values():
        rows = sorted((v, r["run"], r) for (n, v), rs in res.items() if n == g.name for r in rs)
        pts = "; ".join(f"{v} → {r['length']}{'✂' if r['censored'] else '✓' if r['correct'] else '✗'}"
                        f"{'' if r['format_ok'] else '°'}" for v, _, r in rows) or "—"
        best, pstar, model, note = choose(g, res)
        ps = f"{pstar:.3g}" if pstar else "—"
        lines.append(f"| {g.name} | {','.join(map(str, g.rungs))} | {g.param} | {pts} | {model or '—'} | {ps} | "
                     f"{best if best is not None else '—'} | {note} |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("items")
    a.add_argument("--round", type=int, required=True)
    a.add_argument("--only", nargs="*")
    t = sub.add_parser("table")
    t.add_argument("--md")
    args = ap.parse_args()
    {"items": cmd_items, "table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
