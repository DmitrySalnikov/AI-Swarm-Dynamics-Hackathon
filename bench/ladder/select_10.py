"""Second ladder set (2026-10-14): 10 tasks per rung, topics not repeated within a rung.

  uv run --with sympy --with numpy --with scipy --with matplotlib --with reasoning-gym==0.1.19 \\
      python bench/ladder/select_10.py

Eligible: >= 80 % correct runs, CV <= 0.25, mean length in [14 000; 23 000] (the band of fig5_spread_vs_n_band).
Rung 2 (physics) has exactly N topics in the band, so its set is the narrowest N-topic window over all its tasks;
its range [L, U] of mean lengths is the widest of all rungs. The other rungs take the narrowest N-topic window among
their tasks with the mean inside [L, U] (so that all 40 tasks are comparable), falling back to all their tasks if
that is impossible. Window search = spread_vs_n.best_set (exact).
Outputs: set_v2/rung_{1..4}.jsonl (bench format), figures/set_v2_tasks.csv,
figures/fig6_overview_v2.{png,svg} (mean ± std, CV), figures/fig7_runs_v2.{png,svg} (every run).
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "bench" / "ladder"
sys.path[:0] = [str(LADDER), str(LADDER / "generators")]
import plots_ladder as P  # noqa: E402
import select_tasks as S  # noqa: E402
from spread_vs_n import best_set, topic  # noqa: E402

N, LO, HI = 10, 14000, 23000
OUT_SET = LADDER / "set_v2"


def main():
    tasks = [t for t in P.collect() if P.is_clean(t) and LO <= t["mean"] <= HI]
    by_rung = {k: sorted((t for t in tasks if t["rung"] == k), key=lambda t: t["mean"]) for k in range(1, 5)}
    sel = {2: best_set(by_rung[2], N)}
    L, U = min(t["mean"] for t in sel[2]), max(t["mean"] for t in sel[2])
    notes = {}
    for k in (1, 3, 4):
        inside = [t for t in by_rung[k] if L <= t["mean"] <= U]
        sel[k] = best_set(inside, N) or best_set(by_rung[k], N)
        notes[k] = "внутри диапазона ступени 2" if best_set(inside, N) else "диапазон ступени 2 не вместил 10 тем"
    items = S.items()
    OUT_SET.mkdir(exist_ok=True)
    rows = []
    for k in range(1, 5):
        out = []
        for i, t in enumerate(sorted(sel[k], key=lambda t: (topic(t) if k != 4 else t["id"])), 1):
            it = dict(items[t["id"]])
            dom = topic(t) if k != 4 else f"determinant_{i:02d}"
            it.update(id=f"ladder_v2_rung_{k}/{dom}", set=f"ladder_v2_rung_{k}", domain=dom)
            it.pop("data", None)
            it["selection"] = {"candidate_id": t["id"], "n_runs": len(t["runs"]), "mean": round(t["mean"]),
                               "std": round(t["std"]), "cv": round(t["cv"], 3),
                               "max": max(r["length"] for r in t["runs"]),
                               "correct": sum(r["correct"] for r in t["runs"])}
            out.append(it)
            t["final_label"], t["status"], t["letters"] = dom, "final", ""
            rows.append([k, dom, t["id"], t["gen"].name, t["value"], len(t["runs"]), round(t["mean"]), round(t["std"]),
                         round(t["cv"], 3), max(r["length"] for r in t["runs"]), sum(r["correct"] for r in t["runs"])])
        (OUT_SET / f"rung_{k}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out))
    with open(P.OUT / "set_v2_tasks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "topic", "candidate_id", "generator", "param_value", "n_runs", "mean", "std", "cv", "max",
                    "n_correct"])
        w.writerows(rows)
    chosen = sorted((t for k in sel for t in sel[k]), key=lambda t: (t["rung"], t["final_label"]))
    P.fig1(chosen, True, "fig6_overview_v2")
    P.fig2(chosen, True, "fig7_runs_v2")
    for k in range(1, 5):
        m = [t["mean"] for t in sel[k]]
        print(f"rung {k}: {len(sel[k])} tasks, means {min(m):.0f}–{max(m):.0f} (x{max(m) / min(m):.2f}), "
              f"longest run {max(r['length'] for t in sel[k] for r in t['runs'])}"
              + (f"; {notes[k]}" if k in notes else "; рамка для остальных"))
    allm = [t["mean"] for t in chosen]
    print(f"all 40: means {min(allm):.0f}–{max(allm):.0f} (x{max(allm) / min(allm):.2f})")


if __name__ == "__main__":
    main()
