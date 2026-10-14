"""Draft sample of the second ladder (2026-10-12): up to 20 tasks per rung, every topic at most once per rung.

  uv run --with sympy --with numpy --with scipy --with matplotlib --with reasoning-gym==0.1.19 \\
      python bench/ladder/select_v2.py

Eligible task: >= 80 % correct runs, CV <= 0.25, mean length in [LO, HI]. Topic = area (rung 1), physics section
(rung 2), task type (rung 3); rung 4 is one template, so its 20 tasks are different matrices.
Choice: one common target T (scanned over [LO, HI]) for all rungs; in each topic the task whose mean is closest to T;
a rung with more than 20 topics keeps the 20 closest to T. T minimises the range (max - min) of the means of all
chosen tasks. Outputs: figures/selected_v2.csv, figures/fig4_selected_v2.{png,svg}.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "bench" / "ladder"
sys.path[:0] = [str(LADDER), str(LADDER / "generators")]
import plots_ladder as P  # noqa: E402

LO, HI, N = 14000, 23000, 20


def topic(t):
    return t["gen"].area if t["rung"] == 1 else t["gen"].domain if t["rung"] in (2, 3) else t["id"]


def choose(eligible, T):
    sel = []
    for k in range(1, 5):
        best = {}
        for t in (t for t in eligible if t["rung"] == k):
            key = topic(t)
            if key not in best or abs(t["mean"] - T) < abs(best[key]["mean"] - T):
                best[key] = t
        sel += sorted(best.values(), key=lambda t: abs(t["mean"] - T))[:N]
    return sel


def main():
    tasks = P.collect()
    eligible = [t for t in tasks if P.is_clean(t) and LO <= t["mean"] <= HI]
    T = min(range(LO, HI + 1, 250), key=lambda T: (lambda s: max(t["mean"] for t in s) - min(t["mean"] for t in s))(
        choose(eligible, T)))
    sel = choose(eligible, T)
    with open(P.OUT / "selected_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "topic", "question_id", "generator", "param_value", "n_runs", "mean", "std", "cv", "max",
                    "n_correct"])
        for t in sorted(sel, key=lambda t: (t["rung"], t["mean"])):
            Ls = [r["length"] for r in t["runs"]]
            w.writerow([t["rung"], topic(t), t["id"], t["gen"].name, t["value"], len(Ls), round(t["mean"]),
                        round(t["std"]), round(t["cv"], 3), max(Ls), sum(r["correct"] for r in t["runs"])])
    per = [sum(t["rung"] == k for t in sel) for k in range(1, 5)]
    means = [t["mean"] for t in sel]
    title = (f"Черновой выбор: до {N} задач на ступень, темы не повторяются (по ступеням {per[0]} / {per[1]} / {per[2]} / "
             f"{per[3]}); средняя длина {min(means) / 1000:.1f}–{max(means) / 1000:.1f} тыс., общая цель T = {T}")
    P.fig3_pool(sel, "fig4_selected_v2", title=title)
    print(f"T = {T}; per rung {per}; means {min(means):.0f}–{max(means):.0f}; -> figures/selected_v2.csv, fig4_selected_v2.svg")


if __name__ == "__main__":
    main()
