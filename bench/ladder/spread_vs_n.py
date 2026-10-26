"""How the spread of mean lengths grows with the number of tasks n taken per rung (one task per topic).

  uv run --with sympy --with numpy --with scipy --with matplotlib --with reasoning-gym==0.1.19 \\
      python bench/ladder/spread_vs_n.py [--band]   # --band: only tasks with the mean in 14–23 thousand

Candidates: clean tasks of the pool (>= 80 % correct runs, CV <= 0.25), any mean length. Topic = area (rung 1),
physics section (rung 2), task type (rung 3); rung 4 is one template, so every matrix counts as its own topic.
For each rung and n: tasks sorted by mean length; the narrowest window (smallest max/min ratio of the means) that
contains n different topics, found exactly with two pointers; inside the window the two end tasks are kept and the
other topics take the task closest (in log length) to the middle of the window. Reported for the chosen set:
ratio max/min of the means, CV of the means, the longest single run.
Outputs: figures/spread_vs_n.csv, figures/fig5_spread_vs_n.{png,svg}.
"""
import csv
import math
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "bench" / "ladder"
sys.path[:0] = [str(LADDER), str(LADDER / "generators")]
import matplotlib.pyplot as plt  # noqa: E402
import plots_ladder as P  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

MARKERS = {1: ("o", P.BLUE), 2: ("s", P.ORANGE), 3: ("D", P.AQUA), 4: ("^", P.INK2)}


def topic(t):
    return t["gen"].area if t["rung"] == 1 else t["gen"].domain if t["rung"] in (2, 3) else t["id"]


def best_set(ts, n):
    """ts sorted by mean. The n-topic set with the smallest max/min ratio of means, or None."""
    best, cnt, i = None, Counter(), 0
    for j, t in enumerate(ts):
        cnt[topic(t)] += 1
        # move the left end while the window keeps at least n topics
        while len(cnt) >= n and (cnt[topic(ts[i])] > 1 or len(cnt) > n):
            cnt[topic(ts[i])] -= 1
            if cnt[topic(ts[i])] == 0:
                del cnt[topic(ts[i])]
            i += 1
        if len(cnt) >= n:
            ratio = ts[j]["mean"] / ts[i]["mean"]
            if best is None or ratio < best[0]:
                best = (ratio, i, j)
    if best is None:
        return None
    _, i, j = best
    window, mid = ts[i:j + 1], math.sqrt(ts[i]["mean"] * ts[j]["mean"])
    chosen = {topic(ts[i]): ts[i], topic(ts[j]): ts[j]}
    rest = {}
    for t in window:
        k = topic(t)
        if k in chosen:
            continue
        if k not in rest or abs(math.log(t["mean"] / mid)) < abs(math.log(rest[k]["mean"] / mid)):
            rest[k] = t
    extra = sorted(rest.values(), key=lambda t: abs(math.log(t["mean"] / mid)))[:n - len(chosen)]
    return list(chosen.values()) + extra


def main():
    band = "--band" in sys.argv  # only tasks with the mean length inside the 14–23 thousand window
    sfx = "_band" if band else ""
    tasks = [t for t in P.collect() if P.is_clean(t) and (not band or 14000 <= t["mean"] <= 23000)]
    rows, curves = [], {}
    for k in range(1, 5):
        ts = sorted((t for t in tasks if t["rung"] == k), key=lambda t: t["mean"])
        n_topics = len({topic(t) for t in ts})
        for n in range(2, n_topics + 1):
            sel = best_set(ts, n)
            means = [t["mean"] for t in sel]
            longest = max(r["length"] for t in sel for r in t["runs"])
            ratio = max(means) / min(means)
            curves.setdefault(k, []).append((n, ratio, longest))
            rows.append([k, n, n_topics, round(ratio, 3), round(st.stdev(means) / st.mean(means), 3),
                         round(min(means)), round(max(means)), longest,
                         "; ".join(f"{topic(t)}={t['id'].split('/')[1]}" for t in sorted(sel, key=lambda t: t["mean"]))])
    with open(P.OUT / f"spread_vs_n{sfx}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "n", "topics_available", "ratio_max_min_mean", "cv_of_means", "mean_min", "mean_max",
                    "longest_run", "tasks"])
        w.writerows(rows)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    CAP = 2.6  # a few sets with almost all topics reach 3.5–7.9; drawn at the cap with their value written
    for k, pts in curves.items():
        m, c = MARKERS[k]
        ax.plot([p[0] for p in pts], [min(p[1], CAP) for p in pts], "-", color=c, lw=1.6, marker=m, ms=5, mec="white")
        for n, r, _ in pts:
            if r > CAP:
                ax.annotate(f"{r:.1f}".replace(".", ","), (n, CAP), xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=8, color=P.INK)
        ax2.plot([p[0] for p in pts], [p[2] / 1000 for p in pts], "-", color=c, lw=1.6, marker=m, ms=5, mec="white")
    ax.set_ylim(0.95, min(CAP, max(p[1] for pts in curves.values() for p in pts)) + 0.15)
    for a in (ax, ax2):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.axvline(20, color=P.INK2, lw=0.9, ls=":")
        a.grid(color=P.GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.set_xlabel("n — сколько задач (разных тем) берём на ступень")
    ax.set_ylabel("наибольшая средняя / наименьшая средняя")
    ax.set_title("Разброс средних длин в лучшем наборе", loc="left", fontsize=10.5)
    ax2.set_ylabel("самый длинный прогон в наборе, тыс. токенов")
    ax2.set_title("Самый длинный отдельный прогон в этом наборе", loc="left", fontsize=10.5)
    handles = [Line2D([], [], color=MARKERS[k][1], marker=MARKERS[k][0], mec="white", ms=7, lw=1.6,
                      label=f"{P.RUNG_TITLE[k]} (тем: {max(p[0] for p in curves[k])})") for k in curves]
    P.top_matter(fig, "Что будет, если брать больше задач: лучший достижимый разброс при n разных темах "
                      "(≥ 80 % верных, CV ≤ 0,25" + (", средняя 14–23 тыс.)" if band else ")"), handles, ncol=4, top=0.80)
    P.save(fig, f"fig5_spread_vs_n{sfx}")
    for k, pts in curves.items():
        print(f"rung {k}: " + ", ".join(f"n={n}: {r:.2f}" for n, r, _ in pts if n in (5, 10, 12, 14, 16, 18, 20, 22)))


if __name__ == "__main__":
    main()
