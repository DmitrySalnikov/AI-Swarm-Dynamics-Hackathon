"""Illustration of the two "different order" regimes of swarm_pipeline.baseline_sim: cyclic shift vs random Latin square.

Ten agents, ten tasks (letters A–J stand for the tasks of set B). Top: the task of every agent at every position.
Middle: which task follows which, counted over agents. Bottom: the tasks two agents share when one runs a task behind.

  python scripts/plot_latin_vs_cyclic.py artifacts/baseline-v41/report/latin_vs_cyclic.png
"""
import sys
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_pipeline.baseline_sim import random_latin_rows

out = Path(sys.argv[1])
tasks = list("ABCDEFGHIJ")
n = len(tasks)
rng = random.Random(7)
base = rng.sample(tasks, n)
cyclic = [base[i:] + base[:i] for i in range(n)]
latin = random_latin_rows(tasks, random.Random(11))
cmap = plt.get_cmap("tab10")
col = {t: cmap(i) for i, t in enumerate(tasks)}
MARK = "D"                     # one task highlighted in the grids


def grid(ax, rows, title):
    for i, r in enumerate(rows):
        for p, t in enumerate(r):
            ax.add_patch(Rectangle((p, i), 1, 1, color=col[t], alpha=0.9 if t == MARK else 0.45, lw=0))
            ax.text(p + 0.5, i + 0.5, t, ha="center", va="center", fontsize=10,
                    fontweight="bold" if t == MARK else "normal")
            if t == MARK:
                ax.add_patch(Rectangle((p, i), 1, 1, fill=False, ec="black", lw=2))
    ax.set_xlim(0, n); ax.set_ylim(n, 0); ax.set_aspect("equal")
    ax.set_xticks(np.arange(n) + 0.5, range(2, n + 2)); ax.set_yticks(np.arange(n) + 0.5, [f"agent {i}" for i in range(1, n + 1)])
    ax.set_xlabel("position in the agent's sequence (after the opening task)")
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(length=0, labelsize=8)


def successors(ax, rows, title):
    m = np.zeros((n, n), int)
    for r in rows:
        for x, y in zip(r, r[1:]):
            m[tasks.index(x), tasks.index(y)] += 1
    ax.imshow(m, cmap="Greys", vmin=0, vmax=n)
    for i in range(n):
        for j in range(n):
            if m[i, j]:
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=8, color="white" if m[i, j] > 4 else "black")
    ax.set_xticks(range(n), tasks); ax.set_yticks(range(n), tasks)
    ax.set_xlabel("next task"); ax.set_ylabel("task")
    ax.set_title(title, loc="left", fontsize=11)
    ax.tick_params(length=0, labelsize=8)


def lag(ax, rows, title, i=0, j=1, d=1):
    """Agent j runs d tasks behind agent i: which tasks do they work on at the same time?"""
    for k, (r, shift, name) in enumerate(((rows[i], 0, f"agent {i + 1}"), (rows[j], d, f"agent {j + 1}, {d} task late"))):
        for p, t in enumerate(r):
            ax.add_patch(Rectangle((p + shift, k * 1.6), 0.95, 1, color=col[t], alpha=0.75, lw=0))
            ax.text(p + shift + 0.475, k * 1.6 + 0.5, t, ha="center", va="center", fontsize=10)
        ax.text(-0.2, k * 1.6 + 0.5, name, ha="right", va="center", fontsize=9)
    same = 0
    for p in range(d, n):
        if rows[i][p] == rows[j][p - d]:
            same += 1
            ax.add_patch(Rectangle((p - 0.05, -0.15), 1.05, 2.9, fill=False, ec="black", lw=1.8))
    ax.set_xlim(-3.2, n + d + 0.2); ax.set_ylim(2.9, -0.4); ax.axis("off")
    ax.set_title(f"{title}: same task at the same time on {same} of {n - d} steps", loc="left", fontsize=11)


def convoy_stats(rows, d_max=3):
    """Mean number of steps on which two agents coincide when one is d tasks behind, over all ordered pairs."""
    out = []
    for d in range(1, d_max + 1):
        runs = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    best = cur = 0
                    for p in range(d, n):
                        cur = cur + 1 if rows[i][p] == rows[j][p - d] else 0
                        best = max(best, cur)
                    runs.append(best)
        out.append((d, max(runs)))
    return out


fig = plt.figure(figsize=(13, 15.5))
gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 0.35], hspace=0.38, wspace=0.28)
for c, (rows, name) in enumerate(((cyclic, "Cyclic shift"), (latin, "Random Latin square"))):
    grid(fig.add_subplot(gs[0, c]), rows, f"{name}: task of every agent at every position")
    successors(fig.add_subplot(gs[1, c]), rows, f"{name}: which task follows which (count over agents)")
    lag(fig.add_subplot(gs[2, c]), rows, name)
cs, ls = convoy_stats(cyclic), convoy_stats(latin)
fig.suptitle("Two ways to give 10 agents different orders of the same 10 tasks", fontsize=14, y=0.995)
fig.text(0.5, 0.965,
         "Both are Latin squares: at every position each task is held by exactly one agent, and each agent does each task once.\n"
         f"Cyclic: every agent walks the same loop {'→'.join(base)}→… from a different start, so task {MARK} lies on a diagonal "
         "and every task always has the same successor.\n"
         "Latin: the step from one task to the next differs between agents, so successors are spread out.",
         ha="center", va="top", fontsize=10)
fig.text(0.5, 0.012,
         "Why it matters: agents drift apart in tokens. In the cyclic order, an agent that falls d tasks behind lands on its neighbour's "
         "sequence and from then on\nshares every task with it: a persistent convoy "
         f"(longest run of shared tasks over all pairs: {', '.join(f'd={d}: {m}' for d, m in cs)}). "
         "In the random Latin square the same number of coincidences\nis scattered over many pairs, so they are short "
         f"and accidental (longest run: {', '.join(f'd={d}: {m}' for d, m in ls)}).",
         ha="center", va="bottom", fontsize=10)
fig.savefig(out, dpi=130, bbox_inches="tight")
print(out, cs, ls)
