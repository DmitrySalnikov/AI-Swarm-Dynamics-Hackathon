"""Phase-aligned dynamics of the 10 agents of each chat run (vectors from embed_phases.py).

    python3 embedding_eval/dynamics_phases.py              # -> dynamics/<set>_phases.csv, .png  (stretched solve)
    python3 embedding_eval/dynamics_phases.py --extended   # -> dynamics/<set>_phases_ext.csv, .png
      solve phase = 7 real windows per agent: own solve continued by extra solutions (embed_solve_ext.py)

Common clock for all agents of a set:
  negotiation round 1, round 2  |  solve step 1 ... K  |  merge round 1, round 2
K = the largest number of solve windows of any agent in the set. Time is slowed down for shorter solves so that
every agent finishes solving at step K: at solve step s (0-based) an agent with n windows shows its window
round(s * (n - 1) / (K - 1)), i.e. it holds each window for several steps. The axis therefore follows the
longest solve. All 10 agents are active at every step.

Metrics per group (chat run) at each step: the same as dynamics.py (raw and centred; centring = minus the mean
of all phase vectors of the set). speed / drift compare consecutive steps (lag 1). "held" = share of agents
whose vector at this step repeats their vector at the previous step (stretched time).
Plotted: mean over the 10 runs, band = +-1 sd across runs.
"""
import csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import metrics_at, unit, PANELS

HERE = Path(__file__).resolve().parent
CONCAT, OUT = HERE / "concat", HERE / "dynamics"
SKIP = {"n_active", "n_tasks", "top_task_share"}
METRICS = [("held", "Share of agents holding their previous window (stretched time)")] + \
          [(k, t.replace("v(t+2)", "v(t+1)").replace("m(t+2)", "m(t+1)")) for k, t in PANELS if k not in SKIP]
PLOT_METRICS = [m for m in METRICS if m[0] != "held"]
COLOR, STEP = "#2a78d6", 1024


def load_set(bench, extended=False):
    runs = {}
    for k in range(1, 11):
        d = CONCAT / bench / "phases" / f"run_{k}"
        runs[k] = [dict(np.load(d / f"agent_{i}.npz")) for i in range(1, 11)]
        if extended:
            for i, a in enumerate(runs[k], 1):
                a["solve"] = np.load(d / f"agent_{i}_ext.npz")["solve"]
    return runs


def timeline(agents, K):
    """List of steps; each step = (label, phase, [vector of each agent], [held flag of each agent])."""
    steps = [(f"neg {r + 1}", "negotiation", [a["neg"][r] for a in agents], [False] * len(agents)) for r in range(2)]
    prev = [None] * len(agents)
    for s in range(K):
        idx = [round(s * (len(a["solve"]) - 1) / (K - 1)) if K > 1 else 0 for a in agents]
        held = [p == j for p, j in zip(prev, idx)]
        steps.append((f"solve {s + 1}", "solve", [a["solve"][j] for a, j in zip(agents, idx)], held))
        prev = idx
    steps += [(f"merge {r + 1}", "merge", [a["merge"][r] for a in agents], [False] * len(agents)) for r in range(2)]
    return steps


def group_rows(agents, K, variant, mu):
    steps = timeline(agents, K)
    rows = []
    for t, (label, phase, vecs, held) in enumerate(steps):
        X = np.stack(vecs)
        if variant == "centred":
            X = unit(X - mu)
        r = {"step": t, "label": label, "phase": phase, "held": float(np.mean(held)), **metrics_at(X)}
        if t + 1 < len(steps):
            Y = np.stack(steps[t + 1][2])
            if variant == "centred":
                Y = unit(Y - mu)
            r["speed"] = float(np.mean(1 - np.sum(X * Y, 1)))
            r["drift"] = float(unit(X.mean(0)) @ unit(Y.mean(0)))
        else:
            r["speed"] = r["drift"] = np.nan
        rows.append(r)
    return rows


def token_axis(bench, K):
    """x position (generated tokens, window/turn centre) of every step, averaged over all agents of the set.
    Chat turns: mean token length of that round's turn. Solve: K windows of 2048 tokens every 1024."""
    turns = {"neg": [[], []], "merge": [[], []]}
    for p in (CONCAT / bench / "phases").glob("run_*/agent_*.json"):
        if p.stem.endswith("_ext"):
            continue
        meta = json.loads(p.read_text())
        for ph in turns:
            for r, m in enumerate(meta[ph]):
                turns[ph][r].append(m["n_tokens"])
    n1, n2, m1, m2 = (float(np.mean(turns[ph][r])) for ph in ("neg", "merge") for r in (0, 1))
    s0 = n1 + n2
    x = [n1 / 2, n1 + n2 / 2] + [s0 + STEP * s + STEP for s in range(K)]
    s1 = s0 + STEP * (K + 1)
    x += [s1 + m1 / 2, s1 + m1 + m2 / 2]
    return np.array(x), {"negotiation": (0, s0), "solving": (s0, s1), "merge": (s1, s1 + m1 + m2)}


def plot(bench, agg, labels, phases, K, suffix):
    x, spans = token_axis(bench, K)
    fig, axes = plt.subplots(len(PLOT_METRICS), 2, figsize=(11, 2.1 * len(PLOT_METRICS)), sharex=True)
    for c, variant in enumerate(("raw", "centred")):
        mean, sd = agg[variant]
        for r, (key, title) in enumerate(PLOT_METRICS):
            ax = axes[r, c]
            for ph in ("negotiation", "merge"):
                ax.axvspan(*spans[ph], color="#f0efec", lw=0, zorder=0)
            y, e = mean[key], sd[key]
            ax.plot(x, y, color=COLOR, lw=2, marker="o", ms=4, zorder=3)
            ax.fill_between(x, y - e, y + e, color=COLOR, alpha=0.15, lw=0, zorder=2)
            if key == "R" and variant == "centred":
                ax.axhline(1 / np.sqrt(10), color="#8a8984", lw=1, ls=":", label="1/√10 (random directions)")
                ax.legend(fontsize=8, frameon=False)
            ax.set_title(f"{title} — {variant}", fontsize=9, loc="left", color="#0b0b0b", pad=16 if r == 0 else 6)
            ax.grid(color="#e4e3df", lw=0.6); ax.tick_params(labelsize=8, colors="#52514e")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    for c in range(2):
        ax = axes[0, c]
        for ph, (a, b) in spans.items():
            ax.text((a + b) / 2, 1.0, ph, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=8, color="#52514e")
        axes[-1, c].set_xlabel("generated tokens (mean over agents; window / turn centre)", fontsize=9,
                               color="#52514e")
    how = ("solve = own solution continued by new solutions of the same question, " if suffix
           else "solve stretched to ") + f"{K} steps"
    fig.suptitle(f"{bench}: 10 chat runs × 10 agents, phase-aligned (shaded: chat rounds)\n{how}; "
                 f"line: mean over runs, band ±1 sd", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT / f"{bench}_phases{suffix}.png", dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extended", action="store_true", help="use agent_i_ext.npz for the solve phase")
    extended = ap.parse_args().extended
    suffix = "_ext" if extended else ""
    OUT.mkdir(exist_ok=True)
    for bench in ("set_3", "set_4"):
        runs = load_set(bench, extended)
        K = max(len(a["solve"]) for agents in runs.values() for a in agents)
        mu = np.concatenate([np.concatenate([a["neg"], a["solve"], a["merge"]])
                             for agents in runs.values() for a in agents]).mean(0)
        agg, labels, phases = {}, None, None
        with open(OUT / f"{bench}_phases{suffix}.csv", "w", newline="") as f:
            cols = ["variant", "run", "step", "label", "phase"] + [k for k, _ in METRICS]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for variant in ("raw", "centred"):
                per = {k: group_rows(agents, K, variant, mu) for k, agents in runs.items()}
                for k, rows in per.items():
                    for r in rows:
                        w.writerow({"variant": variant, "run": k, **r})
                labels = [r["label"] for r in per[1]]; phases = [r["phase"] for r in per[1]]
                arr = {key: np.array([[r[key] for r in rows] for rows in per.values()], float) for key, _ in METRICS}
                agg[variant] = ({k: np.nanmean(v, 0) for k, v in arr.items()},
                                {k: np.nanstd(v, 0) for k, v in arr.items()})
        plot(bench, agg, labels, phases, K, suffix)
        print(f"{bench}: K = {K} solve steps; wrote {OUT / bench}_phases{suffix}.csv and .png")


if __name__ == "__main__":
    main()
