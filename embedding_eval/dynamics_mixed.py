"""Chat groups next to 10 agents solving the same set independently, on one token axis (0 - 10 000 tokens).

    python3 embedding_eval/dynamics_mixed.py    # -> embedding_eval/dynamics/<set>_mixed.csv, .png

Three lines, each a mean over 10 groups of 10 agents (band +-1 sd across groups):
  chat         the 10 agents of each chat run, phase-aligned (negotiation, solve extended to 7 windows, merge);
               points sit at the mean token position of each turn / window over all agents (dynamics_phases.token_axis)
  sequential   10 agents = 10 of the 20 concat/<set>/sequential files (10 seeded random subsets), windows of 2048
               tokens every 1024, point at the window centre
  shuffled     the same with concat/<set>/shuffled files
Cut at 10 000 generated tokens. Centring: minus the mean of all chat phase vectors and all sequential / shuffled
windows of the set, then re-normalised. speed / drift: consecutive points (lag 1) for every line.
"""
import csv, random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dynamics import metrics_at, unit
from dynamics_phases import load_set, group_rows, token_axis, PLOT_METRICS

HERE = Path(__file__).resolve().parent
CONCAT, OUT = HERE / "concat", HERE / "dynamics"
K, STEP, T_MAX, GROUPS, SIZE = 7, 1024, 10_000, 10, 10
FIG_METRICS = [("R", "R — length of the mean vector"),
               ("var_perp_std", "Spread across the mean direction"),
               ("eff_dim", "Effective dimension")]
PHASE_NAME = {"negotiation": "task distribution", "solving": "solving", "merge": "results exchange"}
LABEL = {"chat": "swarm", "sequential": "independent, problem order is fixed",
         "shuffled": "independent, problem order is shuffled"}
STYLE = {"chat": dict(color="#2a78d6", ls="-", marker="o"),
         "sequential": dict(color="#eb6834", ls="--", marker="s"),
         "shuffled": dict(color="#1baf7a", ls=":", marker="^")}


def solo_rows(files, variant, mu):
    V = [unit(v - mu) if variant == "centred" else v for v in files]
    T = (T_MAX - STEP) // STEP                                   # last window whose centre is <= T_MAX
    rows = []
    for t in range(T + 1):
        X = np.stack([v[t] for v in V])
        r = {"x": t * STEP + STEP, **metrics_at(X)}
        if t + 1 <= T:
            Y = np.stack([v[t + 1] for v in V])
            r["speed"] = float(np.mean(1 - np.sum(X * Y, 1)))
            r["drift"] = float(unit(X.mean(0)) @ unit(Y.mean(0)))
        else:
            r["speed"] = r["drift"] = np.nan
        rows.append(r)
    return rows


def aggregate(groups):
    keys = [k for k, _ in PLOT_METRICS]
    arr = {k: np.array([[r[k] for r in g] for g in groups], float) for k in keys}
    return np.array([r["x"] for r in groups[0]]), {k: np.nanmean(v, 0) for k, v in arr.items()}, \
        {k: np.nanstd(v, 0) for k, v in arr.items()}


def main():
    OUT.mkdir(exist_ok=True)
    for bench in ("set_3", "set_4"):
        runs = load_set(bench, extended=True)
        solo = {k: [np.load(p) for p in sorted((CONCAT / bench / k).glob("run_*.vectors.npy"))]
                for k in ("sequential", "shuffled")}
        chat_vecs = [np.concatenate([a["neg"], a["solve"], a["merge"]]) for ag in runs.values() for a in ag]
        mu = np.concatenate(chat_vecs + solo["sequential"] + solo["shuffled"]).mean(0)
        rng = random.Random(f"{bench}-mixed")
        subsets = [rng.sample(range(len(solo["sequential"])), SIZE) for _ in range(GROUPS)]
        x_chat, spans = token_axis(bench, K)
        data, csv_rows = {}, []
        for variant in ("raw", "centred"):
            groups = []
            for k, agents in runs.items():
                rows = group_rows(agents, K, variant, mu)
                for r, x in zip(rows, x_chat):
                    r["x"] = float(x)
                groups.append([r for r in rows if r["x"] <= T_MAX])
            data[("chat", variant)] = aggregate(groups)
            csv_rows += [{"line": "chat", "variant": variant, "group": k, **r} for k, g in zip(runs, groups) for r in g]
            for kind in ("sequential", "shuffled"):
                groups = [solo_rows([solo[kind][i] for i in sub], variant, mu) for sub in subsets]
                data[(kind, variant)] = aggregate(groups)
                csv_rows += [{"line": kind, "variant": variant, "group": g + 1, **r}
                             for g, rows in enumerate(groups) for r in rows]
        cols = ["line", "variant", "group", "x", "label", "phase"] + [k for k, _ in PLOT_METRICS]
        with open(OUT / f"{bench}_mixed.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(csv_rows)

        H = 2.6 * len(FIG_METRICS) + 0.9                 # figure height, inches (header ~0.9 in)
        fig, axes = plt.subplots(len(FIG_METRICS), 1, figsize=(9, H), sharex=True)
        variant = "raw"
        for r, (key, title) in enumerate(FIG_METRICS):
            ax = axes[r]
            for ph in ("negotiation", "merge"):
                ax.axvspan(spans[ph][0], min(spans[ph][1], T_MAX), color="#f0efec", lw=0, zorder=0)
            for line, st in STYLE.items():
                x, m, sd = data[(line, variant)]
                ax.plot(x, m[key], lw=2, ms=4, label=LABEL[line], zorder=3, **st)
                ax.fill_between(x, m[key] - sd[key], m[key] + sd[key], color=st["color"], alpha=0.12, lw=0)
            ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b", pad=16 if r == 0 else 6)
            ax.grid(color="#e4e3df", lw=0.6); ax.tick_params(labelsize=8, colors="#52514e")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.set_xlim(0, T_MAX)
            ax.set_ylim(bottom=0)                       # no zero suppression
            if key == "R":
                ax.set_ylim(0, 1)
        for ph, (a, b) in spans.items():
            xa, ha = {"negotiation": (a, "left"), "merge": (T_MAX, "right")}.get(ph, ((a + b) / 2, "center"))
            axes[0].text(xa, 1.0, f"swarm: {PHASE_NAME[ph]}", transform=axes[0].get_xaxis_transform(),
                         ha=ha, va="bottom", fontsize=8, color="#52514e")
        axes[-1].set_xlabel("generated tokens", fontsize=9, color="#52514e")
        fig.suptitle(f"{bench}: swarm vs 10 agents solving the set independently, first {T_MAX:,} tokens "
                     f"(raw embeddings)\nSwarm points sit at the mean token position of each turn / window, so their "
                     f"alignment with the independent agents' windows is approximate.\nEach line: mean over 10 groups "
                     f"of 10 agents, band ±1 sd. Shaded: swarm task distribution / results exchange.", fontsize=9, y=1 - 0.1 / H, va="top")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.75 / H), ncol=3, fontsize=9, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.85 / H))
        fig.savefig(OUT / f"{bench}_mixed.png", dpi=110, facecolor="#fcfcfb")
        plt.close(fig)
        print(f"{bench}: wrote {OUT / bench}_mixed.csv and .png")


if __name__ == "__main__":
    main()
