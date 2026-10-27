"""Task timelines of the baseline simulation runs, next to the embedding metrics.

For every rung, one figure with a column per regime. Top: the spread across the mean direction (from
<root>/report/metrics.csv, windows of --width tokens) and the same-task share (the fraction of the agents that work on
the most common task at that token position). Bottom: one row per agent, coloured by the task it works on, against
its own generated tokens (the same axis as the metric plots); white ticks mark task switches. Token positions come
from generated/agent_<i>.generated.index.json (character spans per call) and the DeepSeek-V3 tokenizer.

  python scripts/plot_baseline_timeline.py --root artifacts/baseline-v41 [--width 500]
"""
import argparse, csv, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
from swarm_pipeline.analyze import token_data
from build_baseline_report import label_order, run_label

def segments(run, tok):
    """Per agent: list of (task_id, first token, end token) in call order."""
    out = []
    for f in sorted((run / "generated").glob("agent_*.generated.txt"), key=lambda p: int(p.name.split("_")[1].split(".")[0])):
        text = f.read_text(encoding="utf-8")
        starts = token_data(tok, text)[:, 0]
        index = json.loads(f.with_name(f.name.replace(".txt", ".index.json")).read_text())
        spans = {}
        for piece in index:                                   # reasoning and reply of one call -> one character span
            task, c0, c1 = spans.get(piece["call"], (piece["task_id"], piece["start"], piece["end"]))
            spans[piece["call"]] = (task, min(c0, piece["start"]), max(c1, piece["end"]))
        segs = []
        for call in sorted(spans):
            task, c0, c1 = spans[call]
            t0, t1 = int(np.searchsorted(starts, c0)), int(np.searchsorted(starts, c1))
            if segs and segs[-1][0] == task:              # a repeated call on the same task extends its segment
                segs[-1] = (task, segs[-1][1], t1)
            else:
                segs.append((task, t0, t1))
        out.append(segs)
    return out


def share(segs, x):
    """Fraction of the agents on the most common task at token x (agents that have finished are not counted)."""
    now = [next((s[0] for s in a if s[1] <= x < s[2]), None) for a in segs]
    now = [t for t in now if t is not None]
    return max(now.count(t) for t in set(now)) / len(now) if now else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--width", type=int, default=500)
    a = ap.parse_args()
    tok = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    metrics = list(csv.DictReader(open(a.root / "report" / "metrics.csv")))
    runs = {}
    for d in sorted(a.root.glob("rung_*/*/seed_*")):
        runs[(int(d.parts[-3].split("_")[1]), run_label(d))] = d
    summary = {}
    for rung in sorted({r for r, _ in runs}):
        regimes = label_order({g for r, g in runs if r == rung})
        tasks = sorted({s[0] for g in regimes for a in segments(runs[(rung, g)], tok) for s in a})
        cmap = plt.get_cmap("tab20")
        colour = {t: cmap(i % 20) for i, t in enumerate(tasks)}
        fig, axes = plt.subplots(2, len(regimes), figsize=(7.5 * len(regimes), 8.5), sharex="col",
                                 gridspec_kw={"height_ratios": [1, 1.6]})
        axes = np.atleast_2d(axes).reshape(2, len(regimes))
        for c, regime in enumerate(regimes):
            segs = segments(runs[(rung, regime)], tok)
            end = max(s[2] for a_ in segs for s in a_)
            common = min(a_[-1][2] for a_ in segs)
            xs = np.arange(0, end, 100)
            sh = np.array([share(segs, x) for x in xs])
            summary[(rung, regime)] = segs
            top = axes[0, c]
            pts = [r for r in metrics if r["width"] == str(a.width) and r["rung"] == str(rung) and r["regime"] == regime]
            top.plot([float(p["x"]) for p in pts], [float(p["var_perp_std"]) for p in pts], color="#2a78d6", lw=1.4,
                     label=f"spread across the mean direction (W={a.width})")
            top.plot(xs, sh, color="#eb6834", lw=1.4, label="same-task share")
            top.axvline(common, color="#8a8984", ls=":", lw=1)
            top.set_ylim(0, 1.05); top.set_title(f"rung {rung}, {regime}", loc="left", fontsize=11)
            top.grid(color="#e4e3df", lw=0.6)
            if c == 0:
                top.legend(frameon=False, fontsize=8, loc="lower left")
            ax = axes[1, c]
            for i, a_ in enumerate(segs, 1):
                for task, t0, t1 in a_:
                    ax.barh(i, t1 - t0, left=t0, color=colour[task], height=0.8, edgecolor="white", lw=0.6)
                    if t1 - t0 > 0.06 * end:
                        ax.text((t0 + t1) / 2, i, task.split("/")[1][:12], ha="center", va="center", fontsize=6.5)
            ax.axvline(common, color="#8a8984", ls=":", lw=1)
            ax.set_yticks(range(1, len(segs) + 1)); ax.set_yticklabels([f"agent {i}" for i in range(1, len(segs) + 1)], fontsize=8)
            ax.invert_yaxis(); ax.set_xlabel("generated tokens of each agent")
        fig.suptitle(f"Rung {rung}: task of every agent over its own generated tokens; dotted line = end of the shortest "
                     f"agent (metric plots stop there)", fontsize=10)
        fig.tight_layout()
        fig.savefig(a.root / "report" / f"timeline_rung_{rung}.png", dpi=110)
        plt.close(fig)
    with open(a.root / "report" / "task_switches.json", "w") as f:
        json.dump({f"rung_{r}/{g}": [[{"task": s[0], "start": s[1], "end": s[2]} for s in a_] for a_ in segs]
                   for (r, g), segs in summary.items()}, f, indent=1)
    print(f"wrote timeline_rung_*.png and task_switches.json to {a.root / 'report'}")


if __name__ == "__main__":
    main()
