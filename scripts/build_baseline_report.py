"""Embedding-space metrics of the baseline simulation runs (swarm_pipeline.baseline_sim), by window width.

For every run <root>/rung_<k>/<regime>/seed_<s>/, each agent's stream is its own generated text in call order
(generated/agent_<i>.generated.txt: reasoning + visible replies), tokenized with the DeepSeek-V3 tokenizer and cut
into windows of W tokens with stride W/2 (swarm_pipeline.analyze.windows). Windows are embedded with
qwen3-embedding-4b (analyze.embed: no instruction, unit vectors, cached). At every window index the metrics of
embedding_eval.dynamics.metrics_at are computed over the agents: R (length of the mean vector), the spread across the
mean direction and the effective dimension. Only window indices where every agent still has text are used, so each
run is cut at its shortest agent and every point has the same number of agents.

One figure per (width, rung): three panels (metrics), one line per regime. Outputs in <root>/report/:
metrics_w<W>_rung_<k>.png and metrics.csv.

  python scripts/build_baseline_report.py --root artifacts/baseline-v41 --dry-run   # windows and tokens only
  python scripts/build_baseline_report.py --root artifacts/baseline-v41             # key: OPENROUTER_API_KEY or Keychain
"""
import argparse, csv, json, os, subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding_eval.dynamics import metrics_at
from swarm_pipeline.analyze import embed, token_data, vector_key, windows

METRICS = [("R", "R — length of the mean vector"), ("var_perp_std", "Spread across the mean direction"),
           ("eff_dim", "Effective dimension")]
STYLE = {"same": dict(color="#eb6834", ls="--", marker="s"), "latin": dict(color="#2a78d6", ls="-", marker="o"),
         "cyclic": dict(color="#1baf7a", ls=":", marker="^"), "random": dict(color="#8a5cc2", ls="-.", marker="d"),
         "isolated": dict(color="#52514e", ls="-", marker="v")}   # isolated solutions joined (build_isolated_run.py)
VARIANT = dict(ls=":", marker="x")   # extra runs of a regime (other seeds): same colour, dotted
PRICE = 0.02 / 1e6          # USD per embedded token


def run_label(d):
    """'same' for seed_1, 'same s2 order 1' for seed_2_order_1, ..."""
    regime = d.parts[-2].split("_k")[0]
    return regime if d.name == "seed_1" else regime + " " + d.name.replace("seed_", "s").replace("_order_", " order ")


def style(label):
    base = STYLE[label.split()[0]]
    return base if " " not in label else {**base, **VARIANT}


def label_order(labels):
    return sorted(labels, key=lambda g: (list(STYLE).index(g.split()[0]), g))


def runs(root):
    """{(rung, regime): run dir} for every completed run under root."""
    out = {}
    for d in sorted(root.glob("rung_*/*/seed_*")):
        if json.loads((d / "result.json").read_text()).get("status") == "completed":
            out[(int(d.parts[-3].split("_")[1]), run_label(d))] = d
    return out


def agent_windows(run, width, tok):
    """Per agent: list of windows (text, start, end, x) of its generated stream."""
    out = []
    for f in sorted((run / "generated").glob("agent_*.generated.txt"), key=lambda p: int(p.name.split("_")[1].split(".")[0])):
        text = f.read_text(encoding="utf-8")
        out.append(windows(text, token_data(tok, text), width))
    return out


def openrouter_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and sys.platform == "darwin":
        p = subprocess.run(["security", "find-generic-password", "-s", "openrouter-api", "-a", os.environ.get("USER", ""),
                            "-w"], capture_output=True, text=True)
        key = p.stdout.strip() if p.returncode == 0 else None
    return key


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--widths", type=int, nargs="+", default=[4096, 500])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tok = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    found = runs(a.root)
    wins = {(w, key): agent_windows(d, w, tok) for w in a.widths for key, d in found.items()}
    texts = [x["text"] for ws in wins.values() for agent in ws for x in agent]
    n_tok = sum(x["end"] - x["start"] for ws in wins.values() for agent in ws for x in agent)
    for (w, (rung, regime)), ws in sorted(wins.items()):
        common = min(len(x) for x in ws)
        print(f"W={w:5d} rung {rung} {regime:6s}: agents {len(ws)}, windows per agent {min(len(x) for x in ws)}–"
              f"{max(len(x) for x in ws)}, common {common} (up to token {ws[0][common - 1]['end'] if common else 0:,})")
    print(f"windows {len(texts):,}, unique {len(set(texts)):,}, tokens {n_tok:,}, estimated cost ${n_tok * PRICE:.3f}")
    if a.dry_run:
        return
    key = openrouter_key()
    if not key:
        sys.exit("no OpenRouter key (OPENROUTER_API_KEY or Keychain openrouter-api)")
    os.environ["OPENROUTER_API_KEY"] = key
    out = a.root / "report"
    out.mkdir(parents=True, exist_ok=True)
    vec = embed(texts, out, cache_dir=a.root / "embedding-cache")
    rows = []
    for (w, (rung, regime)), ws in sorted(wins.items()):
        common = min(len(x) for x in ws)
        for t in range(common):
            X = np.stack([vec[vector_key(agent[t]["text"])] for agent in ws])
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            x = float(np.mean([agent[t]["x"] for agent in ws]))
            rows.append({"width": w, "rung": rung, "regime": regime, "window": t, "x": x, **metrics_at(X)})
    with open(out / "metrics.csv", "w", newline="") as f:
        cols = ["width", "rung", "regime", "window", "x", "n_active"] + [k for k, _ in METRICS] + ["spread_total"]
        wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)
    for w in a.widths:
        for rung in sorted({r for (ww, (r, _)) in wins if ww == w}):
            fig, axes = plt.subplots(len(METRICS), 1, figsize=(9, 2.6 * len(METRICS) + 0.8), sharex=True)
            for ax, (k, title) in zip(axes, METRICS):
                for regime in label_order({r["regime"] for r in rows if r["rung"] == rung and r["width"] == w}):
                    pts = [r for r in rows if r["width"] == w and r["rung"] == rung and r["regime"] == regime]
                    ax.plot([p["x"] for p in pts], [p[k] for p in pts], lw=1.6, ms=3 if w >= 2000 else 0, label=regime,
                            **style(regime))
                ax.set_title(title, fontsize=10, loc="left")
                ax.grid(color="#e4e3df", lw=0.6)
                ax.set_ylim(0, 1 if k == "R" else None)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
            axes[0].legend(frameon=False, fontsize=9, ncol=4)
            axes[-1].set_xlabel("generated tokens (window centre, mean over agents)")
            n_tasks = sorted({len(a["order"]) for (r, _), d in found.items() if r == rung
                              for a in json.loads((d / "state.json").read_text())["agents"].values()})
            fig.suptitle(f"Rung {rung}, windows of {w} tokens (stride {w // 2}); 10 independent agents, "
                         f"{'/'.join(map(str, n_tasks))} tasks each; raw embeddings", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"metrics_w{w}_rung_{rung}.png", dpi=110)
            plt.close(fig)
    print(f"wrote {out}/metrics.csv and {len(a.widths) * len({r for (_, (r, _)) in wins})} figures")


if __name__ == "__main__":
    main()
