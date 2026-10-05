"""How the window embeddings of independent agents spread out over time, for different ways of combining files.

    python3 embedding_eval/dynamics.py          # -> embedding_eval/dynamics/<set>.csv, <set>.png

An "agent" is one file; time t is the window index (window t covers tokens [1024 t, 1024 t + 2048)).
At each t the vectors of all files still active form X (N x 2560). Every metric is computed twice:
  raw      - the unit vectors as returned by the model;
  centred  - minus the mean of ALL windows of the set (every file type), then re-normalised to length 1.
Time steps with fewer than MIN_N active files are dropped.

Metrics (unit vectors, so the identities below are exact):
  n_active          N(t)
  R                 |m|, m = mean vector ("standard" mean)
  spread_total      mean |v_i - m|^2 = 1 - R^2
  std_par_std       std of projections v_i . m_hat        (standard axis)
  var_par_std / var_perp_std   spread_total split into the part along m_hat and the rest
  rms_q             sqrt(lambda_1): RMS of projections on the principal axis u of (1/N) sum v v^T ("quadratic" mean)
  mean_q / std_q    mean and std of projections v_i . u   (u oriented so that mean_q >= 0)
  off_axis_q        1 - lambda_1: second moment not captured by u
  eff_dim           participation ratio (sum l)^2 / sum l^2 of the covariance eigenvalues around m
  speed             mean over agents of 1 - cos(v_i(t), v_i(t+2))  (t+2: the next window with no shared text)
  drift             cos(m(t), m(t+2))
  n_tasks / top_task_share   distinct questions at t and the share of agents on the most common one
                    (question of a window = the piece it overlaps most)
"""
import csv, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CONCAT, OUT = HERE / "concat", HERE / "dynamics"
SETS, KINDS = ("set_3", "set_4"), ("sequential", "shuffled")
MIN_N, LAG, STEP = 4, 2, 1024
STYLE = {"sequential": dict(color="#2a78d6", ls="-"), "shuffled": dict(color="#eb6834", ls="--")}


def load(path):
    X = np.load(path.with_suffix(".vectors.npy"))
    wins = json.loads(path.with_suffix(".windows.json").read_text())["windows"]
    pieces = json.loads(path.with_suffix(".index.json").read_text())
    tasks = []
    for w in wins:
        ov = {}
        for j in w["pieces"]:
            p = pieces[j]
            key = p.get("task") or p.get("phase")
            ov[key] = ov.get(key, 0) + min(p["end"], w["char_end"]) - max(p["start"], w["char_start"])
        tasks.append(max(ov, key=ov.get))
    return X, tasks


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-12)


def metrics_at(X):
    N = len(X)
    m = X.mean(0); R = float(np.linalg.norm(m)); mh = m / max(R, 1e-12)
    p = X @ mh
    G = X @ X.T / N                                     # same nonzero spectrum as (1/N) X^T X
    lam, A = np.linalg.eigh(G)
    u = unit(X.T @ A[:, -1]); q = X @ u
    if q.mean() < 0:
        q = -q
    Xc = X - m
    lc = np.clip(np.linalg.eigvalsh(Xc @ Xc.T / N), 0, None)
    spread = 1 - R * R
    return {"n_active": N, "R": R, "spread_total": spread, "std_par_std": float(p.std()),
            "var_par_std": float(p.var()), "var_perp_std": spread - float(p.var()),
            "rms_q": float(np.sqrt(lam[-1])), "mean_q": float(q.mean()), "std_q": float(q.std()),
            "off_axis_q": 1 - float(lam[-1]), "eff_dim": float(lc.sum() ** 2 / max((lc ** 2).sum(), 1e-12))}


def series(files, variant, mu):
    V, T = [], []
    for X, tasks in files:
        V.append(unit(X - mu) if variant == "centred" else X); T.append(tasks)
    rows = []
    for t in range(max(len(v) for v in V)):
        act = [i for i, v in enumerate(V) if len(v) > t]
        if len(act) < MIN_N:
            continue
        X = np.stack([V[i][t] for i in act])
        r = {"t": t, "token_center": t * STEP + STEP, **metrics_at(X)}
        both = [i for i in act if len(V[i]) > t + LAG]
        r["speed"] = float(np.mean([1 - V[i][t] @ V[i][t + LAG] for i in both])) if both else np.nan
        later = [V[i][t + LAG] for i in both]
        r["drift"] = float(unit(X.mean(0)) @ unit(np.mean(later, 0))) if len(both) >= MIN_N else np.nan
        labels = [T[i][t] for i in act]
        r["n_tasks"] = len(set(labels))
        r["top_task_share"] = max(labels.count(x) for x in set(labels)) / len(labels)
        rows.append(r)
    return rows


PANELS = [("n_active", "Active files N"), ("R", "R = |mean vector|"), ("spread_total", "Total spread 1 − R²"),
          ("var_par_std", "Spread along mean direction"), ("var_perp_std", "Spread across mean direction"),
          ("std_par_std", "Std of projections on mean"), ("rms_q", "RMS projection on principal axis"),
          ("mean_q", "Mean projection on principal axis"), ("std_q", "Std of projections on principal axis"),
          ("off_axis_q", "Second moment off principal axis"), ("eff_dim", "Effective dimension"),
          ("speed", "Speed: 1 − cos(v(t), v(t+2))"), ("drift", "Drift: cos(m(t), m(t+2))"),
          ("n_tasks", "Distinct questions"), ("top_task_share", "Share on most common question")]


def plot(bench, data):
    fig, axes = plt.subplots(len(PANELS), 2, figsize=(11, 2.1 * len(PANELS)), sharex=True)
    for c, variant in enumerate(("raw", "centred")):
        for r, (key, title) in enumerate(PANELS):
            ax = axes[r, c]
            for kind in KINDS:
                rows = data[(kind, variant)]
                ax.plot([x["token_center"] for x in rows], [x[key] for x in rows], lw=2, label=kind, **STYLE[kind])
            if key == "R" and variant == "centred":
                rows = data[("sequential", variant)]
                ax.plot([x["token_center"] for x in rows], [1 / np.sqrt(x["n_active"]) for x in rows], lw=1,
                        color="#8a8984", ls=":", label="1/√N (random directions)")
            ax.set_title(f"{title} — {variant}", fontsize=9, loc="left", color="#0b0b0b")
            ax.grid(color="#e4e3df", lw=0.6); ax.tick_params(labelsize=8, colors="#52514e")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            if r == 1:
                ax.legend(fontsize=8, frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel("generated tokens (window centre)", fontsize=9, color="#52514e")
    fig.suptitle(f"{bench}: embedding dynamics of 20 independent agents (windows with N ≥ {MIN_N})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT / f"{bench}.png", dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    for bench in SETS:
        every = [np.load(p) for p in (CONCAT / bench).rglob("*.vectors.npy")]
        mu = np.concatenate(every).mean(0)              # centring reference: all windows of the set
        files = {k: [load(p) for p in sorted((CONCAT / bench / k).glob("run_*.txt"))] for k in KINDS}
        data = {(k, v): series(files[k], v, mu) for k in KINDS for v in ("raw", "centred")}
        with open(OUT / f"{bench}.csv", "w", newline="") as f:
            cols = ["kind", "variant", "t", "token_center"] + [k for k, _ in PANELS if k != "n_active"] + ["n_active"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for (k, v), rows in data.items():
                for r in rows:
                    w.writerow({"kind": k, "variant": v, **{c: r[c] for c in cols[2:]}})
        plot(bench, data)
        print(f"{bench}: wrote {OUT / bench}.csv and .png")


if __name__ == "__main__":
    main()
