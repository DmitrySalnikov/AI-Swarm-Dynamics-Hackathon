"""Diagrams of the ladder selection (step 3 runs only, no calibration runs).

  uv run --with sympy --with numpy --with scipy --with matplotlib --with reasoning-gym==0.1.19 \\
      python bench/ladder/plots_ladder.py

Data and definitions are those of select_tasks.py / REPORT_ladder.md:
- N_t of a run = usage.completion_tokens (reasoning + visible answer);
- per task: mean ⟨N_t⟩, δ(N_t) = sample std (ddof=1), median, CV = δ / ⟨N_t⟩;
- runs: main selection (5–8 per candidate); runs that ended with a transport error are not counted (they were
  redone), a run longer than 60 000 tokens counts as an overrun and as wrong, a run without a visible answer as wrong.
Outputs in bench/ladder/figures/: fig1_overview_{all,final}.{png,svg}, fig2_runs_{all,final}.{png,svg},
ladder_lengths.csv (one row per run).

Stacked bars per sub-problem of the TheoremQA bundles are NOT drawn: the chain of thought keeps returning to
earlier problems (each "Problem k" is mentioned many times, not in order), so the tokens cannot be attributed
to a sub-problem reliably.
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LADDER = ROOT / "bench" / "ladder"
OUT = LADDER / "figures"
sys.path[:0] = [str(LADDER), str(LADDER / "generators")]
import calibrate  # noqa: E402
import select_tasks as S  # noqa: E402

# palette: first three categorical slots of the validated reference palette (all-pairs CVD-safe) + neutral gray;
# every colour is paired with a marker shape, so identity never depends on colour alone
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#eef4fc"
RUNG_TITLE = {1: "1 — далёкие", 2: "2 — физика", 3: "3 — линейная алгебра", 4: "4 — один шаблон"}
REASON = {"медиана": "Д", "CV": "К", "верно": "П"}  # length, CV, correctness; "О" is added for overruns
TARGET, LO, HI, CV_MAX, YMAX = 20000, 10000, 25000, 0.25, 60000

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "svg.fonttype": "none"})


def collect():
    gens = calibrate.generators()
    final = {}
    for k in range(1, 5):
        for l in (LADDER / f"rung_{k}.jsonl").read_text().splitlines():
            t = json.loads(l)
            final[(t["gen"]["name"], t["gen"]["value"], t["gen"]["seed"])] = t["domain"]
    tasks = []
    for it in S.items().values():
        g = gens[it["gen"]["name"]]
        s = S.stats_of(it)
        runs = S.runs_of(it)
        Ls = [r["length"] for r in runs]
        key = (it["gen"]["name"], it["gen"]["value"], it["gen"]["seed"])
        rung = int(it["set"][-1])
        selected = key in final
        letters = "".join(sorted({v for w in s["why"] for kk, v in REASON.items() if w.startswith(kk) or kk in w},
                                 key="ДКП".index)) + ("О" if s["cut"] else "")
        status = ("near" if s["near"] else "final") if selected else ("passed" if s["pass"] else "rejected")
        label = (g.domain + ("_v2" if it["domain"].endswith("_v2") else "")) if rung != 4 else \
            "det_" + it["domain"].rsplit("_", 1)[-1]
        tasks.append(dict(
            id=it["id"], rung=rung, gen=g, value=it["gen"]["value"], area=g.area, label=label, final_label=final.get(key), selected=selected,
            status=status, letters=letters if not selected else "", why="; ".join(s["why"]), runs=runs,
            mean=st.mean(Ls), std=st.stdev(Ls), median=st.median(Ls), cv=st.stdev(Ls) / st.mean(Ls),
            sort=(rung, g.area if rung == 1 else g.domain, it["domain"]),
            pool2=it["gen"]["seed"] % 1000 >= 500))
    return sorted(tasks, key=lambda t: t["sort"])


def write_csv(tasks):
    with open(OUT / "ladder_lengths.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "domain", "question_id", "run_id", "n_tokens", "correct", "overrun", "no_visible_answer",
                    "selected", "status", "reject_reason", "reject_detail", "generator", "param_value"])
        for t in tasks:
            for r in sorted(t["runs"], key=lambda r: r["run"]):
                w.writerow([t["rung"], t["final_label"] or t["label"], t["id"], r["run"], r["length"], int(r["correct"]),
                            int(r["cut"]), int(r["file"].endswith(".failed.json")), int(t["selected"]), t["status"],
                            t["letters"], t["why"] if not t["selected"] else "", t["gen"].name, t["value"]])


def write_task_csv(tasks):
    """One row per task: length statistics over its runs, for choosing the sample (by mean, spread, maximum)."""
    with open(OUT / "pool_tasks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rung", "question_id", "generator", "area_or_type", "param_value", "pool", "n_runs", "mean", "std",
                    "cv", "median", "min", "max", "n_correct", "n_overrun", "in_current_set", "old_status"])
        for t in sorted(tasks, key=lambda t: (t["rung"], t["mean"])):
            Ls = [r["length"] for r in t["runs"]]
            w.writerow([t["rung"], t["id"], t["gen"].name, t["gen"].area if t["rung"] == 1 else t["gen"].domain,
                        t["value"], "pool2" if t["pool2"] else "step3", len(Ls), round(t["mean"]), round(t["std"]),
                        round(t["cv"], 3), round(t["median"]), min(Ls), max(Ls), sum(r["correct"] for r in t["runs"]),
                        sum(r["cut"] for r in t["runs"]), int(t["selected"]), t["status"]])


def is_clean(t):
    return sum(r["correct"] for r in t["runs"]) >= 0.8 * len(t["runs"]) and t["cv"] <= CV_MAX


def fig3_pool(tasks, name, in_band_only=False, title=None):
    """Candidates, each rung sorted by mean length: mean ± std, the longest run, and whether the task is
    'clean' (>= 80 % correct and CV <= 0.25). in_band_only: only clean tasks with the mean in 14–23 thousand."""
    if in_band_only:
        tasks = [t for t in tasks if is_clean(t) and 14000 <= t["mean"] <= 23000]
    custom_title = title
    tasks = sorted(tasks, key=lambda t: (t["rung"], t["mean"]))
    fig, ax = plt.subplots(figsize=(max(12, 0.13 * len(tasks) + 4), 7.2))
    frame(ax, tasks)
    ax.axhspan(14000, 23000, color=BAND, zorder=0)
    ax.text(1.002, 14000, "окно средней\n14–23 тыс.", transform=ax.get_yaxis_transform(), va="bottom", fontsize=8,
            color=INK2)
    for i, t in enumerate(tasks):
        c, m = (BLUE, "o") if is_clean(t) else (GRAY, "v")
        ax.errorbar(i, t["mean"], yerr=t["std"], fmt="none", ecolor=c, elinewidth=1.2, capsize=2, zorder=3)
        ax.plot(i, t["mean"], m, ms=5.5, mfc=c, mec="white", mew=0.7, zorder=4)
        mx = max(r["length"] for r in t["runs"])
        ax.plot(i, min(mx, YMAX), "_", ms=7, mew=1.6, color=ORANGE, zorder=4)
        if mx > YMAX:
            ax.text(i, YMAX + 800, f"{mx // 1000} тыс.", ha="center", va="bottom", fontsize=6.5, color=INK, rotation=90)
    top = max(max(r["length"] for r in t["runs"]) for t in tasks)
    ax.set_ylim(0, YMAX + 6000 if top > YMAX else top * 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f} тыс."))
    ax.set_ylabel("⟨N_t⟩ ± δ(N_t), токенов")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"{t['rung']}-{t['label']}" for t in tasks], rotation=90, fontsize=6 if len(tasks) > 150 else 7)
    handles = [Line2D([], [], marker="o", ls="", mfc=BLUE, mec="white", ms=7, label="≥ 80 % верных и CV ≤ 0,25")]
    if not all(is_clean(t) for t in tasks):
        handles.append(Line2D([], [], marker="v", ls="", mfc=GRAY, mec="white", ms=7,
                              label="меньше 80 % верных или CV > 0,25"))
    handles += [Line2D([], [], marker="_", ls="", color=ORANGE, ms=9, mew=1.6, label="самый длинный прогон задачи"),
                Patch(color=BAND, label="окно средней 14–23 тыс.")]
    per = ", ".join(f"{k}: {sum(t['rung'] == k for t in tasks)}" for k in range(1, 5))
    title = (f"Кандидаты со средней длиной 14–23 тыс., ≥ 80 % верных и CV ≤ 0,25 ({len(tasks)} задач; по ступеням {per})"
             if in_band_only else f"Пул кандидатов ({len(tasks)} задач, {sum(len(t['runs']) for t in tasks)} прогонов)")
    title = custom_title or title + ": внутри ступени — по возрастанию средней длины"
    top_matter(fig, title, handles, ncol=4, top=0.86)
    save(fig, name)


def frame(ax, tasks, top_labels=True):
    """Rung separators and titles, common x axis."""
    xs = range(len(tasks))
    for i in range(1, len(tasks)):
        if tasks[i]["rung"] != tasks[i - 1]["rung"]:
            ax.axvline(i - 0.5, color=INK2, lw=1.0, zorder=1)
    if top_labels:
        for k in range(1, 5):
            idx = [i for i, t in enumerate(tasks) if t["rung"] == k]
            if idx:
                ax.text((idx[0] + idx[-1]) / 2, 1.01, RUNG_TITLE[k], transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=10, color=INK, fontweight="bold")
    ax.set_xlim(-0.7, len(tasks) - 0.3)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def tick_labels(tasks, final_only):
    n, labels = {}, []
    for t in tasks:
        n[t["rung"]] = n.get(t["rung"], 0) + 1
        labels.append(f"{t['rung']}-{n[t['rung']]:02d} {t['final_label'] if final_only else t['label']}")
    return labels


STYLE = {"final": (BLUE, "o", "в итоговом наборе"),
         "near": (BLUE, "D", "в итоговом наборе как «максимально близкая» (CV чуть выше 0,25)"),
         "passed": (AQUA, "s", "прошла фильтры, не выбрана (медианы ступеней / одна задача на область)"),
         "rejected": (GRAY, "v", "отсеяна; буквы — причина: Д длина, К CV, П правильность, О обрывы >60 тыс.")}


def length_band(ax):
    ax.axhspan(LO, HI, color=BAND, zorder=0)
    ax.axhline(TARGET, color=BLUE, lw=1.2, ls="--", zorder=1)
    ax.text(1.002, TARGET, "цель 20 000", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    ax.text(1.002, LO, "допустимо\n10–25 тыс.", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=8, color=INK2)


def fig1(tasks, final_only, name):
    w = max(12, 0.17 * len(tasks) + 4)
    fig, (ax, axc) = plt.subplots(2, 1, figsize=(w, 8.2), sharex=True, gridspec_kw=dict(height_ratios=[4, 1.3], hspace=0.08))
    frame(ax, tasks)
    length_band(ax)
    for i, t in enumerate(tasks):
        c, m, _ = STYLE[t["status"]]
        ax.errorbar(i, t["mean"], yerr=t["std"], fmt="none", ecolor=c, elinewidth=1.4, capsize=2.5, zorder=3)
        ax.plot(i, t["mean"], m, ms=7, mfc=c, mec="white", mew=1.0, zorder=4)
        if t["letters"]:
            ax.text(i, min(t["mean"] + t["std"], YMAX) + 900, t["letters"], ha="center", va="bottom", fontsize=7,
                    color=INK2, rotation=90)
    ax.set_ylim(0, min(YMAX + 6000, max(t["mean"] + t["std"] for t in tasks) * 1.12 + 4000))
    ax.set_ylabel("⟨N_t⟩ ± δ(N_t), токенов")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f} тыс."))
    n_runs = sum(len(t["runs"]) for t in tasks)
    title = (("Итоговые 40 задач" if final_only else f"Все кандидаты основного отбора ({len(tasks)})")
             + f": средняя длина решения и разброс по прогонам ({n_runs} прогонов)")
    statuses = [s for s in STYLE if any(t["status"] == s for t in tasks)]
    handles = [Line2D([], [], marker=STYLE[s][1], ls="", mfc=STYLE[s][0], mec="white", ms=8, label=STYLE[s][2])
               for s in statuses] + [Patch(color=BAND, label="допустимый диапазон длины"),
                                     Line2D([], [], marker="o", ls="", mfc="none", mec=ORANGE, ms=10, mew=1.6,
                                            label="CV в [0,25; 0,26] — пограничные (значение подписано)")]
    top_matter(fig, title, handles, ncol=2, top=0.83)
    # CV panel
    frame(axc, tasks, top_labels=False)
    axc.axhline(CV_MAX, color=ORANGE, lw=1.2, ls="--", zorder=1)
    n_border = 0
    axc.text(1.002, CV_MAX, "порог 0,25", transform=axc.get_yaxis_transform(), va="center", fontsize=8, color=INK2)
    for i, t in enumerate(tasks):
        c, m, _ = STYLE[t["status"]]
        axc.vlines(i, 0, t["cv"], color=c, lw=1.4, zorder=2)
        axc.plot(i, t["cv"], m, ms=5, mfc=c, mec="white", mew=0.8, zorder=3)
        if CV_MAX <= t["cv"] <= 0.26:
            axc.plot(i, t["cv"], "o", ms=12, mfc="none", mec=ORANGE, mew=1.6, zorder=4)
            axc.annotate(f"{t['cv']:.4f}".replace(".", ","), (i, t["cv"]), xytext=(0, 9 + 10 * (n_border % 3)),
                         textcoords="offset points", ha="center", fontsize=7, color=INK,
                         arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5) if n_border % 3 else None)
            n_border += 1
    axc.set_ylim(0, max(0.8, max(t["cv"] for t in tasks) * 1.1))
    axc.set_ylabel("CV")
    axc.set_xticks(range(len(tasks)))
    axc.set_xticklabels(tick_labels(tasks, final_only), rotation=90, fontsize=7 if not final_only else 8)
    save(fig, name)


def fig2(tasks, final_only, name):
    w = max(12, 0.17 * len(tasks) + 4)
    fig, ax = plt.subplots(figsize=(w, 7.2))
    frame(ax, tasks)
    length_band(ax)
    over_txt = []
    for i, t in enumerate(tasks):
        if not final_only and t["selected"]:
            ax.axvspan(i - 0.45, i + 0.45, color="#f3f1ea", zorder=0)
        Ls = [r["length"] for r in t["runs"]]
        ax.vlines(i, min(Ls), min(max(Ls), YMAX), color=INK2, lw=3.5, alpha=0.35, zorder=2)
        rs = sorted(t["runs"], key=lambda r: r["run"])
        for j, r in enumerate(rs):
            dx = (j - (len(rs) - 1) / 2) * min(0.08, 0.6 / max(len(rs), 1))
            y = min(r["length"], YMAX)
            if r["cut"]:
                ax.annotate("", (i + dx, YMAX + 3500), xytext=(i + dx, YMAX - 500),
                            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4), zorder=5)
                over_txt.append((i + dx, r["length"]))
                continue
            if r["file"].endswith(".failed.json"):
                ax.plot(i + dx, y, "X", ms=7, mfc=ORANGE, mec=INK, mew=0.6, zorder=5)
            elif r["correct"]:
                ax.plot(i + dx, y, "o", ms=4.5, mfc=BLUE, mec="white", mew=0.6, zorder=4)
            else:
                ax.plot(i + dx, y, "v", ms=6, mfc=ORANGE, mec="white", mew=0.6, zorder=4)
    for x, v in over_txt:
        ax.text(x, YMAX + 3800, f"{v:,}".replace(",", " "), ha="center", va="bottom", fontsize=7, color=INK, rotation=90)
    runs = [r for t in tasks for r in t["runs"]]
    has_cut, has_empty = any(r["cut"] for r in runs), any(r["file"].endswith(".failed.json") for r in runs)
    has_wrong = any(not r["correct"] and not r["cut"] and not r["file"].endswith(".failed.json") for r in runs)
    if has_cut:
        ax.axhline(YMAX, color=ORANGE, lw=0.8, ls=":", zorder=1)
        ax.set_ylim(0, YMAX + 12000)
    else:
        ax.set_ylim(0, max(r["length"] for r in runs) * 1.1)
    ax.set_yticks([v for v in range(0, int(ax.get_ylim()[1]) + 1, 10000)])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f} тыс."))
    ax.set_ylabel("N_t — длина одного прогона, токенов")
    n_runs = sum(len(t["runs"]) for t in tasks)
    title = (("Итоговые 40 задач" if final_only else f"Все кандидаты основного отбора ({len(tasks)})")
             + f": каждый прогон ({n_runs}); серая полоса — от минимума до максимума")
    handles = [Line2D([], [], marker="o", ls="", mfc=BLUE, mec="white", ms=7, label="верный ответ")]
    if has_wrong:
        handles.append(Line2D([], [], marker="v", ls="", mfc=ORANGE, mec="white", ms=7, label="неверный ответ"))
    if has_empty:
        handles.append(Line2D([], [], marker="X", ls="", mfc=ORANGE, mec=INK, ms=8, label="нет видимого ответа (неверно)"))
    if has_cut:
        handles.append(Line2D([], [], marker=r"$\uparrow$", ls="", color=ORANGE, ms=9,
                              label="длиннее 60 тыс. (обрыв по длине, неверно) — значение подписано"))
    if not final_only:
        handles.append(Patch(color="#f3f1ea", label="задача в итоговом наборе"))
    top_matter(fig, title, handles, ncol=3, top=0.86)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tick_labels(tasks, final_only), rotation=90, fontsize=7 if not final_only else 8)
    save(fig, name)


def top_matter(fig, title, handles, ncol, top):
    """Title and legend above the plot (the rotated task labels take the space below)."""
    fig.subplots_adjust(top=top)
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=11.5, color=INK)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.955), fontsize=8.5, frameon=False,
               ncol=ncol, columnspacing=2.0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = collect()
    final = sorted((t for t in tasks if t["selected"]), key=lambda t: (t["rung"], t["final_label"]))
    write_csv(tasks)
    write_task_csv(tasks)
    fig3_pool(tasks, "fig3_pool_all")
    fig3_pool(tasks, "fig3_pool", in_band_only=True)
    for sub, tt, fo in (("all", tasks, False), ("final", final, True)):
        fig1(tt, fo, f"fig1_overview_{sub}")
        fig2(tt, fo, f"fig2_runs_{sub}")
    for sub, tt in (("все кандидаты", tasks), ("итоговые", final)):
        runs = [r for t in tt for r in t["runs"]]
        print(f"{sub}: задач {len(tt)}, прогонов {len(runs)}, длиннее 60 тыс. {sum(r['cut'] for r in runs)}, "
              f"без видимого ответа {sum(r['file'].endswith('.failed.json') for r in runs)}, "
              f"по ступеням {[sum(t['rung'] == k for t in tt) for k in range(1, 5)]}, "
              f"статусы {dict((s, sum(t['status'] == s for t in tt)) for s in STYLE)}")


if __name__ == "__main__":
    main()
