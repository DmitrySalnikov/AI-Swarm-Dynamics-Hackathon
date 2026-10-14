"""Second pool of ladder candidates (2026-10-12): more tasks per rung, to choose a sample with the smallest spread of
mean lengths between tasks and within a task.

  python bench/ladder/pool_candidates.py        # -> candidates/pool2.jsonl, candidates/pool2_plan.md

Source generators: those whose step-3 candidates had >= 80 % correct runs and CV <= CV_LIMIT (excluding the ones in
SKIP). New instances (new seeds) per generator: NEW_PER_RUNG; rung 4 = RUNG4_NEW new determinants.
Parameter: kept if the mean length of the existing candidates of the generator is within KEEP; otherwise re-fitted to
TARGET tokens (calibrate.predict) on all points of the generator (calibration runs + step-3 runs, value -> length).
Seeds: 1000·rung + 500 + j, so they never repeat earlier tasks.
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bench" / "ladder"), str(ROOT / "bench" / "ladder" / "generators")]
import calibrate  # noqa: E402
import select_tasks as S  # noqa: E402
from common import build  # noqa: E402

TARGET, KEEP, CV_LIMIT = 18000, (15500, 21000), 0.35
NEW_PER_RUNG = {1: 2, 2: 3, 3: 2}
RUNG4_GEN, RUNG4_P, RUNG4_NEW = "la_det_e4", 6, 15
SKIP = {
    "r1d_sara_tax": "0/5 верных",
    "r1d_finmath": "0/5 верных",
    "r2d_tqa_mechanics": "модель узнаёт задачи TheoremQA; низкая длина",
    "r2d_tqa_thermal": "пачка TheoremQA; длина 4–9 тыс.",
    "la_det": "заменён вариантом la_det_e4",
    "la_det_e20": "заменён вариантом la_det_e4",
    "r1_chem_chain": "длина не зависит от p: 20–32 тыс. при p = 6–16",
    "r2_layer_refraction": "длина не зависит от p: 24–40 тыс. при p = 2–12",
    "r2_level_mean_energy": "длина не зависит от p: 20–37 тыс. при p = 14–34",
}
# values set from the observed points where the fit is unreliable (noisy points of different instances)
OVERRIDE_P = {
    "r2_parallel_wires": (5, "по точкам: p = 5 → 13,6 и 18,8 тыс., p = 6–7 → 25–36 тыс."),
    "r2_cart_collisions": (36, "по точкам: p = 40–46 → 25–26 тыс., длина ~ p"),
    "r1_crypto_dh": (701, "по точкам: p = 701 → 15,0 и 15,7 тыс., p = 809 → 39 тыс."),
}
# near misses of rung 2 that failed only by length: re-fitted even though their CV / correctness are fine
FORCE_REFIT = {"r2_cart_collisions", "r2_parallel_wires", "r2_level_mean_energy", "r2_layer_refraction",
               "r2_composite_inertia", "r2_calorimetry"}


def points(name, value_runs):
    """(value, length) of every run of the generator: calibration + step 3."""
    pts = [(v, r["length"]) for (n, v), rs in calibrate.results().items() if n == name for r in rs if r["length"]]
    return pts + value_runs


def main():
    calibrate.TARGET = TARGET  # predict() fits the parameter to calibrate.TARGET
    gens = calibrate.generators()
    stats = S.all_stats()
    by_gen = {}
    for s in stats:
        by_gen.setdefault(s["name"], []).append(s)
    items, md = [], ["| Генератор | Ступени | Было: значение, средняя, CV, верно | Новое значение | Новых задач | Почему |",
                     "|---|---|---|---|---|---|"]
    for name, ss in sorted(by_gen.items()):
        g = gens[name]
        Ls = [L for s in ss for L in s["lengths"]]
        n_runs, n_ok = sum(s["n"] for s in ss), sum(s["correct"] for s in ss)
        mean, cv = st.mean(Ls), st.mean(s["cv"] for s in ss)
        vals = sorted({s["value"] for s in ss})
        was = f"{', '.join(map(str, vals))}; {mean:.0f}; {cv:.2f}; {n_ok}/{n_runs}"
        if name in SKIP:
            md.append(f"| {name} | {','.join(map(str, g.rungs))} | {was} | — | 0 | {SKIP[name]} |")
            continue
        if name == RUNG4_GEN:
            for j in range(RUNG4_NEW):
                items.append(build(g, RUNG4_P, 4500 + j, "ladder_rung_4", f"{name}_x{j + 1:02d}"))
        rungs = [k for k in g.rungs if k in NEW_PER_RUNG]
        if not rungs:
            md.append(f"| {name} | 4 | {was} | {RUNG4_P} | {RUNG4_NEW} | шаблон ступени 4 |")
            continue
        good = n_ok >= 0.8 * n_runs and cv <= CV_LIMIT
        if not good and name not in FORCE_REFIT:
            md.append(f"| {name} | {','.join(map(str, g.rungs))} | {was} | — | 0 | верно < 80 % или CV > {CV_LIMIT} |")
            continue
        if name in OVERRIDE_P:
            value, why = OVERRIDE_P[name]
        elif KEEP[0] <= mean <= KEEP[1] and name not in FORCE_REFIT:
            best = min(ss, key=lambda s: abs(st.mean(s["lengths"]) - TARGET))  # the instance closest to the target
            value, why = best["value"], "средняя в норме"
        else:
            pstar, model = calibrate.predict(points(name, [(s["value"], L) for s in ss for L in s["lengths"]]))
            value = min(g.grid, key=lambda v: abs(v - pstar))
            why = f"подбор на {TARGET} ток. ({model}, p*≈{pstar:.3g})"
        n_new = 0
        for k in rungs:
            for j in range(NEW_PER_RUNG[k]):
                items.append(build(g, value, 1000 * k + 500 + j, f"ladder_rung_{k}", f"{name}_x{j + 1}"))
                n_new += 1
        md.append(f"| {name} | {','.join(map(str, rungs))} | {was} | {value} | {n_new} | {why} |")
    out = calibrate.CAND / "pool2.jsonl"
    out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    (calibrate.CAND / "pool2_plan.md").write_text("\n".join(md) + "\n")
    per = {}
    for it in items:
        per[it["set"]] = per.get(it["set"], 0) + 1
    print(f"{len(items)} new candidates -> {out.relative_to(ROOT)}; per rung {dict(sorted(per.items()))}")


if __name__ == "__main__":
    main()
