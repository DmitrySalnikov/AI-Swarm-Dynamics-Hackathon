"""Candidates for the main selection (step 3): one task per (generator, rung) at the calibrated parameter value;
rung 4 = RUNG4_N instances of one template.

  python bench/ladder/main_candidates.py            # -> candidates/main.jsonl and candidates/main_plan.md

The parameter value is the calibration choice (calibrate.py: choose) unless overridden in OVERRIDE; generators in
EXCLUDE do not go to step 3 (reason given). Seeds: rung k, instance i -> seed 1000·k + i, so a generator shared between
rungs gives different tasks on different rungs.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "bench" / "ladder"), str(ROOT / "bench" / "ladder" / "generators")]
import calibrate  # noqa: E402
from common import build  # noqa: E402

RUNG4_GEN, RUNG4_N = "la_det_e4", 30
DET_GENS = {"la_det", "la_det_e4", "la_det_e20"}  # one of them is used on rungs 1, 3, 4 (RUNG4_GEN)

OVERRIDE = {
    "la_det_e4": (6, "n=6: 15 514 и 17 661 ток.; n=7 по росту ~n³ дал бы ~26 тыс."),
    "bio_gc": (380, "калибровка двухмодальна (300 нт → 5,1–5,5 тыс., 400 нт → 43 тыс.); прежние 4 прогона "
                    "этой модели на 372 нт: медиана 20 227, CV 0,19"),
    "r2_composite_wall": (8, "2 слоя → 4–7 тыс., 15–40 слоёв → 26–27 тыс.; подгонка (22) дала бы ~26 тыс."),
}
EXCLUDE = {
    "r1_ling_numerals": "длина не растёт с p: 3,7–5,8 тыс. при p = 6–25",
    "r2_rc_switching": "уже при минимальном p = 3: 23,8 и 44,5 тыс.; при p = 10–24 — 48–53 тыс.",
    "r2_point_charges": "уже при минимальном p = 2: 32,5 и 40,7 тыс.",
    "r2_decay_chain": "уже при минимальном p = 2: 32,2 и 33,0 тыс.",
    "r2_gas_cycle": "27–33 тыс. при p = 8–16, длина почти не зависит от p",
    "r3_eigen": "скачок: n = 3 → 0,7–0,9 тыс., n = 4–5 → 29–32 тыс.",
    "r3_cholesky": "на верхней границе сетки (n = 10) 7,2–9,1 тыс.; рост ~n^1.1",
    "r2d_ugp_thermodynamics": "UGPhysics: верно 1 из 4; эталоны опираются на неуказанные допущения",
    "r2d_ugp_quantum": "UGPhysics: верно 1 из 4 (только при p = 2)",
    "r2d_ugp_optics": "UGPhysics: верно 0 из 4; есть недоопределённые условия",
    "r2d_ugp_electromagnetism": "UGPhysics: верно 0 из 4; эталон считает катушку длинным соленоидом вопреки условию",
    "r2d_ugp_atomic": "UGPhysics: верно 0 из 4; спорный эталон (NaCl: 2a вместо 2·d111)",
    "r2d_tqa_modern": "TheoremQA modern: верно 0 из 3 при p = 4–5",
}


def plan():
    gens = calibrate.generators()
    res = calibrate.results(gens)
    rows = []
    for g in gens.values():
        if g.name in DET_GENS and g.name != RUNG4_GEN:
            continue
        best, pstar, model, note = calibrate.choose(g, res)
        value, why = OVERRIDE.get(g.name, (best, f"калибровка: {model}, p*≈{pstar:.3g}" if pstar else "калибровка"))
        rows.append(dict(name=g.name, rungs=g.rungs, value=value, why=why, excluded=EXCLUDE.get(g.name), gen=g))
    return rows


def main():
    rows = plan()
    items, md = [], ["| Генератор | Ступени | Значение | Почему | В шаг 3 |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['name']} | {','.join(map(str, r['rungs']))} | {r['value']} | {r['why']} | "
                  f"{'нет — ' + r['excluded'] if r['excluded'] else 'да'} |")
        if r["excluded"]:
            continue
        for k in r["rungs"]:
            if k == 4 and r["name"] != RUNG4_GEN:  # backup templates for rung 4 are run only if the main one fails
                continue
            n = RUNG4_N if k == 4 else 1
            for i in range(n):
                task_id = f"{r['name']}_{i:02d}" if k == 4 else r["name"]
                items.append(build(r["gen"], r["value"], 1000 * k + i, f"ladder_rung_{k}", task_id))
    out = calibrate.CAND / "main.jsonl"
    out.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    (calibrate.CAND / "main_plan.md").write_text("\n".join(md) + "\n")
    per = {}
    for it in items:
        per[it["set"]] = per.get(it["set"], 0) + 1
    print(f"{len(items)} candidates -> {out.relative_to(ROOT)}; per rung: {dict(sorted(per.items()))}")


if __name__ == "__main__":
    main()
