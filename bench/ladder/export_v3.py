"""Third ladder set (new topics only, DeepSeek-V4.1-Flash via Together): bench/ladder/set_v3/rung_{1,2,3,4}.jsonl.

  python bench/ladder/export_v3.py [--allow NAME ...]

Candidates are the tasks in candidates/v41_confirm_*.jsonl and v41_round*.jsonl; each was run 5 times
(embedding_eval/runs/ladder/deepseek-v4.1-flash/main/run_k/ladder_v41/<name>@<value>.json). A task is kept if at
least 4 of 5 runs are correct, the mean length (usage.completion_tokens) is in [LO, HI] and CV <= 0.25; --allow keeps
a named generator that misses only the length window, and --allow-cv one whose CV is at most ALLOW_CV
(the set_v1 selection likewise took a task with CV 0.2522 as the closest available). Topic = area (rung 1) or domain (rungs 2, 3); a task with its
own seed (id ...@<value>_s<seed>) is a rung-4 determinant, topic determinant_NN. Only the first kept task of a topic
is used. Records follow the set_v2 format, with the statistics in `selection`.
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

LADDER = Path(__file__).resolve().parent
ROOT = LADDER.parents[1]
RUNS = ROOT / "embedding_eval" / "runs" / "ladder" / "deepseek-v4.1-flash" / "main"
CONFIRM = sorted(LADDER.glob("candidates/v41_confirm_*.jsonl")) + sorted(LADDER.glob("candidates/v41_round*.jsonl"))
LO, HI, MAX_CV, MIN_CORRECT, ALLOW_CV = 12000, 26000, 0.25, 4, 0.26
sys.path[:0] = [str(LADDER), str(LADDER / "generators")]
from v41_items import TITLES, generators  # noqa: E402


def stats(tid):
    L, ok = [], 0
    for f in sorted(RUNS.glob(f"run_*/ladder_v41/{tid}.json")):
        r = json.loads(f.read_text())
        L.append(r["completion_tokens"]); ok += bool(r["correct"])
    if not L:
        return None
    mean = st.mean(L)
    return dict(n_runs=len(L), mean=round(mean), std=round(st.pstdev(L)), cv=round(st.pstdev(L) / mean, 3), max=max(L),
                correct=ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow", nargs="*", default=[], help="generators kept although outside the length window")
    ap.add_argument("--allow-cv", nargs="*", default=[], help=f"generators kept with CV up to {ALLOW_CV}")
    args = ap.parse_args()
    gens = generators()
    out = {1: [], 2: [], 3: [], 4: []}
    seen = set()
    for path in CONFIRM:
        for line in open(path):
            it = json.loads(line)
            tid = it["id"].split("/")[1]
            if tid in seen:                    # the same task listed in two candidate files
                continue
            seen.add(tid)
            g = gens[it["gen"]["name"]]
            s = stats(tid)
            max_cv = ALLOW_CV if g.name in args.allow_cv else MAX_CV
            ok = s and s["n_runs"] >= 4 and s["correct"] >= MIN_CORRECT * s["n_runs"] / 5 and s["cv"] <= max_cv
            in_window = s and LO <= s["mean"] <= HI
            keep = ok and (in_window or g.name in args.allow)
            print(f"{'keep' if keep else 'drop'}  {tid:24s} {s}")
            if not keep:
                continue
            rung = 4 if re.search(r"@\d+_s\d+$", tid) else next(k for k in (1, 2, 3) if k in g.rungs)
            topic = g.area if rung == 1 else g.domain if rung in (2, 3) else f"determinant_{len(out[4]) + 1:02d}"
            if (rung, topic) in seen:
                print(f"      {tid}: topic {topic} already in rung {rung}, skipped")
                continue
            seen.add((rung, topic))
            rec = {**{k: it[k] for k in ("source", "prompt", "answer", "check", "gen")},
                   "title": TITLES.get(g.name, g.title),
                   "id": f"ladder_v3_rung_{rung}/{topic}", "set": f"ladder_v3_rung_{rung}", "domain": topic,
                   "selection": {"candidate_id": it["id"], "model": "deepseek-ai/DeepSeek-V4.1-Flash", **s}}
            out[rung].append({k: rec[k] for k in ("id", "set", "domain", "title", "source", "prompt", "answer", "check",
                                                  "gen", "selection")})
    (LADDER / "set_v3").mkdir(exist_ok=True)
    for rung, recs in out.items():
        with open(LADDER / "set_v3" / f"rung_{rung}.jsonl", "w") as f:
            for r in sorted(recs, key=lambda r: r["domain"]):
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"rung {rung}: {len(recs)} tasks")


if __name__ == "__main__":
    main()
