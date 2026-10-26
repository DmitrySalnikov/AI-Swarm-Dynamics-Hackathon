"""Task files for the DeepSeek-V4.1-Flash (Together) calibration of new-topic generators.

  python bench/ladder/v41_items.py OUT.jsonl SEED NAME=VALUE[:SEED] [NAME=VALUE[:SEED] ...]

One task per NAME=VALUE (generator name, length parameter), built with the given seed in the set `ladder_v41`,
id `ladder_v41/<name>@<value>`. A per-task seed (NAME=VALUE:SEED) overrides SEED and gives the id
`ladder_v41/<name>@<value>_s<SEED>`, so one generator can supply several tasks (e.g. rung-4 determinants). Runs: embedding_eval/gen_runs.py --provider together41 --items OUT.jsonl.
"""
import importlib
import json
import sys
from pathlib import Path

GEN_DIR = Path(__file__).resolve().parent / "generators"
sys.path[:0] = [str(GEN_DIR)]
from common import build  # noqa: E402

SET = "ladder_v41"
# English titles for the team's generators, whose `title` in the generator code is not in English
TITLES = {"r1_geom_polygon": "Area of a polygon with integer vertices", "r1_nt_order": "Multiplicative order modulo m",
          "r1d_zebra": "Zebra (Einstein) logic puzzle", "r3_adjugate": "Adjugate matrix", "r3_inverse": "Matrix inverse",
          "r3_gram_schmidt": "Gram–Schmidt orthogonalisation", "r3_minpoly": "Minimal polynomial",
          "r3_plu": "LU decomposition with partial pivoting", "r3_smith": "Smith normal form",
          "r3_svd": "Singular values", "r3_signature": "Signature of a quadratic form", "r3_eigen": "Eigenvalues",
          "r3_charpoly": "Characteristic polynomial", "r3_cholesky": "Cholesky decomposition",
          "r3_subspaces": "Sum and intersection of subspaces", "r1_stat_ols": "Least-squares regression",
          "la_det_e4": "Determinant of an integer matrix"}


def generators():
    skip = {"common", "selftest", "fetch_sources"}
    return {g.name: g for p in sorted(GEN_DIR.glob("*.py")) if p.stem not in skip
            for g in getattr(importlib.import_module(p.stem), "GENS", [])}


def main():
    out, seed, specs = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    gens = generators()
    with open(out, "w") as f:
        for s in specs:
            name, value = s.split("=")
            value, _, own = value.partition(":")
            tid = f"{name}@{value}" + (f"_s{own}" if own else "")
            it = build(gens[name], int(value), seed=int(own) if own else seed, set_name=SET, task_id=tid)
            it["title"] = TITLES.get(name, it["title"])
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(specs)} tasks -> {out}")


if __name__ == "__main__":
    main()
