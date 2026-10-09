"""Self-test of all ladder generators (no model calls).

For every generator, several parameter values and seeds:
- the reference answer is accepted by grade() when compared with verify(data) (an independent recomputation);
- a corrupted answer is rejected by grade();
- the task is reproducible (same seed -> same task) and JSON-serialisable.

  uv run --with sympy --with numpy --with scipy --with reasoning-gym==0.1.19 \
      python -W ignore bench/ladder/generators/selftest.py [module ...] [--show NAME]
"""
import importlib
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[2] / "embedding_eval")]
from common import build  # noqa: E402
from gen_runs import grade, parse_num  # noqa: E402

SKIP = {"common", "selftest", "registry", "fetch_sources"}


def all_modules(names=None):
    names = names or sorted(p.stem for p in HERE.glob("*.py") if p.stem not in SKIP)
    return [importlib.import_module(n) for n in names]


def corrupt(item):
    """A wrong answer of the same shape."""
    kind, ans = item["check"]["type"], item["answer"]
    if kind == "num":
        v = parse_num(ans)
        return str(v + max(1, abs(v)) if v == int(v) else v + max(1, abs(v)) / 10)
    if kind == "list":
        parts = ans.split(",")
        if item["check"].get("rel_tol") is not None or item["check"].get("abs_tol") is not None:
            v = parse_num(parts[0], False)
            parts[0] = str(v + max(1, abs(v)))
        else:
            parts[0] = parts[0] + "x"
        return ",".join(parts)
    return ans + "x"


def test_gen(gen, seeds=(0, 1, 2)):
    values = sorted({gen.grid[0], *gen.probe, gen.grid[-1]})
    errors = []
    for p in values:
        for s in seeds:
            try:
                it = build(gen, p, s, "selftest")
                again = build(gen, p, s, "selftest")
                json.dumps(it)
                if it != again:
                    errors.append(f"p={p} s={s}: not reproducible")
                alt = gen.verify(it["data"])
                if not grade(it, alt):
                    errors.append(f"p={p} s={s}: verify gives {alt!r}, answer {it['answer']!r}")
                if grade(it, corrupt(it)):
                    errors.append(f"p={p} s={s}: corrupted answer {corrupt(it)!r} accepted")
            except Exception as e:  # noqa: BLE001
                errors.append(f"p={p} s={s}: {type(e).__name__}: {e}")
    return errors


def main():
    args = sys.argv[1:]
    show = args[args.index("--show") + 1] if "--show" in args else None
    mods = [a for a in args if not a.startswith("--") and a != show]
    names, bad = set(), 0
    for m in all_modules(mods or None):
        for g in getattr(m, "GENS", []):
            assert g.name not in names, f"duplicate generator name {g.name}"
            names.add(g.name)
            assert all(v in g.grid for v in g.probe), f"{g.name}: probe values not in grid"
            if show == g.name:
                it = build(g, g.probe[0], 0, "selftest")
                print(it["prompt"], "\n--- answer:", it["answer"], it["check"])
            err = test_gen(g)
            bad += bool(err)
            print(f"{'FAIL' if err else 'ok  '} {m.__name__}.{g.name:28s} rungs={g.rungs} probe={g.probe}")
            for e in err[:5]:
                print("     ", e)
    print(f"{len(names)} generators, {bad} failing")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
