"""Shared pieces of the ladder task generators.

Every generator module defines `GENS: list[Gen]`. A generator turns (length parameter p, seed) into one task:
- `make(p, rng)` returns dict(prompt=..., answer=..., check=..., data=..., [source=...]); `prompt` is the problem text
  and must state the answer format; `data` is the raw instance (JSON-serialisable) that `verify` gets; an optional
  `source` dict (e.g. ids of the dataset problems used) is merged into the generator's `source`;
- `verify(data)` recomputes the answer by an independent method (different algorithm or library) and returns it
  as a string; selftest.py grades it with the same `grade()` the runs use.
The random generator is seeded from (generator name, p, seed), so a task is fully reproducible.

Check types (graded by embedding_eval/gen_runs.py: grade):
- {"type": "exact"}                      — string equality after whitespace/case normalisation;
- {"type": "num", "rel_tol": t}          — one number; fractions a/b allowed; t = 0 means exact equality;
- {"type": "list"}                       — comma-separated strings, element-wise "exact";
- {"type": "list", "rel_tol": t}         — comma-separated numbers (fractions allowed), element-wise "num".
"""
import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

SUFFIX = "\n\nThink it through, then finish with a final line of the form `ANSWER: <your answer>`."
OWN = {"dataset": "own generator (answer computed by code)", "id": None, "url": None,
       "license": "CC BY-NC-SA 4.0 (this repository)"}


@dataclass
class Gen:
    name: str                 # unique id, e.g. "la_det"; also the task-file stem
    domain: str               # specific label: physics section (rung 2) or task type (rung 3), e.g. "determinant"
    area: str                 # broad field for rung 1 (at most one rung-1 task per area), e.g. "linear_algebra"
    title: str                # short title in Russian, like the titles of set_3/set_4
    param: str                # what the length parameter means, e.g. "matrix size n"
    grid: list                # allowed values of the length parameter, increasing
    probe: tuple              # two values for the first calibration round (low, high guesses)
    make: Callable            # (p, rng) -> dict(prompt, answer, check, data)
    verify: Callable          # (data) -> answer string computed independently
    rungs: tuple = (1,)       # rungs where the generator provides candidates
    source: dict = field(default_factory=lambda: dict(OWN))


def rng_for(name, p, seed):
    return random.Random(f"{name}|{p}|{seed}")


def build(gen, p, seed, set_name, task_id=None, domain=None):
    """One task in the bench format (id, set, domain, title, source, prompt, answer, check) plus `gen` metadata."""
    inst = gen.make(p, rng_for(gen.name, p, seed))
    domain = domain or gen.domain
    return {"id": f"{set_name}/{task_id or domain}", "set": set_name, "domain": task_id or domain,
            "title": gen.title, "source": {**gen.source, **inst.get("source", {})}, "prompt": inst["prompt"].rstrip() + SUFFIX,
            "answer": inst["answer"], "check": inst["check"],
            "gen": {"name": gen.name, "param": gen.param, "value": p, "seed": seed}, "data": inst["data"]}


def frac_str(x):
    """Fraction -> 'p/q' (or 'p' for integers)."""
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def fmt_matrix(rows):
    """Rows of numbers -> aligned text block, one row per line, entries separated by spaces."""
    cells = [[frac_str(v) for v in r] for r in rows]
    w = max(len(c) for r in cells for c in r)
    return "\n".join("[ " + "  ".join(c.rjust(w) for c in r) + " ]" for r in cells)
