"""Generators shared between rungs: integer determinant (rungs 1, 3, 4) and G+C count (rung 1, backup for rung 4)."""
import re
from collections import Counter
from fractions import Fraction
from functools import partial

import sympy

from common import Gen, fmt_matrix


# ---------- determinant of an n x n integer matrix ----------

def _det_gauss(rows):
    a = [[Fraction(v) for v in r] for r in rows]
    n, det = len(a), Fraction(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if a[r][c] != 0), None)
        if piv is None:
            return Fraction(0)
        if piv != c:
            a[c], a[piv] = a[piv], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return det


def make_det(n, rng, lo=-9, hi=9):
    while True:
        rows = [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]
        if any(all(v == 0 for v in r) for r in rows):
            continue
        d = _det_gauss(rows)
        if d != 0:
            break
    prompt = (f"Compute the determinant of the following {n}×{n} integer matrix A.\n\nA =\n{fmt_matrix(rows)}\n\n"
              "Give the exact value as an integer.")
    return dict(prompt=prompt, answer=str(int(d)), check={"type": "num", "rel_tol": 0}, data={"rows": rows})


def verify_det(data):
    return str(sympy.Matrix(data["rows"]).det(method="bareiss"))


# ---------- number of G and C in a DNA sequence ----------

def make_gc(L, rng):
    seq = "".join(rng.choice("ACGT") for _ in range(L))
    c = Counter(seq)
    prompt = (f"Here is a DNA sequence of {L} nucleotides:\n\n{seq}\n\n"
              "How many nucleotides in this sequence are G or C? Give the total count of G plus C as an integer.")
    return dict(prompt=prompt, answer=str(c["G"] + c["C"]), check={"type": "num", "rel_tol": 0}, data={"seq": seq})


def verify_gc(data):
    return str(len(re.findall("[GC]", data["seq"])))


GENS = [
    Gen("la_det", "determinant", "linear_algebra", "Определитель целочисленной матрицы", "matrix size n",
        grid=list(range(4, 13)), probe=(6, 8), make=make_det, verify=verify_det, rungs=(1, 3, 4)),
    # entry-range variants, to bring the length nearer the target than n = 5 / 6 with entries in [-9, 9]
    Gen("la_det_e4", "determinant", "linear_algebra", "Определитель целочисленной матрицы", "matrix size n",
        grid=list(range(4, 13)), probe=(6, 7), make=partial(make_det, lo=-4, hi=4), verify=verify_det, rungs=(1, 3, 4)),
    Gen("la_det_e20", "determinant", "linear_algebra", "Определитель целочисленной матрицы", "matrix size n",
        grid=list(range(4, 13)), probe=(5, 6), make=partial(make_det, lo=-20, hi=20), verify=verify_det, rungs=(1, 3, 4)),
    Gen("bio_gc", "gc_count", "biology", "Число G+C в последовательности ДНК", "sequence length L, nt",
        grid=list(range(100, 1501, 20)), probe=(200, 400), make=make_gc, verify=verify_gc, rungs=(1, 4)),
]
