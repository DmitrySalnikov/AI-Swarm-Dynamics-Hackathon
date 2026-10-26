"""Rung-3 linear algebra generators, part C: matrix norms, compositions of 3-D rotations, the matrix of a linear
operator on polynomials, vector algebra in R^3, permanents, compositions of affine maps, leading principal minors,
the Sylvester equation AX + XB = C.
"""
import itertools
import math
from fractions import Fraction

import numpy as np
import sympy

from common import Gen, frac_str, fmt_matrix
from r2_physics_a import sig

LIST4 = {"type": "list", "rel_tol": 2e-3}
EXACT_LIST = {"type": "list", "rel_tol": 0}


def _list(vals):
    return ",".join(sig(v) for v in vals)


def _mat(rng, n, lo=-9, hi=9):
    return [[rng.randint(lo, hi) for _ in range(n)] for _ in range(n)]


# ---------- matrix norms ----------

def _norm(c):
    A = c["A"]
    if c["kind"] == "one":                      # max column sum
        return max(sum(abs(A[i][j]) for i in range(len(A))) for j in range(len(A[0])))
    if c["kind"] == "inf":                      # max row sum
        return max(sum(abs(x) for x in r) for r in A)
    return math.sqrt(sum(x * x for r in A for x in r))


def make_norms(k, rng):
    ask = {"one": "the 1-norm ‖A‖₁ (maximum absolute column sum)", "inf": "the ∞-norm ‖A‖∞ (maximum absolute row sum)",
           "fro": "the Frobenius norm ‖A‖_F"}
    cases, blocks = [], []
    for j in range(1, k + 1):
        n = rng.randint(4, 6)
        c = dict(kind=rng.choice(list(ask)), A=_mat(rng, n, -20, 20))
        cases.append(c)
        blocks.append(f"{j}. {ask[c['kind']]} of\n{fmt_matrix(c['A'])}")
    prompt = (f"Matrix norms. For each of the following {k} integer matrices find the requested norm:\n\n"
              + "\n\n".join(blocks) + f"\n\nGive the {k} values, each with 4 significant figures (integers may be given "
              "exactly), in the order listed, as plain decimal numbers separated by commas.")
    return dict(prompt=prompt, answer=_list(_norm(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_norms(data):
    return _list(np.linalg.norm(np.array(c["A"], float), {"one": 1, "inf": np.inf, "fro": "fro"}[c["kind"]])
                 for c in data["cases"])


# ---------- compositions of rotations in R^3 ----------

def _rot(axis, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def make_rot(n, rng):
    steps = [(rng.choice("xyz"), rng.choice([-1, 1]) * rng.randint(5, 175)) for _ in range(n)]
    v = [rng.randint(-9, 9) for _ in range(3)]
    if not any(v):
        v[0] = 3
    R = np.eye(3)
    for ax, d in steps:
        R = _rot(ax, d) @ R                     # each rotation acts after the previous ones
    w = R @ np.array(v, float)
    angle = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    lines = [f"{j}. rotation by {d}° about the {ax}-axis" for j, (ax, d) in enumerate(steps, 1)]
    prompt = (f"Rotations of R³ about the coordinate axes; a positive angle is counter-clockwise when looking from the "
              f"positive end of the axis towards the origin (right-hand rule). The following {n} rotations are applied in "
              f"this order, each to the result of the previous ones (all about the fixed coordinate axes):\n"
              + "\n".join(lines) + f"\n\nLet R be the resulting rotation. Find the image R·v of v = ({v[0]}, {v[1]}, {v[2]})"
              " and the rotation angle of R (between 0° and 180°). Give 4 values — the three coordinates of R·v and the "
              "angle in degrees — each with 4 significant figures, as plain decimal numbers separated by commas.")
    return dict(prompt=prompt, answer=_list(list(w) + [angle]), check={"type": "list", "rel_tol": 2e-3, "abs_tol": 2e-3},
                data={"steps": steps, "v": v})


def verify_rot(data):
    # unit quaternions: q = q_n ... q_1, v' = q v q*, angle = 2 acos |w|
    def qmul(a, b):
        w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    q = (1.0, 0.0, 0.0, 0.0)
    for ax, d in data["steps"]:
        h = math.radians(d) / 2
        u = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[ax]
        q = qmul((math.cos(h), *(math.sin(h) * e for e in u)), q)
    vq = (0.0, *map(float, data["v"]))
    w = qmul(qmul(q, vq), (q[0], -q[1], -q[2], -q[3]))
    angle = math.degrees(2 * math.acos(min(1.0, abs(q[0]))))
    return _list([w[1], w[2], w[3], angle])


# ---------- matrix of a linear operator on polynomials ----------

def make_polyop(n, rng):
    a, b, c, s = rng.randint(-3, 3), rng.randint(-4, 4), rng.randint(-5, 5), rng.randint(1, 3)
    # the diagonal entries are b·j + c, so keep them all non-zero (non-singular operator)
    while (a == 0 and b == 0) or any(b * j + c == 0 for j in range(n + 1)):
        a, b, c = rng.randint(-3, 3), rng.randint(-4, 4), rng.randint(-5, 5)
    x = sympy.symbols("x")
    T = lambda p: sympy.expand(a * sympy.diff(p, x, 2) + b * x * sympy.diff(p, x) + c * p.subs(x, x + s))
    M = sympy.zeros(n + 1, n + 1)
    for j in range(n + 1):
        col = sympy.Poly(T(x ** j), x)
        for i in range(n + 1):
            M[i, j] = col.coeff_monomial(x ** i)
    tr, det = M.trace(), M.det()
    p_terms = [rng.randint(-4, 4) for _ in range(n + 1)]
    p = sum(cf * x ** i for i, cf in enumerate(p_terms))
    value = T(p).subs(x, 1)
    term = lambda cf, sym: f"{cf}·{sym}" if cf not in (1, -1) else ("" if cf == 1 else "−") + sym
    expr = " + ".join(t for t in [term(a, "p''(x)") if a else "", term(b, "x·p'(x)") if b else "",
                                  term(c, f"p(x + {s})") if c else ""] if t)
    poly_txt = " + ".join(f"{cf}·x^{i}" for i, cf in enumerate(p_terms) if cf) or "0"
    prompt = (f"Let V be the space of real polynomials of degree at most {n}, with the basis 1, x, x², …, x^{n}. The linear "
              f"operator T: V → V is defined by T(p)(x) = {expr}.\nLet M be the matrix of T in this basis (the j-th column "
              f"holds the coordinates of T(x^j)). Find (1) the trace of M, (2) the determinant of M and (3) the value "
              f"T(q)(1) for q(x) = {poly_txt}. Give the 3 exact integers in this order, separated by commas.")
    return dict(prompt=prompt, answer=f"{tr},{det},{value}", check=EXACT_LIST,
                data={"a": a, "b": b, "c": c, "s": s, "n": n, "q": p_terms})


def verify_polyop(data):
    # matrix entries from closed formulas for each part of T (no symbolic algebra); M is upper triangular
    a, b, c, s, n = data["a"], data["b"], data["c"], data["s"], data["n"]
    M = [[Fraction(0)] * (n + 1) for _ in range(n + 1)]
    for j in range(n + 1):
        if j >= 2:
            M[j - 2][j] += a * j * (j - 1)
        M[j][j] += b * j
        for i in range(j + 1):
            M[i][j] += c * math.comb(j, i) * s ** (j - i)
    tr = sum(M[i][i] for i in range(n + 1))
    det = math.prod(M[i][i] for i in range(n + 1))          # triangular
    q = data["q"]
    val = sum(sum(M[i][j] * q[j] for j in range(n + 1)) for i in range(n + 1))   # T(q)(1) = sum of coordinates
    return f"{tr},{det},{val}"


# ---------- vector algebra in R^3 ----------

def _vec(c):
    P = [np.array(p, float) for p in c["pts"]]
    if c["kind"] == "volume":                   # tetrahedron volume
        return abs(np.dot(P[1] - P[0], np.cross(P[2] - P[0], P[3] - P[0]))) / 6
    if c["kind"] == "area":
        return np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0])) / 2
    if c["kind"] == "dist_line":                # distance from P0 to the line through P1, P2
        d = P[2] - P[1]
        return np.linalg.norm(np.cross(P[0] - P[1], d)) / np.linalg.norm(d)
    if c["kind"] == "dist_plane":               # distance from P0 to the plane through P1, P2, P3
        nrm = np.cross(P[2] - P[1], P[3] - P[1])
        return abs(np.dot(P[0] - P[1], nrm)) / np.linalg.norm(nrm)
    d1, d2 = P[1] - P[0], P[3] - P[2]           # distance between skew lines P0P1 and P2P3
    nrm = np.cross(d1, d2)
    return abs(np.dot(P[2] - P[0], nrm)) / np.linalg.norm(nrm)


def make_vec(k, rng):
    spec = {"volume": (4, "the volume of the tetrahedron with vertices P1, P2, P3, P4"),
            "area": (3, "the area of the triangle with vertices P1, P2, P3"),
            "dist_line": (3, "the distance from P1 to the line through P2 and P3"),
            "dist_plane": (4, "the distance from P1 to the plane through P2, P3, P4"),
            "skew": (4, "the distance between the line through P1, P2 and the line through P3, P4")}
    cases, lines = [], []
    for j in range(1, k + 1):
        kind = rng.choice(list(spec))
        m, txt = spec[kind]
        while True:
            c = dict(kind=kind, pts=[[rng.randint(-9, 9) for _ in range(3)] for _ in range(m)])
            P = np.array(c["pts"], float)
            ok = {"volume": lambda: _vec(c) > 0.5, "area": lambda: _vec(c) > 0.5,
                  "dist_line": lambda: np.linalg.norm(P[2] - P[1]) > 0 and _vec(c) > 0.5,
                  "dist_plane": lambda: np.linalg.norm(np.cross(P[2] - P[1], P[3] - P[1])) > 0 and _vec(c) > 0.5,
                  "skew": lambda: np.linalg.norm(np.cross(P[1] - P[0], P[3] - P[2])) > 0 and _vec(c) > 0.5}[kind]
            if ok():
                break
        pts = ", ".join(f"P{i + 1} = ({p[0]}, {p[1]}, {p[2]})" for i, p in enumerate(c["pts"]))
        cases.append(c)
        lines.append(f"{j}. {pts}; {txt}.")
    prompt = (f"Vector algebra in R³ (standard Euclidean coordinates). Find the requested quantity in each of the following "
              f"{k} independent situations:\n" + "\n".join(lines) + f"\n\nGive the {k} values, each with 4 significant "
              "figures, in the order listed, as plain decimal numbers separated by commas.")
    return dict(prompt=prompt, answer=_list(_vec(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_vec(data):
    # Gram determinants: k-volume of vectors = sqrt(det G); distances as ratios of volumes (exact via Fractions)
    def vol(vs):
        G = sympy.Matrix([[sum(a * b for a, b in zip(u, v)) for v in vs] for u in vs])
        return math.sqrt(float(G.det()))
    out = []
    for c in data["cases"]:
        P = [list(map(int, p)) for p in c["pts"]]
        sub = lambda p, q: [x - y for x, y in zip(p, q)]
        if c["kind"] == "volume":
            out.append(vol([sub(P[1], P[0]), sub(P[2], P[0]), sub(P[3], P[0])]) / 6)
        elif c["kind"] == "area":
            out.append(vol([sub(P[1], P[0]), sub(P[2], P[0])]) / 2)
        elif c["kind"] == "dist_line":
            out.append(vol([sub(P[2], P[1]), sub(P[0], P[1])]) / vol([sub(P[2], P[1])]))
        elif c["kind"] == "dist_plane":
            out.append(vol([sub(P[2], P[1]), sub(P[3], P[1]), sub(P[0], P[1])]) / vol([sub(P[2], P[1]), sub(P[3], P[1])]))
        else:
            d1, d2 = sub(P[1], P[0]), sub(P[3], P[2])
            out.append(vol([d1, d2, sub(P[2], P[0])]) / vol([d1, d2]))
    return _list(out)


# ---------- permanent ----------

def _perm_ryser(A):
    n = len(A)
    total = 0
    for S in range(1, 1 << n):
        cols = [j for j in range(n) if S >> j & 1]
        prod = 1
        for i in range(n):
            prod *= sum(A[i][j] for j in cols)
        total += (-1) ** len(cols) * prod
    return (-1) ** n * total


def make_perm(n, rng):
    A = _mat(rng, n, -3, 4)
    prompt = (f"The permanent of an n×n matrix A is per(A) = Σ over all permutations σ of {{1, …, n}} of "
              f"a(1,σ(1))·a(2,σ(2))·…·a(n,σ(n)) — like the determinant but with every term taken with a plus sign. "
              f"Compute the permanent of the following {n}×{n} integer matrix:\n\n{fmt_matrix(A)}\n\nGive the exact value "
              "as an integer.")
    return dict(prompt=prompt, answer=str(_perm_ryser(A)), check={"type": "num", "rel_tol": 0}, data={"A": A})


def verify_perm(data):
    A = data["A"]
    n = len(A)
    return str(sum(math.prod(A[i][s[i]] for i in range(n)) for s in itertools.permutations(range(n))))


# ---------- compositions of affine maps of the plane ----------

def _affine_steps(rng, n):
    steps = []
    for _ in range(n):
        kind = rng.choice(["shear_x", "shear_y", "reflect", "rotate", "translate", "scale"])
        if kind in ("shear_x", "shear_y"):
            steps.append((kind, rng.choice([-2, -1, 1, 2])))
        elif kind == "reflect":
            steps.append((kind, rng.choice(["x-axis", "y-axis", "line y = x", "line y = −x"])))
        elif kind == "rotate":
            steps.append((kind, rng.choice([90, 180, 270])))
        elif kind == "translate":
            steps.append((kind, (rng.randint(-5, 5), rng.randint(-5, 5))))
        else:
            steps.append((kind, rng.choice(["2", "1/2", "-1"])))
    return steps


def _affine_matrix(step):
    """3x3 homogeneous matrix (Fractions) of one step."""
    kind, p = step
    F = Fraction
    if kind == "shear_x":
        L, t = [[1, p], [0, 1]], (0, 0)
    elif kind == "shear_y":
        L, t = [[1, 0], [p, 1]], (0, 0)
    elif kind == "reflect":
        L, t = {"x-axis": [[1, 0], [0, -1]], "y-axis": [[-1, 0], [0, 1]], "line y = x": [[0, 1], [1, 0]],
                "line y = −x": [[0, -1], [-1, 0]]}[p], (0, 0)
    elif kind == "rotate":
        L, t = {90: [[0, -1], [1, 0]], 180: [[-1, 0], [0, -1]], 270: [[0, 1], [-1, 0]]}[p], (0, 0)
    elif kind == "translate":
        L, t = [[1, 0], [0, 1]], p
    else:
        s = F(p)
        L, t = [[s, 0], [0, s]], (0, 0)
    return [[F(L[0][0]), F(L[0][1]), F(t[0])], [F(L[1][0]), F(L[1][1]), F(t[1])], [F(0), F(0), F(1)]]


def make_affine(n, rng):
    steps = _affine_steps(rng, n)
    pts = [(rng.randint(-6, 6), rng.randint(-6, 6)) for _ in range(3)]
    M = [[Fraction(int(i == j)) for j in range(3)] for i in range(3)]
    for s in steps:
        A = _affine_matrix(s)
        M = [[sum(A[i][k] * M[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    images = [(M[0][0] * x + M[0][1] * y + M[0][2], M[1][0] * x + M[1][1] * y + M[1][2]) for x, y in pts]
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    desc = {"shear_x": lambda p: f"the shear (x, y) ↦ (x + {p}·y, y)", "shear_y": lambda p: f"the shear (x, y) ↦ (x, y + {p}·x)",
            "reflect": lambda p: f"the reflection in the {p}", "rotate": lambda p: f"the rotation by {p}° counter-clockwise "
            "about the origin", "translate": lambda p: f"the translation by the vector ({p[0]}, {p[1]})",
            "scale": lambda p: f"the scaling (x, y) ↦ ({p}·x, {p}·y) about the origin"}
    lines = [f"{j}. {desc[k](p)}" for j, (k, p) in enumerate(steps, 1)]
    names = "ABC"
    pt_txt = ", ".join(f"{names[i]} = ({x}, {y})" for i, (x, y) in enumerate(pts))
    prompt = (f"The following {n} transformations of the plane are applied in this order (each to the result of the "
              f"previous ones):\n" + "\n".join(lines) + f"\n\nLet F be the resulting affine map. Find the images F(A), F(B), "
              f"F(C) of the points {pt_txt}, and the determinant of the linear part of F. Give 7 exact values — the x and y "
              "coordinates of F(A), F(B), F(C), then the determinant — as integers or fractions p/q, separated by commas.")
    ans = [v for im in images for v in im] + [det]
    return dict(prompt=prompt, answer=",".join(frac_str(v) for v in ans), check=EXACT_LIST,
                data={"steps": steps, "pts": pts})


def verify_affine(data):
    # apply each step directly to the points with explicit formulas; determinant as the product of step determinants
    out_pts = [tuple(map(Fraction, p)) for p in data["pts"]]
    det = Fraction(1)
    for kind, p in data["steps"]:
        new = []
        for x, y in out_pts:
            if kind == "shear_x":
                x, y = x + p * y, y
            elif kind == "shear_y":
                x, y = x, y + p * x
            elif kind == "reflect":
                x, y = {"x-axis": (x, -y), "y-axis": (-x, y), "line y = x": (y, x), "line y = −x": (-y, -x)}[p]
            elif kind == "rotate":
                for _ in range(p // 90):
                    x, y = -y, x
            elif kind == "translate":
                x, y = x + p[0], y + p[1]
            else:
                x, y = Fraction(p) * x, Fraction(p) * y
            new.append((x, y))
        out_pts = new
        det *= {"shear_x": 1, "shear_y": 1, "reflect": -1, "rotate": 1, "translate": 1}.get(kind) or Fraction(p) ** 2
    return ",".join(frac_str(v) for v in [c for pt in out_pts for c in pt] + [det])


# ---------- leading principal minors ----------

def make_minors(n, rng):
    while True:
        B = _mat(rng, n, -3, 3)
        A = [[sum(B[k][i] * B[k][j] for k in range(n)) + (rng.randint(-2, 2) if i == j else 0) for j in range(n)]
             for i in range(n)]
        A = [[A[i][j] if i <= j else A[j][i] for j in range(n)] for i in range(n)]       # symmetric
        minors = [sympy.Matrix([r[:k] for r in A[:k]]).det() for k in range(1, n + 1)]
        if all(m != 0 for m in minors):
            break
    pd = "yes" if all(m > 0 for m in minors) else "no"
    prompt = (f"For the symmetric {n}×{n} integer matrix\n\n{fmt_matrix(A)}\n\nfind all {n} leading principal minors Δ1, …, "
              f"Δ{n} (Δk is the determinant of the upper-left k×k submatrix). Give the {n} exact integers in the order "
              "Δ1, …, Δ" + f"{n}, separated by commas.")
    return dict(prompt=prompt, answer=",".join(str(m) for m in minors), check=EXACT_LIST, data={"A": A, "pd": pd})


def verify_minors(data):
    # Gaussian elimination without row exchanges: Δk = product of the first k pivots (exact Fractions)
    A = [[Fraction(x) for x in r] for r in data["A"]]
    n, out, prod = len(A), [], Fraction(1)
    for k in range(n):
        piv = A[k][k]
        prod *= piv
        out.append(prod)
        for i in range(k + 1, n):
            f = A[i][k] / piv
            A[i] = [a - f * b for a, b in zip(A[i], A[k])]
    return ",".join(frac_str(v) for v in out)


# ---------- Sylvester matrix equation AX + XB = C ----------

def make_sylvester(n, rng):
    while True:
        A, B = _mat(rng, n, -4, 4), _mat(rng, n, -4, 4)
        ea = np.linalg.eigvals(np.array(A, float))
        eb = np.linalg.eigvals(np.array(B, float))
        if min(abs(a + b) for a in ea for b in eb) > 0.3:      # spectra of A and -B disjoint: unique solution
            break
    X = _mat(rng, n, -5, 5)
    mul = lambda P, Q: [[sum(P[i][k] * Q[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    AX, XB = mul(A, X), mul(X, B)
    C = [[AX[i][j] + XB[i][j] for j in range(n)] for i in range(n)]
    prompt = (f"Find the {n}×{n} matrix X that satisfies the matrix equation A·X + X·B = C, where\n\nA =\n{fmt_matrix(A)}\n\n"
              f"B =\n{fmt_matrix(B)}\n\nC =\n{fmt_matrix(C)}\n\n(the solution is unique). Give the {n * n} entries of X "
              "row by row as exact numbers separated by commas.")
    return dict(prompt=prompt, answer=",".join(str(x) for r in X for x in r), check=EXACT_LIST,
                data={"A": A, "B": B, "C": C})


def verify_sylvester(data):
    # vectorise: (I ⊗ A + Bᵀ ⊗ I) vec(X) = vec(C), solved exactly with sympy (column-major vec)
    A, B, C = (sympy.Matrix(data[k]) for k in ("A", "B", "C"))
    n = A.shape[0]
    I = sympy.eye(n)
    K = sympy.kronecker_product(I, A) + sympy.kronecker_product(B.T, I)
    x = K.LUsolve(sympy.Matrix([C[i, j] for j in range(n) for i in range(n)]))
    X = [[x[j * n + i] for j in range(n)] for i in range(n)]
    return ",".join(frac_str(Fraction(int(v.p), int(v.q))) for r in X for v in r)


GENS = [
    Gen("r3_norms", "matrix_norms", "LA", "Matrix norms", "number of matrices K", grid=list(range(2, 81)),
        probe=(8, 20), make=make_norms, verify=verify_norms, rungs=(3,)),
    Gen("r3_rotations", "rotation_composition", "LA", "Composition of rotations in R³", "number of rotations n",
        grid=list(range(2, 61)), probe=(6, 12), make=make_rot, verify=verify_rot, rungs=(3,)),
    Gen("r3_poly_operator", "operator_matrix", "LA", "Matrix of a linear operator on polynomials",
        "maximum degree n", grid=list(range(2, 21)), probe=(5, 8), make=make_polyop, verify=verify_polyop, rungs=(3,)),
    Gen("r3_vectors", "vector_algebra", "LA", "Vector algebra in R³", "number of cases K", grid=list(range(2, 81)),
        probe=(8, 20), make=make_vec, verify=verify_vec, rungs=(3,)),
    Gen("r3_permanent", "permanent", "LA", "Permanent of an integer matrix", "matrix size n", grid=list(range(3, 9)),
        probe=(5, 6), make=make_perm, verify=verify_perm, rungs=(3,)),
    Gen("r3_affine", "affine_maps", "LA", "Composition of affine maps of the plane", "number of maps n",
        grid=list(range(2, 81)), probe=(10, 25), make=make_affine, verify=verify_affine, rungs=(3,)),
    Gen("r3_minors", "leading_minors", "LA", "Leading principal minors of a symmetric matrix", "matrix size n",
        grid=list(range(3, 11)), probe=(6, 7), make=make_minors, verify=verify_minors, rungs=(3,)),
    Gen("r3_sylvester", "sylvester_equation", "LA", "Sylvester matrix equation AX + XB = C", "matrix size n",
        grid=list(range(2, 6)), probe=(2, 3), make=make_sylvester, verify=verify_sylvester, rungs=(3,)),
]
