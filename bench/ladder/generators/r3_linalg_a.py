"""Rung-3 linear-algebra generators, part A (task types 2-15; type 1, the determinant, is core.la_det).

Matrices are built backwards (unit-triangular factors, designed RREF, etc.) so that hand elimination stays in small
integers or small-denominator fractions. `make` computes the answer with the Fraction helpers below; `verify` uses
sympy built-ins or a different algorithm.
"""
from fractions import Fraction

import sympy

from common import Gen, fmt_matrix, frac_str


# ---------- shared helpers (also imported by r3_linalg_b) ----------

def mul(A, B):
    return [[sum(a * b for a, b in zip(r, c)) for c in zip(*B)] for r in A]


def eye(n):
    return [[int(i == j) for j in range(n)] for i in range(n)]


def tr(A):
    return [list(c) for c in zip(*A)]


def maxabs(*Ms):
    return max(abs(v) for M in Ms for r in M for v in r)


def ints(M):
    assert all(Fraction(v).denominator == 1 for r in M for v in r)
    return [[int(v) for v in r] for r in M]


def nz(rng, amp):
    return rng.choice([v for v in range(-amp, amp + 1) if v])


def unit_tri(n, rng, lower, amp=1, dens=0.5):
    M = eye(n)
    for i in range(n):
        for j in range(i):
            if rng.random() < dens:
                if lower:
                    M[i][j] = nz(rng, amp)
                else:
                    M[j][i] = nz(rng, amp)
    return M


def unimodular(n, rng, amp=1, dens=0.5):
    """L*U with unit-triangular integer factors and a +-1 diagonal in U: det +-1, elimination pivots +-1."""
    U = unit_tri(n, rng, False, amp, dens)
    for i in range(n):
        U[i] = [v * rng.choice((1, -1)) for v in U[i]] if rng.random() < 0.3 else U[i]
    return mul(unit_tri(n, rng, True, amp, dens), U)


def inverse(A):
    """Gauss-Jordan inverse in Fractions (raises StopIteration if singular)."""
    n = len(A)
    M = [[Fraction(v) for v in r] + [Fraction(int(i == j)) for j in range(n)] for i, r in enumerate(A)]
    for c in range(n):
        p = next(r for r in range(c, n) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        M[c] = [x / M[c][c] for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [r[n:] for r in M]


def rref(A):
    """(RREF as Fractions, list of pivot columns)."""
    M = [[Fraction(v) for v in r] for r in A]
    m, n, piv, r = len(M), len(M[0]), [], 0
    for c in range(n):
        if r == m:
            break
        p = next((i for i in range(r, m) if M[i][c] != 0), None)
        if p is None:
            continue
        M[r], M[p] = M[p], M[r]
        M[r] = [x / M[r][c] for x in M[r]]
        for i in range(m):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    return M, piv


def det(A):
    a = [[Fraction(v) for v in r] for r in A]
    n, d = len(a), Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if a[r][c] != 0), None)
        if p is None:
            return Fraction(0)
        if p != c:
            a[c], a[p] = a[p], a[c]
            d = -d
        d *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [x - f * y for x, y in zip(a[r], a[c])]
    return d


def designed(m, n, r, rng, amp=3, bound=20):
    """m x n integer matrix A = B*C of rank r whose RREF is the integer matrix C (r nonzero rows, zeros below).
    B = first r columns of a unimodular L*U, so elimination without row swaps stays integral with pivots +-1."""
    while True:
        piv = [0] + sorted(rng.sample(range(1, n), r - 1))
        C = [[0] * n for _ in range(r)]
        for i, c in enumerate(piv):
            C[i][c] = 1
            for j in range(c + 1, n):
                if j not in piv and rng.random() < 0.7:
                    C[i][j] = rng.randint(-amp, amp)
        B = [row[:r] for row in unimodular(m, rng, 1, 0.8)]
        A = mul(B, C)
        if maxabs(A) <= bound and all(any(row) for row in A) and all(any(col) for col in zip(*A)):
            return A, C + [[0] * n for _ in range(m - r)], piv


def gram(V):
    """Gram matrix of a list of vectors."""
    return [[sum(a * b for a, b in zip(u, v)) for v in V] for u in V]


def lattice_cols(n, k, rng, gbound, amp=1, bound=9):
    """k integer vectors in R^n (first k columns of a unimodular matrix), Gram determinant <= gbound."""
    while True:
        G = unimodular(n, rng, amp, 0.7)
        perm = rng.sample(range(n), n)
        V = [[G[perm[i]][j] for i in range(n)] for j in range(k)]
        if maxabs(V) <= bound and all(any(v) for v in V) and 1 < det(gram(V)) <= gbound:
            return V


def gs_norms(V):
    """Gram-Schmidt (Fractions): squared norms of the orthogonalised vectors."""
    U, out = [], []
    for v in V:
        u = [Fraction(x) for x in v]
        for w, ww in zip(U, out):
            c = sum(a * b for a, b in zip(v, w)) / ww
            u = [a - c * b for a, b in zip(u, w)]
        U.append(u)
        out.append(sum(a * a for a in u))
    return U, out


def flat(M):
    return ", ".join(frac_str(v) for r in M for v in r)


def joined(xs):
    return ", ".join(frac_str(v) for v in xs)


def vec(name, v):
    return f"{name} = (" + ", ".join(str(x) for x in v) + ")"


def smat(M):
    return sympy.Matrix(M)


def sflat(M):
    return ", ".join(str(v) for v in M)  # sympy Matrix iterates row by row


LIST0 = {"type": "list", "rel_tol": 0}
NUM0 = {"type": "num", "rel_tol": 0}
FRAC = "exact fractions a/b where needed"


def task(intro, mats, ask):
    """Common prompt layout: intro sentence, named matrices, question with answer format."""
    blocks = "\n\n".join(f"{name} =\n{fmt_matrix(M)}" for name, M in mats)
    return f"{intro}\n\n{blocks}\n\n{ask}"


# ---------- 2. inverse of a unimodular matrix ----------

def make_inverse(n, rng):
    while True:
        A = unimodular(n, rng, 2 if n <= 6 else 1, 0.8)
        Ai = ints(inverse(A))
        if maxabs(A) <= 20 and maxabs(Ai) <= 99:
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A (its determinant is ±1, so A⁻¹ has integer entries).",
                  [("A", A)],
                  f"Compute the inverse matrix A⁻¹. Give all {n * n} entries of A⁻¹ row by row (first row left to right, "
                  "then the second row, and so on) as one comma-separated list of integers.")
    return dict(prompt=prompt, answer=flat(Ai), check=LIST0, data={"A": A})


def verify_inverse(data):
    return sflat(smat(data["A"]).inv())


# ---------- 3. rank (plus pivot columns, so the answer is hard to guess) ----------

def make_rank(n, rng):
    r = rng.randint(max(2, n // 2), n - 1)
    A, _, piv = designed(n, n, r, rng)
    prompt = task(f"Consider the following {n}×{n} integer matrix A.", [("A", A)],
                  "Find the rank r of A and its pivot columns: the columns that are not linear combinations of the "
                  "columns to their left (equivalently, the columns containing the leading 1s of the reduced row "
                  "echelon form of A). Give r followed by the 1-based indices of the pivot columns in increasing "
                  "order, as one comma-separated list of integers (r, c_1, …, c_r).")
    return dict(prompt=prompt, answer=joined([r] + [c + 1 for c in piv]), check=LIST0, data={"A": A})


def verify_rank(data):
    M = smat(data["A"])
    piv = M.rref()[1]
    assert len(piv) == M.rank()
    return joined([len(piv)] + [c + 1 for c in piv])


# ---------- 4. eigenvalues (A = P D P^-1) ----------

def make_eigen(n, rng):
    while True:
        D = [rng.randint(-6, 6) for _ in range(n)]
        P = unimodular(n, rng, 1, 0.7)
        A = ints(mul(mul(P, [[D[i] * int(i == j) for j in range(n)] for i in range(n)]), inverse(P)))
        if maxabs(A) <= 20 and len(set(D)) >= n - 1 and sum(v == 0 for r in A for v in r) <= n * n // 3:
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A. All its eigenvalues are integers.", [("A", A)],
                  "Find all eigenvalues of A. List them in increasing order, each repeated according to its algebraic "
                  f"multiplicity ({n} values in total), as one comma-separated list of integers.")
    return dict(prompt=prompt, answer=joined(sorted(D)), check=LIST0, data={"A": A})


def verify_eigen(data):
    return joined(sorted(smat(data["A"]).charpoly().all_roots()))


# ---------- 5. characteristic polynomial ----------

def charpoly(A):
    """Faddeev-LeVerrier: coefficients of det(xI - A), highest degree first."""
    n = len(A)
    c, M = [Fraction(1)], [[Fraction(0)] * n for _ in range(n)]
    for k in range(1, n + 1):
        M = [[x + (c[-1] if i == j else 0) for j, x in enumerate(r)] for i, r in enumerate(mul(A, M))]
        AM = mul(A, M)
        c.append(-sum(AM[i][i] for i in range(n)) / k)
    return c


def make_charpoly(n, rng):
    while True:
        A = [[rng.randint(-4, 4) if rng.random() < 0.7 else 0 for _ in range(n)] for _ in range(n)]
        c = charpoly(A)
        if all(any(r) for r in A) and c[-1] != 0 and max(abs(v) for v in c) <= 3000:
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A.", [("A", A)],
                  f"Compute the characteristic polynomial p(x) = det(xI − A) = x^{n} + c_{n - 1}x^{n - 1} + … + c_1x + "
                  f"c_0. Give all {n + 1} coefficients from the highest degree to the constant term (1, c_{n - 1}, …, "
                  "c_1, c_0), "
                  "including zero coefficients, as one comma-separated list of integers.")
    return dict(prompt=prompt, answer=joined(c), check=LIST0, data={"A": A})


def verify_charpoly(data):
    x = sympy.Symbol("x")
    return joined(int(v) for v in smat(data["A"]).charpoly(x).all_coeffs())


# ---------- 6. LU decomposition without pivoting ----------

def make_lu(n, rng):
    while True:
        L = unit_tri(n, rng, True, 2, 0.8)
        U = [[(nz(rng, 3) if i == j else rng.randint(-3, 3)) if j >= i else 0 for j in range(n)] for i in range(n)]
        A = mul(L, U)
        if maxabs(A) <= 20:
            break
    ans = [L[i][j] for i in range(n) for j in range(i)] + [U[i][j] for i in range(n) for j in range(i, n)]
    prompt = task(f"Consider the following {n}×{n} integer matrix A.", [("A", A)],
                  "Find the LU decomposition A = LU without pivoting (no row exchanges), where L is unit "
                  "lower-triangular (ones on the diagonal) and U is upper-triangular; it exists and is unique, and all "
                  "entries of L and U are integers. Give, as one comma-separated list of integers: first the entries "
                  "of L strictly below the diagonal row by row (L21; L31, L32; L41, L42, L43; …), then the entries of U "
                  "on and above the diagonal row by row (U11, U12, …, U1n; U22, …, U2n; …; Unn).")
    return dict(prompt=prompt, answer=joined(ans), check=LIST0, data={"A": A})


def verify_lu(data):
    L, U, perm = smat(data["A"]).LUdecomposition()
    assert not perm
    n = L.rows
    return joined([L[i, j] for i in range(n) for j in range(i)] + [U[i, j] for i in range(n) for j in range(i, n)])


# ---------- 7. PLU with partial pivoting ----------

def make_plu(n, rng):
    """P A = L U with |multipliers| <= 1/2, so the partial-pivoting choice is strict (never a tie)."""
    half = Fraction(1, 2)
    while True:
        L = [[Fraction(int(i == j)) if j >= i else (rng.choice((half, -half)) if rng.random() < 0.6 else Fraction(0))
              for j in range(n)] for i in range(n)]
        U = [[(2 * nz(rng, 3) if i == j else 2 * rng.randint(-3, 3)) if j >= i else 0 for j in range(n)]
             for i in range(n)]
        LU = ints(mul(L, U))
        perm = rng.sample(range(n), n)          # row k of PA is row perm[k] of A
        A = [None] * n
        for k, i in enumerate(perm):
            A[i] = LU[k]
        if maxabs(A) <= 20 and perm != list(range(n)):
            break
    ans = [i + 1 for i in perm] + [U[i][i] for i in range(n)]
    prompt = task(f"Consider the following {n}×{n} integer matrix A.", [("A", A)],
                  "Perform Gaussian elimination with partial pivoting to obtain PA = LU (P a permutation matrix, L unit "
                  "lower-triangular, U upper-triangular). Pivot rule: at step k (k = 1, …, n), among the rows in "
                  "positions k, …, n of the current matrix choose the one whose entry in column k has the largest "
                  "absolute value (if several tie, the one in the smallest position) and swap it into position k; "
                  "then eliminate the entries below the pivot. Let r_1, …, r_n be the original row numbers (1-based "
                  "row indices in A) of the rows that end up in positions 1, …, n, i.e. row k of PA is row r_k of A. "
                  "Give r_1, …, r_n followed by the diagonal entries U11, …, Unn of U, as one comma-separated list "
                  "(2n numbers; the diagonal entries are integers).")
    return dict(prompt=prompt, answer=joined(ans), check=LIST0, data={"A": A})


def verify_plu(data):
    a = [[Fraction(v) for v in r] for r in data["A"]]
    n, rows = len(a), list(range(len(a)))
    for k in range(n):
        p = max(range(k, n), key=lambda i: (abs(a[i][k]), -i))
        a[k], a[p], rows[k], rows[p] = a[p], a[k], rows[p], rows[k]
        for i in range(k + 1, n):
            f = a[i][k] / a[k][k]
            a[i] = [x - f * y for x, y in zip(a[i], a[k])]
    return joined([i + 1 for i in rows] + [a[i][i] for i in range(n)])


# ---------- 8. solve A x = b ----------

def make_solve(n, rng):
    while True:
        L = unit_tri(n, rng, True, 2, 0.8)
        d = [rng.choice((1, -1)) for _ in range(n)]
        for i in rng.sample(range(n), min(n, 2)):
            d[i] *= rng.choice((2, 3))
        U = [[(d[i] if i == j else rng.randint(-2, 2)) if j >= i else 0 for j in range(n)] for i in range(n)]
        A = mul(L, U)
        b = [rng.randint(-9, 9) for _ in range(n)]
        Ai = inverse(A)
        x = [sum(r * v for r, v in zip(row, b)) for row in Ai]
        if maxabs(A) <= 20 and max(abs(v.numerator) for v in x) <= 200 and any(v.denominator > 1 for v in x):
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A and vector b.",
                  [("A", A), ("b (as a column)", [[v] for v in b])],
                  "Solve the linear system A x = b (it has a unique solution). Give x_1, …, x_n as one comma-separated "
                  f"list of exact numbers (integers or {FRAC}).")
    return dict(prompt=prompt, answer=joined(x), check=LIST0, data={"A": A, "b": b})


def verify_solve(data):
    return sflat(smat(data["A"]).LUsolve(sympy.Matrix(data["b"])))


# ---------- 9. null space basis ----------

def make_null(n, rng):
    k = 2 if n <= 5 else 3
    A, C, piv = designed(n - 1, n, n - k, rng)
    free = [j for j in range(n) if j not in piv]
    basis = []
    for f in free:
        v = [0] * n
        v[f] = 1
        for i, c in enumerate(piv):
            v[c] = -C[i][f]
        basis.append(v)
    prompt = task(f"Consider the following {n - 1}×{n} integer matrix A.", [("A", A)],
                  "Find the canonical basis of the null space {x : A x = 0}: take the free variables of the reduced "
                  "row echelon form of A; for each free variable x_f build the solution with x_f = 1 and all other free "
                  "variables 0. List these basis vectors in order of increasing free-variable index f, and write each "
                  "vector as its n entries x_1, …, x_n. Give all vectors one after another as one comma-separated "
                  "list of integers (k·n numbers for a k-dimensional null space).")
    return dict(prompt=prompt, answer=", ".join(joined(v) for v in basis), check=LIST0, data={"A": A})


def verify_null(data):
    return ", ".join(sflat(v) for v in smat(data["A"]).nullspace())


# ---------- 10. RREF ----------

def make_rref(m, rng):
    n = m + 2
    r = m if rng.random() < 0.5 else m - 1
    A, C, _ = designed(m, n, r, rng)
    prompt = task(f"Consider the following {m}×{n} integer matrix A.", [("A", A)],
                  "Compute the reduced row echelon form R of A (leading entries equal to 1, zeros above and below "
                  f"every leading 1, zero rows at the bottom). Give all {m * n} entries of R row by row (first row left to "
                  f"right, then the second row, and so on, including zero rows) as one comma-separated list of exact "
                  f"numbers (integers or {FRAC}).")
    return dict(prompt=prompt, answer=flat(C), check=LIST0, data={"A": A})


def verify_rref(data):
    return sflat(smat(data["A"]).rref()[0])


# ---------- 11. matrix power A^4 ----------

POW = 4


def make_power(n, rng):
    while True:
        A = [[rng.randint(-2, 2) if rng.random() < 0.6 else 0 for _ in range(n)] for _ in range(n)]
        A4 = mul(mul(A, A), mul(A, A))
        if all(any(r) for r in A) and 20 <= maxabs(A4) <= 300 * n // 3:
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A.", [("A", A)],
                  f"Compute the matrix power A^{POW} = A·A·A·A. Give all {n * n} entries of A^{POW} row by row (first row "
                  "left to right, then the second row, and so on) as one comma-separated list of integers.")
    return dict(prompt=prompt, answer=flat(A4), check=LIST0, data={"A": A})


def verify_power(data):
    return sflat(smat(data["A"]) ** POW)


# ---------- 12. Gram-Schmidt: squared norms ----------

def make_gs(k, rng):
    n = k + 1
    while True:
        V = lattice_cols(n, k, rng, 10 * k * k, 1, 4)
        U, nn = gs_norms(V)
        if max(v.denominator for u in U for v in u) <= 12 * k and any(v.denominator > 1 for v in nn):
            break
    prompt = (f"Apply the Gram–Schmidt process (standard dot product, no normalisation) to the following {k} vectors "
              f"of R^{n}, in the given order:\n\n" + "\n".join(vec(f"v{i + 1}", v) for i, v in enumerate(V)) +
              "\n\nThat is, u1 = v1 and u_j = v_j − Σ_{i<j} (v_j·u_i)/(u_i·u_i) u_i. Give the squared lengths "
              f"u1·u1, …, u{k}·u{k} (in this order) as one comma-separated list of exact numbers (integers or {FRAC}).")
    return dict(prompt=prompt, answer=joined(nn), check=LIST0, data={"V": V})


def verify_gs(data):
    V = data["V"]
    G = smat(gram(V))
    minors = [1] + [G[:j, :j].det() for j in range(1, len(V) + 1)]   # |u_j|^2 = G_j / G_{j-1}
    return joined(Fraction(int(minors[j]), int(minors[j - 1])) for j in range(1, len(V) + 1))


# ---------- 13. orthogonal projection onto a span ----------

def make_proj(k, rng):
    n = k + 2
    while True:
        V = lattice_cols(n, k, rng, 40, 1, 3)
        b = [rng.randint(-5, 5) for _ in range(n)]
        U, nn = gs_norms(V)
        p = [Fraction(0)] * n
        for u, uu in zip(U, nn):
            c = sum(x * y for x, y in zip(b, u)) / uu
            p = [x + c * y for x, y in zip(p, u)]
        if any(v.denominator > 1 for v in p) and p != b and max(abs(v.numerator) for v in p) <= 300:
            break
    prompt = (f"In R^{n} with the standard dot product, let W be the subspace spanned by the {k} linearly independent "
              "vectors\n\n" + "\n".join(vec(f"w{i + 1}", v) for i, v in enumerate(V)) +
              f"\n\nand let\n\n{vec('b', b)}\n\nCompute the orthogonal projection of b onto W. Give its {n} coordinates "
              f"as one comma-separated list of exact numbers (integers or {FRAC}).")
    return dict(prompt=prompt, answer=joined(p), check=LIST0, data={"V": V, "b": b})


def verify_proj(data):
    M = sympy.Matrix(data["V"]).T
    return sflat(M * (M.T * M).inv() * M.T * sympy.Matrix(data["b"]))


# ---------- 14. least squares ----------

def make_lsq(n, rng):
    m = n + 2
    while True:
        cols = lattice_cols(m, n, rng, 50, 1, 3)
        A = tr(cols)
        b = [rng.randint(-6, 6) for _ in range(m)]
        x = [sum(r * v for r, v in zip(row, [sum(a * c for a, c in zip(col, b)) for col in cols]))
             for row in inverse(gram(cols))]
        if any(v.denominator > 1 for v in x) and max(abs(v.numerator) for v in x) <= 300:
            break
    prompt = task(f"Consider the overdetermined linear system A x = b with the following {m}×{n} integer matrix A "
                  "(its columns are linearly independent) and vector b.",
                  [("A", A), ("b (as a column)", [[v] for v in b])],
                  "Find the least-squares solution x, i.e. the vector minimising ‖A x − b‖² (Euclidean norm). Give "
                  f"x_1, …, x_n as one comma-separated list of exact numbers (integers or {FRAC}).")
    return dict(prompt=prompt, answer=joined(x), check=LIST0, data={"A": A, "b": b})


def verify_lsq(data):
    return sflat(smat(data["A"]).pinv() * sympy.Matrix(data["b"]))


# ---------- 15. adjugate ----------

def make_adj(n, rng):
    while True:
        L = unit_tri(n, rng, True, 2, 0.8)
        d = [rng.choice((1, -1)) for _ in range(n)]
        d[rng.randrange(n)] *= rng.choice((2, 3))
        U = [[(d[i] if i == j else rng.randint(-2, 2)) if j >= i else 0 for j in range(n)] for i in range(n)]
        A = mul(L, U)
        D = det(A)
        adj = ints([[D * v for v in r] for r in inverse(A)])
        if maxabs(A) <= 20 and maxabs(adj) <= 150:
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A.", [("A", A)],
                  "Compute the adjugate (classical adjoint) adj(A), the transpose of the cofactor matrix, so that "
                  f"A·adj(A) = det(A)·I. Give all {n * n} entries of adj(A) row by row (first row left to right, then the "
                  "second row, and so on) as one comma-separated list of integers.")
    return dict(prompt=prompt, answer=flat(adj), check=LIST0, data={"A": A})


def verify_adj(data):
    return sflat(smat(data["A"]).adjugate(method="berkowitz"))


LA = "linear_algebra"
GENS = [
    Gen("r3_inverse", "inverse", LA, "Обратная матрица", "matrix size n",
        grid=list(range(3, 10)), probe=(4, 5), make=make_inverse, verify=verify_inverse, rungs=(3,)),
    Gen("r3_rank", "rank", LA, "Ранг и базисные столбцы матрицы", "matrix size n",
        grid=list(range(3, 13)), probe=(6, 8), make=make_rank, verify=verify_rank, rungs=(3,)),
    Gen("r3_eigen", "eigenvalues", LA, "Собственные значения матрицы", "matrix size n",
        grid=list(range(3, 9)), probe=(4, 5), make=make_eigen, verify=verify_eigen, rungs=(3,)),
    Gen("r3_charpoly", "char_polynomial", LA, "Характеристический многочлен", "matrix size n",
        grid=list(range(3, 9)), probe=(4, 5), make=make_charpoly, verify=verify_charpoly, rungs=(3,)),
    Gen("r3_lu", "lu", LA, "LU-разложение", "matrix size n",
        grid=list(range(3, 11)), probe=(5, 7), make=make_lu, verify=verify_lu, rungs=(3,)),
    Gen("r3_plu", "plu_pivoting", LA, "LU-разложение с выбором главного элемента", "matrix size n",
        grid=list(range(3, 10)), probe=(5, 7), make=make_plu, verify=verify_plu, rungs=(3,)),
    Gen("r3_solve", "linear_system", LA, "Решение системы линейных уравнений", "matrix size n",
        grid=list(range(3, 11)), probe=(5, 7), make=make_solve, verify=verify_solve, rungs=(3,)),
    Gen("r3_nullspace", "null_space", LA, "Базис ядра матрицы", "number of columns n",
        grid=list(range(3, 12)), probe=(6, 8), make=make_null, verify=verify_null, rungs=(3,)),
    Gen("r3_rref", "rref", LA, "Приведённый ступенчатый вид", "number of rows m (columns m+2)",
        grid=list(range(2, 9)), probe=(4, 6), make=make_rref, verify=verify_rref, rungs=(3,)),
    Gen("r3_power", "matrix_power", LA, "Степень матрицы", "matrix size n (power 4)",
        grid=list(range(2, 9)), probe=(3, 5), make=make_power, verify=verify_power, rungs=(3,)),
    Gen("r3_gram_schmidt", "gram_schmidt", LA, "Ортогонализация Грама–Шмидта", "number of vectors k (in R^(k+1))",
        grid=list(range(2, 8)), probe=(3, 4), make=make_gs, verify=verify_gs, rungs=(3,)),
    Gen("r3_projection", "projection", LA, "Ортогональная проекция на подпространство",
        "number of spanning vectors k (in R^(k+2))",
        grid=list(range(1, 8)), probe=(3, 4), make=make_proj, verify=verify_proj, rungs=(3,)),
    Gen("r3_least_squares", "least_squares", LA, "Метод наименьших квадратов", "number of unknowns n (n+2 equations)",
        grid=list(range(2, 8)), probe=(3, 4), make=make_lsq, verify=verify_lsq, rungs=(3,)),
    Gen("r3_adjugate", "adjugate", LA, "Присоединённая матрица", "matrix size n",
        grid=list(range(3, 9)), probe=(4, 5), make=make_adj, verify=verify_adj, rungs=(3,)),
]
