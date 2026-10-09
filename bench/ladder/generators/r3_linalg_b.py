"""Rung-3 linear-algebra generators, part B (task types 16-28). Helpers come from r3_linalg_a."""
from fractions import Fraction
from itertools import combinations
from math import gcd

import sympy

from common import Gen, frac_str
from r3_linalg_a import (FRAC, LIST0, NUM0, det, eye, gram, gs_norms, ints, inverse, joined, maxabs, mul, nz, rref,
                         sflat, smat, task, tr, unimodular, unit_tri, vec)


def diag(d):
    return [[d[i] if i == j else 0 for j in range(len(d))] for i in range(len(d))]


def sparse_enough(A, frac=3):
    return sum(v == 0 for r in A for v in r) <= len(A) * len(A[0]) // frac


def jordan_blocks(n, rng):
    """Random Jordan structure: list of (eigenvalue, block size) with fewer than n distinct eigenvalues; not scalar."""
    while True:
        d = rng.randint(1, min(3, n - 1))
        lams = sorted(rng.sample(range(-3, 4), d))
        cuts = sorted(rng.sample(range(1, n), d - 1))
        mults = [b - a for a, b in zip([0] + cuts, cuts + [n])]
        blocks = []
        for lam, m in zip(lams, mults):
            sizes = []
            while m:
                sizes.append(rng.randint(1, m))
                m -= sizes[-1]
            blocks += [(lam, s) for s in sorted(sizes, reverse=True)]
        if d > 1 or len(blocks) < n:
            return blocks


def similar_to_jordan(n, rng):
    """(A, blocks) with A = P J P^-1, P unimodular, small dense entries."""
    while True:
        blocks = jordan_blocks(n, rng)
        J, i = [[0] * n for _ in range(n)], 0
        for lam, s in blocks:
            for k in range(s):
                J[i + k][i + k] = lam
                if k:
                    J[i + k - 1][i + k] = 1
            i += s
        P = unimodular(n, rng, 1, 0.7)
        A = ints(mul(mul(P, J), inverse(P)))
        if maxabs(A) <= 20 and sparse_enough(A):
            return A, blocks


def eig_line(blocks):
    lams = sorted({lam for lam, _ in blocks})
    if len(lams) == 1:
        return f"Its only eigenvalue is {lams[0]}."
    return "Its distinct eigenvalues (listed without multiplicity) are: " + ", ".join(str(v) for v in lams) + "."


# ---------- 16. Jordan form block sizes ----------

def make_jordan(n, rng):
    A, blocks = similar_to_jordan(n, rng)
    prompt = task(f"Consider the following {n}×{n} integer matrix A. {eig_line(blocks)}", [("A", A)],
                  "Determine the Jordan normal form of A. Describe every Jordan block by the pair (eigenvalue, block "
                  "size). Order the blocks by eigenvalue in increasing order and, for equal eigenvalues, by block size "
                  "in decreasing order, and write all pairs one after another as one comma-separated list of integers "
                  "λ, size, λ, size, … (for example, blocks J_2(−1), J_1(−1), J_3(4) give −1, 2, −1, 1, 4, 3).")
    return dict(prompt=prompt, answer=joined(v for b in blocks for v in b), check=LIST0, data={"A": A})


def verify_jordan(data):
    J = smat(data["A"]).jordan_form()[1]
    n, blocks, i = J.rows, [], 0
    while i < n:
        s = 1
        while i + s < n and J[i + s - 1, i + s] == 1:
            s += 1
        blocks.append((int(J[i, i]), s))
        i += s
    blocks.sort(key=lambda b: (b[0], -b[1]))
    return joined(v for b in blocks for v in b)


# ---------- 17. Smith normal form ----------

def make_smith(n, rng):
    while True:
        d, cur = [], 1
        for i in range(n):
            if i >= n // 2 or rng.random() < 0.3:
                cur *= rng.choice((1, 2, 2, 3))
            d.append(cur)
        if rng.random() < 0.3:
            d[-1] = 0
        A = mul(mul(unimodular(n, rng, 1, 0.7), diag(d)), unimodular(n, rng, 1, 0.7))
        if d[-2] > 1 and maxabs(A) <= 20 and sparse_enough(A):
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A.", [("A", A)],
                  "Compute the Smith normal form of A over the integers: the diagonal matrix diag(d_1, …, d_n) "
                  "obtained from A by invertible integer row and column operations, with d_i ≥ 0 and d_i dividing "
                  f"d_(i+1) for every i (zeros, if any, come last). Give the invariant factors d_1, …, d_{n} as one "
                  "comma-separated list of non-negative integers.")
    return dict(prompt=prompt, answer=joined(d), check=LIST0, data={"A": A})


def verify_smith(data):
    from sympy.matrices.normalforms import invariant_factors
    f = [abs(int(v)) for v in invariant_factors(smat(data["A"]), domain=sympy.ZZ)]
    return joined(f + [0] * (len(data["A"]) - len(f)))


# ---------- 18. minimal polynomial ----------

def make_minpoly(n, rng):
    A, blocks = similar_to_jordan(n, rng)
    coef = [1]
    for lam in sorted({lam for lam, _ in blocks}):
        for _ in range(max(s for l2, s in blocks if l2 == lam)):
            coef = [a - lam * b for a, b in zip(coef + [0], [0] + coef)]   # multiply by (x - lam)
    prompt = task(f"Consider the following {n}×{n} integer matrix A. {eig_line(blocks)}", [("A", A)],
                  "Find the minimal polynomial m(x) of A: the monic polynomial of least degree with m(A) = 0. Give its "
                  "coefficients from the highest degree down to the constant term (starting with the leading 1 and "
                  "including zero coefficients) as one comma-separated list of integers.")
    return dict(prompt=prompt, answer=joined(coef), check=LIST0, data={"A": A})


def verify_minpoly(data):
    """Krylov in matrix space: the first power A^d that is a combination of I, A, ..., A^(d-1)."""
    A = smat(data["A"])
    pows = [sympy.eye(A.rows)]
    while True:
        pows.append(pows[-1] * A)
        K = sympy.Matrix.hstack(*[p.reshape(A.rows * A.rows, 1) for p in pows])
        ns = K.nullspace()
        if ns:
            v = ns[0] / ns[0][-1]
            return joined(v[i] for i in range(len(pows) - 1, -1, -1))


# ---------- 19. signature of a quadratic form ----------

def make_signature(n, rng):
    while True:
        z = rng.randint(0, n // 3)
        p = rng.randint(0, n - z)
        signs = [1] * p + [-1] * (n - z - p)
        rng.shuffle(signs)
        D = [s * rng.choice((1, 1, 2, 3)) for s in signs] + [0] * z
        M = unit_tri(n, rng, False, 2, 0.7)
        A = mul(mul(tr(M), diag(D)), M)
        if maxabs(A) <= 20 and sparse_enough(A) and 0 < p < n:
            break
    q = n - z - p
    prompt = task(f"Consider the real quadratic form Q(x) = xᵀ A x in variables x_1, …, x_{n}, given by the following "
                  f"symmetric {n}×{n} integer matrix A.", [("A", A)],
                  "Find the signature of Q, i.e. the number p of positive and the number q of negative coefficients "
                  "in any diagonal form of Q obtained by a real change of variables (Sylvester's law of inertia; for "
                  "example via Lagrange's method of completing squares). Give p, q as a comma-separated list of two "
                  "integers.")
    return dict(prompt=prompt, answer=joined([p, q]), check=LIST0, data={"A": A})


def verify_signature(data):
    """Descartes' rule of signs is exact for the real-rooted characteristic polynomial of a symmetric matrix."""
    x = sympy.Symbol("x")
    c = [int(v) for v in smat(data["A"]).charpoly(x).all_coeffs()]
    while c[-1] == 0:
        c.pop()
    nzc = [v for v in c if v]
    p = sum(a * b < 0 for a, b in zip(nzc, nzc[1:]))
    return joined([p, len(c) - 1 - p])


# ---------- 20. Cholesky decomposition ----------

def make_cholesky(n, rng):
    while True:
        L = [[(rng.choice((1, 1, 2, 3)) if i == j else (nz(rng, 2) if rng.random() < 0.7 else 0))
              if j <= i else 0 for j in range(n)] for i in range(n)]
        A = mul(L, tr(L))
        if maxabs(A) <= 25:
            break
    prompt = task(f"Consider the following symmetric positive definite {n}×{n} integer matrix A.", [("A", A)],
                  "Compute its Cholesky decomposition A = L·Lᵀ, where L is lower-triangular with positive diagonal "
                  "entries (all entries of L are integers). Give the entries of L on and below the diagonal row by row "
                  "(L11; L21, L22; L31, L32, L33; …) as one comma-separated list of integers.")
    return dict(prompt=prompt, answer=joined(L[i][j] for i in range(n) for j in range(i + 1)), check=LIST0,
                data={"A": A})


def verify_cholesky(data):
    L = smat(data["A"]).cholesky(hermitian=False)
    return joined(L[i, j] for i in range(L.rows) for j in range(i + 1))


# ---------- 21. coordinates in a new basis ----------

def make_coords(n, rng):
    while True:
        B = unimodular(n, rng, 2 if n <= 5 else 1, 0.8)
        cs = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(2)]
        vs = [[sum(B[i][j] * c[j] for j in range(n)) for i in range(n)] for c in cs]
        if maxabs(B) <= 9 and maxabs(vs) <= 25 and sparse_enough(B, 4):
            break
    cols = tr(B)
    prompt = (f"The vectors\n\n" + "\n".join(vec(f"b{j + 1}", c) for j, c in enumerate(cols)) +
              f"\n\nform a basis of R^{n}. Find the coordinates of the vectors\n\n{vec('v', vs[0])}\n{vec('w', vs[1])}"
              f"\n\nin this basis, i.e. the numbers c_1, …, c_{n} with v = c_1·b1 + … + c_{n}·b{n}, and likewise for w. "
              f"Give the {n} coordinates of v followed by the {n} coordinates of w as one comma-separated list of "
              f"exact numbers (integers or {FRAC}).")
    return dict(prompt=prompt, answer=", ".join(joined(c) for c in cs), check=LIST0, data={"B": B, "v": vs})


def verify_coords(data):
    B = smat(data["B"])
    return ", ".join(sflat(B.solve(sympy.Matrix(v))) for v in data["v"])


# ---------- 22. traces of powers ----------

KTR = 4


def make_traces(n, rng):
    while True:
        A = [[rng.randint(-2, 2) if rng.random() < 0.6 else 0 for _ in range(n)] for _ in range(n)]
        P, ts = eye(n), []
        for _ in range(KTR):
            P = mul(P, A)
            ts.append(sum(P[i][i] for i in range(n)))
        if all(any(r) for r in A) and maxabs(mul(A, A)) <= 25 and len(set(ts)) == KTR:
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A.", [("A", A)],
                  f"Compute the traces of its powers: tr(A), tr(A²), tr(A³), tr(A⁴). Give these {KTR} numbers in this "
                  "order as one comma-separated list of integers.")
    return dict(prompt=prompt, answer=joined(ts), check=LIST0, data={"A": A})


def verify_traces(data):
    """Newton's identities from the characteristic polynomial coefficients."""
    x = sympy.Symbol("x")
    c = [int(v) for v in smat(data["A"]).charpoly(x).all_coeffs()][1:]
    c += [0] * KTR
    p = []
    for k in range(1, KTR + 1):
        p.append(-k * c[k - 1] - sum(c[i - 1] * p[k - i - 1] for i in range(1, k)))
    return joined(p)


# ---------- 23. sum and intersection of subspaces ----------

def make_subspaces(n, rng):
    while True:
        dU, dW = rng.randint(2, n - 1), rng.randint(2, n - 1)
        dI = rng.randint(max(0, dU + dW - n), min(dU, dW) - 1)
        G = tr(unimodular(n, rng, 1, 0.7))                       # basis vectors g_1..g_n
        bu = G[:dU]
        bw = G[:dI] + G[dU:dU + dW - dI]
        spans = []
        for basis in (bu, bw):
            S = [[sum(c * g[i] for c, g in zip(cf, basis)) for i in range(n)]
                 for cf in ([rng.choice((-1, 0, 1)) for _ in basis] for _ in range(len(basis) + 1))]
            spans.append(S)
        if (maxabs(*spans) <= 9 and all(any(v) for S in spans for v in S)
                and [len(rref(S)[1]) for S in spans] == [dU, dW]):
            break
    U, W = spans
    dims = [dU, dW, dU + dW - dI, dI]
    prompt = (f"In R^{n}, let U be the subspace spanned by the vectors\n\n" +
              "\n".join(vec(f"u{i + 1}", v) for i, v in enumerate(U)) +
              "\n\nand let W be the subspace spanned by the vectors\n\n" +
              "\n".join(vec(f"w{i + 1}", v) for i, v in enumerate(W)) +
              "\n\n(the spanning vectors need not be linearly independent). Find dim U, dim W, dim(U + W) and "
              "dim(U ∩ W). Give these four numbers in this order as one comma-separated list of integers.")
    return dict(prompt=prompt, answer=joined(dims), check=LIST0, data={"U": U, "W": W})


def verify_subspaces(data):
    U, W = sympy.Matrix(data["U"]).T, sympy.Matrix(data["W"]).T
    ker = sympy.Matrix.hstack(U, -W).nullspace()                 # U a = W b  <=>  (a, b) in the kernel
    inter = [U * v[:U.cols, :] for v in ker]
    di = sympy.Matrix.hstack(*inter).rank() if inter else 0
    return joined([U.rank(), W.rank(), sympy.Matrix.hstack(U, W).rank(), di])


# ---------- 24. operator matrix in a new basis ----------

def make_change_basis(n, rng):
    while True:
        B = unimodular(n, rng, 1, 0.8)
        A = [[rng.randint(-4, 4) if rng.random() < 0.7 else 0 for _ in range(n)] for _ in range(n)]
        C = ints(mul(mul(inverse(B), A), B))
        if (maxabs(B) <= 9 and maxabs(C) <= 30 and sparse_enough(B, 4) and sparse_enough(C)
                and all(any(r) for r in A) and all(any(c) for c in zip(*A))):
            break
    prompt = task(f"A linear operator on R^{n} has the matrix A in the standard basis. The columns of the matrix B "
                  "form a new basis (det B = ±1).", [("A", A), ("B", B)],
                  "Find the matrix of the operator in the new basis, C = B⁻¹·A·B. Give all "
                  f"{n * n} entries of C row by row (first row left to right, then the second row, and so on) as one "
                  "comma-separated list of integers.")
    return dict(prompt=prompt, answer=", ".join(joined(r) for r in C), check=LIST0, data={"A": A, "B": B})


def verify_change_basis(data):
    B = smat(data["B"])
    return sflat(B.solve(smat(data["A"]) * B))


# ---------- 25. QR: squared diagonal of R ----------

def make_qr(n, rng):
    while True:
        L = unit_tri(n, rng, True, 1, 0.7)
        d = [rng.choice((1, -1)) for _ in range(n)]
        d[rng.randrange(n)] *= rng.choice((2, 3))
        U = [[(d[i] if i == j else rng.randint(-1, 1)) if j >= i else 0 for j in range(n)] for i in range(n)]
        A = mul(L, U)
        perm = rng.sample(range(n), n)
        A = [[r[j] for j in perm] for r in A]
        Us, nn = gs_norms(tr(A))
        if (maxabs(A) <= 9 and sparse_enough(A, 4) and max(v.denominator for u in Us for v in u) <= 10 * n
                and sum(v.denominator > 1 for v in nn) >= min(2, n - 1)):
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A (it is invertible).", [("A", A)],
                  "Let A = Q·R be its QR decomposition (Gram–Schmidt on the columns of A, left to right), where Q is "
                  "orthogonal and R is upper-triangular with positive diagonal entries. Give the squares of the "
                  f"diagonal entries of R, R11², …, R{n}{n}², in this order as one comma-separated list of exact "
                  f"numbers (integers or {FRAC}).")
    return dict(prompt=prompt, answer=joined(nn), check=LIST0, data={"A": A})


def verify_qr(data):
    """R_jj^2 = det(G_j) / det(G_(j-1)) for the leading Gram minors of the columns."""
    M = smat(data["A"])
    G = M.T * M
    minors = [1] + [G[:j, :j].det() for j in range(1, M.cols + 1)]
    return joined(Fraction(int(minors[j]), int(minors[j - 1])) for j in range(1, M.cols + 1))


# ---------- 26. trace of a product chain ----------

CH = 3


def make_chain(K, rng):
    """Matrices are drawn one at a time (nonsingular, no zero row) so that every partial product stays moderate."""
    def draw():
        while True:
            M = [[rng.randint(-2, 2) if rng.random() < 0.6 else 0 for _ in range(CH)] for _ in range(CH)]
            if all(any(r) for r in M) and det(M) != 0:
                return M
    while True:
        Ms = [draw()]
        P = Ms[0]
        for _ in range(K - 1):
            for _ in range(500):
                M = draw()
                Q = mul(P, M)
                if 3 <= maxabs(Q) <= 60:
                    break
            else:
                break
            Ms.append(M)
            P = Q
        t = sum(P[i][i] for i in range(CH))
        if len(Ms) == K and t != 0:
            break
    names = [f"A{i + 1}" for i in range(K)]
    prompt = task(f"Consider the following {K} integer {CH}×{CH} matrices.", list(zip(names, Ms)),
                  f"Compute the trace of the product {'·'.join(names)} (multiplied in this order). Give the exact "
                  "value as an integer.")
    return dict(prompt=prompt, answer=str(t), check=NUM0, data={"Ms": Ms})


def verify_chain(data):
    P = sympy.eye(CH)
    for M in reversed(data["Ms"]):
        P = smat(M) * P
    return str(P.trace())


# ---------- 27. Gram determinant ----------

def make_gramdet(k, rng):
    n = k + 1
    while True:
        V = [[rng.randint(-2, 2) if rng.random() < 0.7 else 0 for _ in range(n)] for _ in range(k)]
        G = gram(V)
        g = det(G)
        if g != 0 and maxabs(G) <= 20 and g <= 20000 and all(any(v) for v in V) and all(any(c) for c in zip(*V)):
            break
    prompt = (f"Consider the following {k} vectors in R^{n}:\n\n" +
              "\n".join(vec(f"v{i + 1}", v) for i, v in enumerate(V)) +
              f"\n\nCompute the Gram determinant det(G), where G is the {k}×{k} matrix of pairwise dot products "
              f"G_ij = v_i·v_j (this is the squared {k}-dimensional volume of the parallelotope spanned by the "
              "vectors). Give the exact value as an integer.")
    return dict(prompt=prompt, answer=frac_str(g), check=NUM0, data={"V": V})


def verify_gramdet(data):
    """Cauchy-Binet: sum of squared maximal minors of the k x n matrix."""
    M = smat(data["V"])
    return str(sum(M[:, list(c)].det() ** 2 for c in combinations(range(M.cols), M.rows)))


# ---------- 28. singular values ----------

def orth_vectors(n, k, rng):
    """k mutually orthogonal primitive integer vectors with entries in [-2, 2]."""
    W, fails = [], 0
    while len(W) < k:
        if W:
            R, piv = rref(W)
            free = [f for f in range(n) if f not in piv]
            basis = []
            for f in free:
                b = [Fraction(0)] * n
                b[f] = Fraction(1)
                for i, c in enumerate(piv):
                    b[c] = -R[i][f]
                basis.append(b)
            cf = [rng.choice((-1, 0, 1)) for _ in basis]
            v = [sum(c * b[i] for c, b in zip(cf, basis)) for i in range(n)]
        else:
            v = [Fraction(rng.choice((-1, 0, 1))) for _ in range(n)]
        den = 1
        for x in v:
            den = den * x.denominator // gcd(den, x.denominator)
        v = [int(x * den) for x in v]
        g = 0
        for x in v:
            g = gcd(g, x)
        if g and max(abs(x) for x in v) <= 2 * g and sum(x != 0 for x in v) >= 2:
            W.append([x // g for x in v])
        else:
            fails += 1
            if fails > 30:
                W, fails = [], 0
    return W


def signed_perm(n, rng):
    p = rng.sample(range(n), n)
    return [[rng.choice((1, -1)) if j == p[i] else 0 for j in range(n)] for i in range(n)]


def make_svd(n, rng):
    """A = S1 (r I + sum_j t_j w_j w_j^T) S2 with orthogonal integer w_j and signed permutations S1, S2:
    singular values |r + t_j |w_j|^2| and |r| (the rest)."""
    while True:
        k = rng.randint(max(1, n - 2), n - 1)
        W = orth_vectors(n, k, rng)
        r = rng.choice((-3, -2, -1, 1, 2, 3))
        t = [rng.choice((-2, -1, 1, 2)) for _ in W]
        S = [[r * int(i == j) + sum(tj * w[i] * w[j] for tj, w in zip(t, W)) for j in range(n)] for i in range(n)]
        A = mul(mul(signed_perm(n, rng), S), signed_perm(n, rng))
        sv = sorted([abs(r + tj * sum(x * x for x in w)) for tj, w in zip(t, W)] + [abs(r)] * (n - k), reverse=True)
        if maxabs(A) <= 20 and sparse_enough(A) and min(sv) > 0 and len(set(sv)) >= min(n, 3):
            break
    prompt = task(f"Consider the following {n}×{n} integer matrix A. All its singular values are integers.",
                  [("A", A)],
                  "Find the singular values of A (the square roots of the eigenvalues of AᵀA). List all of them in "
                  f"decreasing order, each repeated according to its multiplicity ({n} values in total), as one "
                  "comma-separated list of integers.")
    return dict(prompt=prompt, answer=joined(sv), check=LIST0, data={"A": A})


def verify_svd(data):
    M = smat(data["A"])
    return joined(sorted((sympy.sqrt(k) for k in (M.T * M).charpoly().all_roots()), reverse=True))


LA = "linear_algebra"
GENS = [
    Gen("r3_jordan", "jordan_form", LA, "Жорданова форма", "matrix size n",
        grid=list(range(3, 9)), probe=(3, 5), make=make_jordan, verify=verify_jordan, rungs=(3,)),
    Gen("r3_smith", "smith_form", LA, "Нормальная форма Смита", "matrix size n",
        grid=list(range(3, 9)), probe=(4, 6), make=make_smith, verify=verify_smith, rungs=(3,)),
    Gen("r3_minpoly", "minimal_polynomial", LA, "Минимальный многочлен матрицы", "matrix size n",
        grid=list(range(3, 9)), probe=(3, 5), make=make_minpoly, verify=verify_minpoly, rungs=(3,)),
    Gen("r3_signature", "signature", LA, "Сигнатура квадратичной формы", "number of variables n",
        grid=list(range(3, 11)), probe=(5, 7), make=make_signature, verify=verify_signature, rungs=(3,)),
    Gen("r3_cholesky", "cholesky", LA, "Разложение Холецкого", "matrix size n",
        grid=list(range(3, 11)), probe=(5, 7), make=make_cholesky, verify=verify_cholesky, rungs=(3,)),
    Gen("r3_coords", "basis_coordinates", LA, "Координаты в новом базисе", "dimension n",
        grid=list(range(3, 11)), probe=(5, 7), make=make_coords, verify=verify_coords, rungs=(3,)),
    Gen("r3_traces", "power_traces", LA, "Следы степеней матрицы", "matrix size n (powers 1..4)",
        grid=list(range(3, 11)), probe=(4, 5), make=make_traces, verify=verify_traces, rungs=(3,)),
    Gen("r3_subspaces", "subspace_sum_intersection", LA, "Сумма и пересечение подпространств", "dimension n",
        grid=list(range(3, 11)), probe=(4, 6), make=make_subspaces, verify=verify_subspaces, rungs=(3,)),
    Gen("r3_change_basis", "change_of_basis", LA, "Матрица оператора в новом базисе", "matrix size n",
        grid=list(range(2, 8)), probe=(3, 4), make=make_change_basis, verify=verify_change_basis, rungs=(3,)),
    Gen("r3_qr", "qr", LA, "QR-разложение", "matrix size n",
        grid=list(range(2, 8)), probe=(3, 4), make=make_qr, verify=verify_qr, rungs=(3,)),
    Gen("r3_chain_trace", "product_trace", LA, "След произведения цепочки матриц", "number of 3×3 matrices K",
        grid=list(range(2, 65)), probe=(4, 6), make=make_chain, verify=verify_chain, rungs=(3,)),
    Gen("r3_gram_det", "gram_determinant", LA, "Определитель Грама", "number of vectors k (in R^(k+1))",
        grid=list(range(2, 9)), probe=(4, 5), make=make_gramdet, verify=verify_gramdet, rungs=(3,)),
    Gen("r3_svd", "singular_values", LA, "Сингулярные числа матрицы", "matrix size n",
        grid=list(range(3, 9)), probe=(3, 4), make=make_svd, verify=verify_svd, rungs=(3,)),
]
