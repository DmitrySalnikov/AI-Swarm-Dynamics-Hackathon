"""Rung-1 generators for fields not covered by the other r1_* modules.

Astronomy, probability (absorbing Markov chains), graph theory (maximum flow), accounting (inventory costing),
epidemiology (discrete SIR), navigation (great circles), pharmacology (repeated IV doses), demography (life tables),
meteorology (humidity and dew point).
"""
import math
from collections import deque
from fractions import Fraction

import mpmath
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow

from common import Gen, frac_str
from r2_physics_a import sig

LIST4 = {"type": "list", "rel_tol": 2e-3}
FORMAT = ("Give the {k} requested values, each with 4 significant figures, in the order listed, as plain decimal numbers "
          "separated by commas.")


def _list(vals):
    return ",".join(sig(v) for v in vals)


# ---------- astronomy ----------

def _astro(c):
    if c["kind"] == "parallax":                # distance in light years
        return 1000 / c["p"] * 3.2616
    if c["kind"] == "absmag":
        return c["m"] - 5 * math.log10(c["d"] / 10)
    if c["kind"] == "lum":                     # L1 / L2
        return 10 ** (0.4 * (c["M2"] - c["M1"]))
    return math.sqrt(c["a"] ** 3 / c["Ms"])    # period in years


def make_astro(k, rng):
    cases, lines = [], []
    for j in range(1, k + 1):
        kind = rng.choice(["parallax", "absmag", "lum", "kepler"])
        if kind == "parallax":
            c = dict(kind=kind, p=rng.randint(5, 900) / 10)
            s = f"a star has a parallax of {c['p']} milliarcseconds; its distance in light years"
        elif kind == "absmag":
            c = dict(kind=kind, m=rng.randint(-150, 1500) / 100, d=rng.randint(2, 5000))
            s = f"a star of apparent magnitude {c['m']} is at a distance of {c['d']} pc; its absolute magnitude"
        elif kind == "lum":
            M1 = rng.randint(-500, 1200) / 100
            c = dict(kind=kind, M1=M1, M2=round(M1 + rng.choice([-1, 1]) * rng.randint(10, 750) / 100, 2))   # ratio 1e-3..1e3
            s = (f"star X has absolute magnitude {c['M1']} and star Y has absolute magnitude {c['M2']}; the ratio of the "
                 "luminosity of X to that of Y")
        else:
            c = dict(kind=kind, a=rng.randint(10, 4000) / 100, Ms=rng.randint(10, 300) / 100)
            s = (f"a planet orbits a star of {c['Ms']} solar masses on an orbit with semi-major axis {c['a']} AU; its "
                 "orbital period in years")
        cases.append(c)
        lines.append(f"{j}. {s}.")
    prompt = (f"Astronomy. Use: 1 pc = 3.2616 light years; distance in pc = 1 / (parallax in arcseconds); absolute magnitude "
              f"M = m − 5·log10(d / 10 pc); a difference of 5 magnitudes is a factor of exactly 100 in luminosity; "
              f"Kepler's third law in the form T² = a³ / M (T in years, a in AU, M in solar masses, planet mass "
              f"negligible). Find the requested quantity in each of the following {k} independent situations:\n"
              + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_astro(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_astro(data):
    # SI units and flux ratios instead of the textbook shortcuts
    pc, ly = 3.0857e16, 9.4607e15                      # m; their ratio is 3.2616
    out = []
    for c in data["cases"]:
        if c["kind"] == "parallax":
            out.append(pc / math.tan(math.radians(c["p"] / 1000 / 3600)) * math.radians(1 / 3600) / ly)
        elif c["kind"] == "absmag":
            flux_ratio = (c["d"] / 10) ** 2            # flux at 10 pc / observed flux
            out.append(c["m"] - 2.5 * np.log10(flux_ratio))
        elif c["kind"] == "lum":
            out.append(np.exp((c["M2"] - c["M1"]) * np.log(100) / 5))
        else:
            gm, au, yr = 1.32712440018e20 * c["Ms"], 1.495978707e11, 2 * math.pi * math.sqrt(1.495978707e11 ** 3 / 1.32712440018e20)
            out.append(2 * math.pi * math.sqrt((c["a"] * au) ** 3 / gm) / yr)
    return _list(out)


# ---------- probability: absorbing Markov chain ----------

def _chain_solve(P, n):
    """Exact: x_i = P(absorbed in W | start i), t_i = expected steps, by Gaussian elimination over Fractions."""
    def solve(rhs):
        A = [[Fraction(int(i == j)) - P[i][j] for j in range(n)] + [rhs[i]] for i in range(n)]
        for col in range(n):
            piv = next(r for r in range(col, n) if A[r][col] != 0)
            A[col], A[piv] = A[piv], A[col]
            for r in range(n):
                if r != col and A[r][col] != 0:
                    f = A[r][col] / A[col][col]
                    A[r] = [a - f * b for a, b in zip(A[r], A[col])]
        return [A[i][n] / A[i][i] for i in range(n)]
    win = solve([P[i][n] for i in range(n)])
    steps = solve([Fraction(1)] * n)
    return win, steps


def make_markov(n, rng):
    while True:
        P = []
        for i in range(n):
            targets = rng.sample(range(n + 2), rng.randint(2, 4))
            if n not in targets and n + 1 not in targets and rng.random() < 0.5:
                targets[0] = rng.choice([n, n + 1])
            parts = [1] * len(targets)
            for _ in range(10 - len(targets)):
                parts[rng.randrange(len(targets))] += 1
            row = [Fraction(0)] * (n + 2)
            for t, w in zip(targets, parts):
                row[t] += Fraction(w, 10)
            P.append(row)
        M = np.array([[float(P[i][j]) for j in range(n)] for i in range(n)])
        if max(abs(np.linalg.eigvals(M))) < 0.97:     # every state reaches absorption, chain not too slow
            break
    win, steps = _chain_solve(P, n)
    name = lambda s: f"S{s + 1}" if s < n else ("W" if s == n else "L")
    lines = [f"from {name(i)}: " + ", ".join(f"to {name(j)} with probability {float(P[i][j]):g}"
                                              for j in range(n + 2) if P[i][j]) for i in range(n)]
    prompt = (f"A token moves on the states S1, …, S{n}, W and L. Each step it moves from its current state according to "
              f"the transition probabilities below (a move to the same state is allowed); W and L are absorbing (the "
              f"token stays there forever).\n" + "\n".join(lines) + "\n\nThe token starts in S1. Find (1) the probability "
              "that it is eventually absorbed in W and (2) the expected number of steps until it is absorbed (in W or "
              "L). Give the 2 values, each with 4 significant figures, as plain decimal numbers separated by commas.")
    return dict(prompt=prompt, answer=_list([win[0], steps[0]]), check=LIST4,
                data={"P": [[frac_str(x) for x in r] for r in P], "n": n})


def verify_markov(data):
    # fundamental matrix by iterating the chain until the transient mass vanishes (floating point)
    n = data["n"]
    P = np.eye(n + 2)                          # W and L absorb
    P[:n] = [[float(Fraction(x)) for x in r] for r in data["P"]]
    dist, win, steps = np.eye(n + 2)[0], 0.0, 0.0
    for _ in range(200000):
        steps += dist[:n].sum()
        dist = dist @ P
        if dist[:n].sum() < 1e-15:
            break
    win = dist[n]
    return _list([win, steps])


# ---------- graph theory: maximum flow ----------

def _edmonds_karp(n, edges, s, t):
    cap = {}
    adj = [set() for _ in range(n)]
    for u, v, c in edges:
        cap[(u, v)] = cap.get((u, v), 0) + c
        cap.setdefault((v, u), 0)
        adj[u].add(v); adj[v].add(u)
    flow = 0
    while True:
        prev = [-1] * n
        prev[s] = s
        q = deque([s])
        while q and prev[t] < 0:
            u = q.popleft()
            for v in sorted(adj[u]):
                if prev[v] < 0 and cap[(u, v)] > 0:
                    prev[v] = u
                    q.append(v)
        if prev[t] < 0:
            return flow
        b, v = math.inf, t
        while v != s:
            b = min(b, cap[(prev[v], v)]); v = prev[v]
        v = t
        while v != s:
            cap[(prev[v], v)] -= b; cap[(v, prev[v])] += b; v = prev[v]
        flow += b


def make_flow(n, rng):
    edges, seen = [], set()
    for v in range(1, n):                     # a spanning path of forward edges keeps the sink reachable
        u = rng.randrange(0, v)
        edges.append((u, v, rng.randint(3, 40))); seen.add((u, v))
    # arcs never enter s or leave t, and at most one arc per pair of nodes
    while len(edges) < min(int(2.5 * n), (n - 1) * (n - 2) // 2 + n - 1):
        u, v = rng.sample(range(n), 2)
        if (u, v) in seen or (v, u) in seen or v == 0 or u == n - 1:
            continue
        edges.append((u, v, rng.randint(3, 40))); seen.add((u, v))
    rng.shuffle(edges)
    name = lambda x: "s" if x == 0 else "t" if x == n - 1 else f"v{x}"
    lines = [f"{name(u)} → {name(v)}: {c}" for u, v, c in edges]
    prompt = (f"A directed network has {n} nodes: the source s, the sink t and the nodes v1, …, v{n - 2}. The arcs and "
              f"their capacities are:\n" + "\n".join(lines) + "\n\nFind the value of the maximum flow from s to t. Give the "
              "exact integer.")
    return dict(prompt=prompt, answer=str(_edmonds_karp(n, edges, 0, n - 1)), check={"type": "num", "rel_tol": 0},
                data={"n": n, "edges": edges})


def verify_flow(data):
    n, e = data["n"], data["edges"]
    M = np.zeros((n, n), dtype=np.int32)
    for u, v, c in e:
        M[u, v] += c
    return str(maximum_flow(csr_matrix(M), 0, n - 1).flow_value)


# ---------- accounting: inventory costing ----------

def _inventory(ops, opening):
    """Perpetual FIFO, perpetual LIFO and moving weighted average: cost of goods sold (exact Fractions)."""
    res = []
    for method in ("fifo", "lifo"):
        lots, cogs = [list(opening)], Fraction(0)
        for op in ops:
            if op[0] == "buy":
                lots.append([op[1], Fraction(op[2]).limit_denominator(1000)])
            else:
                need = op[1]
                while need:
                    lot = lots[0] if method == "fifo" else lots[-1]
                    take = min(need, lot[0])
                    cogs += take * lot[1]; lot[0] -= take; need -= take
                    if lot[0] == 0:
                        lots.pop(0 if method == "fifo" else -1)
        res.append(cogs)
    qty, value, cogs = opening[0], opening[0] * opening[1], Fraction(0)
    for op in ops:
        if op[0] == "buy":
            qty += op[1]; value += op[1] * Fraction(op[2]).limit_denominator(1000)
        else:
            avg = value / qty
            cogs += op[1] * avg; value -= op[1] * avg; qty -= op[1]
    res.append(cogs)
    return res


def make_inventory(k, rng):
    opening = [rng.randint(20, 200), Fraction(rng.randint(500, 5000), 100)]
    ops, stock = [], opening[0]
    for _ in range(k):
        if stock < 30 or rng.random() < 0.5:
            q = rng.randint(10, 150)
            ops.append(("buy", q, round(float(opening[1]) * rng.uniform(0.7, 1.4), 2)))
            stock += q
        else:
            q = rng.randint(5, stock - 1)
            ops.append(("sell", q)); stock -= q
    if not any(op[0] == "sell" for op in ops):   # at least one sale, so the cost of goods sold is not zero
        ops[-1] = ("sell", rng.randint(5, opening[0] - 1))
    lines = [f"{j}. purchase of {op[1]} units at ${op[2]:.2f} per unit" if op[0] == "buy" else f"{j}. sale of {op[1]} units"
             for j, op in enumerate(ops, 1)]
    vals = _inventory(ops, opening)
    prompt = (f"A shop starts the month with {opening[0]} units of a product carried at ${float(opening[1]):.2f} per unit. "
              f"During the month it has the following {k} transactions, in this order:\n" + "\n".join(lines) + "\n\nUsing a "
              "perpetual inventory system, find the cost of goods sold for the month under (1) FIFO, (2) LIFO and (3) the "
              "moving weighted-average method (recompute the average cost after each purchase; do not round intermediate "
              "results). Give the 3 amounts in dollars, rounded to the cent, as plain decimal numbers separated by commas.")
    return dict(prompt=prompt, answer=",".join(f"{float(v):.2f}" for v in vals), check={"type": "list", "rel_tol": 2e-5},
                data={"opening": [opening[0], frac_str(opening[1])], "ops": ops})


def verify_inventory(data):
    # unit-by-unit simulation with floats: every unit carries its own cost
    q0, c0 = data["opening"][0], float(Fraction(data["opening"][1]))
    out = []
    for method in ("fifo", "lifo"):
        units, cogs = deque([c0] * q0), 0.0
        for op in data["ops"]:
            if op[0] == "buy":
                units.extend([op[2]] * op[1])
            else:
                for _ in range(op[1]):
                    cogs += units.popleft() if method == "fifo" else units.pop()
        out.append(cogs)
    units = [c0] * q0                          # average: re-price every unit in stock to the mean after a purchase
    cogs = 0.0
    for op in data["ops"]:
        if op[0] == "buy":
            units += [op[2]] * op[1]
            units = [sum(units) / len(units)] * len(units)
        else:
            cogs += sum(units[:op[1]]); units = units[op[1]:]
    out.append(cogs)
    return ",".join(f"{v:.2f}" for v in out)


# ---------- epidemiology: discrete SIR ----------

def _sir(c, prec=None):
    S, I, R = c["N"] - c["I0"], c["I0"], 0
    if prec:
        S, I, R = mpmath.mpf(S), mpmath.mpf(I), mpmath.mpf(0)
    for _ in range(c["T"]):
        new_inf = c["beta"] * S * I / c["N"]
        new_rec = c["gamma"] * I
        S, I, R = S - new_inf, I + new_inf - new_rec, R + new_rec
    return S, I, R


def make_sir(T, rng):
    c = dict(N=rng.randint(2000, 200000), I0=rng.randint(1, 50), beta=rng.randint(20, 90) / 100, gamma=rng.randint(5, 40) / 100,
             T=T)
    c["beta"] = max(c["beta"], round(c["gamma"] * 1.3 + 0.01, 2))
    S, I, R = _sir(c)
    prompt = (f"A discrete-time SIR epidemic model: a closed population of N = {c['N']} people; on day 0 there are "
              f"{c['I0']} infected people and everyone else is susceptible. Each day the numbers are updated simultaneously "
              f"from the previous day's values:\n  new infections = β·S·I/N with β = {c['beta']},\n  new recoveries = γ·I "
              f"with γ = {c['gamma']},\n  S ← S − new infections, I ← I + new infections − new recoveries, R ← R + new "
              f"recoveries.\nNumbers are not rounded to whole people. Find S, I and R after {T} days. Give the 3 values, "
              "each with 4 significant figures, in this order, as plain decimal numbers separated by commas.")
    return dict(prompt=prompt, answer=_list([S, I, R]), check=LIST4, data=c)


def verify_sir(data):
    # 50-digit arithmetic; R checked through conservation S + I + R = N
    with mpmath.workdps(50):
        S, I, _ = _sir(data, prec=True)
        return _list([S, I, data["N"] - S - I])


# ---------- navigation: great circles ----------

R_EARTH = 6371.0


def _gc(c):
    p1, l1, p2, l2 = map(math.radians, (c["lat1"], c["lon1"], c["lat2"], c["lon2"]))
    if c["kind"] == "dist":
        h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
        return 2 * R_EARTH * math.asin(math.sqrt(h))
    y = math.sin(l2 - l1) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(l2 - l1)
    return math.degrees(math.atan2(y, x)) % 360


def make_gc(k, rng):
    cases, lines = [], []
    fmt = lambda lat, lon: f"{abs(lat)}° {'N' if lat >= 0 else 'S'}, {abs(lon)}° {'E' if lon >= 0 else 'W'}"
    for j in range(1, k + 1):
        while True:
            c = dict(kind=rng.choice(["dist", "bearing"]), lat1=rng.randint(-700, 700) / 10, lon1=rng.randint(-1800, 1800) / 10,
                     lat2=rng.randint(-700, 700) / 10, lon2=rng.randint(-1800, 1800) / 10)
            if 300 < _gc(dict(c, kind="dist")) < 19000 and (c["kind"] == "dist" or 1 < _gc(c) < 359):
                break
        ask = "great-circle distance in km" if c["kind"] == "dist" else \
            "initial true bearing (course) of the great-circle route from the first point, in degrees from 0 to 360"
        cases.append(c)
        lines.append(f"{j}. from {fmt(c['lat1'], c['lon1'])} to {fmt(c['lat2'], c['lon2'])}; {ask}.")
    prompt = (f"Navigation on a spherical Earth of radius 6371 km. For each of the following {k} pairs of points find the "
              "requested quantity of the shortest (great-circle) route:\n" + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_gc(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_gc(data):
    # 3-D unit vectors: angle via atan2(|a×b|, a·b); bearing from the local north/east components of the tangent
    out = []
    for c in data["cases"]:
        p1, l1, p2, l2 = map(math.radians, (c["lat1"], c["lon1"], c["lat2"], c["lon2"]))
        a = np.array([math.cos(p1) * math.cos(l1), math.cos(p1) * math.sin(l1), math.sin(p1)])
        b = np.array([math.cos(p2) * math.cos(l2), math.cos(p2) * math.sin(l2), math.sin(p2)])
        if c["kind"] == "dist":
            out.append(R_EARTH * math.atan2(np.linalg.norm(np.cross(a, b)), a @ b))
        else:
            tangent = b - (a @ b) * a
            north = np.array([-math.sin(p1) * math.cos(l1), -math.sin(p1) * math.sin(l1), math.cos(p1)])
            east = np.array([-math.sin(l1), math.cos(l1), 0])
            out.append(math.degrees(math.atan2(tangent @ east, tangent @ north)) % 360)
    return _list(out)


# ---------- pharmacology: repeated IV bolus doses ----------

def _pk(c):
    k = math.log(2) / c["half"]
    n = int(c["t"] // c["tau"]) + 1                 # doses given at 0, tau, ..., up to t
    n = min(n, c["ndoses"])
    return sum(c["D"] / c["V"] * math.exp(-k * (c["t"] - i * c["tau"])) for i in range(n))


def make_pk(k, rng):
    cases, lines = [], []
    for j in range(1, k + 1):
        c = dict(D=rng.randint(5, 100) * 10, V=rng.randint(10, 80), half=rng.randint(10, 240) / 10, tau=rng.choice([4, 6, 8, 12, 24]),
                 ndoses=rng.randint(2, 8))
        last = (c["ndoses"] - 1) * c["tau"]
        while True:                            # not at a dosing moment, and still a measurable concentration
            c["t"] = round(rng.uniform(0.5, last + 2 * c["half"]), 1)
            if abs(c["t"] / c["tau"] - round(c["t"] / c["tau"])) >= 0.02 and _pk(c) >= 0.05:
                break
        cases.append(c)
        lines.append(f"{j}. a dose of {c['D']} mg every {c['tau']} h, {c['ndoses']} doses in total starting at t = 0; volume "
                     f"of distribution {c['V']} L, elimination half-life {c['half']} h; plasma concentration in mg/L at "
                     f"t = {c['t']} h.")
    prompt = (f"Pharmacokinetics, one-compartment model with instantaneous intravenous bolus doses and first-order "
              f"elimination (each dose instantly raises the concentration by dose / volume; concentrations from separate "
              f"doses add up). In each of the following {k} independent regimens find the requested concentration:\n"
              + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_pk(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_pk(data):
    # integrate dC/dt = -k C between doses, adding D / V at each dosing time
    out = []
    for c in data["cases"]:
        k, C, t0 = math.log(2) / c["half"], 0.0, 0.0
        for i in range(c["ndoses"]):
            td = i * c["tau"]
            if td > c["t"]:
                break
            if td > t0:
                C = solve_ivp(lambda t, y: -k * y, (t0, td), [C], rtol=1e-12, atol=1e-14).y[0, -1]
            C += c["D"] / c["V"]; t0 = td
        C = solve_ivp(lambda t, y: -k * y, (t0, c["t"]), [C], rtol=1e-12, atol=1e-14).y[0, -1]
        out.append(C)
    return _list(out)


# ---------- demography: life table ----------

def make_life(n, rng):
    q = [round(rng.uniform(0.005, 0.05) * (1 + 6 * x / n) ** 2 / 4, 4) for x in range(n)]
    q = [min(max(v, 0.001), 0.6) for v in q] + [1.0]
    l = [100000.0]
    for v in q[:-1]:
        l.append(l[-1] * (1 - v))
    Lx = [l[x] - 0.5 * l[x] * q[x] for x in range(n + 1)]
    e0 = sum(Lx) / l[0]
    age = n // 2
    e_mid = sum(Lx[age:]) / l[age]
    lines = [f"age {x}: q = {q[x]}" for x in range(n)]
    prompt = (f"A cohort life table starts with 100000 newborns. The probability of dying within the year of age x (between "
              f"exact ages x and x + 1) is q_x:\n" + "\n".join(lines) + f"\nage {n}: q = 1 (everyone alive at age {n} dies "
              f"within that year).\nAssume deaths are spread uniformly within each year of age, so a person dying in a year "
              f"lives half of it on average. Find (1) the life expectancy at birth e_0 and (2) the life expectancy at exact "
              f"age {age}, in years. Give the 2 values, each with 4 significant figures, as plain decimal numbers separated "
              "by commas.")
    return dict(prompt=prompt, answer=_list([e0, e_mid]), check=LIST4, data={"q": q, "age": age})


def verify_life(data):
    # e_x = 0.5 + sum_{y > x} l_y / l_x for uniform deaths within the year (person-years via survivors)
    q, age = data["q"], data["age"]
    l = [1.0]
    for v in q[:-1]:
        l.append(l[-1] * (1 - v))
    e = lambda x: 0.5 + sum(l[x + 1:]) / l[x]
    return _list([e(0), e(age)])


# ---------- meteorology: humidity ----------

def _es(T):
    return 6.112 * math.exp(17.62 * T / (243.12 + T))     # hPa, Magnus formula


def _met(c):
    if c["kind"] == "dew":
        g = math.log(c["RH"] / 100) + 17.62 * c["T"] / (243.12 + c["T"])
        return 243.12 * g / (17.62 - g)
    if c["kind"] == "rh":
        return 100 * _es(c["Td"]) / _es(c["T"])
    if c["kind"] == "w":                       # mixing ratio, g/kg
        e = c["RH"] / 100 * _es(c["T"])
        return 622 * e / (c["p"] - e)
    g = math.log(c["RH"] / 100) + 17.62 * c["T"] / (243.12 + c["T"])   # cloud base height, m
    return 125 * (c["T"] - 243.12 * g / (17.62 - g))


def make_met(k, rng):
    ask = {"dew": "dew point in °C", "rh": "relative humidity in %", "w": "water-vapour mixing ratio in g/kg",
           "lcl": "estimated cloud-base height in m"}
    cases, lines = [], []
    for j in range(1, k + 1):
        c = dict(kind=rng.choice(list(ask)), T=rng.randint(-150, 400) / 10, RH=rng.randint(15, 98), p=rng.randint(700, 1040))
        if c["kind"] == "rh":
            c["Td"] = round(c["T"] - rng.randint(5, 250) / 10, 1)
            s = f"air temperature {c['T']} °C, dew point {c['Td']} °C"
        elif c["kind"] == "w":
            s = f"air temperature {c['T']} °C, relative humidity {c['RH']} %, pressure {c['p']} hPa"
        else:
            s = f"air temperature {c['T']} °C, relative humidity {c['RH']} %"
        cases.append(c)
        lines.append(f"{j}. {s}; {ask[c['kind']]}.")
    prompt = (f"Meteorology. Use the Magnus formula for the saturation vapour pressure over water, e_s(T) = 6.112·exp(17.62·T"
              f" / (243.12 + T)) hPa (T in °C); relative humidity RH = 100·e / e_s(T); the dew point is the temperature at "
              f"which e_s equals the actual vapour pressure e; the mixing ratio is w = 622·e / (p − e) g/kg; the cloud-base "
              f"height is estimated as 125 m per °C of the difference between temperature and dew point. Find the requested "
              f"quantity in each of the following {k} independent situations:\n" + "\n".join(lines) + "\n\n"
              + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_met(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_met(data):
    # dew point by root finding on e_s(Td) = e instead of the inverted formula
    out = []
    for c in data["cases"]:
        if c["kind"] == "rh":
            out.append(100 * _es(c["Td"]) / _es(c["T"]))
            continue
        e = c["RH"] / 100 * _es(c["T"])
        if c["kind"] == "w":
            out.append(622 * e / (c["p"] - e))
            continue
        td = brentq(lambda x: _es(x) - e, -90, 60, xtol=1e-12)
        out.append(td if c["kind"] == "dew" else 125 * (c["T"] - td))
    return _list(out)


GENS = [
    Gen("r1_astro", "stellar_astronomy", "astronomy", "Stellar magnitudes, parallax, Kepler's third law", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 30), make=make_astro, verify=verify_astro),
    Gen("r1_markov", "absorbing_chain", "probability", "Absorbing Markov chain", "number of transient states n",
        grid=list(range(2, 13)), probe=(4, 6), make=make_markov, verify=verify_markov),
    Gen("r1_maxflow", "max_flow", "graph_theory", "Maximum flow in a network", "number of nodes n",
        grid=list(range(4, 41)), probe=(8, 14), make=make_flow, verify=verify_flow),
    Gen("r1_inventory", "inventory_costing", "accounting", "Cost of goods sold: FIFO, LIFO, moving average",
        "number of transactions K", grid=list(range(2, 81)), probe=(8, 20), make=make_inventory, verify=verify_inventory),
    Gen("r1_sir", "sir_model", "epidemiology", "Discrete SIR model", "number of days T",
        grid=list(range(2, 121)), probe=(10, 25), make=make_sir, verify=verify_sir),
    Gen("r1_greatcircle", "great_circle", "navigation", "Great-circle distance and initial course", "number of cases K",
        grid=list(range(2, 81)), probe=(8, 20), make=make_gc, verify=verify_gc),
    Gen("r1_pk_doses", "repeated_doses", "pharmacology", "Drug concentration after repeated doses",
        "number of cases K", grid=list(range(2, 81)), probe=(8, 20), make=make_pk, verify=verify_pk),
    Gen("r1_life_table", "life_table", "demography", "Life table: life expectancy",
        "number of ages n", grid=list(range(5, 121)), probe=(20, 50), make=make_life, verify=verify_life),
    Gen("r1_humidity", "humidity", "meteorology", "Air humidity and dew point", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 30), make=make_met, verify=verify_met),
]
