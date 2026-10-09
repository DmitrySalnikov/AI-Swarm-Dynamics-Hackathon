"""Rung-2 physics generators, part A: circuits, optics, thermodynamics, statistical, atomic and quantum physics."""
import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import sympy
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from common import Gen, frac_str

EXACT = {"type": "num", "rel_tol": 0}
SIG4 = {"type": "num", "rel_tol": 2e-3}


def sig(x, n=6):
    """Number -> decimal string with n significant figures, never in exponent notation."""
    s = f"{float(x):.{n}g}"
    return format(Decimal(s), "f") if "e" in s else s


def far(x):
    """False near -10, where selftest's corruption v -> 1.1·v + 1 stays within the 2e-3 tolerance."""
    return abs(float(x) + 10) > 0.3


def _solve(a, b):
    """Gaussian elimination over Fractions."""
    n = len(b)
    m = [list(map(Fraction, r)) + [Fraction(v)] for r, v in zip(a, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


# ---------- 1. equivalent resistance of a series-parallel network ----------

UNIT_SPLITS = {2: [(2, 2)], 3: [(3, 3, 3), (2, 4, 4), (2, 3, 6)],
               4: [(4, 4, 4, 4), (2, 4, 8, 8), (2, 6, 6, 6), (3, 3, 6, 6), (2, 4, 6, 12), (2, 5, 5, 10),
                   (3, 4, 4, 6), (3, 3, 4, 12), (2, 3, 12, 12)]}


def _sizes(n, rng, kmax=4):
    k = rng.randint(2, min(kmax, n))
    cuts = sorted(rng.sample(range(1, n), k - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [n])]


def _sp_int(r, n, kind, rng):
    """Sub-network of n resistors (each 1..60 ohm) with integer resistance r whose sub-networks are integer too;
    raises ValueError when the random choices lead to an impossible resistor."""
    if n == 1:
        if not 1 <= r <= 60:
            raise ValueError
        return r
    if kind == "S":
        if r < 2:
            raise ValueError
        sizes = _sizes(n, rng, min(4, r))
        cuts = sorted(rng.sample(range(1, r), len(sizes) - 1))
        vals = sorted(b - a for a, b in zip([0] + cuts, cuts + [r]))
        order = sorted(range(len(sizes)), key=lambda i: (sizes[i] == 1, rng.random()))   # largest parts to leaves
    else:
        sizes = _sizes(n, rng)
        vals = [r * c for c in sorted(rng.choice(UNIT_SPLITS[len(sizes)]))]
        order = sorted(range(len(sizes)), key=lambda i: (sizes[i] != 1, rng.random()))   # smallest parts to leaves
    v = [0] * len(sizes)
    for i, x in zip(order, vals):
        v[i] = x
    return [kind, [_sp_int(x, m, "P" if kind == "S" else "S", rng) for x, m in zip(v, sizes)]]


def _sp_tree(n, rng, kind):
    """Free combination at the top (generally a fraction) of sub-networks with integer resistances."""
    other = "P" if kind == "S" else "S"
    sizes, kids = _sizes(n, rng), []
    for m in sizes:
        while True:
            try:
                kids.append(rng.randint(1, 30) if m == 1 else _sp_int(rng.randint(2, 30), m, other, rng))
                break
            except ValueError:
                pass
    if kind == "S" and max(sizes) > 1:
        i = sizes.index(max(sizes))
        kids[i] = _sp_tree(sizes[i], rng, "P")
    return [kind, kids]


def _sp_text(t):
    return str(t) if isinstance(t, int) else t[0] + "(" + ", ".join(_sp_text(c) for c in t[1]) + ")"


def _sp_value(t):
    if isinstance(t, int):
        return Fraction(t)
    vals = [_sp_value(c) for c in t[1]]
    return sum(vals) if t[0] == "S" else 1 / sum(1 / v for v in vals)


def make_sp(n, rng):
    while True:
        tree = _sp_tree(n, rng, rng.choice("SP"))
        r = _sp_value(tree)
        if r.denominator > 1 and max(r.numerator, r.denominator) < 10**5:
            break
    prompt = (f"A two-terminal resistor network of {n} resistors is written in the following notation: an integer is a "
              "single resistor with that resistance in ohms; S(x, y, …) means the sub-networks x, y, … connected in "
              "series; P(x, y, …) means the sub-networks x, y, … connected in parallel (all between the same two nodes)."
              f"\n\nNetwork = {_sp_text(tree)}\n\n"
              "Find the equivalent resistance of the whole network between its two terminals, in ohms. "
              "Give the exact value as a reduced fraction p/q (or an integer if it is a whole number).")
    return dict(prompt=prompt, answer=frac_str(r), check=EXACT, data={"tree": tree})


def verify_sp(data):
    # nodal analysis: build the graph, solve the Laplacian system with 1 A injected at node 0, node 1 grounded
    edges, count = [], [2]

    def add(t, a, b):
        if isinstance(t, int):
            edges.append((a, b, t))
        elif t[0] == "P":
            for c in t[1]:
                add(c, a, b)
        else:
            inner = list(range(count[0], count[0] + len(t[1]) - 1))
            count[0] += len(inner)
            nodes = [a] + inner + [b]
            for c, x, y in zip(t[1], nodes, nodes[1:]):
                add(c, x, y)

    add(data["tree"], 0, 1)
    n = count[0]
    lap = sympy.zeros(n, n)
    for a, b, r in edges:
        g = sympy.Rational(1, r)
        lap[a, a] += g
        lap[b, b] += g
        lap[a, b] -= g
        lap[b, a] -= g
    keep = [i for i in range(n) if i != 1]
    rhs = sympy.zeros(n - 1, 1)
    rhs[0] = 1
    v = lap.extract(keep, keep).LUsolve(rhs)[0]
    return frac_str(Fraction(int(v.p), int(v.q)))


# ---------- 2. ladder network: current in one branch ----------

def _ladder_currents(r, R, E):
    n = len(r)
    a = [[0] * (n + 1) for _ in range(n + 1)]
    b = [Fraction(E[i], R[i]) for i in range(n + 1)]
    for i in range(n + 1):
        a[i][i] += Fraction(1, R[i])
        if i > 0:
            a[i][i] += Fraction(1, r[i - 1])
            a[i][i - 1] -= Fraction(1, r[i - 1])
        if i < n:
            a[i][i] += Fraction(1, r[i])
            a[i][i + 1] -= Fraction(1, r[i])
    v = _solve(a, b)
    return [(v[i] - E[i]) / R[i] for i in range(n + 1)]


def make_ladder(n, rng):
    while True:
        r = [rng.randint(1, 20) for _ in range(n)]
        R = [rng.randint(1, 20) for _ in range(n + 1)]
        E = [rng.choice([-1, 1]) * rng.randint(1, 24) if rng.random() < 0.4 else 0 for _ in range(n + 1)]
        if not any(E):
            continue
        cur = _ladder_currents(r, R, E)
        big = max(abs(x) for x in cur)
        ok = [i for i in range(n + 1) if abs(cur[i]) >= big / 8 and abs(cur[i]) >= Fraction(1, 1000) and far(cur[i])]
        if ok:
            k = rng.choice(ok)
            break
    lines = []
    for i in range(n + 1):
        if i > 0:
            lines.append(f"- between A{i - 1} and A{i}: resistor {r[i - 1]} Ω")
        bat = (f" in series with an ideal battery of EMF {abs(E[i])} V whose positive terminal faces "
               f"{'A' + str(i) if E[i] > 0 else 'G'}") if E[i] else " (no battery)"
        lines.append(f"- branch {i}, from A{i} down to G: resistor {R[i]} Ω{bat}")
    prompt = (f"A planar ladder network has {n} loops. Its top row consists of nodes A0, A1, …, A{n}; at the bottom "
              "there is a single common wire G of zero resistance. Neighbouring top nodes are joined by resistors, "
              "and every top node Ai is joined to G by a vertical branch i. The elements are:\n"
              + "\n".join(lines) +
              f"\n\nAll batteries are ideal (no internal resistance). Find the current in branch {k}, counted "
              f"positive when it flows from A{k} through the branch down to G (negative if it flows from G up to A{k}). "
              "Give the value in amperes with 4 significant figures, including its sign.")
    return dict(prompt=prompt, answer=sig(cur[k]), check=SIG4, data={"r": r, "R": R, "E": E, "k": k})


def verify_ladder(data):
    # mesh analysis: clockwise loop currents J1..Jn, loop i bounded by branches i-1 and i
    r, R, E, k = data["r"], data["R"], data["E"], data["k"]
    n = len(r)
    a, b = np.zeros((n, n)), np.zeros(n)
    for i in range(1, n + 1):
        row = i - 1
        a[row, row] += r[i - 1] + R[i] + R[i - 1]
        if i < n:
            a[row, row + 1] -= R[i]
        if i > 1:
            a[row, row - 1] -= R[i - 1]
        b[row] = E[i - 1] - E[i]
    j = np.concatenate([[0.0], np.linalg.solve(a, b), [0.0]])
    return sig(j[k] - j[k + 1])


# ---------- 3. RC circuit with a sequence of switching intervals ----------

def make_rc(n, rng):
    C = rng.randint(1, 10)
    while True:
        steps = []
        for _ in range(n):
            R = rng.randint(1, 20)
            tau = R * C
            t = max(1, round(tau * rng.uniform(0.2, 2.0)))
            steps.append([R, rng.randint(-20, 20), t])
        u, peak = 0.0, 0.0
        for R, U, t in steps:
            u = U + (u - U) * math.exp(-t / (R * C))
            peak = max(peak, abs(U))
        if abs(u) >= max(0.5, 0.1 * peak) and far(u):
            break
    lines = [f"{j}. duration {t} ms: R = {R} kΩ, U = {U} V" for j, (R, U, t) in enumerate(steps, 1)]
    prompt = (f"A capacitor of capacitance C = {C} μF has its bottom plate permanently connected to ground. Initially it "
              "is uncharged (capacitor voltage u = 0 V). Then it goes through the following consecutive time intervals; "
              "during each interval its top plate is connected through a resistor R to an ideal source that holds a "
              "terminal at the constant potential U relative to ground (U = 0 means the capacitor simply discharges "
              "through R; U may be negative). Switching between intervals is instantaneous and the capacitor voltage "
              "is continuous. Note that R·C with R in kΩ and C in μF is in ms.\n\n" + "\n".join(lines) +
              "\n\nFind the capacitor voltage u (top plate relative to ground) at the end of the last interval, in "
              "volts, with 4 significant figures and its sign.")
    return dict(prompt=prompt, answer=sig(u), check=SIG4, data={"C": C, "steps": steps})


def verify_rc(data):
    # numerical integration of C du/dt = (U - u) / R
    u, C = 0.0, data["C"]
    for R, U, t in data["steps"]:
        sol = solve_ivp(lambda _, y: [(U - y[0]) / (R * C)], (0, t), [u], method="DOP853", rtol=1e-11, atol=1e-12)
        u = sol.y[0, -1]
    return sig(u)


# ---------- 4. coaxial system of thin lenses ----------

def _lens_options(s):
    """(f, s') with integer image distance s' for object distance s."""
    out, s2 = [], s * s
    for d in range(1, abs(s) + 1):
        if s2 % d:
            continue
        for k in {d, -d, s2 // d, -(s2 // d)}:
            f, img = s - k, s2 // k - s
            if f != 0 and 5 <= abs(f) <= 60 and 3 <= abs(img) <= 120:
                out.append((f, img))
    return sorted(set(out))


def make_lenses(n, rng):
    while True:
        s0 = rng.randint(10, 60)
        s, fs, ds, ok = s0, [], [], True
        for i in range(n):
            opts = _lens_options(s)
            if not opts:
                ok = False
                break
            f, img = rng.choice(opts)
            fs.append(f)
            if i == n - 1:
                break
            ds_ok = [d for d in range(5, 51) if abs(d - img) >= 3 and _lens_options(d - img)]
            if not ds_ok:
                ok = False
                break
            d = rng.choice(ds_ok)
            ds.append(d)
            s = d - img
        if ok:
            break
    lines = [f"lens {i + 1}: f = {f} cm" + (f"; distance from lens {i + 1} to lens {i + 2}: {ds[i]} cm" if i < n - 1
                                             else "") for i, f in enumerate(fs)]
    prompt = (f"A coaxial optical system consists of {n} thin lenses in air, numbered 1 to {n} in the direction of "
              f"light propagation (left to right). A point object is on the axis {s0} cm to the left of lens 1.\n\n"
              + "\n".join(lines) +
              "\n\nSign convention: for every lens use 1/s + 1/s' = 1/f, with f > 0 for a converging and f < 0 for a "
              "diverging lens; s is the object distance, positive when the object is to the left of the lens (real "
              "object) and negative when it is to the right of the lens (virtual object, i.e. the light arrives "
              "converging toward a point behind the lens); s' is the image distance, positive when the image is to "
              "the right of the lens (real) and negative when it is to the left (virtual). The image formed by each "
              "lens is the object for the next lens (if that image lies beyond the next lens, it is a virtual object "
              "for it); ignore apertures and aberrations.\n\n"
              f"Find the position of the final image formed by lens {n}: its signed distance from lens {n} in cm "
              f"(positive if it is to the right of lens {n}, negative if to the left). Give the exact value as an "
              "integer or a reduced fraction.")
    return dict(prompt=prompt, answer=str(img), check=EXACT, data={"s0": s0, "f": fs, "d": ds})


def verify_lenses(data):
    # ray-transfer (ABCD) matrices from the object plane; image where the B element of T(x)·M vanishes
    m = sympy.Matrix([[1, data["s0"]], [0, 1]])
    for i, f in enumerate(data["f"]):
        m = sympy.Matrix([[1, 0], [sympy.Rational(-1, f), 1]]) * m
        if i < len(data["d"]):
            m = sympy.Matrix([[1, data["d"][i]], [0, 1]]) * m
    x = -m[0, 1] / m[1, 1]
    return frac_str(Fraction(int(x.p), int(x.q)))


# ---------- 5. lateral displacement of a ray in a stack of plane layers ----------

def make_layers(n, rng):
    th = rng.randint(15, 75)
    layers = [[round(rng.uniform(1.3, 2.4), 2), rng.randint(2, 30)] for _ in range(n)]
    t0 = math.radians(th)
    shift = 0.0
    for nn, d in layers:
        ti = math.asin(math.sin(t0) / nn)
        shift += d * (math.tan(t0) - math.tan(ti))
    ans = shift * math.cos(t0)
    lines = [f"layer {i + 1}: refractive index {nn:.2f}, thickness {d} mm" for i, (nn, d) in enumerate(layers)]
    prompt = (f"A narrow light ray in air (refractive index exactly 1) falls on the flat top surface of a stack of {n} "
              f"transparent plane-parallel layers at an angle of incidence of {th}° (measured from the normal). The "
              "layers are listed from top to bottom; they are in direct contact, all interfaces are parallel, and below "
              "the last layer there is air again.\n\n" + "\n".join(lines) +
              "\n\nThe ray passes through all layers (ignore reflections) and emerges into the air below. Find the "
              "lateral displacement of the ray: the perpendicular distance between the straight line of the incident "
              "ray (extended through the stack) and the emergent ray. Give it in mm with 4 significant figures.")
    return dict(prompt=prompt, answer=sig(ans), check=SIG4, data={"theta": th, "layers": layers})


def verify_layers(data):
    # vector form of Snell's law, ray traced interface by interface; distance between the two lines
    t0 = math.radians(data["theta"])
    d0 = np.array([math.sin(t0), -math.cos(t0)])
    normal = np.array([0.0, 1.0])
    pos, d, n1 = np.zeros(2), d0.copy(), 1.0
    for n2, thick in data["layers"] + [[1.0, None]]:
        eta = n1 / n2
        ci = -normal @ d
        d = eta * d + (eta * ci - math.sqrt(1 - eta**2 * (1 - ci**2))) * normal
        if thick is not None:
            pos = pos + d * (thick / -d[1])
        n1 = n2
    return sig(abs(pos[0] * d0[1] - pos[1] * d0[0]))


# ---------- 6. net work of an ideal-gas cycle ----------

def _cycle_states(p1, v1, gam, procs):
    """Walk the processes; returns list of (p, v) states and works (kPa·L = J)."""
    p, v, states, works = p1, v1, [(p1, v1)], []
    for kind, x in procs:
        if kind == "isobaric":
            w, v = p * (x - v), x
        elif kind == "isochoric":
            w, p = 0.0, x
        elif kind == "isothermal":
            w, p, v = p * v * math.log(x / v), p * v / x, x
        else:
            p2 = p * (v / x) ** gam
            w, p, v = (p * v - p2 * x) / (gam - 1), p2, x
        states.append((p, v))
        works.append(w)
    return states, works


def make_cycle(n, rng):
    gam = rng.choice([5 / 3, 7 / 5])
    while True:
        p1, v1 = rng.randint(100, 400), rng.randint(2, 10)
        p, v, procs, prev, ok = p1, v1, [], None, True
        for _ in range(n - 2):
            kind = rng.choice([k for k in ("isobaric", "isochoric", "isothermal", "adiabatic") if k != prev])
            if kind == "isochoric":
                x = round(p * rng.choice([rng.uniform(0.4, 0.85), rng.uniform(1.2, 2.5)]))
            else:
                x = round(v * rng.choice([rng.uniform(0.5, 0.85), rng.uniform(1.2, 2.0)]))
            procs.append([kind, x])
            states, _ = _cycle_states(p1, v1, gam, procs)
            p, v = states[-1]
            if not (30 <= p <= 3000 and 1 <= v <= 40):
                ok = False
                break
            prev = kind
        if not ok or abs(p - p1) < 0.1 * p1 or abs(v - v1) < 0.1 * v1:
            continue
        first = {"isobaric": "isochoric", "isochoric": "isobaric"}.get(prev) or rng.choice(["isobaric", "isochoric"])
        tail = [["isobaric", v1], ["isochoric", p1]] if first == "isobaric" else [["isochoric", p1], ["isobaric", v1]]
        procs += tail
        states, works = _cycle_states(p1, v1, gam, procs)
        if any(abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9 for a, b in zip(states, states[1:])):
            continue
        wnet = sum(works)
        if abs(wnet) >= 0.1 * sum(abs(w) for w in works) and abs(wnet) >= 10 and far(wnet):
            break
    names = {"isobaric": "isobaric", "isochoric": "isochoric", "isothermal": "isothermal",
             "adiabatic": "reversible adiabatic"}
    lines = []
    for i, (kind, x) in enumerate(procs):
        a, b = i + 1, (i + 2 if i < n - 1 else 1)
        target = f"p{b} = {x} kPa" if kind == "isochoric" else f"V{b} = {x} L"
        lines.append(f"{a}→{b}: {names[kind]}, until {target}" + (" (back to state 1)" if b == 1 else ""))
    gtxt = "5/3" if gam > 1.5 else "7/5"
    prompt = (f"An ideal gas with adiabatic index γ = Cp/Cv = {gtxt} performs a closed cycle of {n} quasi-static "
              f"processes. State 1: p1 = {p1} kPa, V1 = {v1} L. The processes are:\n" + "\n".join(lines) +
              "\n\nFind the net work done BY the gas over one full cycle, in joules (1 kPa·L = 1 J). It is positive if "
              "the gas does net positive work and negative otherwise. Give the value with 4 significant figures "
              "and its sign.")
    return dict(prompt=prompt, answer=sig(wnet), check=SIG4,
                data={"gamma": gtxt, "p1": p1, "v1": v1, "procs": procs})


def verify_cycle(data):
    # work as the numerical integral of p dV along each process
    gam = float(Fraction(data["gamma"]))
    p, v, total = data["p1"], data["v1"], 0.0
    for kind, x in data["procs"]:
        if kind == "isochoric":
            p = x
            continue
        if kind == "isobaric":
            law = lambda vv, c=p: c
        elif kind == "isothermal":
            law = lambda vv, c=p * v: c / vv
        else:
            law = lambda vv, c=p * v**gam: c / vv**gam
        total += quad(law, v, x, epsabs=1e-12, epsrel=1e-12)[0]
        p, v = law(x), x
    return sig(total)


# ---------- 7. calorimetry with phase changes ----------

CW, CICE, CSTEAM, LF, LV = 4186, 2100, 2010, 334000, 2260000
METALS = {"aluminium": 897, "copper": 385, "iron": 449, "lead": 129, "silver": 235, "brass": 380, "zinc": 388,
          "glass": 840, "granite": 790, "tin": 228}


def _calor_tf(bodies):
    # linear heat balance assuming final state is liquid water at 0 < Tf < 100
    a, b = Fraction(0), Fraction(0)    # sum of heat gained = a*Tf + b
    for kind, m, t, c in bodies:
        m = Fraction(m, 1000)
        if kind == "water":
            a, b = a + m * CW, b - m * CW * t
        elif kind == "ice":
            a, b = a + m * CW, b + m * CICE * (0 - t) + m * LF
        elif kind == "steam":
            a, b = a + m * CW, b + m * CSTEAM * (100 - t) - m * LV - m * CW * 100
        else:
            a, b = a + m * c, b - m * c * t
    return -b / a


def make_calor(n, rng):
    while True:
        bodies = []
        for _ in range(n):
            kind = rng.choices(["water", "metal", "ice", "steam"], [3, 4, 1.5, 1.2])[0]
            if kind == "water":
                bodies.append(["water", rng.randint(50, 500), rng.randint(1, 99), CW])
            elif kind == "ice":
                bodies.append(["ice", rng.randint(10, 150), rng.randint(-40, 0), CICE])
            elif kind == "steam":
                bodies.append(["steam", rng.randint(5, 40), rng.randint(100, 140), CSTEAM])
            else:
                name = rng.choice(sorted(METALS))
                bodies.append([name, rng.randint(50, 1000), rng.randint(-20, 300), METALS[name]])
        if any(b[0] in ("water", "ice", "steam") for b in bodies):
            tf = _calor_tf(bodies)
            if 5 <= tf <= 95:
                break
    desc = {"water": "liquid water", "ice": "ice", "steam": "water vapour (steam)"}
    lines = [f"{i + 1}. {desc.get(k, k)}: mass {m} g, initial temperature {t} °C" +
             (f", specific heat {c} J/(kg·K)" if k not in desc else "") for i, (k, m, t, c) in enumerate(bodies)]
    prompt = (f"The following {n} bodies are brought into thermal contact inside a thermally insulated container of "
              "negligible heat capacity, at a pressure of 1 atm (water melts at 0 °C and boils at 100 °C):\n"
              + "\n".join(lines) +
              f"\n\nConstants: specific heat of liquid water 4186 J/(kg·K), of ice 2100 J/(kg·K), of steam "
              "2010 J/(kg·K); latent heat of fusion of ice 3.34·10^5 J/kg; latent heat of vaporisation of water "
              "2.26·10^6 J/kg. The other bodies are solids that do not change phase; their specific heats are given. It is guaranteed that in the final equilibrium all the ice has "
              "melted and all the steam has condensed, and the final temperature is strictly between 0 °C and 100 °C."
              "\n\nFind the final equilibrium temperature in °C with 4 significant figures.")
    return dict(prompt=prompt, answer=sig(tf), check=SIG4, data={"bodies": bodies})


def verify_calor(data):
    # total enthalpy (liquid water at 0 °C as reference) conserved; root of H(T) in the liquid region
    def h_init(kind, m, t, c):
        m /= 1000
        if kind == "ice":
            return m * (CICE * t - LF)
        if kind == "water":
            return m * CW * t
        if kind == "steam":
            return m * (CW * 100 + LV + CSTEAM * (t - 100))
        return m * c * t

    def h_liquid(kind, m, t, c, temp):
        return m / 1000 * (CW if kind in ("ice", "water", "steam") else c) * temp

    h0 = sum(h_init(*b) for b in data["bodies"])
    g = lambda temp: sum(h_liquid(*b, temp) for b in data["bodies"]) - h0
    return sig(brentq(g, 1e-9, 100 - 1e-9, xtol=1e-12))


# ---------- 8. mean energy of an N-level system ----------

def make_levels(n, rng):
    top = max(3.0, min(8.0, 0.12 * n + 2))
    eps = sorted(rng.sample(range(5, int(top * 100) + 1), n - 1))
    levels = [[0, rng.randint(1, 5)]] + [[e, rng.randint(1, 9)] for e in eps]   # energies in units of kT/100
    z = sum(g * math.exp(-e / 100) for e, g in levels)
    u = sum(g * e / 100 * math.exp(-e / 100) for e, g in levels) / z
    lines = [f"level {i}: ε/kT = {e / 100:.2f}, degeneracy g = {g}" for i, (e, g) in enumerate(levels)]
    prompt = (f"A quantum system in thermal equilibrium at temperature T has {n} energy levels. Their energies ε "
              "(given in units of kT, measured from the ground level) and degeneracies g are:\n" + "\n".join(lines) +
              "\n\nUsing the canonical (Boltzmann) distribution, the probability of a level is proportional to "
              "g·exp(−ε/kT). Find the mean energy ⟨ε⟩ of the system in units of kT, with 4 significant figures.")
    return dict(prompt=prompt, answer=sig(u), check=SIG4, data={"levels": levels})


def verify_levels(data):
    # <E> = -d ln Z / d beta at beta = 1/kT, symbolically
    beta = sympy.Symbol("beta")
    z = sum(g * sympy.exp(-beta * sympy.Rational(e, 100)) for e, g in data["levels"])
    return sig(sympy.N(-sympy.diff(sympy.log(z), beta).subs(beta, 1), 20))


# ---------- 9. wavelengths of transitions in hydrogen-like ions ----------

RYD = 1.0973731568e7


def make_hlike(k, rng):
    cases = []
    for _ in range(k):
        z, nf = rng.randint(1, 8), rng.randint(1, 5)
        cases.append([z, rng.randint(nf + 1, nf + 6), nf])
    lam = [1e9 / (RYD * z * z * (1 / nf**2 - 1 / ni**2)) for z, ni, nf in cases]
    lines = [f"{j}. Z = {z}, transition n = {ni} → n = {nf}" for j, (z, ni, nf) in enumerate(cases, 1)]
    prompt = (f"For each of the following {k} transitions in a hydrogen-like ion (one electron, nuclear charge Z), find "
              "the wavelength of the emitted photon. Use the Rydberg formula 1/λ = R·Z²·(1/n_f² − 1/n_i²) with "
              "R = 1.0973731568·10^7 m⁻¹ (infinitely heavy nucleus; ignore reduced-mass, fine-structure and all other "
              "corrections).\n\n" + "\n".join(lines) +
              f"\n\nGive the {k} wavelengths in nanometres, each with 4 significant figures, in the order listed, "
              "as plain decimal numbers (no units, no exponent notation, no thousands separators) separated by commas.")
    return dict(prompt=prompt, answer=",".join(sig(x) for x in lam), check={"type": "list", "rel_tol": 2e-3},
                data={"cases": cases})


def verify_hlike(data):
    # Bohr energy levels from fundamental constants (CODATA 2018), lambda = hc / dE
    me, e, eps0, h, c = 9.1093837015e-31, 1.602176634e-19, 8.8541878128e-12, 6.62607015e-34, 299792458
    level = lambda z, n: -me * e**4 * z**2 / (8 * eps0**2 * h**2 * n**2)
    return ",".join(sig(h * c / (level(z, ni) - level(z, nf)) * 1e9) for z, ni, nf in data["cases"])


# ---------- 10. counting states of a particle in a cubic box ----------

def make_box(p, rng):
    emax = p - rng.randrange(0, 10)
    m = math.isqrt(emax)
    cnt = sum(1 for x in range(1, m + 1) for y in range(1, m + 1) for z in range(1, m + 1)
              if x * x + y * y + z * z <= emax)
    prompt = ("A particle is confined in a cubic box with impenetrable walls. Its energy levels are "
              "E = (n_x² + n_y² + n_z²)·E1, where n_x, n_y, n_z = 1, 2, 3, … are positive integers and E1 = h²/(8mL²). "
              "Every ordered triple (n_x, n_y, n_z) is a distinct quantum state (ignore spin); for example, (1, 1, 2), "
              "(1, 2, 1) and (2, 1, 1) are three different states.\n\n"
              f"How many quantum states have energy E ≤ {emax}·E1? Give the count as an integer.")
    return dict(prompt=prompt, answer=str(cnt), check=EXACT, data={"emax": emax})


def verify_box(data):
    # cube of the theta series sum q^(n^2), n >= 1; sum of coefficients up to emax
    emax = data["emax"]
    theta = np.zeros(emax + 1, dtype=np.int64)
    for n in range(1, math.isqrt(emax) + 1):
        theta[n * n] = 1
    cube = np.convolve(np.convolve(theta, theta)[:emax + 1], theta)[:emax + 1]
    return str(int(cube.sum()))


GENS = [
    Gen("r2_resistor_sp", "dc_circuits", "physics", "Эквивалентное сопротивление цепи", "number of resistors N",
        grid=list(range(4, 401, 2)), probe=(14, 32), make=make_sp, verify=verify_sp, rungs=(1, 2)),
    Gen("r2_ladder_mesh", "dc_circuits_mesh", "physics", "Ток в ветви лестничной цепи", "number of loops N",
        grid=list(range(2, 37)), probe=(6, 12), make=make_ladder, verify=verify_ladder, rungs=(2,)),
    Gen("r2_rc_switching", "transients", "physics", "Напряжение на конденсаторе после переключений",
        "number of switching intervals N", grid=list(range(3, 81)), probe=(10, 24), make=make_rc, verify=verify_rc,
        rungs=(2,)),
    Gen("r2_thin_lenses", "geometric_optics", "physics", "Изображение в системе тонких линз", "number of lenses N",
        grid=list(range(3, 121)), probe=(10, 24), make=make_lenses, verify=verify_lenses, rungs=(1, 2)),
    Gen("r2_layer_refraction", "refraction", "physics", "Смещение луча в слоистой пластине", "number of layers N",
        grid=[2, 3] + list(range(4, 101, 2)), probe=(12, 30), make=make_layers, verify=verify_layers, rungs=(2,)),
    Gen("r2_gas_cycle", "thermodynamics", "physics", "Работа газа за цикл", "number of processes N",
        grid=list(range(3, 41)), probe=(8, 16), make=make_cycle, verify=verify_cycle, rungs=(2,)),
    Gen("r2_calorimetry", "calorimetry", "physics", "Уравнение теплового баланса", "number of bodies N",
        grid=list(range(3, 71)), probe=(10, 24), make=make_calor, verify=verify_calor, rungs=(2,)),
    Gen("r2_level_mean_energy", "statistical_physics", "physics", "Средняя энергия многоуровневой системы",
        "number of levels N", grid=list(range(4, 121, 2)), probe=(14, 34), make=make_levels, verify=verify_levels,
        rungs=(2,)),
    Gen("r2_hydrogenlike_lines", "atomic_physics", "physics", "Длины волн водородоподобных ионов",
        "number of transitions K", grid=list(range(3, 121)), probe=(12, 32), make=make_hlike, verify=verify_hlike,
        rungs=(2,)),
    Gen("r2_box_states", "quantum_mechanics", "physics", "Число состояний в кубическом ящике",
        "energy bound E_max in units of E1 (actual value p minus 0..9)", grid=list(range(20, 601, 10)),
        probe=(80, 200), make=make_box, verify=verify_box, rungs=(2,)),
]
