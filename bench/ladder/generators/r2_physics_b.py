"""Rung-2 physics generators, part B: mechanics, electromagnetism, nuclear physics, relativity, waves, fluids, heat."""
import math
from fractions import Fraction

import mpmath
import numpy as np
import scipy.linalg
import sympy
from scipy.integrate import quad, solve_ivp

from common import Gen, frac_str
from r2_physics_a import EXACT, SIG4, sig

LIST4 = {"type": "list", "rel_tol": 2e-3}


def _sym_frac(x):
    x = sympy.nsimplify(x)
    return Fraction(int(x.p), int(x.q))


# ---------- 11. sequence of collisions of carts ----------

def _collide(m1, v1, m2, v2, kind):
    if kind == "inelastic":
        v = (m1 * v1 + m2 * v2) / (m1 + m2)
        return v, v
    return ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2), ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)


def make_coll(k, rng):
    n = k + 2
    while True:
        mass = [rng.randint(1, 3) for _ in range(n)]
        groups = [[i] for i in range(n)]
        vel = [Fraction(rng.randint(-6, 6)) for _ in range(n)]
        v0 = [int(v) for v in vel]
        events = []
        for _ in range(k):
            cand = []      # approaching neighbours whose collision keeps every denominator <= 6
            for g in range(len(groups) - 1):
                if vel[g] > vel[g + 1]:
                    m1, m2 = sum(mass[i] for i in groups[g]), sum(mass[i] for i in groups[g + 1])
                    for kind in ("elastic", "inelastic"):
                        a, b = _collide(m1, vel[g], m2, vel[g + 1], kind)
                        if max(x.denominator for x in vel[:g] + [a, b] + vel[g + 2:]) <= 6:
                            cand.append((kind, g, a, b))
            if not cand:
                break
            elastic = [c for c in cand if c[0] == "elastic"]
            kind, g, a, b = rng.choice(elastic if elastic and rng.random() < 0.75 else cand)
            events.append([groups[g][-1] + 1, kind])
            if kind == "inelastic":
                groups[g:g + 2] = [groups[g] + groups[g + 1]]
                vel[g:g + 2] = [a]
            else:
                vel[g], vel[g + 1] = a, b
        if len(events) == k:
            break
    final = [vel[g] for g, grp in enumerate(groups) for _ in grp]
    lines = [f"{j}. carts {a} and {a + 1}: " + ("elastic collision" if kind == "elastic"
                                                 else "perfectly inelastic collision (they stick together)")
             for j, (a, kind) in enumerate(events, 1)]
    carts = "\n".join(f"cart {i + 1}: mass {m} kg, initial velocity {v} m/s" for i, (m, v) in enumerate(zip(mass, v0)))
    prompt = (f"{n} carts move along a straight horizontal frictionless track. They are numbered 1 to {n} from left to "
              "right; velocities are positive to the right and negative to the left.\n" + carts +
              f"\n\nThe following {k} collisions take place, in exactly this order, and no others:\n" + "\n".join(lines) +
              "\n\nCarts that have stuck together move afterwards as one rigid body. When a cart belonging to such a "
              "body is named in a later collision, the whole body takes part (the collision is between the whole body "
              "on the left and the whole body on the right). Every collision is one-dimensional and instantaneous; in "
              "each listed collision the two bodies approach each other.\n\n"
              f"Find the velocities of carts 1, 2, …, {n} right after the last listed collision, in m/s. Give exact "
              "values (integers or reduced fractions p/q, with a minus sign for leftward motion), in the order of the "
              "cart numbers, separated by commas.")
    return dict(prompt=prompt, answer=",".join(frac_str(v) for v in final), check={"type": "list", "rel_tol": 0},
                data={"mass": mass, "v0": v0, "events": events})


def verify_coll(data):
    # solve the conservation laws with sympy for each collision
    mass = data["mass"]
    body = list(range(len(mass)))           # body id of each cart
    vel = {i: sympy.Integer(v) for i, v in enumerate(data["v0"])}
    x = sympy.Symbol("x")
    for a, kind in data["events"]:
        b1, b2 = body[a - 1], body[a]
        m1 = sum(mass[i] for i in range(len(mass)) if body[i] == b1)
        m2 = sum(mass[i] for i in range(len(mass)) if body[i] == b2)
        p = m1 * vel[b1] + m2 * vel[b2]
        if kind == "inelastic":
            vel[b1] = sympy.solve(sympy.Eq((m1 + m2) * x, p), x)[0]
            body = [b1 if q == b2 else q for q in body]
        else:
            e = m1 * vel[b1] ** 2 + m2 * vel[b2] ** 2
            roots = sympy.roots(sympy.Poly(m1 * x**2 + m2 * ((p - m1 * x) / m2) ** 2 - e, x))
            sx = next(r for r in roots if r != vel[b1])
            sy = (p - m1 * sx) / m2
            vel[b1], vel[b2] = sx, sy
    return ",".join(frac_str(_sym_frac(vel[body[i]])) for i in range(len(mass)))


# ---------- 12. block sliding over track segments with friction ----------

G = 9.8


def _seg_acc(kind, ang, mu):
    th = math.radians(ang)
    if kind == "flat":
        return -mu * G
    if kind == "up":
        return -G * (math.sin(th) + mu * math.cos(th))
    return G * (math.sin(th) - mu * math.cos(th))


def make_track(n, rng):
    while True:
        v0 = rng.randint(6, 15)
        v2, segs = v0 * v0, []
        for _ in range(n):
            for _ in range(60):
                kind = rng.choice(["flat", "up", "down", "down"])
                ang = 0 if kind == "flat" else rng.randint(5, 35)
                seg = [kind, ang, round(rng.uniform(0.05, 0.4), 2), round(rng.uniform(1, 12), 1)]
                w2 = v2 + 2 * _seg_acc(*seg[:3]) * seg[3]
                if 9 <= w2 <= 400:
                    break
            else:
                break
            segs.append(seg)
            v2 = w2
        if len(segs) == n:
            break
    desc = {"flat": "horizontal", "up": "uphill incline", "down": "downhill incline"}
    lines = [f"{i + 1}. {desc[k]}" + (f" at {a}° to the horizontal" if k != "flat" else "") +
             f", length {L} m (measured along the surface), kinetic friction coefficient μ = {mu:.2f}"
             for i, (k, a, mu, L) in enumerate(segs)]
    prompt = (f"A small block starts at the beginning of segment 1 with speed {v0} m/s and slides along a track made "
              f"of {n} consecutive straight segments (in the direction of motion):\n" + "\n".join(lines) +
              "\n\n\"Uphill\" means the block moves up the incline, \"downhill\" means it moves down. The block stays in "
              "contact with the surface, passes every junction without any loss of speed, and there is no air "
              "resistance. Use g = 9.8 m/s². It is guaranteed that the block never stops on the track.\n\n"
              "Find the speed of the block at the end of the last segment, in m/s, with 4 significant figures.")
    return dict(prompt=prompt, answer=sig(math.sqrt(v2)), check=SIG4, data={"v0": v0, "segs": segs})


def verify_track(data):
    # integrate the equations of motion segment by segment until the segment length is covered
    v = float(data["v0"])
    for kind, ang, mu, L in data["segs"]:
        a = _seg_acc(kind, ang, mu)
        ev = lambda t, y, L=L: y[0] - L
        ev.terminal = True
        sol = solve_ivp(lambda t, y, a=a: [y[1], a], (0, 1e3), [0.0, v], events=ev, rtol=1e-11, atol=1e-12,
                        max_step=L / v / 8)
        v = sol.y_events[0][0][1]
    return sig(v)


# ---------- 13. moment of inertia of a composite body ----------

def make_inertia(n, rng):
    px, py = rng.randint(-3, 3), rng.randint(-3, 3)
    parts, total, lines = [], Fraction(0), []
    for i in range(n):
        kind = rng.choice(["point", "rod", "disk", "ring", "plate"])
        m, cx, cy = rng.randint(1, 9), rng.randint(-5, 5), rng.randint(-5, 5)
        if kind == "point":
            part, icm = [kind, m, cx, cy], Fraction(0)
            text = f"point mass {m} kg at ({cx}, {cy})"
        elif kind == "rod":
            x2, y2 = cx, cy
            while (x2, y2) == (cx, cy):
                x2, y2 = rng.randint(-5, 5), rng.randint(-5, 5)
            part = [kind, m, cx, cy, x2, y2]
            icm = Fraction(m * ((x2 - cx) ** 2 + (y2 - cy) ** 2), 12)
            text = f"thin uniform rod of mass {m} kg with ends at ({cx}, {cy}) and ({x2}, {y2})"
            cx, cy = Fraction(cx + x2, 2), Fraction(cy + y2, 2)
        elif kind in ("disk", "ring"):
            r = rng.randint(1, 3)
            part = [kind, m, cx, cy, r]
            icm = Fraction(m * r * r, 2 if kind == "disk" else 1)
            text = (f"uniform solid disk of mass {m} kg and radius {r} m" if kind == "disk"
                    else f"thin uniform ring (hoop) of mass {m} kg and radius {r} m") + f", centre at ({cx}, {cy})"
        else:
            a, b = rng.randint(1, 4), rng.randint(1, 4)
            part = [kind, m, cx, cy, a, b]
            icm = Fraction(m * (a * a + b * b), 12)
            text = (f"thin uniform rectangular plate of mass {m} kg, {a} m × {b} m (sides parallel to the x and y "
                    f"axes), centre at ({cx}, {cy})")
        parts.append(part)
        lines.append(f"{i + 1}. {text}")
        total += icm + m * ((cx - px) ** 2 + (cy - py) ** 2)
    prompt = (f"A rigid flat body lies in the xy-plane (coordinates in metres) and consists of the following {n} parts "
              "(parts may overlap; they are all rigidly joined):\n" + "\n".join(lines) +
              f"\n\nFind the moment of inertia of the whole body about the axis perpendicular to the xy-plane through "
              f"the point ({px}, {py}). Use the standard results about an axis through the centre of mass and "
              "perpendicular to the plane: thin rod mL²/12, solid disk mR²/2, thin ring mR², rectangular plate "
              "m(a² + b²)/12, together with the parallel-axis theorem. Give the exact value in kg·m² as a reduced "
              "fraction p/q (or an integer).")
    return dict(prompt=prompt, answer=frac_str(total), check=EXACT, data={"axis": [px, py], "parts": parts})


def verify_inertia(data):
    # direct integration of r^2 dm over each part with sympy
    px, py = data["axis"]
    t, s, rho, phi = sympy.symbols("t s rho phi", real=True)
    d2 = lambda x, y: (x - px) ** 2 + (y - py) ** 2
    total = sympy.Integer(0)
    for kind, m, cx, cy, *rest in data["parts"]:
        if kind == "point":
            total += m * d2(cx, cy)
        elif kind == "rod":
            x2, y2 = rest
            total += m * sympy.integrate(d2(cx + t * (x2 - cx), cy + t * (y2 - cy)), (t, 0, 1))
        elif kind == "ring":
            r = rest[0]
            total += sympy.Rational(m, 2) / sympy.pi * sympy.integrate(
                sympy.expand(d2(cx + r * sympy.cos(phi), cy + r * sympy.sin(phi))), (phi, 0, 2 * sympy.pi))
        elif kind == "disk":
            r = rest[0]
            inner = sympy.integrate(sympy.expand(d2(cx + rho * sympy.cos(phi), cy + rho * sympy.sin(phi)) * rho),
                                    (phi, 0, 2 * sympy.pi))
            total += m / (sympy.pi * r**2) * sympy.integrate(inner, (rho, 0, r))
        else:
            a, b = rest
            total += sympy.Rational(m, a * b) * sympy.integrate(
                d2(t, s), (t, cx - sympy.Rational(a, 2), cx + sympy.Rational(a, 2)),
                (s, cy - sympy.Rational(b, 2), cy + sympy.Rational(b, 2)))
    return frac_str(_sym_frac(sympy.simplify(total)))


# ---------- 14. point charges: potential and field ----------

KE = 8.99e9


def _points(n, rng, lo=1, hi=9, unit=""):
    while True:
        P = (rng.randint(-10, 10), rng.randint(-10, 10))
        pts = []
        while len(pts) < n:
            x, y = rng.randint(-10, 10), rng.randint(-10, 10)
            if (x - P[0]) ** 2 + (y - P[1]) ** 2 >= 4 and all((x, y) != (a, b) for a, b, _ in pts):
                pts.append((x, y, rng.choice([-1, 1]) * rng.randint(lo, hi)))
        yield P, pts


def make_charges(n, rng):
    for P, pts in _points(n, rng):
        V, E, sv, se = 0.0, np.zeros(2), 0.0, 0.0
        for x, y, q in pts:
            d = np.array([P[0] - x, P[1] - y]) / 100
            r = np.linalg.norm(d)
            V += KE * q * 1e-9 / r
            E += KE * q * 1e-9 * d / r**3
            sv += abs(KE * q * 1e-9 / r)
            se += KE * abs(q) * 1e-9 / r**2
        if abs(V) >= 0.2 * sv and np.linalg.norm(E) >= 0.2 * se:
            break
    lines = [f"q{i + 1} = {q} nC at ({x}, {y})" for i, (x, y, q) in enumerate(pts)]
    prompt = (f"{n} point charges are fixed in vacuum in the xy-plane (coordinates in centimetres):\n" + "\n".join(lines) +
              f"\n\nConsider the point P = ({P[0]}, {P[1]}). Use k = 1/(4πε0) = 8.99·10^9 N·m²/C² and take the "
              "potential to be zero at infinity. Find (1) the electric potential V at P in volts (with its sign) and "
              "(2) the magnitude of the electric field E at P in V/m. Give both with 4 significant figures as plain "
              "decimal numbers (no exponent notation, no thousands separators), in the form `V, E`.")
    return dict(prompt=prompt, answer=f"{sig(V)},{sig(np.linalg.norm(E))}", check=LIST4,
                data={"P": list(P), "charges": [list(c) for c in pts]})


def verify_charges(data):
    # E = -grad V, by numerical differentiation of the potential (mpmath, high precision)
    mpmath.mp.dps = 30
    pts = data["charges"]
    pot = lambda x, y: sum(KE * q * mpmath.mpf(10) ** -9 / mpmath.sqrt(((x - a) / 100) ** 2 + ((y - b) / 100) ** 2)
                           for a, b, q in pts)
    x0, y0 = data["P"]
    ex = -mpmath.diff(lambda x: pot(x, y0), x0) * 100
    ey = -mpmath.diff(lambda y: pot(x0, y), y0) * 100
    return f"{sig(pot(x0, y0))},{sig(mpmath.sqrt(ex**2 + ey**2))}"


# ---------- 15. magnetic field of parallel wires ----------

MU0 = 4e-7 * math.pi


def make_wires(n, rng):
    for P, pts in _points(n, rng, 1, 20):
        B, sb = np.zeros(2), 0.0
        for x, y, i in pts:
            dx, dy = (P[0] - x) / 100, (P[1] - y) / 100
            r2 = dx * dx + dy * dy
            B += MU0 * i / (2 * math.pi * r2) * np.array([-dy, dx])
            sb += MU0 * abs(i) / (2 * math.pi * math.sqrt(r2))
        if np.linalg.norm(B) >= 0.2 * sb:
            break
    lines = [f"wire {j + 1}: through ({x}, {y}), current {abs(i)} A " + ("out of the page (+z)" if i > 0
                                                                         else "into the page (−z)")
             for j, (x, y, i) in enumerate(pts)]
    prompt = (f"{n} infinitely long straight thin wires are perpendicular to the xy-plane (the page) and pass through "
              "the following points (coordinates in centimetres):\n" + "\n".join(lines) +
              f"\n\nThe wires are in vacuum, μ0 = 4π·10⁻⁷ T·m/A. Each wire produces the field of magnitude "
              "μ0·I/(2πr) circling it by the right-hand rule: for a current out of the page the field lines go "
              "counterclockwise as seen by a reader looking at the page (x-axis to the right, y-axis up, +z toward the "
              "reader). Find the magnitude of the total magnetic field at the "
              f"point P = ({P[0]}, {P[1]}), in microtesla (μT), with 4 significant figures.")
    return dict(prompt=prompt, answer=sig(np.linalg.norm(B) * 1e6), check=SIG4,
                data={"P": list(P), "wires": [list(w) for w in pts]})


def verify_wires(data):
    # Biot-Savart law integrated numerically along each wire
    B = np.zeros(2)
    for x, y, i in data["wires"]:
        dx, dy = (data["P"][0] - x) / 100, (data["P"][1] - y) / 100
        integral = quad(lambda z: (dx * dx + dy * dy + z * z) ** -1.5, -np.inf, np.inf, epsabs=0, epsrel=1e-12)[0]
        B += MU0 * i / (4 * math.pi) * integral * np.array([-dy, dx])
    return sig(np.linalg.norm(B) * 1e6)


# ---------- 16. radioactive decay chain ----------

def _bateman_terms(halves, t):
    lam = [math.log(2) / h for h in halves]
    n = len(lam)
    pref = math.prod(lam[:-1])
    return [pref * math.exp(-lam[i] * t) / math.prod(lam[j] - lam[i] for j in range(n) if j != i) for i in range(n)]


def make_decay(n, rng):
    while True:
        ratio, base = rng.uniform(1.3, 1.6), rng.uniform(0.5, 2)     # well separated half-lives, shuffled
        halves = [round(base * ratio ** (i + rng.uniform(-0.2, 0.2)), 2) for i in range(n)]
        rng.shuffle(halves)
        t = round(sum(halves) * rng.uniform(0.6, 1.5), 1)
        terms = _bateman_terms(halves, t)
        res = sum(terms)
        if res >= 1e-3 and sum(abs(x) for x in terms) <= 30 * res:
            break
    chain = " → ".join(f"X{i + 1}" for i in range(n)) + " → (stable)"
    lines = [f"X{i + 1}: half-life {h} h" for i, h in enumerate(halves)]
    prompt = (f"A radioactive decay chain {chain} consists of {n} radioactive nuclides (each decays only into the next "
              "one):\n" + "\n".join(lines) +
              f"\n\nAt t = 0 the sample contains N0 nuclei of X1 and no other members of the chain. Decay constants are "
              f"λ = ln 2 / T½. You may use the Bateman solution: N_n(t) = N0·(λ1·λ2·…·λ_(n−1))·Σ_(i=1..n) "
              f"e^(−λ_i·t) / Π_(j=1..n, j≠i) (λ_j − λ_i).\n\nFind the ratio N_{n}(t)/N0 (the fraction of the "
              f"initial nuclei present as X{n}) at t = {t} h, with 4 significant figures.")
    return dict(prompt=prompt, answer=sig(res), check=SIG4, data={"halves": halves, "t": t})


def verify_decay(data):
    # matrix exponential of the rate matrix of the linear ODE system dN/dt = A N
    lam = [math.log(2) / h for h in data["halves"]]
    n = len(lam)
    a = np.zeros((n, n))
    for i in range(n):
        a[i, i] = -lam[i]
        if i:
            a[i, i - 1] = lam[i - 1]
    n0 = np.zeros(n)
    n0[0] = 1
    return sig((scipy.linalg.expm(a * data["t"]) @ n0)[-1])


# ---------- 17. relativistic velocity addition ----------

DOPPLER = [Fraction(a, b) for a, b in [(2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (3, 2), (4, 3), (5, 2), (5, 3), (5, 4),
                                        (6, 5), (7, 3), (7, 4), (7, 5), (9, 4)]]


def make_relvel(n, rng):
    while True:
        us, u, kprod = [], Fraction(0), Fraction(1)
        for _ in range(n):
            for _ in range(100):
                k = rng.choice(DOPPLER) ** rng.choice([1, -1])
                if max((kprod * k).numerator, (kprod * k).denominator) <= 10**4:
                    break
            kprod *= k
            v = (k - 1) / (k + 1)
            us.append(v)
            u = (u + v) / (1 + u * v)
        if u != 0:
            break
    lines = [f"S{i + 1} relative to S{i}: {frac_str(v)}" for i, v in enumerate(us)]
    prompt = (f"Inertial frames S0, S1, …, S{n} have parallel axes and move along their common x-axis. The velocity of "
              f"each frame relative to the previous one is given in units of c (positive = along +x):\n"
              + "\n".join(lines) +
              f"\n\nUsing special relativity, find the velocity of frame S{n} relative to S0, in units of c. Give the "
              "exact value as a reduced fraction p/q with its sign.")
    return dict(prompt=prompt, answer=frac_str(u), check=EXACT, data={"u": [frac_str(v) for v in us]})


def verify_relvel(data):
    # longitudinal Doppler factors (1+u)/(1-u) multiply along the chain
    k = sympy.Integer(1)
    for v in data["u"]:
        v = sympy.Rational(v)
        k *= (1 + v) / (1 - v)
    return frac_str(_sym_frac((k - 1) / (k + 1)))


# ---------- 18. Doppler effect cases ----------

VS = 343


def _doppler(case):
    kind, f, a, b = case          # a, b: signed speeds, + = toward the other party
    f = Fraction(f)
    if kind == "S":
        return f * VS / (VS - a)
    if kind == "O":
        return f * (VS + a) / VS
    if kind == "SO":
        return f * (VS + b) / (VS - a)
    return f * (VS + a) / (VS - a)


def make_doppler(k, rng):
    cases = []
    for _ in range(k):
        kind = rng.choice(["S", "O", "SO", "R"])
        sp = lambda: rng.choice([-1, 1]) * rng.randint(5, 90)
        cases.append([kind, rng.randint(200, 2000), sp(), sp() if kind == "SO" else 0])
    ta = lambda x: f"at {x} m/s toward {{}}" if x > 0 else f"at {-x} m/s away from {{}}"
    lines = []
    for j, (kind, f, a, b) in enumerate(cases, 1):
        if kind == "S":
            s = f"a source emitting {f} Hz moves " + ta(a).format("a stationary observer") + "; frequency heard by the observer"
        elif kind == "O":
            s = f"an observer moves " + ta(a).format(f"a stationary source emitting {f} Hz") + "; frequency heard by the observer"
        elif kind == "SO":
            s = (f"a source emitting {f} Hz moves " + ta(a).format("the observer") + ", and the observer moves "
                 + ta(b).format("the source") + "; frequency heard by the observer")
        else:
            s = (f"a stationary source emitting {f} Hz sends sound to a large flat reflector that moves "
                 + ta(a).format("the source") + "; frequency of the reflected sound received back at the source")
        lines.append(f"{j}. {s}.")
    prompt = (f"Sound travels in still air at {VS} m/s. In each of the following {k} independent situations all motion is "
              "along the straight line joining the source, the observer and (if present) the reflector, all speeds are "
              "constant, and any reflector re-emits the sound it receives as a moving source.\n" + "\n".join(lines) +
              f"\n\nGive the {k} requested frequencies in Hz, each with 4 significant figures, in the order listed, as "
              "plain decimal numbers separated by commas.")
    return dict(prompt=prompt, answer=",".join(sig(_doppler(c)) for c in cases), check=LIST4, data={"cases": cases})


def verify_doppler(data):
    # kinematics of two successive wavefronts (exact, with Fractions)
    def arrival(xs, te, v_obs, x_obs0, direction):
        # wavefront emitted at te from xs travels with speed VS in `direction`; meets x_obs0 + v_obs t
        return (direction * (x_obs0 - xs) + VS * te) / (VS - direction * v_obs)

    out = []
    for kind, f, a, b in data["cases"]:
        T, D = Fraction(1, f), Fraction(1000)
        vs = a if kind in ("S", "SO") else 0          # source at x = 0 initially, moves along +x when approaching
        vo = -(a if kind == "O" else b if kind == "SO" else 0)
        if kind == "R":
            vo = -a
        times = []
        for te in (Fraction(0), T):
            t1 = arrival(vs * te, te, vo, D, 1)
            if kind == "R":
                t1 = arrival(D + vo * t1, t1, 0, Fraction(0), -1)
            times.append(t1)
        out.append(sig(1 / (times[1] - times[0])))
    return ",".join(out)


# ---------- 19. pipeline with Bernoulli equation and losses ----------

RHO = 1000


def make_pipe(n, rng):
    while True:
        q = rng.randint(5, 30)                      # L/s
        p1 = rng.randint(200, 400)                  # kPa
        area = lambda: rng.randint(math.ceil(10 * q / 6), math.floor(10 * q / 0.5))
        A, z, K = [area()], [round(rng.uniform(0, 30), 1)], []
        p = Fraction(p1 * 1000)
        for i in range(n - 1):
            for _ in range(60):
                a2, dz, k = area(), round(rng.uniform(-8, 8), 1), round(rng.uniform(0.1, 1.5), 1)
                v1, v2 = Fraction(10 * q, A[-1]), Fraction(10 * q, a2)
                p2 = (p + RHO * (v1**2 - v2**2) / 2 + RHO * Fraction("9.8") * Fraction(str(-dz))
                      - Fraction(str(k)) * RHO * v1**2 / 2)
                if 120000 <= p2 <= 450000:
                    break
            else:
                break
            A.append(a2)
            z.append(round(z[-1] + dz, 1))
            K.append(k)
            p = p2
        if len(A) == n:
            break
    lines = [f"section {i + 1}: cross-section {A[i]} cm², centreline elevation {z[i]} m" +
             (f"; loss coefficient of the connection to section {i + 2}: K = {K[i]}" if i < n - 1 else "")
             for i in range(n)]
    prompt = (f"Water (density 1000 kg/m³, incompressible) flows steadily at a volume flow rate of {q} L/s through a "
              f"pipeline of {n} sections connected one after another:\n" + "\n".join(lines) +
              f"\n\nIn every section the speed is uniform over the cross-section (v = Q/A). Between section i and "
              "section i + 1 the flow loses pressure Δp_loss = K·ρ·v_i²/2, where v_i is the speed in the upstream "
              "section i; otherwise the Bernoulli equation holds: p + ρv²/2 + ρgz is conserved. Use g = 9.8 m/s². The "
              f"absolute pressure in section 1 is {p1} kPa.\n\nFind the absolute pressure in section {n}, in kPa, with "
              "4 significant figures.")
    return dict(prompt=prompt, answer=sig(p / 1000), check=SIG4, data={"q": q, "p1": p1, "A": A, "z": z, "K": K})


def verify_pipe(data):
    # march along the pipe in terms of total head H = p/(rho g) + z + v^2/(2g)
    v = [data["q"] / 1000 / (a / 1e4) for a in data["A"]]
    z = data["z"]
    H = data["p1"] * 1000 / (RHO * G) + z[0] + v[0] ** 2 / (2 * G)
    for i, k in enumerate(data["K"]):
        H -= k * v[i] ** 2 / (2 * G)
    return sig((H - z[-1] - v[-1] ** 2 / (2 * G)) * RHO * G / 1000)


# ---------- 20. heat conduction through a composite wall ----------

MATERIALS = {"brick": 0.72, "concrete": 1.40, "stone": 2.30, "wood": 0.13, "plaster": 0.50, "gypsum board": 0.17,
             "glass wool": 0.040, "polystyrene foam": 0.035, "cork": 0.043, "plywood": 0.12, "clay": 1.10}


def make_wall(n, rng):
    while True:
        layers = []
        for _ in range(n):
            name = rng.choice(sorted(MATERIALS))
            k = MATERIALS[name]
            opts = [L for L in range(5, 301, 5) if 0.02 <= L / 1000 / k <= 1.0]
            layers.append([name, k, rng.choice(opts)])
        hin, hout = rng.randint(5, 10), rng.randint(15, 30)
        tin, tout = rng.randint(18, 25), rng.randint(-25, 5)
        rs = [1 / hin] + [L / 1000 / k for _, k, L in layers] + [1 / hout]
        q = (tin - tout) / sum(rs)
        j = rng.randint(1, n - 1)
        tj = tin - q * sum(rs[:j + 1])
        if abs(tj) >= 5:
            break
    lines = [f"layer {i + 1}: {name}, thickness {L} mm, thermal conductivity {k} W/(m·K)"
             for i, (name, k, L) in enumerate(layers)]
    prompt = (f"A flat composite wall consists of {n} layers in perfect thermal contact, listed from the inside to the "
              "outside:\n" + "\n".join(lines) +
              f"\n\nThe inside air is at {tin} °C with a convective heat-transfer coefficient of {hin} W/(m²·K) at the "
              f"inner surface; the outside air is at {tout} °C with a coefficient of {hout} W/(m²·K) at the outer "
              "surface. Heat flow is steady and one-dimensional (no radiation, no edge effects).\n\nFind (1) the heat "
              "flux through the wall in W/m² (from inside to outside) and (2) the temperature in °C at the interface "
              f"between layer {j} and layer {j + 1}. Give both with 4 significant figures, in the form `q, T`.")
    return dict(prompt=prompt, answer=f"{sig(q)},{sig(tj)}", check=LIST4,
                data={"layers": layers, "h": [hin, hout], "T": [tin, tout], "j": j})


def verify_wall(data):
    # thermal network: unknown surface/interface temperatures from energy balances, solved as a linear system
    layers, (hin, hout), (tin, tout), j = data["layers"], data["h"], data["T"], data["j"]
    n = len(layers)
    g = [k / (L / 1000) for _, k, L in layers]
    a, b = np.zeros((n + 1, n + 1)), np.zeros(n + 1)
    for i in range(n + 1):
        left = hin if i == 0 else g[i - 1]
        right = hout if i == n else g[i]
        a[i, i] = left + right
        if i == 0:
            b[i] += hin * tin
        else:
            a[i, i - 1] = -left
        if i == n:
            b[i] += hout * tout
        else:
            a[i, i + 1] = -right
    temps = np.linalg.solve(a, b)
    return f"{sig(hin * (tin - temps[0]))},{sig(temps[j])}"


GENS = [
    Gen("r2_cart_collisions", "collisions", "physics", "Последовательность столкновений тележек",
        "number of collisions K", grid=list(range(3, 121)), probe=(8, 18), make=make_coll, verify=verify_coll,
        rungs=(2,)),
    Gen("r2_friction_track", "mechanics_friction", "physics", "Брусок на трассе с трением", "number of segments N",
        grid=list(range(3, 91)), probe=(10, 26), make=make_track, verify=verify_track, rungs=(2,)),
    Gen("r2_composite_inertia", "rigid_body", "physics", "Момент инерции составного тела", "number of parts N",
        grid=list(range(3, 121)), probe=(14, 36), make=make_inertia, verify=verify_inertia, rungs=(2,)),
    Gen("r2_point_charges", "electrostatics", "physics", "Потенциал и поле системы зарядов", "number of charges N",
        grid=list(range(2, 61)), probe=(8, 20), make=make_charges, verify=verify_charges, rungs=(2,)),
    Gen("r2_parallel_wires", "magnetism", "physics", "Магнитное поле параллельных токов", "number of wires N",
        grid=list(range(2, 61)), probe=(8, 20), make=make_wires, verify=verify_wires, rungs=(2,)),
    Gen("r2_decay_chain", "nuclear", "physics", "Цепочка радиоактивных распадов", "chain length N",
        grid=list(range(2, 15)), probe=(4, 7), make=make_decay, verify=verify_decay, rungs=(2,)),
    Gen("r2_velocity_addition", "special_relativity", "physics", "Релятивистское сложение скоростей",
        "number of frames N", grid=list(range(3, 91)), probe=(10, 28), make=make_relvel, verify=verify_relvel,
        rungs=(2,)),
    Gen("r2_doppler_cases", "waves_acoustics", "physics", "Эффект Доплера", "number of cases K",
        grid=list(range(3, 121)), probe=(12, 32), make=make_doppler, verify=verify_doppler, rungs=(2,)),
    Gen("r2_pipeline_bernoulli", "fluid_dynamics", "physics", "Давление в трубопроводе", "number of sections N",
        grid=list(range(3, 101)), probe=(12, 30), make=make_pipe, verify=verify_pipe, rungs=(2,)),
    Gen("r2_composite_wall", "heat_transfer", "physics", "Теплопередача через многослойную стенку",
        "number of layers N", grid=list(range(2, 151)), probe=(15, 40), make=make_wall, verify=verify_wall,
        rungs=(2,)),
]
