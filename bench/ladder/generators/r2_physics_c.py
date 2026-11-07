"""Rung-2 physics generators, part C: physics sections not covered by parts A and B.

Induction, damped and driven oscillations, projectile motion, orbits, charge sharing between capacitors, AC circuits,
polarisers, the photoelectric effect, kinetic theory of gases, buoyancy. Each task is a list of K independent cases
(or a chain of K elements); the answer is a comma-separated list of numbers with 4 significant figures.
"""
import math

import mpmath
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar

from common import Gen
from r2_physics_a import sig

LIST4 = {"type": "list", "rel_tol": 2e-3}
G = 9.8
FORMAT = ("Give the {k} requested values, each with 4 significant figures, in the order listed, as plain decimal numbers "
          "separated by commas.")


def _list(vals):
    return ",".join(sig(v) for v in vals)


# ---------- 1. electromagnetic induction ----------

def _induction(c):
    kind = c["kind"]
    if kind == "linear":                       # |I| = N A |dB/dt| / R
        return c["N"] * c["A"] * 1e-4 * abs(c["b"]) / c["R"]
    if kind == "sine":                         # |emf(t)| = N A Bm w |cos(w t)|
        w = 2 * math.pi * c["f"]
        return abs(c["N"] * c["A"] * 1e-4 * c["Bm"] * w * math.cos(w * c["t"]))
    if kind == "rotate":                       # flux = N B A cos(w t) -> |emf| = N B A w |sin(w t)|
        w = 2 * math.pi * c["f"]
        return abs(c["N"] * c["B"] * c["A"] * 1e-4 * w * math.sin(w * c["t"]))
    return (c["B"] * c["L"]) ** 2 * c["v"] / c["R"]   # rod: F = B I L, I = B L v / R


def make_induction(k, rng):
    cases, lines = [], []
    for j in range(1, k + 1):
        kind = rng.choice(["linear", "sine", "rotate", "rod"])
        if kind == "linear":
            c = dict(kind=kind, N=rng.randint(20, 500), A=rng.randint(10, 400), b=rng.choice([-1, 1]) * rng.randint(2, 90) / 100,
                     R=rng.randint(2, 60))
            s = (f"a flat coil of {c['N']} turns and area {c['A']} cm² (resistance {c['R']} Ω, ends joined) lies perpendicular "
                 f"to a uniform magnetic field that changes at a constant rate of {c['b']} T/s; magnitude of the induced "
                 "current in A")
        elif kind == "sine":
            c = dict(kind=kind, N=rng.randint(20, 500), A=rng.randint(10, 400), Bm=rng.randint(5, 80) / 100,
                     f=rng.randint(5, 120), t=rng.randint(1, 99) / 1000)
            while abs(math.cos(2 * math.pi * c["f"] * c["t"])) < 0.1:          # keep the answer away from zero
                c["t"] = rng.randint(1, 99) / 1000
            s = (f"a flat coil of {c['N']} turns and area {c['A']} cm² lies perpendicular to a uniform magnetic field "
                 f"B(t) = {c['Bm']}·sin(2π·{c['f']}·t) T (t in s); magnitude of the induced emf in V at t = {c['t']} s")
        elif kind == "rotate":
            c = dict(kind=kind, N=rng.randint(20, 500), A=rng.randint(10, 400), B=rng.randint(5, 80) / 100,
                     f=rng.randint(5, 120), t=rng.randint(1, 99) / 1000)
            while abs(math.sin(2 * math.pi * c["f"] * c["t"])) < 0.1:
                c["t"] = rng.randint(1, 99) / 1000
            s = (f"a flat coil of {c['N']} turns and area {c['A']} cm² rotates at {c['f']} revolutions per second about an "
                 f"axis perpendicular to a uniform {c['B']} T magnetic field; at t = 0 the plane of the coil is "
                 f"perpendicular to the field; magnitude of the induced emf in V at t = {c['t']} s")
        else:
            c = dict(kind=kind, B=rng.randint(10, 150) / 100, L=rng.randint(10, 150) / 100, v=rng.randint(5, 200) / 10,
                     R=rng.randint(1, 40) / 2)
            s = (f"a conducting rod of length {c['L']} m slides at a constant {c['v']} m/s along two parallel rails joined "
                 f"by a {c['R']} Ω resistor (other resistances negligible), in a uniform {c['B']} T field perpendicular to "
                 "the plane of the rails; magnitude of the magnetic force on the rod in N")
        cases.append(c)
        lines.append(f"{j}. {s}.")
    prompt = (f"Electromagnetic induction. In each of the following {k} independent situations find the requested quantity "
              "(ignore self-inductance):\n" + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_induction(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_induction(data):
    # emf = -d(flux)/dt by numerical differentiation (mpmath); rod: force from dissipated power, F = P / v
    out = []
    for c in data["cases"]:
        kind = c["kind"]
        if kind == "linear":
            flux = lambda t: c["N"] * c["A"] * 1e-4 * c["b"] * t
            out.append(abs(mpmath.diff(flux, 0.3)) / c["R"])
        elif kind == "sine":
            flux = lambda t: c["N"] * c["A"] * 1e-4 * c["Bm"] * mpmath.sin(2 * mpmath.pi * c["f"] * t)
            out.append(abs(mpmath.diff(flux, c["t"])))
        elif kind == "rotate":
            flux = lambda t: c["N"] * c["B"] * c["A"] * 1e-4 * mpmath.cos(2 * mpmath.pi * c["f"] * t)
            out.append(abs(mpmath.diff(flux, c["t"])))
        else:
            emf = c["B"] * c["L"] * c["v"]
            out.append(emf ** 2 / c["R"] / c["v"])
    return _list(out)


# ---------- 2. damped and driven oscillations ----------

def _osc(c):
    m, k, b = c["m"], c["k"], c["b"]
    if c["kind"] == "free":                    # x(0) = x0, v(0) = 0, underdamped
        g = b / (2 * m)
        wd = math.sqrt(k / m - g * g)
        t = c["t"]
        return math.exp(-g * t) * c["x0"] * (math.cos(wd * t) + g / wd * math.sin(wd * t))
    w = c["w"]                                 # steady-state amplitude under F0 cos(w t)
    return c["F0"] / math.sqrt((k - m * w * w) ** 2 + (b * w) ** 2)


def make_osc(k_cases, rng):
    cases, lines = [], []
    for j in range(1, k_cases + 1):
        m, k = rng.randint(1, 40) / 10, rng.randint(5, 400)
        bcrit = 2 * math.sqrt(k * m)
        b = round(rng.uniform(0.03, 0.6) * bcrit, 2)
        if rng.random() < 0.6:
            c = dict(kind="free", m=m, k=k, b=b, x0=rng.randint(2, 30) / 100, t=rng.randint(5, 300) / 100)
            while not (b / (2 * m) * c["t"] <= 3 and abs(_osc(c)) >= 0.1 * c["x0"] * math.exp(-b / (2 * m) * c["t"])):
                c["t"] = rng.randint(5, 300) / 100
            s = (f"a {m} kg mass on a spring of stiffness {k} N/m with a linear damping force −{b}·v (N, v in m/s) is "
                 f"released from rest at x = {c['x0']} m; position x in m at t = {c['t']} s")
        else:
            w0 = math.sqrt(k / m)
            c = dict(kind="driven", m=m, k=k, b=b, F0=rng.randint(1, 50), w=round(w0 * rng.uniform(0.3, 1.7), 1))
            s = (f"a {m} kg mass on a spring of stiffness {k} N/m with a linear damping force −{b}·v is driven by the force "
                 f"{c['F0']}·cos({c['w']}·t) N; amplitude of the steady-state oscillation in m")
        cases.append(c)
        lines.append(f"{j}. {s}.")
    prompt = (f"Oscillations. In each of the following {k_cases} independent situations the motion is one-dimensional and "
              "x is measured from the equilibrium position; find the requested quantity:\n" + "\n".join(lines) + "\n\n"
              + FORMAT.format(k=k_cases))
    return dict(prompt=prompt, answer=_list(_osc(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_osc(data):
    # numerical integration of m x'' + b x' + k x = F(t); steady state from the complex response
    out = []
    for c in data["cases"]:
        m, k, b = c["m"], c["k"], c["b"]
        if c["kind"] == "free":
            sol = solve_ivp(lambda t, y: [y[1], -(b * y[1] + k * y[0]) / m], (0, c["t"]), [c["x0"], 0],
                            rtol=1e-11, atol=1e-13, method="DOP853")
            out.append(sol.y[0, -1])
        else:
            out.append(abs(c["F0"] / complex(k - m * c["w"] ** 2, b * c["w"])))
    return _list(out)


# ---------- 3. projectile motion ----------

def _proj(c):
    th = math.radians(c["angle"])
    vx, vy = c["v"] * math.cos(th), c["v"] * math.sin(th)
    if c["kind"] == "range":
        t = (vy + math.sqrt(vy * vy + 2 * G * c["h"])) / G
        return vx * t
    if c["kind"] == "hmax":
        return c["h"] + max(vy, 0) ** 2 / (2 * G)
    t = (vy + math.sqrt(vy * vy + 2 * G * c["h"])) / G   # impact speed
    return math.hypot(vx, vy - G * t)


def make_proj(k, rng):
    cases, lines = [], []
    ask = {"range": "horizontal distance from the launch point to the landing point in m",
           "hmax": "greatest height above the ground reached in m", "speed": "speed just before hitting the ground in m/s"}
    for j in range(1, k + 1):
        c = dict(kind=rng.choice(list(ask)), v=rng.randint(30, 600) / 10, angle=rng.randint(-40, 80), h=rng.randint(0, 900) / 10)
        if c["kind"] == "hmax" and c["angle"] <= 0:
            c["angle"] = rng.randint(5, 80)
        if c["h"] == 0 and c["angle"] <= 0:
            c["h"] = rng.randint(10, 900) / 10
        where = f"from a height of {c['h']} m above level ground" if c["h"] > 0 else "from level ground"
        dirn = f"{c['angle']}° above" if c["angle"] >= 0 else f"{-c['angle']}° below"
        cases.append(c)
        lines.append(f"{j}. a ball is launched {where} at {c['v']} m/s, {dirn} the horizontal; {ask[c['kind']]}.")
    prompt = (f"Projectile motion near the Earth's surface (g = 9.8 m/s², no air resistance). In each of the following {k} "
              "independent situations find the requested quantity:\n" + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_proj(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_proj(data):
    # numerical integration of the equations of motion until y = 0
    out = []
    for c in data["cases"]:
        th = math.radians(c["angle"])
        hit = lambda t, y: y[1]
        hit.terminal, hit.direction = True, -1
        top = lambda t, y: y[3]
        sol = solve_ivp(lambda t, y: [y[2], y[3], 0, -G], (0, 100), [0, c["h"], c["v"] * math.cos(th), c["v"] * math.sin(th)],
                        events=[hit, top], rtol=1e-11, atol=1e-11, dense_output=True)
        yend = sol.y_events[0][-1]
        if c["kind"] == "range":
            out.append(yend[0])
        elif c["kind"] == "speed":
            out.append(math.hypot(yend[2], yend[3]))
        else:
            out.append(sol.y_events[1][0][1] if len(sol.y_events[1]) else c["h"])
    return _list(out)


# ---------- 4. satellite orbits ----------

GRAV = 6.674e-11


def _orbit(c):
    mu = GRAV * c["M"] * 1e24
    rp, ra = (c["R"] + c["hp"]) * 1e3, (c["R"] + c["ha"]) * 1e3
    a = (rp + ra) / 2
    if c["kind"] == "period":
        return 2 * math.pi * math.sqrt(a ** 3 / mu) / 60
    r = rp if c["kind"] == "vp" else ra
    return math.sqrt(mu * (2 / r - 1 / a)) / 1e3


def make_orbit(k, rng):
    bodies = [("planet A", 5.97, 6371), ("planet B", 0.642, 3390), ("planet C", 4.87, 6052), ("moon D", 0.0735, 1737),
              ("planet E", 1898, 69911), ("planet F", 86.8, 25362), ("planet G", 568, 58232), ("planet H", 102, 24622)]
    ask = {"period": "orbital period in minutes", "vp": "speed at the lowest point of the orbit in km/s",
           "va": "speed at the highest point of the orbit in km/s"}
    cases, lines = [], []
    for j in range(1, k + 1):
        name, M, R = rng.choice(bodies)
        hp = rng.randint(100, 3000) if R < 10000 else rng.randint(2000, 60000)
        ha = hp + rng.choice([0, rng.randint(50, 4 * R)])
        c = dict(kind=rng.choice(list(ask)), M=M, R=R, hp=hp, ha=ha)
        orbit = (f"a circular orbit at an altitude of {hp} km" if ha == hp else
                 f"an elliptical orbit with lowest altitude {hp} km and highest altitude {ha} km")
        cases.append(c)
        lines.append(f"{j}. a satellite around {name} (mass {M}·10^24 kg, radius {R} km) moves on {orbit}; {ask[c['kind']]}.")
    prompt = (f"Gravitation and orbits (G = 6.674·10^-11 N·m²/kg², bodies are spherical, altitudes are measured from the "
              f"surface). In each of the following {k} independent situations find the requested quantity:\n"
              + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_orbit(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_orbit(data):
    # period: integral of r^2 / h over one revolution; speeds from angular momentum h = r v (at apsides)
    out = []
    for c in data["cases"]:
        mu = GRAV * c["M"] * 1e24
        rp, ra = (c["R"] + c["hp"]) * 1e3, (c["R"] + c["ha"]) * 1e3
        e = (ra - rp) / (ra + rp)
        p = rp * (1 + e)
        h = math.sqrt(mu * p)
        if c["kind"] == "period":
            T, _ = quad(lambda f: (p / (1 + e * math.cos(f))) ** 2 / h, 0, 2 * math.pi, epsabs=0, epsrel=1e-12)
            out.append(T / 60)
        else:
            out.append(h / (rp if c["kind"] == "vp" else ra) / 1e3)
    return _list(out)


# ---------- 5. charge sharing between capacitors ----------

def _share(c):
    C1, C2, V1, V2 = c["C1"], c["C2"], c["V1"], c["V2"]
    s = 1 if c["pol"] == "like" else -1
    Vf = (C1 * V1 + s * C2 * V2) / (C1 + C2)
    if c["kind"] == "V":
        return abs(Vf)
    return (0.5 * (C1 * V1 ** 2 + C2 * V2 ** 2) - 0.5 * (C1 + C2) * Vf ** 2) / 1000   # energy dissipated, mJ


def make_share(k, rng):
    cases, lines = [], []
    for j in range(1, k + 1):
        c = dict(C1=rng.randint(1, 990) / 10, C2=rng.randint(1, 990) / 10, V1=rng.randint(5, 400), V2=rng.randint(0, 400),
                 pol=rng.choice(["like", "opposite"]), kind=rng.choice(["V", "E"]))
        while _share(c) < 0.01 or (c["pol"] == "opposite" and c["C1"] * c["V1"] == c["C2"] * c["V2"]):
            c["V2"] = rng.randint(0, 400)
        join = ("positive plate to positive plate" if c["pol"] == "like" else "positive plate of each to the negative plate "
                "of the other")
        ask = "magnitude of the final common voltage in V" if c["kind"] == "V" else "energy dissipated in the connection in mJ"
        cases.append(c)
        lines.append(f"{j}. a {c['C1']} µF capacitor charged to {c['V1']} V and a {c['C2']} µF capacitor charged to "
                     f"{c['V2']} V are disconnected from their sources and then connected to each other, {join}; {ask}.")
    prompt = (f"Capacitors. In each of the following {k} independent situations the capacitors are ideal and isolated, and the "
              "connecting wires have negligible capacitance; find the requested quantity:\n" + "\n".join(lines) + "\n\n"
              + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_share(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_share(data):
    # final voltages Va, Vb from a linear system (charge on the joined node conserved, Vb = s Va);
    # loss = C1 C2 (V1 - s V2)^2 / (2 (C1 + C2)), µF·V² = µJ -> mJ
    out = []
    for c in data["cases"]:
        C1, C2, V1, V2 = c["C1"], c["C2"], c["V1"], c["V2"]
        s = 1 if c["pol"] == "like" else -1
        Va, _ = np.linalg.solve(np.array([[C1, s * C2], [s, -1]], float), [C1 * V1 + s * C2 * V2, 0])
        if c["kind"] == "V":
            out.append(abs(Va))
        else:
            out.append(C1 * C2 * (V1 - s * V2) ** 2 / (2 * (C1 + C2)) / 1000)
    return _list(out)


# ---------- 6. AC circuits ----------

def _ac(c):
    w = 2 * math.pi * c["f"]
    X = w * c["L"] * 1e-3 - 1 / (w * c["C"] * 1e-6)
    Z = math.sqrt(c["R"] ** 2 + X ** 2)
    I = c["V"] / Z
    if c["kind"] == "I":
        return I
    if c["kind"] == "P":
        return I * I * c["R"]
    if c["kind"] == "phase":
        return math.degrees(math.atan2(X, c["R"]))
    return I * w * c["L"] * 1e-3              # rms voltage across the inductor


def make_ac(k, rng):
    ask = {"I": "rms current in A", "P": "average power in W",
           "phase": "phase angle in degrees by which the source voltage leads the current (negative if it lags)",
           "VL": "rms voltage across the inductor in V"}
    cases, lines = [], []
    for j in range(1, k + 1):
        c = dict(kind=rng.choice(list(ask)), R=rng.randint(5, 300), L=rng.randint(5, 900), C=rng.randint(1, 200),
                 V=rng.randint(10, 240), f=rng.choice([50, 60, rng.randint(20, 2000)]))
        cases.append(c)
        lines.append(f"{j}. a series circuit of a {c['R']} Ω resistor, a {c['L']} mH inductor and a {c['C']} µF capacitor "
                     f"is driven by a sinusoidal source of {c['V']} V rms at {c['f']} Hz; {ask[c['kind']]}.")
    prompt = (f"AC circuits (ideal components, steady state). In each of the following {k} independent situations find the "
              "requested quantity:\n" + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_ac(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_ac(data):
    # complex phasors; power as Re(V I*)
    out = []
    for c in data["cases"]:
        w = 2 * np.pi * c["f"]
        Z = c["R"] + 1j * w * c["L"] * 1e-3 + 1 / (1j * w * c["C"] * 1e-6)
        I = c["V"] / Z
        out.append({"I": abs(I), "P": (c["V"] * I.conjugate()).real, "phase": np.degrees(np.angle(Z)),
                    "VL": abs(I * 1j * w * c["L"] * 1e-3)}[c["kind"]])
    return _list(out)


# ---------- 7. chain of polarisers ----------

def make_pol(n, rng):
    els, lines = [], []
    angle = rng.randint(0, 179)
    for j in range(1, n + 1):
        if j > 1:                              # small turns keep the intensity from collapsing along the chain
            angle = (angle + rng.choice([-1, 1]) * rng.randint(1, 20)) % 180
        el = dict(angle=angle, t=rng.choice([1.0] * 5 + [rng.randint(90, 99) / 100]))
        els.append(el)
        loss = "" if el["t"] == 1 else f", and it transmits only {round(el['t'] * 100)} % of the light polarised along its axis"
        lines.append(f"{j}. polariser with its transmission axis at {el['angle']}°{loss}")
    report = [j for j in range(1, n + 1) if j % 4 == 0 or j == n]
    I, out = 100.0, []
    for j, el in enumerate(els, 1):           # Malus's law, angle from the previous polariser
        I *= el["t"] * (0.5 if j == 1 else math.cos(math.radians(el["angle"] - els[j - 2]["angle"])) ** 2)
        if j in report:
            out.append(I)
    prompt = (f"Unpolarised light of intensity 100 W/m² passes in turn through the following {n} linear polarisers "
              "(angles of the transmission axes measured from the same reference direction). Unless stated otherwise a "
              "polariser is ideal: it fully transmits light polarised along its axis and blocks the perpendicular "
              "component.\n" + "\n".join(lines) + "\n\nGive the intensity in W/m² of the light leaving polarisers "
              + ", ".join(map(str, report)) + f" ({len(report)} values), each with 4 significant figures, in this order, as "
              "plain decimal numbers separated by commas.")
    return dict(prompt=prompt, answer=_list(out), check=LIST4,
                data={"els": els, "report": report})


def verify_pol(data):
    # Jones calculus, averaged over two orthogonal input polarisations (unpolarised light)
    out = {}
    for e in ([1, 0], [0, 1]):
        E = np.sqrt(50) * np.array(e, complex)
        for j, el in enumerate(data["els"], 1):
            a = np.radians(el["angle"])
            u = np.array([np.cos(a), np.sin(a)])
            E = np.sqrt(el["t"]) * np.outer(u, u) @ E
            if j in data["report"]:
                out[j] = out.get(j, 0) + float(np.vdot(E, E).real)
    return _list(out[j] for j in data["report"])


# ---------- 8. photoelectric effect ----------

H, C_LIGHT, E_CH, M_E = 6.626e-34, 2.998e8, 1.602e-19, 9.109e-31
METALS = [("caesium", 2.14), ("potassium", 2.30), ("sodium", 2.36), ("calcium", 2.87), ("magnesium", 3.66),
          ("aluminium", 4.08), ("zinc", 4.33), ("iron", 4.50), ("copper", 4.65), ("silver", 4.73), ("nickel", 5.01),
          ("gold", 5.10), ("platinum", 5.65)]


def _photo(c):
    E = H * C_LIGHT / (c["lam"] * 1e-9)               # J
    K = E - c["W"] * E_CH
    if c["kind"] == "V":
        return K / E_CH
    if c["kind"] == "v":
        return math.sqrt(2 * K / M_E) / 1e3            # km/s
    if c["kind"] == "lam0":
        return H * C_LIGHT / (c["W"] * E_CH) * 1e9      # nm
    return H / math.sqrt(2 * M_E * K) * 1e9            # de Broglie wavelength of the fastest electrons, nm


def make_photo(k, rng):
    ask = {"V": "stopping potential in V", "v": "maximum speed of the emitted electrons in km/s",
           "lam0": "threshold (cut-off) wavelength of the metal in nm",
           "dB": "de Broglie wavelength of the fastest emitted electrons in nm"}
    cases, lines = [], []
    for j in range(1, k + 1):
        name, W = rng.choice(METALS)
        lam0 = H * C_LIGHT / (W * E_CH) * 1e9
        lam = rng.randint(100, int(lam0 * 0.92))
        c = dict(kind=rng.choice(list(ask)), W=W, lam=lam)
        cases.append(c)
        lines.append(f"{j}. light of wavelength {lam} nm falls on {name} (work function {W} eV); {ask[c['kind']]}.")
    prompt = (f"Photoelectric effect (h = 6.626·10^-34 J·s, c = 2.998·10^8 m/s, e = 1.602·10^-19 C, electron mass "
              f"9.109·10^-31 kg; non-relativistic electrons). In each of the following {k} independent situations find the "
              "requested quantity:\n" + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_photo(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_photo(data):
    # everything in eV and nm: hc = 1239.84... eV·nm from the same constants; m c^2 in eV
    hc = H * C_LIGHT / E_CH * 1e9
    mc2 = M_E * C_LIGHT ** 2 / E_CH
    out = []
    for c in data["cases"]:
        K = hc / c["lam"] - c["W"]
        out.append({"V": K, "v": C_LIGHT * math.sqrt(2 * K / mc2) / 1e3, "lam0": hc / c["W"],
                    "dB": hc / math.sqrt(2 * mc2 * K)}[c["kind"]])
    return _list(out)


# ---------- 9. kinetic theory of gases ----------

KB, NA = 1.381e-23, 6.022e23
GASES = [("helium", 4.003, 2.2e-10), ("neon", 20.18, 2.6e-10), ("argon", 39.95, 3.4e-10), ("nitrogen", 28.01, 3.7e-10),
         ("oxygen", 32.00, 3.6e-10), ("hydrogen", 2.016, 2.7e-10), ("carbon dioxide", 44.01, 4.5e-10),
         ("xenon", 131.3, 4.9e-10), ("methane", 16.04, 3.8e-10)]


def _gas(c):
    m = c["M"] * 1e-3 / NA
    T = c["T"]
    if c["kind"] == "rms":
        return math.sqrt(3 * KB * T / m)
    if c["kind"] == "mean":
        return math.sqrt(8 * KB * T / (math.pi * m))
    if c["kind"] == "mp":
        return math.sqrt(2 * KB * T / m)
    lam = KB * T / (math.sqrt(2) * math.pi * c["d"] ** 2 * c["p"])
    return lam * 1e9 if c["kind"] == "mfp" else math.sqrt(8 * KB * T / (math.pi * m)) / lam / 1e9   # nm or 1e9/s


def make_gas(k, rng):
    ask = {"rms": "root-mean-square speed in m/s", "mean": "mean speed in m/s", "mp": "most probable speed in m/s",
           "mfp": "mean free path in nm", "freq": "mean collision frequency of one molecule in units of 10^9 per second"}
    cases, lines = [], []
    for j in range(1, k + 1):
        name, M, d = rng.choice(GASES)
        c = dict(kind=rng.choice(list(ask)), M=M, d=d, T=rng.randint(60, 1200), p=rng.randint(10, 500) * 1000)
        cases.append(c)
        extra = (f", pressure {c['p'] // 1000} kPa, molecular diameter {d * 1e10:.1f}·10^-10 m"
                 if c["kind"] in ("mfp", "freq") else "")
        lines.append(f"{j}. {name} (molar mass {M} g/mol) at {c['T']} K{extra}; {ask[c['kind']]}.")
    prompt = (f"Kinetic theory of an ideal gas with a Maxwell speed distribution (k_B = 1.381·10^-23 J/K, N_A = "
              f"6.022·10^23 1/mol; the mean free path is λ = k_B·T / (√2·π·d²·p) and the collision frequency is the mean "
              f"speed divided by λ). In each of the following {k} independent situations find the requested quantity:\n"
              + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_gas(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_gas(data):
    # moments and the maximum of the Maxwell distribution found numerically
    out = []
    for c in data["cases"]:
        m = c["M"] * 1e-3 / NA
        a = math.sqrt(KB * c["T"] / m)
        f = lambda v: (v / a) ** 2 * math.exp(-(v / a) ** 2 / 2)     # unnormalised, in units of a
        norm = quad(lambda x: x * x * math.exp(-x * x / 2), 0, 40)[0]
        mean = a * quad(lambda x: x ** 3 * math.exp(-x * x / 2), 0, 40)[0] / norm
        if c["kind"] == "rms":
            out.append(a * math.sqrt(quad(lambda x: x ** 4 * math.exp(-x * x / 2), 0, 40)[0] / norm))
        elif c["kind"] == "mean":
            out.append(mean)
        elif c["kind"] == "mp":
            out.append(minimize_scalar(lambda v: -f(v), bounds=(0.5 * a, 3 * a), method="bounded",
                                       options={"xatol": 1e-9 * a}).x)
        else:
            n = c["p"] / (KB * c["T"])
            lam = 1 / (math.sqrt(2) * n * math.pi * c["d"] ** 2)
            out.append(lam * 1e9 if c["kind"] == "mfp" else mean / lam / 1e9)
    return _list(out)


# ---------- 10. buoyancy ----------

LIQ = [("water", 1000), ("sea water", 1025), ("glycerine", 1260), ("ethanol", 789), ("olive oil", 910), ("mercury", 13600)]
SOLIDS = [("pine wood", 500), ("oak", 750), ("ice", 917), ("aluminium", 2700), ("iron", 7870), ("copper", 8960),
          ("lead", 11340), ("cork", 240), ("polyethylene", 940), ("granite", 2750)]


def _buoy(c):
    if c["kind"] == "float":                   # floating cube: depth submerged, cm
        return c["a"] * c["rs"] / c["rl"]
    if c["kind"] == "tension":                 # fully submerged, hanging on a string, N
        V = c["m"] / c["rs"]
        return abs(c["m"] * G - c["rl"] * V * G)
    if c["kind"] == "two":                     # cube floating at oil/water interface: depth in the lower liquid, cm
        rl, ru, rs, a = c["rl"], c["ru"], c["rs"], c["a"]
        return a * (rs - ru) / (rl - ru)
    V = c["m"] / c["rs"]                       # apparent weight on a scale (N) of a fully submerged sinking body
    return c["m"] * G - c["rl"] * V * G


def make_buoy(k, rng):
    cases, lines = [], []
    for j in range(1, k + 1):
        kind = rng.choice(["float", "tension", "two", "apparent"])
        if kind == "float":
            (ls, rl), (ss, rs) = rng.choice(LIQ), rng.choice([s for s in SOLIDS if s[1] < 1000])
            if rs >= rl:
                ls, rl = "glycerine", 1260
            c = dict(kind=kind, rl=rl, rs=rs, a=rng.randint(5, 60))
            s = (f"a solid {ss} cube (density {rs} kg/m³) of edge {c['a']} cm floats upright in {ls} (density {rl} kg/m³); "
                 "depth of the bottom face below the liquid surface in cm")
        elif kind == "tension":
            (ls, rl), (ss, rs) = rng.choice(LIQ[:5]), rng.choice(SOLIDS)
            c = dict(kind=kind, rl=rl, rs=rs, m=rng.randint(5, 900) / 100)
            how = "a string tied to the bottom of the container" if rs < rl else "a string from above"
            s = (f"a {c['m']} kg solid {ss} body (density {rs} kg/m³) is held fully submerged in {ls} (density {rl} kg/m³) "
                 f"by {how}; tension in the string in N")
        elif kind == "two":
            ru = rng.choice([700, 789, 850, 910])
            rl = rng.choice([1000, 1025, 1260])
            rs = rng.randint(ru + 20, rl - 20)
            c = dict(kind=kind, rl=rl, ru=ru, rs=rs, a=rng.randint(5, 40))
            s = (f"a solid cube of edge {c['a']} cm and density {rs} kg/m³ floats upright at the interface between a lower "
                 f"liquid of density {rl} kg/m³ and a deep upper layer of oil of density {ru} kg/m³ (the cube is entirely "
                 "below the top of the oil); depth of the cube's bottom face below the interface in cm")
        else:
            (ls, rl), (ss, rs) = rng.choice(LIQ[:5]), rng.choice([s for s in SOLIDS if s[1] > 1300])
            c = dict(kind=kind, rl=rl, rs=rs, m=rng.randint(5, 900) / 100)
            s = (f"a {c['m']} kg solid {ss} body (density {rs} kg/m³) rests on the bottom of a container of {ls} "
                 f"(density {rl} kg/m³), fully submerged; normal force from the bottom on the body in N")
        cases.append(c)
        lines.append(f"{j}. {s}.")
    prompt = (f"Hydrostatics and buoyancy (g = 9.8 m/s², liquids at rest, strings are massless). In each of the following "
              f"{k} independent situations find the requested quantity:\n" + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_buoy(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_buoy(data):
    # equilibrium found numerically: root of (buoyant force - weight) in the submerged depth; forces by direct sums
    out = []
    for c in data["cases"]:
        if c["kind"] == "float":
            a = c["a"] / 100
            out.append(100 * brentq(lambda d: c["rl"] * a * a * d - c["rs"] * a ** 3, 0, a))
        elif c["kind"] == "two":
            a = c["a"] / 100
            out.append(100 * brentq(lambda d: (c["rl"] * d + c["ru"] * (a - d)) * a * a - c["rs"] * a ** 3, 0, a))
        else:
            W, B = c["m"] * G, c["rl"] * G * c["m"] / c["rs"]
            out.append(abs(B - W))
    return _list(out)


GENS = [
    Gen("r2_induction", "em_induction", "physics", "Electromagnetic induction", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 25), make=make_induction, verify=verify_induction, rungs=(2,)),
    Gen("r2_oscillations", "oscillations", "physics", "Damped and driven oscillations", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 25), make=make_osc, verify=verify_osc, rungs=(2,)),
    Gen("r2_projectiles", "projectile_motion", "physics", "Projectile motion", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 25), make=make_proj, verify=verify_proj, rungs=(2,)),
    Gen("r2_orbits", "gravitation", "physics", "Satellite orbits", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 25), make=make_orbit, verify=verify_orbit, rungs=(2,)),
    Gen("r2_charge_sharing", "capacitors", "physics", "Charge sharing between capacitors",
        "number of cases K", grid=list(range(2, 121)), probe=(10, 25), make=make_share, verify=verify_share, rungs=(2,)),
    Gen("r2_ac_rlc", "ac_circuits", "physics", "Series RLC AC circuit", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 25), make=make_ac, verify=verify_ac, rungs=(2,)),
    Gen("r2_polarisers", "polarisation", "physics", "Chain of polarisers", "number of polarisers N",
        grid=list(range(2, 201)), probe=(20, 50), make=make_pol, verify=verify_pol, rungs=(2,)),
    Gen("r2_photoelectric", "photoelectric", "physics", "Photoelectric effect", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 25), make=make_photo, verify=verify_photo, rungs=(2,)),
    Gen("r2_kinetic_gas", "kinetic_theory", "physics", "Kinetic theory of gases", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 25), make=make_gas, verify=verify_gas, rungs=(2,)),
    Gen("r2_buoyancy", "hydrostatics", "physics", "Buoyancy and Archimedes' principle", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 25), make=make_buoy, verify=verify_buoy, rungs=(2,)),
]
