"""Rung-1 generators, second batch: information theory (Huffman codes), operations research (queues),
materials science (cubic crystals and X-ray diffraction).
"""
import heapq
import math
from collections import deque

import numpy as np

from common import Gen
from r2_physics_a import sig

LIST4 = {"type": "list", "rel_tol": 2e-3}
FORMAT = ("Give the {k} requested values, each with 4 significant figures, in the order listed, as plain decimal numbers "
          "separated by commas.")


def _list(vals):
    return ",".join(sig(v) for v in vals)


# ---------- information theory: entropy and Huffman codes ----------

def _huffman_len(counts):
    """Average codeword length (bits per symbol) of a binary Huffman code, via a heap."""
    total = sum(counts)
    heap = [(c, i) for i, c in enumerate(counts)]
    heapq.heapify(heap)
    cost, nxt = 0, len(counts)
    while len(heap) > 1:
        a, _ = heapq.heappop(heap)
        b, _ = heapq.heappop(heap)
        cost += a + b                          # every merge adds one bit to all symbols below it
        heapq.heappush(heap, (a + b, nxt)); nxt += 1
    return cost / total


def _info(c):
    counts = c["counts"]
    if c["kind"] == "huffman":
        return _huffman_len(counts)
    n = sum(counts)
    H = -sum(x / n * math.log2(x / n) for x in counts)
    return H if c["kind"] == "entropy" else _huffman_len(counts) - H      # redundancy


def make_info(k, rng):
    ask = {"entropy": "the Shannon entropy of the source in bits per symbol",
           "huffman": "the average codeword length of an optimal binary Huffman code in bits per symbol",
           "redundancy": "the redundancy of an optimal binary Huffman code (average codeword length minus entropy) in "
                         "bits per symbol"}
    cases, lines = [], []
    for j in range(1, k + 1):
        m = rng.randint(4, 9)
        c = dict(kind=rng.choice(list(ask)), counts=[rng.randint(1, 60) for _ in range(m)])
        if c["kind"] == "redundancy":
            while _info(c) < 0.01:
                c["counts"] = [rng.randint(1, 60) for _ in range(m)]
        letters = "ABCDEFGHI"[:m]
        freq = ", ".join(f"{s}: {x}" for s, x in zip(letters, c["counts"]))
        cases.append(c)
        lines.append(f"{j}. a source emits the symbols {letters[0]}–{letters[-1]} with relative frequencies proportional "
                     f"to {freq} (out of {sum(c['counts'])}); {ask[c['kind']]}.")
    prompt = (f"Information theory. The symbols of a memoryless source occur with probabilities equal to their counts "
              f"divided by the total count. Find the requested quantity in each of the following {k} independent "
              f"situations:\n" + "\n".join(lines) + "\n\n" + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_info(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_info(data):
    # Huffman by the two-queue method on sorted weights; entropy with natural logarithms
    out = []
    for c in data["cases"]:
        counts, n = c["counts"], sum(c["counts"])
        leaves, merged, cost = deque(sorted(counts)), deque(), 0

        def pop():
            if not merged or (leaves and leaves[0] <= merged[0]):
                return leaves.popleft()
            return merged.popleft()
        while len(leaves) + len(merged) > 1:
            s = pop() + pop()
            cost += s; merged.append(s)
        L = cost / n
        H = -sum(x / n * np.log(x / n) for x in counts) / np.log(2)
        out.append({"entropy": H, "huffman": L, "redundancy": L - H}[c["kind"]])
    return _list(out)


# ---------- operations research: M/M/c queues ----------

def _erlang_c(lam, mu, c):
    a = lam / mu
    rho = a / c
    s = sum(a ** k / math.factorial(k) for k in range(c))
    top = a ** c / math.factorial(c) / (1 - rho)
    return top / (s + top)


def _queue(q):
    lam, mu, c = q["lam"], q["mu"], q["c"]
    pw = _erlang_c(lam, mu, c)
    Lq = pw * (lam / mu) / (c - lam / mu)
    return {"pwait": pw, "Lq": Lq, "Wq": Lq / lam * 60, "W": (Lq / lam + 1 / mu) * 60, "L": Lq + lam / mu}[q["kind"]]


def make_queue(k, rng):
    ask = {"pwait": "probability that an arriving customer has to wait", "Lq": "average number of customers waiting in "
           "the queue", "Wq": "average waiting time in the queue in minutes", "W": "average time in the system in "
           "minutes", "L": "average number of customers in the system"}
    cases, lines = [], []
    for j in range(1, k + 1):
        c = rng.randint(1, 5)
        mu = rng.randint(4, 30)
        lam = round(c * mu * rng.uniform(0.3, 0.92), 1)
        q = dict(kind=rng.choice(list(ask)), lam=lam, mu=mu, c=c)
        cases.append(q)
        srv = "a single server" if c == 1 else f"{c} identical servers sharing one queue"
        lines.append(f"{j}. customers arrive at {lam} per hour and are served by {srv}, each serving {mu} customers per "
                     f"hour on average; {ask[q['kind']]}.")
    prompt = (f"Queueing theory. Every system below is an M/M/c queue in steady state: Poisson arrivals, exponential "
              f"service times, c identical servers, one first-come-first-served queue of unlimited length. Find the "
              f"requested quantity in each of the following {k} independent systems:\n" + "\n".join(lines) + "\n\n"
              + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_queue(q) for q in cases), check=LIST4, data={"cases": cases})


def verify_queue(data):
    # stationary distribution of the birth-death chain, truncated far in the tail, by balance equations
    out = []
    for q in data["cases"]:
        lam, mu, c = q["lam"], q["mu"], q["c"]
        N = 3000
        p = np.zeros(N + 1); p[0] = 1.0
        for n in range(N):
            p[n + 1] = p[n] * lam / (min(n + 1, c) * mu)
        p /= p.sum()
        n = np.arange(N + 1)
        Lq = float(((n - c).clip(min=0) * p).sum())
        L = float((n * p).sum())
        pw = float(p[c:].sum())
        out.append({"pwait": pw, "Lq": Lq, "Wq": Lq / lam * 60, "W": L / lam * 60, "L": L}[q["kind"]])
    return _list(out)


# ---------- materials science: cubic crystals and X-ray diffraction ----------

ELEMENTS = [("aluminium", "FCC", 26.98, 4.050), ("copper", "FCC", 63.55, 3.615), ("nickel", "FCC", 58.69, 3.524),
            ("silver", "FCC", 107.87, 4.086), ("gold", "FCC", 196.97, 4.078), ("lead", "FCC", 207.2, 4.950),
            ("iron", "BCC", 55.85, 2.866), ("chromium", "BCC", 52.00, 2.885), ("tungsten", "BCC", 183.84, 3.165),
            ("molybdenum", "BCC", 95.95, 3.147), ("sodium", "BCC", 22.99, 4.291), ("polonium", "SC", 209.0, 3.359)]
NA = 6.022e23
ATOMS = {"SC": 1, "BCC": 2, "FCC": 4}


def _allowed(s, h, k, l):
    if s == "BCC":
        return (h + k + l) % 2 == 0
    if s == "FCC":
        return len({h % 2, k % 2, l % 2}) == 1
    return True


def _cryst(c):
    s, M, a = c["s"], c["M"], c["a"]
    if c["kind"] == "density":                 # g/cm^3
        return ATOMS[s] * M / NA / (a * 1e-8) ** 3
    if c["kind"] == "radius":                  # atomic radius in Å from touching atoms
        return {"SC": a / 2, "BCC": a * math.sqrt(3) / 4, "FCC": a * math.sqrt(2) / 4}[s]
    h, k, l = c["hkl"]
    d = a / math.sqrt(h * h + k * k + l * l)
    if c["kind"] == "d":
        return d
    return 2 * math.degrees(math.asin(c["lam"] / (2 * d)))     # 2θ in degrees


def make_cryst(k, rng):
    ask = {"density": "theoretical density in g/cm³", "radius": "atomic radius in Å (nearest atoms touching)",
           "d": "spacing of the ({h} {k} {l}) planes in Å",
           "two_theta": "first-order Bragg diffraction angle 2θ in degrees for the ({h} {k} {l}) planes with X-rays of "
                        "wavelength {lam} Å"}
    cases, lines = [], []
    for j in range(1, k + 1):
        name, s, M, a = rng.choice(ELEMENTS)
        c = dict(kind=rng.choice(list(ask)), s=s, M=M, a=a)
        if c["kind"] in ("d", "two_theta"):
            while True:
                hkl = [rng.randint(0, 4) for _ in range(3)]
                if any(hkl) and _allowed(s, *hkl):
                    break
            c["hkl"] = hkl
            c["lam"] = rng.choice([1.5406, 0.7093, 1.7890, 2.2909])
            if c["kind"] == "two_theta" and c["lam"] / (2 * a / math.sqrt(sum(x * x for x in hkl))) >= 0.98:
                c["kind"] = "d"
        cases.append(c)
        q = ask[c["kind"]].format(h=c.get("hkl", [0])[0], k=c.get("hkl", [0, 0])[1], l=c.get("hkl", [0, 0, 0])[2],
                                  lam=c.get("lam"))
        lines.append(f"{j}. {name} ({ {'SC': 'simple cubic', 'BCC': 'body-centred cubic', 'FCC': 'face-centred cubic'}[s]}, "
                     f"lattice constant {a} Å, molar mass {M} g/mol); {q}.")
    prompt = (f"Materials science: cubic crystals of a single element (N_A = 6.022·10^23 1/mol, 1 Å = 10^-8 cm). Find the "
              f"requested quantity in each of the following {k} independent situations:\n" + "\n".join(lines) + "\n\n"
              + FORMAT.format(k=k))
    return dict(prompt=prompt, answer=_list(_cryst(c) for c in cases), check=LIST4, data={"cases": cases})


def verify_cryst(data):
    # explicit atom positions in the cell: atoms per cell by counting shared corner/face/body sites, nearest-neighbour
    # distance by search, plane spacing from the reciprocal lattice vector
    basis = {"SC": [(0, 0, 0)], "BCC": [(0, 0, 0), (.5, .5, .5)], "FCC": [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]}
    out = []
    for c in data["cases"]:
        a = c["a"]
        B = np.array(basis[c["s"]])
        if c["kind"] == "density":
            out.append(len(B) * c["M"] / NA / (a * 1e-8) ** 3)
        elif c["kind"] == "radius":
            pts = np.array([b + np.array(t) for b in B for t in np.ndindex(3, 3, 3)]) - 1
            dist = np.linalg.norm(pts, axis=1) * a
            out.append(dist[dist > 1e-9].min() / 2)
        else:
            g = np.array(c["hkl"]) / a                      # reciprocal vector of a cubic lattice
            d = 1 / np.linalg.norm(g)
            out.append(d if c["kind"] == "d" else 2 * np.degrees(np.arcsin(c["lam"] / (2 * d))))
    return _list(out)


GENS = [
    Gen("r1_huffman", "huffman_entropy", "information_theory", "Entropy and Huffman codes", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 30), make=make_info, verify=verify_info),
    Gen("r1_queues", "mmc_queues", "operations_research", "M/M/c queues", "number of cases K",
        grid=list(range(2, 121)), probe=(10, 30), make=make_queue, verify=verify_queue),
    Gen("r1_crystals", "cubic_crystals", "materials_science", "Cubic crystals and X-ray diffraction",
        "number of cases K", grid=list(range(2, 121)), probe=(10, 30), make=make_cryst, verify=verify_cryst),
]
