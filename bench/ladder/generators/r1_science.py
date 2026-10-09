"""Rung-1 generators, natural sciences: mRNA translation, one-locus selection, heated salt mixtures,
a reaction chain with yields, buffer pH."""
import math
import re
from decimal import ROUND_HALF_UP, Decimal, getcontext
from fractions import Fraction

import sympy

from common import Gen


def _round(x, d):
    """Fraction -> Fraction rounded half up to d decimals (x >= 0)."""
    s = 10 ** d
    return Fraction(math.floor(Fraction(x) * s + Fraction(1, 2)), s)


def _dec(x, d):
    """Fraction -> decimal string with d decimals (rounded half up)."""
    n = math.floor(Fraction(x) * 10 ** d + Fraction(1, 2))
    return f"{n // 10 ** d}.{n % 10 ** d:0{d}d}" if d else str(n)


def _margin(x, d):
    """Distance of x * 10^d from the nearest rounding boundary (k + 1/2), in units of the last digit."""
    f = Fraction(x) * 10 ** d
    return abs(f - math.floor(f) - Fraction(1, 2))


# ---------- 1. translation of an mRNA, counts of three amino acids ----------

CODONS = {
    "Ala": "GCU GCC GCA GCG", "Arg": "CGU CGC CGA CGG AGA AGG", "Asn": "AAU AAC", "Asp": "GAU GAC",
    "Cys": "UGU UGC", "Gln": "CAA CAG", "Glu": "GAA GAG", "Gly": "GGU GGC GGA GGG", "His": "CAU CAC",
    "Ile": "AUU AUC AUA", "Leu": "UUA UUG CUU CUC CUA CUG", "Lys": "AAA AAG", "Met": "AUG",
    "Phe": "UUU UUC", "Pro": "CCU CCC CCA CCG", "Ser": "UCU UCC UCA UCG AGU AGC", "Thr": "ACU ACC ACA ACG",
    "Trp": "UGG", "Tyr": "UAU UAC", "Val": "GUU GUC GUA GUG", "Stop": "UAA UAG UGA",
}
AA_OF = {c: aa for aa, cs in CODONS.items() for c in cs.split()}


def make_translation(n, rng):
    sense = sorted(c for c, aa in AA_OF.items() if aa != "Stop")
    many = sorted(aa for aa, cs in CODONS.items() if aa not in ("Stop", "Met", "Trp"))
    while True:
        codons = ["AUG"] + [rng.choice(sense) for _ in range(n - 2)] + [rng.choice(CODONS["Stop"].split())]
        targets = rng.sample(many, 3)
        protein = [AA_OF[c] for c in codons[:-1]]
        counts = [protein.count(a) for a in targets]
        if min(counts) >= 1:
            break
    table = "\n".join(f"  {aa}: {', '.join(cs.split())}" for aa, cs in CODONS.items())
    prompt = (f"An mRNA of {n} codons is written below 5'→3', codon by codon (spaces separate the codons). "
              "Translation starts at the first codon (AUG, which encodes Met) and proceeds codon by codon "
              "until the stop codon at the end; the stop codon adds no amino acid. There are no other stop "
              "codons in the sequence.\n\nStandard genetic code (RNA codons):\n"
              f"{table}\n\nmRNA:\n{' '.join(codons)}\n\n"
              f"How many residues of {targets[0]}, {targets[1]} and {targets[2]} does the encoded protein contain? "
              f"Give three integers separated by commas, in this order: {', '.join(targets)}.")
    return dict(prompt=prompt, answer=",".join(map(str, counts)), check={"type": "list", "rel_tol": 0},
                data={"mrna": "".join(codons), "targets": targets})


def verify_translation(data):
    ncbi = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"  # NCBI table 1, order UCAG
    three = dict(A="Ala", R="Arg", N="Asn", D="Asp", C="Cys", Q="Gln", E="Glu", G="Gly", H="His", I="Ile",
                 L="Leu", K="Lys", M="Met", F="Phe", P="Pro", S="Ser", T="Thr", W="Trp", Y="Tyr", V="Val")
    s, prot = data["mrna"], ""
    for i in range(0, len(s), 3):
        a = ncbi[sum("UCAG".index(b) * 4 ** (2 - k) for k, b in enumerate(s[i:i + 3]))]
        if a == "*":
            break
        prot += a
    one = {v: k for k, v in three.items()}
    return ",".join(str(prot.count(one[t])) for t in data["targets"])


# ---------- 2. one-locus selection, frequency of A after G generations ----------

def make_selection(g, rng):
    vals = [Fraction(v, 100) for v in range(60, 101, 5)]
    while True:
        kind = rng.choice(["dir", "dir", "over"])
        w = sorted(rng.sample(vals, 3), reverse=True)
        if kind == "dir":
            w11, w12, w22 = w if rng.random() < 0.5 else w[::-1]
        else:
            w12, w11, w22 = w[0], *rng.sample(w[1:], 2)
        p = Fraction(rng.randint(10, 90), 100)
        p0, ok = p, True
        for _ in range(g):
            q = 1 - p
            wbar = p * p * w11 + 2 * p * q * w12 + q * q * w22
            x = (p * p * w11 + p * q * w12) / wbar
            if _margin(x, 4) < Fraction(1, 50):
                ok = False
            p = _round(x, 4)
            if not Fraction(5, 100) <= p <= Fraction(95, 100):
                ok = False
        if ok and p >= Fraction(1, 4) and p != p0:
            break
    f = lambda v: _dec(v, 2)
    prompt = (f"A diploid population has one locus with two alleles, A and a. Genotype fitnesses (relative "
              f"viabilities) are w(AA) = {f(w11)}, w(Aa) = {f(w12)}, w(aa) = {f(w22)}. Generations are discrete "
              "and non-overlapping, mating is random (Hardy–Weinberg proportions among zygotes), there is no "
              "mutation, migration or drift. If p is the frequency of A in one generation, the frequency in the "
              "next generation is\n\n"
              "  p' = (p²·w(AA) + p·q·w(Aa)) / (p²·w(AA) + 2·p·q·w(Aa) + q²·w(aa)),  where q = 1 − p.\n\n"
              f"The initial frequency is p₀ = {f(p0)}. Rounding rule: after computing each generation, round p' "
              "to 4 decimal places (round half up) and use this rounded value for the next generation.\n\n"
              f"What is the frequency of A after {g} generations (p_{g})? Give it rounded to 4 decimal places.")
    return dict(prompt=prompt, answer=_dec(p, 4), check={"type": "num", "rel_tol": 1e-3},
                data={"w": [f(w11), f(w12), f(w22)], "p0": f(p0), "g": g})


def verify_selection(data):
    getcontext().prec = 40
    w11, w12, w22 = map(Decimal, data["w"])
    p = Decimal(data["p0"])
    for _ in range(data["g"]):
        q = 1 - p
        wbar = p * p * w11 + 2 * p * q * w12 + q * q * w22
        p = (p + p * q * (p * (w11 - w12) + q * (w12 - w22)) / wbar).quantize(Decimal("0.0001"), ROUND_HALF_UP)
    return str(p)


# ---------- 3. mass percentages in heated two-component mixtures ----------

ATOMS = {"H": "1.008", "C": "12.011", "O": "15.999", "Na": "22.990", "Mg": "24.305", "S": "32.06",
         "Cl": "35.45", "K": "39.098", "Ca": "40.078", "Cu": "63.546", "Zn": "65.38"}
# formula, composition, residue formula, residue composition, moles of residue per mole, reaction text
SALTS = [
    ("NaHCO3", dict(Na=1, H=1, C=1, O=3), "Na2CO3", dict(Na=2, C=1, O=3), Fraction(1, 2),
     "2 NaHCO3 → Na2CO3 + H2O + CO2"),
    ("KHCO3", dict(K=1, H=1, C=1, O=3), "K2CO3", dict(K=2, C=1, O=3), Fraction(1, 2),
     "2 KHCO3 → K2CO3 + H2O + CO2"),
    ("MgCO3", dict(Mg=1, C=1, O=3), "MgO", dict(Mg=1, O=1), 1, "MgCO3 → MgO + CO2"),
    ("CaCO3", dict(Ca=1, C=1, O=3), "CaO", dict(Ca=1, O=1), 1, "CaCO3 → CaO + CO2"),
    ("ZnCO3", dict(Zn=1, C=1, O=3), "ZnO", dict(Zn=1, O=1), 1, "ZnCO3 → ZnO + CO2"),
    ("Mg(OH)2", dict(Mg=1, O=2, H=2), "MgO", dict(Mg=1, O=1), 1, "Mg(OH)2 → MgO + H2O"),
    ("Ca(OH)2", dict(Ca=1, O=2, H=2), "CaO", dict(Ca=1, O=1), 1, "Ca(OH)2 → CaO + H2O"),
    ("CuSO4·5H2O", dict(Cu=1, S=1, O=9, H=10), "CuSO4", dict(Cu=1, S=1, O=4), 1, "CuSO4·5H2O → CuSO4 + 5 H2O"),
    ("Na2CO3·10H2O", dict(Na=2, C=1, O=13, H=20), "Na2CO3", dict(Na=2, C=1, O=3), 1,
     "Na2CO3·10H2O → Na2CO3 + 10 H2O"),
    ("NaCl", dict(Na=1, Cl=1), "NaCl", dict(Na=1, Cl=1), 1, "NaCl does not change"),
    ("K2CO3", dict(K=2, C=1, O=3), "K2CO3", dict(K=2, C=1, O=3), 1, "K2CO3 does not change"),
]


def _molar(comp):
    return sum(Fraction(ATOMS[e]) * k for e, k in comp.items())


def _res_frac(s):
    return _molar(s[3]) * s[4] / _molar(s[1])


def make_mixture(k, rng):
    samples, used = [], set()
    while len(samples) < k:
        s1, s2 = rng.sample(SALTS, 2)
        r1, r2 = _res_frac(s1), _res_frac(s2)
        if abs(r1 - r2) < Fraction(1, 4):
            continue
        m = Fraction(rng.randint(500, 3000), 100)
        w = Fraction(rng.randint(15, 85), 100)
        res = _round(m * (w * r1 + (1 - w) * r2), 2)
        pct = 100 * (res / m - r2) / (r1 - r2)
        if not 5 < pct < 95 or _margin(pct, 1) < Fraction(3, 10):
            continue
        samples.append(dict(a=s1[0], b=s2[0], m=_dec(m, 2), res=_dec(res, 2), pct=_dec(pct, 1)))
        used |= {s1[0], s2[0]}
    reactions = "\n".join(f"  {s[0]}: {s[5]}" for s in SALTS if s[0] in used)
    elems = sorted({e for s in SALTS if s[0] in used for e in s[1]}, key=lambda e: Fraction(ATOMS[e]))
    lines = "\n".join(f"{i}. {s['m']} g of a mixture of {s['a']} and {s['b']}; mass of the residue {s['res']} g."
                      for i, s in enumerate(samples, 1))
    prompt = ("Each sample below is a mixture of exactly two compounds. Every sample is heated strongly until its "
              "mass no longer changes. On heating the compounds behave as follows (water vapour and carbon "
              "dioxide escape completely; the solid products shown are stable and remain in the residue; "
              f"nothing else happens):\n{reactions}\n\n"
              f"Atomic masses: {', '.join(f'{e} = {ATOMS[e]}' for e in elems)}.\n\n"
              f"Samples (initial mass of the mixture; mass of the solid residue after heating):\n{lines}\n\n"
              "For each sample, find the mass percentage of the FIRST-named compound in the original mixture. "
              "Use the atomic masses above and keep at least 5 significant figures in intermediate results. "
              f"Give the {k} percentages rounded to 1 decimal place, separated by commas, in the order of the "
              "samples.")
    return dict(prompt=prompt, answer=",".join(s["pct"] for s in samples), check={"type": "list", "rel_tol": 0.003},
                data={"samples": [{x: s[x] for x in ("a", "b", "m", "res")} for s in samples]})


def _parse_formula(f):
    """'CuSO4·5H2O' or 'Mg(OH)2' -> {element: count}."""
    total = {}
    for part in f.split("·"):
        mult = re.match(r"(\d*)(.*)", part)
        k0, part = int(mult[1] or 1), mult[2]
        stack = [{}]
        for tok in re.findall(r"[A-Z][a-z]?\d*|\(|\)\d*", part):
            if tok == "(":
                stack.append({})
            elif tok.startswith(")"):
                grp, n = stack.pop(), int(tok[1:] or 1)
                for e, c in grp.items():
                    stack[-1][e] = stack[-1].get(e, 0) + c * n
            else:
                e, n = re.match(r"([A-Z][a-z]?)(\d*)", tok).groups()
                stack[-1][e] = stack[-1].get(e, 0) + int(n or 1)
        for e, c in stack[0].items():
            total[e] = total.get(e, 0) + c * k0
    return total


def verify_mixture(data):
    mass = lambda f: sum(sympy.Rational(ATOMS[e]) * c for e, c in _parse_formula(f).items())
    rx = {s[0]: s[5] for s in SALTS}

    def residue_per_gram(f):
        text = rx[f]
        if "does not change" in text:
            return sympy.Integer(1)
        left, right = text.split("→")
        n_in = int(re.match(r"\s*(\d*)", left)[1] or 1)
        prod = right.split("+")[0].strip()
        return mass(prod) / (n_in * mass(f))

    out = []
    for s in data["samples"]:
        x, y = sympy.symbols("x y")
        sol = sympy.solve([x + y - sympy.Rational(s["m"]),
                           x * residue_per_gram(s["a"]) + y * residue_per_gram(s["b"]) - sympy.Rational(s["res"])],
                          [x, y])
        out.append(f"{float(100 * sol[x] / sympy.Rational(s['m'])):.1f}")
    return ",".join(out)


# ---------- 4. multistep synthesis with yields and limiting reagents ----------

def make_chain(s, rng):
    while True:
        steps, n, ok = [], Fraction(1), True
        mx = [Fraction(rng.randint(6000, 40000), 100) for _ in range(s + 1)]
        for i in range(s):
            a = rng.choice([1, 1, 1, 2])
            b = 2 * a if n < Fraction(1, 2) else (rng.choice([1, 1, 1, 2]) if n < 2 else 1)
            c = rng.choice([1, 1, 2, 3])
            if math.gcd(a, b, c) > 1:
                c += 1
            mr = Fraction(rng.randint(1800, 25000), 100)
            y = Fraction(rng.randint(70, 98), 100)
            lim_r = rng.random() < 0.35
            factor = Fraction(rng.randint(60, 90), 100) if lim_r else Fraction(rng.randint(115, 200), 100)
            r_mol = n / a * c * factor
            steps.append(dict(a=a, b=b, c=c, mr=mr, y=y, r_mol=r_mol))
            n = min(n / a, r_mol / c) * b * y
        scale = Fraction(rng.randint(1000, 9000), 100) / (n * mx[-1])
        m0 = _round(mx[0] * scale, 2)
        for st in steps:
            st["mR"] = _round(st["r_mol"] * scale * st["mr"], 2)
        # recompute exactly from the stated (rounded) numbers, check limiting-reagent margins
        n = m0 / mx[0]
        for st in steps:
            ex, rr = n / st["a"], st["mR"] / st["mr"] / st["c"]
            if not (rr > ex * Fraction(11, 10) or rr < ex * Fraction(9, 10)):
                ok = False
            n = min(ex, rr) * st["b"] * st["y"]
        final = n * mx[-1]
        if ok and 10 <= final <= 100 and m0 < 10 ** 6:
            break
    nm = lambda i: f"X{i + 1}"
    coef = lambda k, x: f"{k} {x}" if k > 1 else x
    lines = [f"Molar masses (g/mol): " + ", ".join(f"{nm(i)} = {_dec(mx[i], 2)}" for i in range(s + 1)) + ", "
             + ", ".join(f"R{i + 1} = {_dec(st['mr'], 2)}" for i, st in enumerate(steps)) + "."]
    for i, st in enumerate(steps):
        lines.append(f"Step {i + 1}: {coef(st['a'], nm(i))} + {coef(st['c'], f'R{i + 1}')} → {coef(st['b'], nm(i + 1))}"
                     f" (+ by-products); {_dec(st['mR'], 2)} g of R{i + 1} is used; yield {int(st['y'] * 100)}%.")
    prompt = (f"A synthesis proceeds in {s} steps. X1 is the starting material; in each step the whole amount of "
              "product obtained in the previous step reacts with the stated mass of a reagent R. The yield of a "
              "step is the percentage of the theoretical amount of product, and the theoretical amount is "
              "determined by the limiting reactant of that step (the reactant that would be used up first "
              "according to the stoichiometric coefficients). By-products do not matter.\n\n"
              f"Starting mass of X1: {_dec(m0, 2)} g.\n" + "\n".join(lines) +
              f"\n\nWhat mass of X{s + 1} is obtained at the end? Keep at least 6 significant figures in "
              "intermediate results and give the final mass in grams rounded to 2 decimal places.")
    data = {"m0": _dec(m0, 2), "M": [_dec(v, 2) for v in mx],
            "steps": [{"a": st["a"], "b": st["b"], "c": st["c"], "MR": _dec(st["mr"], 2), "mR": _dec(st["mR"], 2),
                       "y": int(st["y"] * 100)} for st in steps]}
    return dict(prompt=prompt, answer=_dec(final, 2), check={"type": "num", "rel_tol": 1e-3}, data=data)


def verify_chain(data):
    mass, M = float(data["m0"]), [float(v) for v in data["M"]]
    for i, st in enumerate(data["steps"]):
        # extent of reaction in mol from each reactant's mass
        xi = min(mass / (st["a"] * M[i]), float(st["mR"]) / (st["c"] * float(st["MR"])))
        mass = xi * st["b"] * M[i + 1] * st["y"] / 100
    return f"{mass:.2f}"


# ---------- 5. pH of buffer solutions (Henderson–Hasselbalch) ----------

BUFFERS = [  # acid form, base form, pKa
    ("acetic acid (CH3COOH)", "sodium acetate (CH3COONa)", "4.76"),
    ("formic acid (HCOOH)", "sodium formate (HCOONa)", "3.75"),
    ("lactic acid", "sodium lactate", "3.86"),
    ("benzoic acid", "sodium benzoate", "4.20"),
    ("propanoic acid", "sodium propanoate", "4.87"),
    ("ammonium chloride (NH4Cl)", "ammonia (NH3)", "9.25"),
    ("sodium dihydrogen phosphate (NaH2PO4)", "disodium hydrogen phosphate (Na2HPO4)", "7.21"),
    ("TRIS hydrochloride (TRIS-H+ Cl-)", "TRIS base", "8.07"),
    ("hypochlorous acid (HClO)", "sodium hypochlorite (NaClO)", "7.53"),
]


def _buf_ratio(b):
    na = Fraction(b["Va"]) * Fraction(b["ca"])
    nb = Fraction(b["Vb"]) * Fraction(b["cb"])
    add = Fraction(b["add"])
    if b["what"] == "HCl":
        na, nb = na + add, nb - add
    elif b["what"] == "NaOH":
        na, nb = na - add, nb + add
    return na, nb


def make_buffer(k, rng):
    concs = ["0.050", "0.100", "0.150", "0.200", "0.250", "0.300", "0.400", "0.500"]
    items = []
    while len(items) < k:
        acid, base, pka = rng.choice(BUFFERS)
        b = dict(acid=acid, base=base, pka=pka, Va=f"{rng.randrange(10, 101, 5)}.0", ca=rng.choice(concs),
                 Vb=f"{rng.randrange(10, 101, 5)}.0", cb=rng.choice(concs), what=rng.choice(["none", "HCl", "NaOH"]),
                 add="0")
        if b["what"] != "none":
            b["add"] = f"{rng.randint(1, 12) / 2:.1f}"
        na, nb = _buf_ratio(b)
        if na <= 0 or nb <= 0 or not Fraction(1, 5) <= nb / na <= 5:
            continue
        ph = Fraction(pka) + Fraction(math.log10(nb / na))
        if _margin(ph, 2) < Fraction(1, 5):
            continue
        b["pH"] = f"{float(ph):.2f}"
        items.append(b)
    lines = []
    for i, b in enumerate(items, 1):
        t = (f"{i}. {b['Va']} mL of {b['ca']} M {b['acid']} is mixed with {b['Vb']} mL of {b['cb']} M {b['base']}"
             f" (pKa = {b['pka']})")
        t += {"none": ".", "HCl": f"; then {b['add']} mmol of HCl is added.",
              "NaOH": f"; then {b['add']} mmol of NaOH is added."}[b["what"]]
        lines.append(t)
    prompt = ("Compute the pH of each of the following buffer solutions at 25 °C. Use the Henderson–Hasselbalch "
              "equation pH = pKa + log10( n(base form) / n(acid form) ), where n are the amounts (in mmol) of the "
              "base form and the acid form AFTER any added strong acid or strong base has reacted completely "
              "(HCl converts base form into acid form, NaOH converts acid form into base form, mole for mole). "
              "Both forms are in the same volume, so volumes cancel; ignore activity coefficients and water "
              "autoionisation. The pKa given refers to the acid form.\n\n" + "\n".join(lines) +
              f"\n\nGive the {k} pH values rounded to 2 decimal places, separated by commas, in the order given.")
    return dict(prompt=prompt, answer=",".join(b["pH"] for b in items), check={"type": "list", "rel_tol": 0.002},
                data={"buffers": [{x: b[x] for x in ("pka", "Va", "ca", "Vb", "cb", "what", "add")} for b in items]})


def verify_buffer(data):
    out = []
    for b in data["buffers"]:
        acid = float(b["Va"]) * float(b["ca"])
        base = float(b["Vb"]) * float(b["cb"])
        d = float(b["add"]) * {"none": 0, "HCl": 1, "NaOH": -1}[b["what"]]
        out.append(f"{float(b['pka']) + (math.log(base - d) - math.log(acid + d)) / math.log(10):.2f}")
    return ",".join(out)


GENS = [
    Gen("r1_bio_translation", "mrna_translation", "biology", "Трансляция мРНК: подсчёт аминокислот",
        "number of codons", grid=list(range(20, 1601, 10)), probe=(80, 200), make=make_translation,
        verify=verify_translation),
    Gen("r1_gen_selection", "selection_one_locus", "genetics", "Отбор в однолокусной модели",
        "number of generations", grid=list(range(2, 51)), probe=(5, 14), make=make_selection,
        verify=verify_selection),
    Gen("r1_chem_mixture", "salt_mixture_heating", "chemistry", "Состав смесей солей по потере массы",
        "number of samples", grid=list(range(1, 21)), probe=(3, 7), make=make_mixture, verify=verify_mixture),
    Gen("r1_chem_chain", "synthesis_chain_yields", "chemistry", "Многостадийный синтез с выходами",
        "number of steps", grid=list(range(2, 61)), probe=(6, 16), make=make_chain, verify=verify_chain),
    Gen("r1_chem_buffer", "buffer_ph", "chemistry", "pH буферных растворов",
        "number of buffer solutions", grid=list(range(1, 121)), probe=(4, 10), make=make_buffer,
        verify=verify_buffer),
]
