"""Rung-1 generators, social sciences and language: loan balance, IRR, Cournot oligopoly, deadweight loss of a tax,
invented numerals, invented alphabetical order, OLS regression."""
import math
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import numpy as np
import sympy

from common import Gen, frac_str


def _cents(c):
    return f"{c // 100}.{c % 100:02d}"


# ---------- 6. loan balance with rate changes and prepayments ----------

def make_loan(n, rng):
    while True:
        bal0 = rng.randint(1_000_000, 6_000_000)                     # cents
        m = rng.randrange(30, 76, 5)                                  # monthly rate in units of 0.01 %
        r = m / 10000
        pay = math.ceil(bal0 / 100 * r / (1 - (1 + r) ** -rng.choice([180, 240, 300]))) * 100
        months = rng.sample(range(2, n + 1), min(n - 1, max(1, n // 6)))
        events, rate_seen = {}, False
        for k in sorted(months):
            if not rate_seen and k > n // 3 or rng.random() < 0.35:
                events[k] = ("rate", None)
                rate_seen = True
            else:
                events[k] = ("extra", rng.randint(100, 2500) * 100 + rng.choice([0, 0, rng.randint(1, 99)]))
        bal, cur, ok = bal0, m, True
        for k in range(1, n + 1):
            ev = events.get(k)
            if ev and ev[0] == "rate":
                cur = rng.choice([v for v in range(30, 76, 5) if 0 < abs(v - cur) <= 15])
                events[k] = ("rate", cur)
            num = bal * cur
            if num % 10000 == 5000:
                ok = False
            interest = (num + 5000) // 10000
            if interest >= pay:
                ok = False
            bal += interest - pay
            if ev and ev[0] == "extra":
                bal -= ev[1]
        if ok and bal > bal0 // 5:
            break
    pct = lambda mm: f"{mm / 100:.2f}%"
    lines = []
    for k in sorted(events):
        t, v = events[k]
        lines.append(f"- Starting with month {k}, the monthly interest rate becomes {pct(v)} (annual {pct(12 * v)})."
                     if t == "rate" else f"- At the end of month {k}, after the regular payment, an extra "
                                         f"prepayment of ${_cents(v)} is made.")
    prompt = (f"A loan has an outstanding balance of ${_cents(bal0)} at the start of month 1. The nominal annual "
              f"interest rate is {pct(12 * m)}, i.e. the monthly interest rate is {pct(m)}. Every month the "
              "following happens, in this order:\n"
              "  1. interest = (balance at the start of the month) × (monthly rate in effect for that month), "
              "rounded to the nearest cent (half a cent rounds up);\n"
              f"  2. the interest is added to the balance and the regular payment of ${_cents(pay)} is subtracted;\n"
              "  3. if an extra prepayment is scheduled for that month, it is subtracted as well.\n"
              "The balance after step 3 is the balance at the start of the next month. The regular payment stays "
              f"${_cents(pay)} for the whole period. Scheduled events:\n" + "\n".join(lines) +
              f"\n\nWhat is the outstanding balance at the end of month {n} (after all steps of month {n})? "
              "Give the amount in dollars with exactly two decimals, without the $ sign and without thousands "
              "separators (e.g. 12345.67).")
    data = {"bal0": bal0, "m": m, "pay": pay, "n": n,
            "events": [[k, events[k][0], events[k][1]] for k in sorted(events)]}
    return dict(prompt=prompt, answer=_cents(bal), check={"type": "num", "rel_tol": 0}, data=data)


def verify_loan(data):
    bal, rate, pay = Decimal(data["bal0"]) / 100, Decimal(data["m"]) / 10000, Decimal(data["pay"]) / 100
    ev = {k: (t, v) for k, t, v in data["events"]}
    for k in range(1, data["n"] + 1):
        if ev.get(k, ("",))[0] == "rate":
            rate = Decimal(ev[k][1]) / 10000
        bal = bal + (bal * rate).quantize(Decimal("0.01"), ROUND_HALF_UP) - pay
        if ev.get(k, ("",))[0] == "extra":
            bal -= Decimal(ev[k][1]) / 100
    return str(bal)


# ---------- 7. internal rate of return ----------

def _npv(r, flows):
    return sum(cf / (1 + r) ** t for t, cf in enumerate(flows))


def make_irr(n, rng):
    while True:
        target = rng.uniform(0.04, 0.25)
        base = rng.randint(5, 40) * 100
        flows = [base + rng.randint(-20, 20) * 10 * rng.choice([1, 2, 3]) for _ in range(n)]
        flows = [max(100, f) for f in flows]
        inv = round(_npv(target, [0] + flows) / 100) * 100
        cf = [-inv] + flows
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if _npv(mid, cf) > 0 else (lo, mid)
        irr = (lo + hi) / 2 * 100
        f = irr * 100 - math.floor(irr * 100)
        if 2 < irr < 40 and abs(f - 0.5) > 0.1:
            break
    lines = "\n".join(f"  t = {t}: {'+' if c > 0 else '−'}{abs(c)}" for t, c in enumerate(cf))
    prompt = (f"An investment has the following yearly cash flows (in dollars; t = 0 is today, the flow at year t "
              f"occurs at the end of year t):\n{lines}\n\n"
              "Find the internal rate of return (IRR), i.e. the annual rate r > 0 such that "
              "NPV = Σ CF_t / (1 + r)^t = 0 (annual compounding). There is exactly one such rate because the flows "
              "change sign only once. Give r as a percentage rounded to 2 decimal places (e.g. 12.34 for 12.34%).")
    return dict(prompt=prompt, answer=f"{irr:.2f}", check={"type": "num", "rel_tol": 1e-3}, data={"cf": cf})


def verify_irr(data):
    roots = np.roots(data["cf"][::-1])           # polynomial in x = 1/(1+r), highest power first
    x = [z.real for z in roots if abs(z.imag) < 1e-9 and 0 < z.real < 1]
    assert len(x) == 1
    return f"{(1 / x[0] - 1) * 100:.2f}"


# ---------- 8. Cournot oligopoly with inactive high-cost firms ----------

def make_cournot(n, rng):
    while True:
        m = max(1, round(n * rng.uniform(0.55, 0.8)))
        if m == n:
            m -= 1
        act = [rng.randint(20, 120) for _ in range(m)]
        target = max(act) + rng.randint(2, 20) + rng.random()
        a = round(target * (m + 1) - sum(act))
        price = Fraction(a + sum(act), m + 1)
        if a <= 2 * price or any(c > price - 1 for c in act):
            continue
        lo = math.ceil(price + 1)
        ina = [rng.randint(lo, lo + 40) for _ in range(n - m)]
        if price.denominator == 1 and rng.random() < 0.7:
            continue
        break
    b = rng.choice([1, 2, 3, 4, 5])
    costs = act + ina
    rng.shuffle(costs)
    lines = "\n".join(f"  Firm {i}: c = {c}" for i, c in enumerate(costs, 1))
    prompt = (f"{n} firms compete in quantities (Cournot) in a market with inverse demand P = {a} − {b}·Q, where Q is "
              "the total quantity. Each firm i has a constant marginal cost c_i and no fixed cost, and chooses its "
              "quantity q_i ≥ 0 simultaneously to maximise its own profit (P − c_i)·q_i. Marginal costs:\n"
              f"{lines}\n\nIn the Nash equilibrium some high-cost firms may find it optimal to produce zero. Find the "
              "equilibrium market price P. Give it exactly, as a reduced fraction p/q (or an integer).")
    return dict(prompt=prompt, answer=frac_str(price), check={"type": "num", "rel_tol": 0},
                data={"a": a, "b": b, "c": costs})


def verify_cournot(data):
    a, b, c = data["a"], data["b"], sorted(data["c"])
    found = []
    for k in range(1, len(c) + 1):
        Q = Fraction(sum(a - ci for ci in c[:k]), b * (k + 1))     # from summing the FOCs of k active firms
        P = a - b * Q
        q = [Fraction(a - ci, b) - Q for ci in c[:k]]
        if all(x > 0 for x in q) and all(P - ci <= 0 for ci in c[k:]):
            found.append(P)
    assert len(found) == 1
    return frac_str(found[0])


# ---------- 9. deadweight loss of a per-unit tax with piecewise-linear curves ----------

def make_dwl(n, rng):
    dq = 10
    while True:
        dD = [rng.randint(1, 6) for _ in range(n)]
        dS = [rng.randint(1, 6) for _ in range(n)]
        cum = [0]
        for x, y in zip(dD, dS):
            cum.append(cum[-1] + x + y)
        g0 = rng.randint(cum[n - 1] + 1, cum[n] - 1)
        g = [g0 - c for c in cum]                                    # demand minus supply at breakpoints
        seg = 0 if n < 4 or g[0] - g[1] >= 2 else 1
        if g[seg] - g[seg + 1] < 2:
            continue
        t = rng.randint(g[seg + 1] + 1, g[seg] - 1)
        if t <= 0 or seg + 1 >= n:
            continue
        s0 = max(5, sum(dD) - g0 + 5) + rng.randint(0, 20)
        S = [s0]
        for y in dS:
            S.append(S[-1] + y)
        D = [S[k] + g[k] for k in range(n + 1)]
        break

    def cross(level):  # Q where g(Q) = level
        k = next(i for i in range(n) if g[i] >= level > g[i + 1] or g[i] > level >= g[i + 1])
        return k, k * dq + Fraction(g[k] - level, g[k] - g[k + 1]) * dq
    kt, qt = cross(t)
    k0, q0 = cross(0)
    gq = lambda Q: Fraction(g[min(int(Q // dq), n - 1)]) + (Q - min(int(Q // dq), n - 1) * dq) * Fraction(
        g[min(int(Q // dq), n - 1) + 1] - g[min(int(Q // dq), n - 1)], dq)
    pts = [qt] + [k * dq for k in range(n + 1) if qt < k * dq < q0] + [q0]
    area = sum((pts[i + 1] - pts[i]) * (gq(pts[i]) + gq(pts[i + 1])) / 2 for i in range(len(pts) - 1))
    rows = "\n".join(f"  Q = {k * dq:4d}:  demand price {D[k]:4d},  supply price {S[k]:4d}" for k in range(n + 1))
    prompt = ("In a competitive market the demand curve (price buyers are willing to pay) and the supply curve "
              "(price sellers require) are piecewise linear: they pass through the points below and are straight "
              "lines between consecutive points (prices in dollars, quantities in units).\n"
              f"{rows}\n\nThe government introduces a per-unit tax of ${t} paid by sellers. Without the tax the "
              "market trades the quantity Q0 at which demand price = supply price. With the tax it trades the "
              "quantity Qt at which demand price − supply price = tax. The deadweight loss is the area between the "
              "demand curve (above) and the supply curve (below) for quantities from Qt to Q0, i.e. "
              "∫ from Qt to Q0 of (demand price − supply price) dQ.\n\n"
              "Compute the deadweight loss exactly. Give it as a reduced fraction p/q (or an integer).")
    return dict(prompt=prompt, answer=frac_str(area), check={"type": "num", "rel_tol": 0},
                data={"dq": dq, "D": D, "S": S, "t": t})


def verify_dwl(data):
    dq, D, S, t = data["dq"], data["D"], data["S"], data["t"]
    n, x = len(D) - 1, sympy.Symbol("x")
    line = lambda P, k: P[k] + (P[k + 1] - P[k]) * (x - k * dq) / sympy.Integer(dq)

    def solve(level):
        for k in range(n):
            r = sympy.solve(sympy.Eq(line(D, k) - line(S, k), level), x)
            if r and k * dq <= r[0] <= (k + 1) * dq:
                return r[0], k
    (qt, kt), (q0, k0) = solve(t), solve(0)
    upper = [(qt, line(D, kt).subs(x, qt))] + [(k * dq, D[k]) for k in range(n + 1) if qt < k * dq < q0]
    lower = [(qt, line(S, kt).subs(x, qt))] + [(k * dq, S[k]) for k in range(n + 1) if qt < k * dq < q0]
    poly = lower + [(q0, line(S, k0).subs(x, q0))] + upper[::-1]     # counter-clockwise outline
    area = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1])) / 2
    return str(sympy.Rational(area))


# ---------- 10. numerals of an invented language ----------

CONS, VOWS = "kmnprstlvbdgh", "aeiou"


def _num_word(v, base, words, conn):
    """words: {1..base-1, base, base^2: syllable}."""
    terms, d = [], [v // base ** 2, v // base % base, v % base]
    for i, pw in ((0, base ** 2), (1, base)):
        if d[i]:
            terms.append(("" if d[i] == 1 else words[d[i]]) + words[pw])
    if d[2]:
        terms.append(words[d[2]])
    return f" {conn} ".join(terms)


def _solve_numerals(examples, questions, conn, limit=2):
    """All atom-value assignments consistent with the examples (grammar: terms joined by the connector, a term is
    a digit syllable or [digit syllable] + power syllable; bases 3..12, powers base and base^2). Returns the set of
    answer tuples for the questions (stops after `limit` different tuples)."""
    parse = lambda w: [[t[i:i + 2] for i in range(0, len(t), 2)] for t in w.split(f" {conn} ")]
    ex = [(parse(w), v) for w, v in examples]
    qs = [parse(w) for w in questions]
    order = []
    for terms, _ in sorted(ex, key=lambda e: sum(map(len, e[0]))):
        for t in terms:
            order += [a for a in t if a not in order]
    for terms in qs:
        for t in terms:
            order += [a for a in t if a not in order]
    answers = set()

    def value(terms, val, base):
        tot, last = 0, None
        for t in terms:
            if len(t) == 2:
                d, p = val[t[0]], val[t[1]]
                if not (2 <= d < base and p in (base, base ** 2)):
                    return None
                tv, place = d * p, p
            else:
                tv = val[t[0]]
                place = tv if tv in (base, base ** 2) else 1
            if last is not None and place >= last:
                return None
            last, tot = place, tot + tv
        return tot

    for base in range(3, 13):
        dom = list(range(1, base)) + [base, base ** 2]

        def rec(i, val, used):
            if len(answers) >= limit:
                return
            for terms, v in ex:
                if all(a in val for t in terms for a in t) and value(terms, val, base) != v:
                    return
            if i == len(order):
                res = tuple(value(q, val, base) for q in qs)
                if None not in res:
                    answers.add(res)
                return
            for x in dom:
                if x not in used:
                    val[order[i]] = x
                    rec(i + 1, val, used | {x})
                    del val[order[i]]
        rec(0, {}, frozenset())
    return answers


def make_numerals(k, rng):
    while True:
        base = rng.choice([5, 6, 7, 8])
        syl = rng.sample([c + v for c in CONS for v in VOWS], base + 2)
        conn = syl.pop()
        words = {d: syl[d - 1] for d in range(1, base)}
        words[base], words[base ** 2] = syl[base - 1], syl[base]
        top = base ** 3 - 1
        exv = sorted(set(rng.sample(range(1, base * 2), 4) + rng.sample(range(base, top + 1), base + 4)))
        examples = [(_num_word(v, base, words, conn), v) for v in exv]
        seen = {t[i:i + 2] for w, _ in examples for t in w.split(f" {conn} ") for i in range(0, len(t), 2)}
        if not seen >= set(syl[:base + 1]):
            continue
        rest = [v for v in range(base + 1, top + 1) if v not in exv]
        if len(rest) < k:
            continue
        qv = rng.sample(rest, k)
        questions = [_num_word(v, base, words, conn) for v in qv]
        if len(_solve_numerals(examples, questions, conn)) == 1:
            break
    ex_lines = "\n".join(f"{w} = {v}" for w, v in examples)
    q_lines = "\n".join(f"{i}. {w}" for i, w in enumerate(questions, 1))
    prompt = ("Here are some numbers written in an invented language (every number is written as one or more parts "
              f"joined by the word \"{conn}\"; spaces separate words):\n{ex_lines}\n\n"
              f"Using the system these examples reveal, write the following numbers in digits:\n{q_lines}\n\n"
              f"Give the {k} numbers as integers, separated by commas, in the order given.")
    return dict(prompt=prompt, answer=",".join(map(str, qv)), check={"type": "list", "rel_tol": 0},
                data={"examples": examples, "questions": questions, "conn": conn})


def verify_numerals(data):
    sols = _solve_numerals([tuple(e) for e in data["examples"]], data["questions"], data["conn"])
    assert len(sols) == 1
    return ",".join(map(str, next(iter(sols))))


# ---------- 11. sorting words by an invented alphabetical order ----------

def make_alphabet(n, rng):
    letters = rng.sample("abdefghiklmnoprstuvz", 12)
    vows = [c for c in letters if c in "aeiou"] or [letters[0]]
    cons = [c for c in letters if c not in vows]
    syll = lambda: rng.choice(cons) + rng.choice(vows) if cons else rng.choice(letters) * 2

    words = set()
    while len(words) < n:
        if words and rng.random() < 0.55:
            w0 = rng.choice(sorted(words))
            w = w0[:rng.randint(1, min(4, len(w0) - 1))] + "".join(rng.choice(letters) for _ in range(rng.randint(1, 4)))
        else:
            w = "".join(syll() for _ in range(rng.randint(1, 3))) + rng.choice(["", rng.choice(letters)])
        if len(w) >= 3 and not any(w.startswith(x) or x.startswith(w) for x in words):
            words.add(w)
    words = sorted(words)
    rng.shuffle(words)
    rank = {c: i for i, c in enumerate(letters)}
    srt = sorted(words, key=lambda w: [rank[c] for c in w])
    prompt = ("An invented language uses only the following 12 letters, and its alphabetical order is (from first to "
              f"last):\n\n  {' < '.join(letters)}\n\n"
              "Words are compared letter by letter from the left: at the first position where they differ, the word "
              "whose letter comes earlier in this alphabet comes first. (No word in the list is a beginning of "
              f"another.)\n\nSort the following {n} words in increasing alphabetical order of this language:\n\n"
              f"{', '.join(words)}\n\n"
              "Give the sorted words separated by commas, first word first.")
    return dict(prompt=prompt, answer=",".join(srt), check={"type": "list"},
                data={"letters": letters, "words": words})


def verify_alphabet(data):
    # map the invented order onto 'a', 'b', ... and merge-sort the translated strings
    tr = str.maketrans("".join(data["letters"]), "abcdefghijkl")
    items = [(w.translate(tr), w) for w in data["words"]]

    def msort(xs):
        if len(xs) <= 1:
            return xs
        left, right, out = msort(xs[:len(xs) // 2]), msort(xs[len(xs) // 2:]), []
        while left and right:
            out.append((left if left[0][0] < right[0][0] else right).pop(0))
        return out + left + right
    return ",".join(w for _, w in msort(items))


# ---------- 12. OLS regression with exact coefficients ----------

def make_ols(n, rng):
    while True:
        b0, b1 = rng.randint(-30, 30), rng.uniform(-4, 4)
        xs = [rng.randint(0, 25) for _ in range(n)]
        ys = [round(b0 + b1 * x + rng.gauss(0, 8)) for x in xs]
        sx, sy = sum(xs), sum(ys)
        sxx, sxy = sum(x * x for x in xs), sum(x * y for x, y in zip(xs, ys))
        den = n * sxx - sx * sx
        if den == 0:
            continue
        slope = Fraction(n * sxy - sx * sy, den)
        icpt = (sy - slope * sx) / n
        if slope != 0 and icpt != 0:
            break
    pts = "\n".join(f"  ({x}, {y})" for x, y in zip(xs, ys))
    prompt = (f"Fit the straight line y = a + b·x to the following {n} data points (x, y) by ordinary least squares, "
              f"i.e. choose a and b to minimise Σ (y_i − a − b·x_i)²:\n{pts}\n\n"
              "Give the slope b and the intercept a exactly, as reduced fractions p/q (or integers), separated by a "
              "comma, in the order: b, a.")
    return dict(prompt=prompt, answer=f"{frac_str(slope)},{frac_str(icpt)}", check={"type": "list", "rel_tol": 0},
                data={"x": xs, "y": ys})


def verify_ols(data):
    X = sympy.Matrix([[1, x] for x in data["x"]])
    y = sympy.Matrix(data["y"])
    a, b = (X.T * X).LUsolve(X.T * y)
    return f"{b},{a}"


GENS = [
    Gen("r1_fin_loan", "loan_balance", "finance", "Остаток по кредиту с досрочными платежами",
        "number of months", grid=list(range(3, 121)), probe=(12, 30), make=make_loan, verify=verify_loan),
    Gen("r1_fin_irr", "irr", "finance", "Внутренняя норма доходности", "number of cash flows after t=0",
        grid=list(range(2, 41)), probe=(5, 12), make=make_irr, verify=verify_irr),
    Gen("r1_econ_cournot", "cournot_oligopoly", "economics", "Равновесие Курно с неактивными фирмами",
        "number of firms", grid=list(range(3, 201)), probe=(40, 120), make=make_cournot, verify=verify_cournot),
    Gen("r1_econ_dwl", "tax_deadweight_loss", "economics", "Безвозвратные потери от налога",
        "number of linear segments", grid=list(range(3, 121)), probe=(15, 45), make=make_dwl, verify=verify_dwl),
    Gen("r1_ling_numerals", "invented_numerals", "linguistics", "Числительные выдуманного языка",
        "number of words to decode", grid=list(range(1, 61)), probe=(6, 25), make=make_numerals,
        verify=verify_numerals),
    Gen("r1_ling_alphabet", "invented_alphabet_sort", "linguistics", "Сортировка слов по выдуманному алфавиту",
        "number of words", grid=list(range(4, 101)), probe=(12, 30), make=make_alphabet, verify=verify_alphabet),
    Gen("r1_stat_ols", "ols_regression", "statistics", "Точная МНК-регрессия", "number of data points",
        grid=list(range(4, 151)), probe=(15, 40), make=make_ols, verify=verify_ols),
]
