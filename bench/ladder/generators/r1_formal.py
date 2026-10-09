"""Rung-1 generators, formal fields: textbook cryptography (Diffie–Hellman, Vigenère, RSA), calendars (weekday,
Roman dates), just-intonation melody, multiplicative order, stack machine, polygon area."""
import datetime
import math
from collections import deque
from fractions import Fraction

import sympy
from sympy.ntheory import discrete_log, n_order

from common import Gen, frac_str


# ---------- 13. Diffie–Hellman shared secret (discrete log by hand) ----------

def _is_primitive_root(g, p):
    return all(pow(g, (p - 1) // f, p) != 1 for f in sympy.primefactors(p - 1))


def make_dh(p, rng):
    roots = [g for g in range(2, 40) if _is_primitive_root(g, p)]
    g = rng.choice(roots[:4])
    while True:
        a, b = (rng.randint((p - 1) // 2, 9 * (p - 1) // 10) for _ in range(2))
        A, B = pow(g, a, p), pow(g, b, p)
        K = pow(B, a, p)
        if a != b and K not in (0, 1, A, B):
            break
    prompt = (f"Alice and Bob run a textbook Diffie–Hellman key exchange with the public prime p = {p} and the public "
              f"generator g = {g} (a primitive root modulo p). Alice picks a secret exponent a and publishes "
              f"A = g^a mod p = {A}; Bob picks a secret exponent b and publishes B = g^b mod p = {B} "
              f"(1 ≤ a, b ≤ p − 2). The shared secret is K = g^(a·b) mod p = B^a mod p = A^b mod p.\n\n"
              "Recover the shared secret K (you will need to find one of the secret exponents, e.g. by computing "
              "successive powers of g modulo p). Give K as an integer between 0 and p − 1.")
    return dict(prompt=prompt, answer=str(K), check={"type": "num", "rel_tol": 0},
                data={"p": p, "g": g, "A": A, "B": B})


def verify_dh(data):
    p, g, A, B = data["p"], data["g"], data["A"], data["B"]
    return str(pow(B, int(discrete_log(p, A, g)), p))


# ---------- 14. Vigenère decryption with a given key ----------

WORDS = """the and that have with this from they will would there their what about which when make like time
just know take people into year your good some could them other than then look only come over think also back
after work first well even want because these give most river stone garden window market winter summer bridge
mountain silver forest island letter morning evening village teacher doctor student kitchen orange yellow purple
green little large small early late quiet happy simple strong gentle bright north south east west water fire
earth wind light night paper music number family friend animal flower travel follow answer between under always
never often together around before during against another country question picture example important different
children learn write read listen speak open close begin finish carry bring build clean count cover cross dance
dream drink drive enter fall feel fight fill find fly grow hold hope hunt jump keep kind king lake land laugh
lead leave lift live lose love meet move name need note ocean plant play pull push rain reach rest ride ring rise
road rock roof room rule run sail salt sand seat seed send ship shop sing sit sleep snow song sound star stay step
storm story sugar table tail talk tall team test tree true turn voice wait walk wall warm wash wave wear wheel
wild wing wise wood word yard young""".split()


def make_vigenere(n, rng):
    plain = ""
    while len(plain) < n:
        plain += rng.choice(WORDS).upper()
    key = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(4, 8)))
    cipher = "".join(chr((ord(c) + ord(key[i % len(key)]) - 130) % 26 + 65) for i, c in enumerate(plain))
    groups = " ".join(cipher[i:i + 5] for i in range(0, len(cipher), 5))
    prompt = ("A message (English words written together, letters only, no spaces) was encrypted with the Vigenère "
              "cipher. Letters are numbered A = 0, B = 1, …, Z = 25; the key is repeated over the whole message, and "
              "the i-th ciphertext letter is C_i = (P_i + K_i) mod 26, where K_i is the key letter at position i "
              f"(so decryption is P_i = (C_i − K_i) mod 26).\n\nKey: {key}\n\n"
              f"Ciphertext ({len(cipher)} letters, written in groups of five only for readability; the key continues "
              f"across group boundaries):\n{groups}\n\n"
              "Decrypt the message. Give the plaintext as one string of uppercase letters without spaces.")
    return dict(prompt=prompt, answer=plain, check={"type": "exact"}, data={"key": key, "cipher": cipher})


def verify_vigenere(data):
    abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    square = {k: abc[abc.index(k):] + abc[:abc.index(k)] for k in abc}   # tabula recta rows
    key = data["key"]
    return "".join(abc[square[key[i % len(key)]].index(c)] for i, c in enumerate(data["cipher"]))


# ---------- 15. textbook RSA decryption ----------

def make_rsa(k, rng):
    primes = list(sympy.primerange(31, 98))
    while True:
        p, q = rng.sample(primes, 2)
        phi = (p - 1) * (q - 1)
        es = [e for e in (3, 5, 7, 11, 13, 17, 19, 23) if math.gcd(e, phi) == 1]
        if es:
            break
    n, e = p * q, rng.choice(es)
    d = pow(e, -1, phi)
    msgs = []
    while len(msgs) < k:
        m = rng.randint(2, n - 2)
        c = pow(m, e, n)
        if m not in msgs and c != m and math.gcd(m, n) == 1:
            msgs.append(m)
    cs = [pow(m, e, n) for m in msgs]
    prompt = (f"Textbook RSA with small primes: p = {p}, q = {q}, n = p·q = {n}, public exponent e = {e}. "
              "The private exponent d is the inverse of e modulo φ(n) = (p − 1)(q − 1), taken in the range "
              "1 ≤ d < φ(n). A ciphertext c decrypts to m = c^d mod n.\n\n"
              f"Decrypt the following {k} ciphertexts:\n{', '.join(map(str, cs))}\n\n"
              f"Give the {k} plaintext numbers (each between 0 and n − 1), separated by commas, in the order of the "
              "ciphertexts.")
    return dict(prompt=prompt, answer=",".join(map(str, msgs)), check={"type": "list", "rel_tol": 0},
                data={"p": p, "q": q, "e": e, "c": cs})


def verify_rsa(data):
    n = data["p"] * data["q"]
    table = {pow(m, data["e"], n): m for m in range(n)}       # brute-force inverse of encryption
    return ",".join(str(table[c]) for c in data["c"])


# ---------- 16. day of the week for Gregorian dates ----------

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def make_weekday(k, rng):
    lo, hi = datetime.date(1600, 1, 1).toordinal(), datetime.date(2400, 12, 31).toordinal()
    dates = [datetime.date.fromordinal(rng.randint(lo, hi)) for _ in range(k)]
    lines = "\n".join(f"{i}. {d.day} {MONTHS[d.month - 1]} {d.year}" for i, d in enumerate(dates, 1))
    prompt = ("For each of the following dates in the Gregorian calendar (extended to all years shown; leap years are "
              "years divisible by 4, except years divisible by 100 that are not divisible by 400), find the day of "
              f"the week.\n\n{lines}\n\n"
              f"Give the {k} weekday names in English (Monday, Tuesday, …), separated by commas, in the order of the "
              "dates.")
    return dict(prompt=prompt, answer=",".join(DAYS[d.weekday()] for d in dates), check={"type": "list"},
                data={"dates": [[d.year, d.month, d.day] for d in dates]})


def verify_weekday(data):
    out = []
    for y, m, d in data["dates"]:
        if m < 3:                                                    # Zeller's congruence
            m, y = m + 12, y - 1
        h = (d + 13 * (m + 1) // 5 + y + y // 4 - y // 100 + y // 400) % 7   # 0 = Saturday
        out.append(["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][h])
    return ",".join(out)


# ---------- 17. Roman calendar dates -> modern dates ----------

LAT_ACC = ["Ianuarias", "Februarias", "Martias", "Apriles", "Maias", "Iunias", "Iulias", "Augustas", "Septembres",
           "Octobres", "Novembres", "Decembres"]
LAT_ABL = ["Ianuariis", "Februariis", "Martiis", "Aprilibus", "Maiis", "Iuniis", "Iuliis", "Augustis",
           "Septembribus", "Octobribus", "Novembribus", "Decembribus"]
MDAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _nones(m):  # m: 1..12
    return 7 if m in (3, 5, 7, 10) else 5


def _roman_num(n):
    out = ""
    for v, s in ((10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")):
        while n >= v:
            out, n = out + s, n - v
    return out


def _to_roman(d, m):
    non, ide = _nones(m), _nones(m) + 8
    if d == 1:
        return f"Kalendis {LAT_ABL[m - 1]}"
    if d == non:
        return f"Nonis {LAT_ABL[m - 1]}"
    if d == ide:
        return f"Idibus {LAT_ABL[m - 1]}"
    if d < non:
        cnt, ref = non - d + 1, f"Nonas {LAT_ACC[m - 1]}"
    elif d < ide:
        cnt, ref = ide - d + 1, f"Idus {LAT_ACC[m - 1]}"
    else:
        cnt, ref = MDAYS[m - 1] - d + 2, f"Kalendas {LAT_ACC[m % 12]}"
    return f"pridie {ref}" if cnt == 2 else f"ante diem {_roman_num(cnt)} {ref}"


def make_roman(k, rng):
    days = [(d, m) for m in range(1, 13) for d in range(1, MDAYS[m - 1] + 1)]
    while True:
        picks = rng.sample(days, k)
        named = sum(1 for d, m in picks if not _to_roman(d, m).startswith("ante"))
        if named <= max(1, k // 4):
            break
    lines = "\n".join(f"{i}. {_to_roman(d, m)}" for i, (d, m) in enumerate(picks, 1))
    table = "\n".join(f"  {i + 1:2d}. {MONTHS[i]:9s} — {MDAYS[i]} days; accusative {LAT_ACC[i]}, ablative "
                      f"{LAT_ABL[i]}; Nones on the {_nones(i + 1)}th, Ides on the {_nones(i + 1) + 8}th"
                      for i in range(12))
    prompt = ("Convert the following Roman calendar dates to modern day.month form. The year is a common (non-leap) "
              "year of the Julian/Gregorian month lengths. Rules:\n"
              "- Each month has three named days: the Kalends (Kalendae) = the 1st, the Nones (Nonae) and the Ides "
              "(Idus), on the days given in the table below.\n"
              "- \"Kalendis/Nonis/Idibus <month>\" (ablative) is the named day itself.\n"
              "- \"ante diem N Kalendas/Nonas/Idus <month>\" (N a Roman numeral, month in the accusative) counts "
              "backwards INCLUSIVELY from that named day: the named day itself is day 1, so the date is the named "
              "day minus (N − 1) days. \"pridie\" (= ante diem II) is the day immediately before the named day.\n"
              "- Days after the Ides of a month are counted towards the Kalends (the 1st) of the NEXT month, so "
              "\"Kalendas\" with a month name always refers to the 1st of that named month and the date falls in "
              "the month before it (for Kalendas Ianuarias: in December).\n\n"
              f"Months:\n{table}\n\nDates:\n{lines}\n\n"
              f"Give the {k} dates in the form DD.MM (two digits each, e.g. 05.03 for 5 March), separated by commas, "
              "in the order given.")
    return dict(prompt=prompt, answer=",".join(f"{d:02d}.{m:02d}" for d, m in picks), check={"type": "list"},
                data={"roman": [_to_roman(d, m) for d, m in picks]})


def verify_roman(data):
    val = {"I": 1, "V": 5, "X": 10}
    out = []
    for s in data["roman"]:
        w = s.split()
        mon = (LAT_ACC + LAT_ABL).index(w[-1]) % 12 + 1
        if w[0] == "ante":
            v = [val[c] for c in w[2]]
            back = sum(-x if x < y else x for x, y in zip(v, v[1:] + [0])) - 1
        else:
            back = 1 if w[0] == "pridie" else 0
        kind = w[-2][:3]                                             # Kal / Non / Idu / Idi
        day = 1 if kind == "Kal" else {"Non": 0, "Idu": 8, "Idi": 8}[kind] + (7 if mon in (3, 5, 7, 10) else 5)
        out.append((datetime.date(2023, mon, day) - datetime.timedelta(days=back)).strftime("%d.%m"))
    return ",".join(out)


# ---------- 18. just-intonation melody ----------

RATIOS = [Fraction(*r) for r in ((1, 1), (16, 15), (9, 8), (6, 5), (5, 4), (4, 3), (45, 32), (3, 2), (8, 5), (5, 3),
                                 (16, 9), (15, 8), (2, 1))]
NAT = ["C", "D", "E", "F", "G", "A", "B"]
PC = [0, 2, 4, 5, 7, 9, 11]


def _semis(note):
    return 12 * (int(note[1:]) + 1) + PC[NAT.index(note[0])]


def make_just(n, rng):
    starts = [("A4", 440), ("G4", 392), ("C5", 528), ("E4", 330), ("D4", 294), ("F4", 352)]
    lo, hi = 7 * 3 + 4, 7 * 5 + 4          # diatonic indices G3 .. G5 (index = 7*octave + letter)
    while True:
        start, f0 = rng.choice(starts)
        idx = 7 * int(start[1]) + NAT.index(start[0])
        notes, f, ok = [start], Fraction(f0), True
        for _ in range(n - 1):
            step = rng.choice([0, 1, 1, 2, 2, 3, 4, 7, -1, -1, -2, -2, -3, -4, -7])
            if not lo <= idx + step <= hi:
                step = -step
            idx += step
            nt = f"{NAT[idx % 7]}{idx // 7}"
            dist = _semis(nt) - _semis(notes[-1])
            f = f * RATIOS[dist] if dist >= 0 else f / RATIOS[-dist]
            notes.append(nt)
            if max(f.numerator, f.denominator) > 10 ** 9:
                ok = False
                break
        if ok and f.denominator > 1:
            break
    table = "\n".join(f"  {i:2d} semitones: {r.numerator}/{r.denominator}" for i, r in enumerate(RATIOS))
    prompt = (f"A melody is played starting on {start} tuned to {f0} Hz. Every next note is tuned relative to the note "
              "immediately before it, using the just-intonation ratio of the interval between them: multiply by the "
              "ratio when the melody goes up, divide when it goes down, keep the frequency for a repeated note. "
              "Intervals are measured in semitones (C4 = middle C; within an octave C–D, D–E, F–G, G–A, A–B are 2 "
              "semitones and E–F, B–C are 1 semitone; the octave number increases from B to C). Use exactly these "
              f"ratios:\n{table}\n\nThe melody ({n} notes):\n{' '.join(notes)}\n\n"
              "What is the frequency of the last note? Give it as a reduced fraction a/b in Hz (or an integer if "
              "b = 1).")
    return dict(prompt=prompt, answer=frac_str(f), check={"type": "num", "rel_tol": 0}, data={"notes": notes, "f0": f0})


def verify_just(data):
    midi = lambda s: 12 * int(s[1:]) + {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[s[0]]
    ratio = {0: 1, 1: sympy.Rational(16, 15), 2: sympy.Rational(9, 8), 3: sympy.Rational(6, 5),
             4: sympy.Rational(5, 4), 5: sympy.Rational(4, 3), 6: sympy.Rational(45, 32), 7: sympy.Rational(3, 2),
             8: sympy.Rational(8, 5), 9: sympy.Rational(5, 3), 10: sympy.Rational(16, 9), 11: sympy.Rational(15, 8),
             12: 2}
    f = sympy.Integer(data["f0"])
    for a, b in zip(data["notes"], data["notes"][1:]):
        d = midi(b) - midi(a)
        f = f * ratio[d] if d >= 0 else f / ratio[-d]
    return str(f)


# ---------- 19. multiplicative orders ----------

def make_order(k, rng):
    pairs, ords = [], []
    while len(pairs) < k:
        m = rng.randint(15, 250)
        a = rng.randint(2, m - 2)
        if math.gcd(a, m) != 1 or (a, m) in pairs:
            continue
        x, o = a % m, 1
        while x != 1:
            x, o = x * a % m, o + 1
        if o >= 3:
            pairs.append((a, m))
            ords.append(o)
    lines = "\n".join(f"{i}. a = {a}, m = {m}" for i, (a, m) in enumerate(pairs, 1))
    prompt = ("The multiplicative order of a modulo m (for gcd(a, m) = 1) is the smallest positive integer k such "
              f"that a^k ≡ 1 (mod m). Find the multiplicative order for each of the following pairs:\n{lines}\n\n"
              f"Give the {k} orders as integers, separated by commas, in the order given.")
    return dict(prompt=prompt, answer=",".join(map(str, ords)), check={"type": "list", "rel_tol": 0},
                data={"pairs": pairs})


def verify_order(data):
    return ",".join(str(n_order(a, m)) for a, m in data["pairs"])


# ---------- 20. stack machine ----------

def make_stack(n, rng):
    while True:
        prog, st = [], []
        while len(prog) < n:
            opts = []
            if len(st) < 6:
                opts += ["PUSH"] * 3 + ["DUP"] * bool(st)
            if len(st) >= 2:
                opts += ["ADD", "SUB", "MUL", "SWAP", "OVER", "POP"] + ["ADD", "SUB"]
            if len(prog) >= n - 3 and len(st) >= 2:
                opts = [o for o in opts if o in ("ADD", "SUB", "MUL")]
            op = rng.choice(opts)
            if op == "PUSH":
                v = rng.randint(1, 20)
                st.append(v)
                prog.append(f"PUSH {v}")
                continue
            if op == "MUL" and abs(st[-1] * st[-2]) > 5000:
                continue
            if op in ("ADD", "SUB", "MUL"):
                b, a = st.pop(), st.pop()
                st.append(a + b if op == "ADD" else a - b if op == "SUB" else a * b)
            elif op == "DUP":
                st.append(st[-1])
            elif op == "SWAP":
                st[-1], st[-2] = st[-2], st[-1]
            elif op == "OVER":
                st.append(st[-2])
            elif op == "POP":
                st.pop()
            prog.append(op)
        if st and st[-1] != 0 and max(map(abs, st)) < 10 ** 6:
            break
    lines = "\n".join(f"{i:3d}. {ins}" for i, ins in enumerate(prog, 1))
    prompt = ("A stack machine holds a stack of integers, initially empty. It executes the instructions below once, "
              "in order (there are no jumps). Instruction semantics:\n"
              "  PUSH k — push the integer k;\n"
              "  ADD — pop x (the top), pop y, push y + x;\n"
              "  SUB — pop x (the top), pop y, push y − x;\n"
              "  MUL — pop x (the top), pop y, push y × x;\n"
              "  DUP — push a copy of the top element;\n"
              "  OVER — push a copy of the element just below the top;\n"
              "  SWAP — exchange the two top elements;\n"
              "  POP — remove the top element.\n"
              "Every instruction is valid at the point where it is executed (the stack never underflows).\n\n"
              f"Program ({n} instructions):\n{lines}\n\n"
              "What integer is on top of the stack after the last instruction? Give it as an integer.")
    return dict(prompt=prompt, answer=str(st[-1]), check={"type": "num", "rel_tol": 0}, data={"prog": prog})


def verify_stack(data):
    import operator
    s = deque()                                                      # top of the stack at the left end
    binop = {"ADD": operator.add, "SUB": operator.sub, "MUL": operator.mul}
    for ins in data["prog"]:
        w = ins.split()
        if w[0] == "PUSH":
            s.appendleft(int(w[1]))
        elif w[0] in binop:
            x = s.popleft()
            s.appendleft(binop[w[0]](s.popleft(), x))
        elif w[0] == "DUP":
            s.appendleft(s[0])
        elif w[0] == "OVER":
            s.appendleft(s[1])
        elif w[0] == "SWAP":
            x, y = s.popleft(), s.popleft()
            s.extendleft([x, y])
        elif w[0] == "POP":
            s.popleft()
    return str(s[0])


# ---------- 21. area of a simple lattice polygon ----------

def make_polygon(n, rng):
    R = 10 + n // 2
    while True:
        pts = []
        for i in range(n):
            ang = 2 * math.pi * (i + rng.uniform(0.15, 0.85)) / n
            r = rng.uniform(0.35, 1.0) * R
            pts.append((round(r * math.cos(ang)), round(r * math.sin(ang))))
        cr = [pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)]
        angs = [math.atan2(y, x) % (2 * math.pi) for x, y in pts]
        if (len(set(pts)) == n and (0, 0) not in pts and all(c > 0 for c in cr)
                and all(angs[i] < angs[i + 1] for i in range(n - 1))):
            break
    dx, dy = rng.randint(-5, 30), rng.randint(-5, 30)
    pts = [(x + dx, y + dy) for x, y in pts]
    shift = rng.randrange(n)
    pts = pts[shift:] + pts[:shift]
    area = Fraction(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1] for i in range(n)), 2)
    lines = "\n".join(f"  P{i}: ({x}, {y})" for i, (x, y) in enumerate(pts, 1))
    prompt = (f"A simple (non-self-intersecting) polygon has {n} vertices with integer coordinates, listed in "
              f"counter-clockwise order (P{n} is joined back to P1):\n{lines}\n\n"
              "Find the area of the polygon. The exact area is an integer or a half-integer; give it exactly "
              "(e.g. 1234 or 1234.5).")
    ans = str(area.numerator) if area.denominator == 1 else f"{area.numerator // 2}.5"
    return dict(prompt=prompt, answer=ans, check={"type": "num", "rel_tol": 0}, data={"pts": pts})


def verify_polygon(data):
    a = sympy.Polygon(*[sympy.Point(x, y) for x, y in data["pts"]]).area
    a = abs(sympy.Rational(a))
    return str(a.p // 2) + ".5" if a.q == 2 else str(a)


GENS = [
    Gen("r1_crypto_dh", "diffie_hellman", "cryptography", "Диффи–Хеллман: общий ключ через дискретный логарифм",
        "prime modulus p", grid=[53, 101, 151, 199, 251, 307, 401, 503, 601, 701, 809, 907, 1009, 1201, 1409, 1601,
                                 1801, 2003, 2503, 3001],
        probe=(401, 1009), make=make_dh, verify=verify_dh),
    Gen("r1_crypto_vigenere", "vigenere_decrypt", "cryptography", "Расшифровка шифра Виженера по ключу",
        "ciphertext length, letters", grid=list(range(20, 1001, 20)), probe=(100, 260), make=make_vigenere,
        verify=verify_vigenere),
    Gen("r1_crypto_rsa", "rsa_decrypt", "cryptography", "Учебный RSA: расшифровка", "number of ciphertexts",
        grid=list(range(1, 31)), probe=(3, 8), make=make_rsa, verify=verify_rsa),
    Gen("r1_cal_weekday", "weekday", "calendar", "День недели для григорианских дат", "number of dates",
        grid=list(range(1, 121)), probe=(5, 15), make=make_weekday, verify=verify_weekday),
    Gen("r1_cal_roman", "roman_dates", "calendar", "Римские даты", "number of dates",
        grid=list(range(2, 161)), probe=(6, 16), make=make_roman, verify=verify_roman),
    Gen("r1_music_just", "just_intonation", "music", "Чистый строй: частота последней ноты", "number of notes",
        grid=list(range(5, 121)), probe=(15, 40), make=make_just, verify=verify_just),
    Gen("r1_nt_order", "multiplicative_order", "number_theory", "Мультипликативный порядок по модулю",
        "number of (a, m) pairs", grid=list(range(1, 31)), probe=(3, 8), make=make_order, verify=verify_order),
    Gen("r1_cs_stack", "stack_machine", "computer_science", "Исполнение программы стековой машины",
        "number of instructions", grid=list(range(10, 1201, 10)), probe=(50, 130), make=make_stack,
        verify=verify_stack),
    Gen("r1_geom_polygon", "polygon_area", "geometry", "Площадь многоугольника с целыми вершинами",
        "number of vertices", grid=list(range(4, 201)), probe=(24, 60), make=make_polygon, verify=verify_polygon),
]
