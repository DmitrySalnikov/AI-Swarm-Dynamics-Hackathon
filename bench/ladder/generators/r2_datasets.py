"""Rung 2 (physics sections) from datasets: bundles of p TheoremQA / UGPhysics problems of one subtopic.

The pools are compact files built by fetch_sources.py (bench/ladder/sources/*.json). A bundle is a fixed-seed
sample of p problems from one pool; the answer is the list of the dataset answers in problem order. Dataset answers
cannot be recomputed, so verify() only re-reads them from the source record (TheoremQA: by id from the pool file;
UGPhysics: re-parses the original LaTeX `\\boxed{...}` answer by a second, independent converter).
"""
import json
import re
from fractions import Fraction
from functools import cache
from pathlib import Path

from common import Gen

SOURCES = Path(__file__).resolve().parents[1] / "sources"
GRID, PROBE = list(range(2, 9)), (2, 4)
TQA_TOL, UGP_TOL = 0.04, 0.01     # TheoremQA's own evaluation uses 4% (number_utils.within_eps); UGPhysics 1%


@cache
def load(name):
    return json.loads((SOURCES / name).read_text())


def bundle_prompt(kind, label, items):
    """items: (problem text, unit line) pairs."""
    parts = [f"Below are {len(items)} independent {kind} problems on {label}. Solve each of them.\n"]
    for i, (text, unit) in enumerate(items, 1):
        parts.append(f"Problem {i}.\n{text}\n[{unit}]\n")
    parts.append(f"Give the final answer as a comma-separated list of {len(items)} numbers, one per problem in "
                 "order (Problem 1 first). Write each number as a plain decimal or in e-notation (e.g. 6.2e-5), "
                 "without units and without thousands separators.")
    return "\n".join(parts)


# ---------- TheoremQA ----------

TQA_LABELS = {"mechanics": "mechanics (kinematics, dynamics, gravitation, fluids)",
              "modern": "modern physics (atomic, nuclear, particle physics, relativity)",
              "thermal": "thermal, statistical and condensed-matter physics"}


def make_tqa(group):
    def make(p, rng):
        pool = load("theoremqa_physics.json")["pools"][group]
        picks = rng.sample(pool, p)
        items = [(x["question"], f"Answer unit: {x['unit']}") for x in picks]
        return dict(prompt=bundle_prompt("physics", TQA_LABELS[group], items),
                    answer=",".join(str(x["answer"]) for x in picks),
                    check={"type": "list", "rel_tol": TQA_TOL},
                    data={"dataset": "TheoremQA", "group": group, "ids": [x["id"] for x in picks]})
    return make


def verify_tqa(data):
    by_id = {x["id"]: x for x in load("theoremqa_physics.json")["pools"][data["group"]]}
    return ",".join(str(by_id[i]["answer"]) for i in data["ids"])


# ---------- UGPhysics ----------

UGP_LABELS = {"thermodynamics": "thermodynamics", "quantum": "quantum mechanics",
              "optics": "optics (wave and geometrical optics)",
              "electromagnetism": "electromagnetism (classical electromagnetism and electrodynamics)",
              "atomic": "atomic physics"}


def ugp_unit(x):
    return f"Answer unit: ${x['unit']}$" if x["unit"] else "The answer is a pure number (no unit)"


def make_ugp(group):
    def make(p, rng):
        pool = load("ugphysics.json")["pools"][group]
        picks = rng.sample(pool, p)
        items = [(x["problem"], ugp_unit(x)) for x in picks]
        return dict(prompt=bundle_prompt("undergraduate physics", UGP_LABELS[group], items),
                    answer=",".join(x["answer"] for x in picks),
                    check={"type": "list", "rel_tol": UGP_TOL},
                    data={"dataset": "UGPhysics", "group": group, "indices": [x["index"] for x in picks]})
    return make


def _latex_value(s):
    """Second converter for the boxed LaTeX answers: evaluate the number with Fraction arithmetic."""
    body = s[s.index("\\boxed{") + 7:s.rindex("}")]
    body = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", body).replace("\\times", "*")
    body = re.sub(r"\^\{?(-?\d+)\}?", r"**\1", body).replace(" ", "").replace("\\,", "").replace("{,}", "")
    assert re.fullmatch(r"[-+*/().\d]+", body), body
    tokens = re.findall(r"\*\*|\d+\.?\d*|[-+*/()]", body)
    pos = 0

    def expr():
        nonlocal pos
        v = term()
        while pos < len(tokens) and tokens[pos] in "+-":
            op, pos = tokens[pos], pos + 1
            v = v + term() if op == "+" else v - term()
        return v

    def term():
        nonlocal pos
        v = power()
        while pos < len(tokens) and tokens[pos] in ("*", "/"):
            op, pos = tokens[pos], pos + 1
            v = v * power() if op == "*" else v / power()
        return v

    def power():
        nonlocal pos
        v = atom()
        if pos < len(tokens) and tokens[pos] == "**":
            pos += 1
            sign = -1 if tokens[pos] == "-" else 1
            pos += tokens[pos] in "+-"
            v = v ** (sign * int(tokens[pos]))
            pos += 1
        return v

    def atom():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        if t == "-":
            return -atom()
        if t == "(":
            v = expr()
            pos += 1
            return v
        return Fraction(t)

    return expr()


def verify_ugp(data):
    by_index = {x["index"]: x for x in load("ugphysics.json")["pools"][data["group"]]}
    vals = [_latex_value(by_index[i]["answer_latex"]) for i in data["indices"]]
    return ",".join(f"{v.numerator}/{v.denominator}" for v in vals)


# ---------- registry ----------

def _src(name, ids, url, lic):
    return {"dataset": name, "id": ids, "url": url, "license": lic}


TQA_URL = "https://huggingface.co/datasets/TIGER-Lab/TheoremQA"
UGP_URL = "https://huggingface.co/datasets/UGPhysics/ugphysics"
TQA_SUB = {"mechanics": "Kinetics, Classic/Celestial/Fluid mechanics", "modern": "Atomic physics, Quantum, Particle, "
           "Relativity", "thermal": "Thermodynamics, Statistical physics, Condensed matter physics"}
UGP_SUB = {"thermodynamics": "Thermodynamics", "quantum": "QuantumMechanics", "optics": "WaveOptics, GeometricalOptics",
           "electromagnetism": "ClassicalElectromagnetism, Electrodynamics", "atomic": "AtomicPhysics"}


def _gen(name, domain, title, make, verify, src):
    return Gen(name, domain, "physics", title, "number of problems in the bundle p", grid=GRID, probe=PROBE,
               make=make, verify=verify, rungs=(2,), source=src)


GENS = [
    _gen("r2d_tqa_mechanics", "mechanics", "Пакет задач TheoremQA: механика", make_tqa("mechanics"), verify_tqa,
         _src("TheoremQA", f"physics, subfields {TQA_SUB['mechanics']}; problem ids in task data", TQA_URL, "MIT")),
    _gen("r2d_tqa_modern", "modern_physics", "Пакет задач TheoremQA: атомная, ядерная физика и СТО",
         make_tqa("modern"), verify_tqa,
         _src("TheoremQA", f"physics, subfields {TQA_SUB['modern']}; problem ids in task data", TQA_URL, "MIT")),
    _gen("r2d_tqa_thermal", "thermal_condensed", "Пакет задач TheoremQA: тепловая и статистическая физика",
         make_tqa("thermal"), verify_tqa,
         _src("TheoremQA", f"physics, subfields {TQA_SUB['thermal']}; problem ids in task data", TQA_URL, "MIT")),
] + [
    _gen(f"r2d_ugp_{g}", dom, title, make_ugp(g), verify_ugp,
         _src("UGPhysics", f"subject {UGP_SUB[g]}, EN, NV answers; problem indices in task data", UGP_URL,
              "CC BY-NC-SA 4.0"))
    for g, dom, title in [
        ("thermodynamics", "thermodynamics", "Пакет задач UGPhysics: термодинамика"),
        ("quantum", "quantum_mechanics", "Пакет задач UGPhysics: квантовая механика"),
        ("optics", "optics", "Пакет задач UGPhysics: оптика"),
        ("electromagnetism", "electromagnetism", "Пакет задач UGPhysics: электромагнетизм"),
        ("atomic", "atomic_physics", "Пакет задач UGPhysics: атомная физика"),
    ]
]
