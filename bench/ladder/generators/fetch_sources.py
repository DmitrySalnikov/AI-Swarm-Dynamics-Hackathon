"""Download dataset sources into bench/ladder/.cache/ and build compact pools in bench/ladder/sources/*.json.

  uv run -q --with pyarrow python -I bench/ladder/generators/fetch_sources.py download
  uv run -q --with pyarrow python -I bench/ladder/generators/fetch_sources.py build

Downloads are treated as data only: each source goes to its own directory under .cache/ and nothing from it is
imported. The one exception is FinanceMath `python_solution`: plain-arithmetic snippets are run by `run_snippet`
(AST whitelist, restricted builtins, separate isolated interpreter, timeout, no network / no file writes).
This module is not a generator; GENS is empty so that selftest.py can import it.
"""
import ast
import glob
import json
import random
import re
import subprocess
import sys
import tarfile
import urllib.request
from fractions import Fraction
from pathlib import Path

GENS = []

LADDER = Path(__file__).resolve().parents[1]
CACHE, OUT = LADDER / ".cache", LADDER / "sources"

TQA_URL = "https://huggingface.co/datasets/TIGER-Lab/TheoremQA/resolve/main/data/test-00000-of-00001.parquet"
TQA_GH = "https://raw.githubusercontent.com/wenhuchen/TheoremQA/main/"   # same 800 problems + id/field/subfield
UGP_URL = "https://huggingface.co/datasets/UGPhysics/ugphysics/resolve/main/{}/en.jsonl"
UGP_SUBJECTS = ["Thermodynamics", "QuantumMechanics", "GeometricalOptics", "WaveOptics", "Electrodynamics",
                "ClassicalElectromagnetism", "AtomicPhysics", "Relativity"]
SARA_URL = "https://nlp.jhu.edu/law/sara/sara.tar.gz"
FM_URL = "https://raw.githubusercontent.com/yale-nlp/FinanceMath/main/data/{}.json"


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print("get", url)
        urllib.request.urlretrieve(url, dest)


def download():
    fetch(TQA_URL, CACHE / "theoremqa" / "test.parquet")
    fetch(TQA_GH + "theoremqa_test.json", CACHE / "theoremqa_gh" / "theoremqa_test.json")
    fetch(TQA_GH + "LICENSE", CACHE / "theoremqa_gh" / "LICENSE")
    for s in UGP_SUBJECTS:
        fetch(UGP_URL.format(s), CACHE / "ugphysics" / f"{s}.en.jsonl")
    fetch(SARA_URL, CACHE / "sara_tar" / "sara.tar.gz")
    out = CACHE / "sara"
    if not out.exists():
        with tarfile.open(CACHE / "sara_tar" / "sara.tar.gz") as t:
            t.extractall(out, filter="data")
    for part in ("validation", "test"):
        fetch(FM_URL.format(part), CACHE / "financemath" / f"{part}.json")


def dump(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


# ---------- TheoremQA: physics problems grouped by the dataset's own `subfield` ----------

TQA_GROUPS = {
    "mechanics": ["Kinetics", "Classic mechanics", "Celestial mechanics", "Fluid mechanics"],
    "modern": ["Atomic physics", "Quantum", "Particle", "Relativity"],
    "thermal": ["Thermodynamics", "Statistical physics", "Condensed matter physics"],
}
# Answer unit / which quantity is asked, written by us after reading each problem (the dataset has no unit field).
TQA_UNITS = {
    "wenhuchen/Fluid_mechanics2.json": "cubic feet (ft^3)",
    "panlu/energy_conservation1.json": "units of 10^4 m/s",
    "panlu/angular_frequency3.json": "Hz",
    "panlu/center_of_gravity2.json": "m",
    "wenhuchen/Fluid_mechanics1.json": "m",
    "panlu/friction1.json": "dimensionless; give the coefficient of kinetic friction",
    "panlu/work_energy1.json": "m/s; give the signed x-velocity",
    "wenhuchen/kepler's_law3.json": "Earth days",
    "panlu/black_hole1.json": "give the number X",
    "panlu/angular_frequency1.json": "units of 10^7 rad/s; give the angular frequency",
    "wenhuchen/kinetics1.json": "degrees",
    "xinyi/momentum.json": "dimensionless fraction",
    "xinyi/rotation.json": "cm",
    "xinyi/angular_momentum.json": "rad/s",
    "panlu/angular_frequency2.json": "s",
    "panlu/rigid-body3.json": "rad/s",
    "panlu/kepler’s_third_law1.json": "units of 10^11 m",
    "panlu/density1.json": "units of 10^5 N; give the weight of the equal volume of water",
    "panlu/rigid-body2.json": "m/s^2",
    "panlu/physical_pendulum1.json": "m/s",
    "wenhuchen/kinetics4.json": "m/s (lengths in m, g in m/s^2)",
    "xinyi/work_energy_theorem.json": "m/s",
    "xinyi/potential_energy.json": "m",
    "xinyi/newtons_laws_3.json": "dimensionless",
    "wenhuchen/kepler's_law1.json": "days",
    "panlu/fluid_pressure1.json": "units of 10^5 Pa",
    "wenhuchen/kinetics3.json": "days",
    "panlu/pojectile_motion2.json": "m",
    "panlu/gravitational_force1.json": "give the number X",
    "panlu/fluid_flow1.json": "m/s",
    "xinyi/newtons_laws_1.json": "m/s^2",
    "panlu/center_of_gravity1.json": "kg",
    "wenhuchen/kepler's_law2.json": "give the number X",
    "panlu/uniform_circular_motion1.json": "m",
    "panlu/force_and_power1.json": "hp",
    "panlu/circular_orbit1.json": "units of 10^10 J",
    "wenhuchen/kinetics2.json": "years",
    "panlu/center_of_mass1.json": "m",
    "panlu/uniform_circular_motion2.json": "m/s^2",
    "panlu/young’s_modulus1.json": "mm; give the elongation",
    "panlu/kepler’s_third_law2.json": "give the number X",
    "panlu/gravitational_force2.json": "m/s^2",
    "tonyxia/atom1.json": "dimensionless fraction",
    "tonyxia/relativity3.json": "MeV",
    "tonyxia/particle2.json": "decays per day",
    "tonyxia/atom3.json": "dimensionless ratio",
    "wenhuchen/relativity2.json": "km",
    "tonyxia/particle5.json": "fraction of the speed of light (dimensionless)",
    "tonyxia/particle3.json": "MeV",
    "tonyxia/nuclear6.json": "give the number X",
    "tonyxia/nuclear3.json": "MeV",
    "tonyxia/particle1.json": "decays per year",
    "wenhuchen/relativity3.json": "µs",
    "tonyxia/atom4.json": "N/m",
    "tonyxia/nuclear1.json": "dimensionless probability",
    "tonyxia/relativity4.json": "V",
    "panlu/molecule_vibration1.json": "units of 10^11 Hz",
    "tonyxia/particle4.json": "TeV",
    "tonyxia/particle6.json": "GeV",
    "tonyxia/relativity1.json": "picoseconds lost per second, as a positive number",
    "tonyxia/relativity2.json": "m/s",
    "tonyxia/statisticalphysics1.json": "eV (take room temperature as 293 K)",
    "panlu/molar_heat_capacity2.json": "g",
    "tonyxia/statisticalphysics4.json": "m/s (take T = 293 K)",
    "tonyxia/semiconductor6.json": "dimensionless fraction",
    "panlu/molar_heat_capacity1.json": "°C",
    "tonyxia/photoelectric2.json": "V",
    "tonyxia/semiconductor3.json": "nm",
    "tonyxia/statisticalphysics2.json": "J",
    "tonyxia/statisticalphysics6.json": "kelvin (the 'in eV' in the question is a typo: give the Fermi temperature in K)",
    "tonyxia/statisticalphysics3.json": "m/s (take T = 293 K)",
    "panlu/thermal_stress1.json": "units of 10^4 N; give the force on each wall, negative for compression",
    "tonyxia/photoelectric1.json": "eV",
    "tonyxia/semiconductor2.json": "nm",
    "tonyxia/semiconductor5.json": "dimensionless fraction",
    "panlu/volume_thermal_expansion1.json": "cm^3",
    "tonyxia/semiconductor1.json": "V/K",
    "tonyxia/statisticalphysics5.json": "eV",
    "panlu/linear_expansion1.json": "m",
}
TQA_EXCLUDED = {
    "panlu/liquid_compressibility1.json": "reference -0.8 contradicts the data (dV = -8e-4 m^3 = -8 in the stated unit)",
    "panlu/rigid-body1.json": "the ladder angle is missing from the text",
    "panlu/pojectile_motion1.json": "numbers are missing from the text",
    "tonyxia/nuclear2.json": "1-significant-digit reference 0.2; 1-exp(-n*sigma*x) gives 0.18",
    "tonyxia/quantum2.json": "uncertainty-principle convention (hbar vs hbar/2) decides the answer",
    "tonyxia/quantum3.json": "uncertainty-principle convention decides the answer",
    "tonyxia/quantum4.json": "uncertainty-principle convention decides the answer",
    "tonyxia/atom2.json": "uncertainty-principle convention decides the answer",
    "wenhuchen/relativity1.json": "asks for two values; the reference matches the second",
    "tonyxia/nuclear4.json": "'106-kg' is a garbled 10^6 kg",
    "tonyxia/nuclear5.json": "'106-kg' is a garbled 10^6 kg",
    "tonyxia/quantum1.json": "classical-estimate assumptions (absorbing area) unspecified",
    "tonyxia/quantum5.json": "tunnelling approximation unspecified; reference depends on it",
    "tonyxia/semiconductor4.json": "exp(59) is too sensitive to kT for a 1-significant-digit reference",
}


def build_theoremqa():
    import pyarrow.parquet as pq
    gh = json.loads((CACHE / "theoremqa_gh" / "theoremqa_test.json").read_text())
    hf = {r["Question"].strip(): r["Answer"] for r in pq.read_table(CACHE / "theoremqa" / "test.parquet").to_pylist()}
    pools, mismatch = {g: [] for g in TQA_GROUPS}, []
    for x in gh:
        group = next((g for g, subs in TQA_GROUPS.items() if x["field"] == "Physics" and x["subfield"] in subs), None)
        if group is None or x["Answer_type"] not in ("float", "integer") or x["Picture"] or x["id"] in TQA_EXCLUDED:
            continue
        q = x["Question"].strip()
        if q not in hf or float(hf[q]) != float(x["Answer"]):
            mismatch.append(x["id"])
            continue
        pools[group].append({"id": x["id"], "subfield": x["subfield"], "question": q, "answer": hf[q],
                             "unit": TQA_UNITS[x["id"]]})
    dump("theoremqa_physics.json", {
        "dataset": "TheoremQA", "url": "https://huggingface.co/datasets/TIGER-Lab/TheoremQA",
        "metadata_url": "https://github.com/wenhuchen/TheoremQA (theoremqa_test.json: id, field, subfield)",
        "license": "MIT", "license_text": (CACHE / "theoremqa_gh" / "LICENSE").read_text(),
        "note": "question/answer from the HF test split; `unit` written by the ladder authors; physics only, "
                "numeric answers, no pictures; groups by the dataset's subfield",
        "groups": TQA_GROUPS, "excluded": TQA_EXCLUDED, "pools": pools})
    print("theoremqa:", {g: len(v) for g, v in pools.items()}, "HF/GH mismatches:", mismatch)


# ---------- UGPhysics: English, single numerical-value (NV) answers ----------

UGP_GROUPS = {
    "thermodynamics": ["Thermodynamics"],
    "quantum": ["QuantumMechanics"],
    "optics": ["WaveOptics", "GeometricalOptics"],
    "electromagnetism": ["ClassicalElectromagnetism", "Electrodynamics"],
    "atomic": ["AtomicPhysics"],
}
UGP_POOL = 60          # problems kept per group (fixed-seed sample of the eligible ones)
UGP_MAXLEN = 1500      # characters


def latex_number(s):
    """'\\boxed{1.6 \\times 10^{17}}' -> '1.6e17'; only plain decimals, a×10^k and integer fractions; else None."""
    m = re.fullmatch(r"\s*(?:\\\[)?\s*\\boxed\{(.*)\}\s*(?:\\\])?\s*", s, re.S)
    if not m:
        return None
    b = m[1].replace(" ", "").replace("\\,", "").replace("{,}", "")
    if mm := re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:\\times10\^\{?(-?\d+)\}?)?", b):
        return mm[1] + (f"e{int(mm[2])}" if mm[2] else "")
    if mm := re.fullmatch(r"(-?)\\[dt]?frac\{(\d+)\}\{(\d+)\}", b):
        return f"{mm[1]}{mm[2]}/{mm[3]}"
    return None


def sig_digits(num):
    mant = num.split("e")[0].lstrip("-")
    if "/" in mant:
        return 9
    digits = mant.replace(".", "").lstrip("0")
    return len(digits if "." in mant else digits.rstrip("0"))


def build_ugphysics():
    pools, stats = {}, {}
    bad = re.compile(r"figure|fig\.|diagram|shown|\(1\)|\(a\)|\(i\)|\(2\)|estimat|rough|guess|order of magnitude"
                     r"|already|previous|preceding|above|part \(|\\text\{\s*\}|\\mathrm\{\s*\}", re.I)
    pure = re.compile(r"how many|ratio|by what factor|number of", re.I)
    for g, subjects in UGP_GROUPS.items():
        ok = []
        for s in subjects:
            for line in (CACHE / "ugphysics" / f"{s}.en.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r["answer_type"] != "NV" or r["is_multiple_answer"] or r["language"] != "EN":
                    continue
                num = latex_number(r["answers"])
                if (num is None or Fraction(num) == 0 or sig_digits(num) < 2 or len(r["problem"]) > UGP_MAXLEN
                        or bad.search(r["problem"])):
                    continue
                unit = (r["unit"] or "").replace("\\text{}", "").strip()
                if not unit and not pure.search(r["problem"]):   # unit unknown: keep only pure numbers
                    continue
                if unit.endswith(("/", "\\cdot")) or re.search(r"\\(text|mathrm)\{\s*\}", unit):  # damaged unit
                    continue
                ok.append({"index": r["index"], "subject": r["subject"], "topic": r["topic"],
                           "problem": r["problem"].strip(), "answer": num, "answer_latex": r["answers"],
                           "unit": unit or None})
        ok.sort(key=lambda r: r["index"])
        pools[g] = sorted(random.Random(f"ugphysics|{g}").sample(ok, min(UGP_POOL, len(ok))), key=lambda r: r["index"])
        stats[g] = (len(ok), len(pools[g]))
    dump("ugphysics.json", {
        "dataset": "UGPhysics", "url": "https://huggingface.co/datasets/UGPhysics/ugphysics",
        "license": "CC BY-NC-SA 4.0",
        "attribution": "Xu et al., UGPhysics: A Comprehensive Benchmark for Undergraduate Physics Reasoning with "
                       "Large Language Models (2025); problems copied unchanged, answers converted from LaTeX",
        "filter": "EN, answer_type NV, single answer, plain numeric answer with >= 2 significant digits, nonzero, "
                  "no figure / multi-part / estimate / back-reference wording, no damaged (empty) LaTeX units, unit "
                  f"given (or a pure number: count/ratio/factor), <= {UGP_MAXLEN} chars; fixed-seed sample of "
                  f"{UGP_POOL} per group",
        "groups": UGP_GROUPS, "pools": pools})
    print("ugphysics (eligible, kept):", stats)


# ---------- SARA v1: numeric tax cases ----------

SARA_SECTIONS = ["section1", "section2", "section63", "section68", "section151", "section152", "section3301",
                 "section3306", "section7703"]


def build_sara():
    root = CACHE / "sara" / "sara"
    cases, skipped = [], {}
    for f in sorted(glob.glob(str(root / "cases" / "tax_case_*.pl")), key=lambda p: int(re.findall(r"\d+", p)[-1])):
        name, t = Path(f).stem, Path(f).read_text()
        text = re.search(r"% Text\n(.*?)\n\n% Question", t, re.S)[1]
        text = "\n".join(l[2:] if l.startswith("% ") else l.lstrip("%") for l in text.splitlines()).strip()
        q = re.search(r"% Question\n% (How much tax does (\w+) have to pay in (\d+)\?) \$(\d+)", t)
        test = re.search(r":- tax\((\w+),(\d+),(\d+)\)\.", t)
        if not q or not test:
            skipped[name] = "unparsed"
            continue
        if q[2].lower() != test[1] or q[3] != test[2] or q[4] != test[3]:
            skipped[name] = f"question {q[0]!r} disagrees with the Prolog test {test[0]!r}"
            continue
        cases.append({"case": name, "text": text, "question": q[1], "answer": int(q[4]),
                      "prolog_test": test[0], "split": None})
    splits = {s: set((root / "splits" / s).read_text().split()) for s in ("train", "test")}
    for c in cases:
        c["split"] = next(s for s, names in splits.items() if c["case"] in names)
    dump("sara.json", {
        "dataset": "SARA v1 (StAtutory Reasoning Assessment)", "url": "https://nlp.jhu.edu/law/sara/",
        "license": "MIT-style (sara/LICENSE)", "license_text": (root / "LICENSE").read_text(),
        "attribution": "Holzenberger, Blair-Stanek, Van Durme, A Dataset for Statutory Reasoning in Tax Law "
                       "Entailment and Question Answering (2020)",
        "statutes": {s: (root / "statutes" / "source" / s).read_text().strip() for s in SARA_SECTIONS},
        "skipped": skipped, "cases": cases})
    print("sara:", len(cases), "cases, skipped", skipped)


# ---------- FinanceMath (reserve): sandboxed recomputation of python_solution ----------

SAFE_BUILTINS = ["abs", "min", "max", "sum", "range", "len", "round", "int", "float", "pow", "zip", "enumerate",
                 "sorted", "list", "dict", "tuple", "bool", "map", "filter", "reversed", "any", "all"]
RUNNER = r"""
import ast, json, math, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
src = sys.stdin.read()
tree = ast.parse(src)
for n in ast.walk(tree):
    if isinstance(n, ast.Import):
        assert all(a.name == "math" and a.asname is None for a in n.names), "import"
    elif isinstance(n, ast.ImportFrom):
        assert n.module == "math", "import"
    elif isinstance(n, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef, ast.With, ast.Try,
                        ast.Raise, ast.Delete, ast.Await, ast.Yield, ast.YieldFrom)):
        raise AssertionError(type(n).__name__)
    elif isinstance(n, ast.Attribute):
        assert not n.attr.startswith("_"), "attr"
    elif isinstance(n, ast.Name):
        assert not n.id.startswith("__"), "name"
safe = {k: __builtins__.__dict__[k] if hasattr(__builtins__, "__dict__") else __builtins__[k] for k in NAMES}
def imp(name, *a, **k):
    assert name == "math"
    return math
safe["__import__"] = imp
env = {"__builtins__": safe}
exec(compile(tree, "<snippet>", "exec"), env)
print(json.dumps(float(env["solution"]())))
""".replace("NAMES", repr(SAFE_BUILTINS))


def run_snippet(src, timeout=10):
    """Run a FinanceMath `def solution()` snippet; returns float or None (rejected / failed)."""
    cmd = [sys.executable, "-I", "-S", "-c", RUNNER]
    if Path("/usr/bin/sandbox-exec").exists():
        cmd = ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)(deny file-write*)"] + cmd
    try:
        r = subprocess.run(cmd, input=src, capture_output=True, text=True, timeout=timeout, env={}, cwd="/")
    except subprocess.TimeoutExpired:
        return None
    return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None


FM_TOPIC = "Quantitative Analysis & Valuation"


def build_financemath():
    rows = [r for part in ("validation", "test")
            for r in json.loads((CACHE / "financemath" / f"{part}.json").read_text())]
    keep, stats = [], {"topic": 0, "rejected_or_failed": 0, "mismatch": 0, "small": 0}
    for r in rows:
        if r["topic"] != FM_TOPIC:
            continue
        stats["topic"] += 1
        gt = float(r["ground_truth"])
        if abs(gt) < 1:      # rates given as decimals vs percent are ambiguous; keep |answer| >= 1
            stats["small"] += 1
            continue
        v = run_snippet(r["python_solution"])
        if v is None:
            stats["rejected_or_failed"] += 1
        elif abs(v - gt) > 0.005 * abs(gt):
            stats["mismatch"] += 1
        else:
            tables = r["tables"] or []
            if isinstance(tables, str):     # some records hold the list as its Python repr
                tables = ast.literal_eval(tables)
            keep.append({"question_id": r["question_id"], "question": r["question"], "tables": tables,
                         "ground_truth": str(r["ground_truth"]), "python_solution": r["python_solution"]})
    dump("financemath.json", {
        "dataset": "FinanceMath", "url": "https://github.com/yale-nlp/FinanceMath (data/)",
        "license": "MIT (per the HF dataset card yale-nlp/FinanceMath; no LICENSE file on GitHub)",
        "attribution": "Zhao et al., FinanceMath: Knowledge-Intensive Math Reasoning in Finance Domains (ACL 2024)",
        "filter": f"topic {FM_TOPIC!r}, |ground_truth| >= 1, python_solution passes the AST whitelist and its "
                  "sandboxed result is within 0.5% of ground_truth", "stats": stats, "pool": keep})
    print("financemath:", len(keep), stats)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "download":
        download()
    elif cmd == "build":
        for f in (build_theoremqa, build_ugphysics, build_sara, build_financemath):
            if len(sys.argv) < 3 or f.__name__.endswith(sys.argv[2]):
                f()
