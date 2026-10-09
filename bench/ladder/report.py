"""Builds bench/ladder/REPORT_ladder.md from the run files, ledgers and tables (every number comes from here).

  python bench/ladder/report.py      # after calibrate.py table, select_tasks.py stats/embed/final
"""
import glob
import json
import statistics as st
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "embedding_eval"))
from gen_runs import extract  # noqa: E402

LADDER = ROOT / "bench" / "ladder"
RUNS = ROOT / "embedding_eval" / "runs" / "ladder" / "deepseek-v4-flash"
EMB_LEDGER = ROOT / "embedding_eval" / "cache" / "ladder" / "spend.jsonl"
PY = ["uv", "run", "-q", "--with", "sympy", "--with", "numpy", "--with", "scipy", "--with", "reasoning-gym==0.1.19",
      "python", "-W", "ignore"]


def run(*args):
    return subprocess.run(PY + list(args), cwd=ROOT, capture_output=True, text=True, check=True).stdout


def ledger():
    rows = [json.loads(l) for l in (RUNS / "spend.jsonl").read_text().splitlines() if l.strip()]
    calib = [r for r in rows if r["id"].startswith("ladder_calib/")]
    main = [r for r in rows if not r["id"].startswith("ladder_calib/")]
    emb = [json.loads(l) for l in EMB_LEDGER.read_text().splitlines() if l.strip()] if EMB_LEDGER.exists() else []
    return rows, calib, main, emb


def run_files(sub):
    out = []
    for f in glob.glob(str(RUNS / sub / "run_*" / "*" / "*.json")):
        out.append((f, json.loads(Path(f).read_text())))
    return out


def facts():
    rows, calib, main, emb = ledger()
    fs = run_files("calib") + run_files("main")
    ok = [r for f, r in fs if not f.endswith(".failed.json")]
    over = [r for f, r in fs if (r.get("completion_tokens") or 0) > 60000]
    return dict(
        total=sum(r["cost"] for r in rows), calib=sum(r["cost"] for r in calib), main=sum(r["cost"] for r in main),
        est_rows=sum(r.get("estimated", False) for r in rows), est_cost=sum(r["cost"] for r in rows if r.get("estimated")),
        stalled=[r for r in rows if "stalled" in r.get("note", "")],
        emb_usd=sum(r["usd"] for r in emb), emb_tok=sum(r["tokens"] for r in emb),
        n_calib=sum(1 for f, _ in run_files("calib")), n_main=sum(1 for f, _ in run_files("main")),
        over=len(over), over_max=max((r["completion_tokens"] for r in over), default=0),
        over_stop=sum(r["finish_reason"] == "stop" for r in over),
        no_answer_line=sum(1 for r in ok if extract(r["response"]) is None and extract(r["response"], True)),
        empty_visible=sum(1 for f, r in fs if f.endswith(".failed.json") and not r.get("error")),
        tok_s=st.median(r["completion_tokens"] / r["seconds"] for r in ok if r.get("seconds") and r.get("completion_tokens")),
    )


def main():
    selftest = run(str(LADDER / "generators" / "selftest.py")).strip().splitlines()[-1]
    calib_table = run(str(LADDER / "calibrate.py"), "table")
    stats = run(str(LADDER / "select_tasks.py"), "stats")
    final = run(str(LADDER / "select_tasks.py"), "final")
    plan = (LADDER / "candidates" / "main_plan.md").read_text()
    F = facts()
    stats_table = "\n".join(l for l in stats.splitlines() if l.startswith("|"))
    stats_sum = "\n".join(f"- {l}" for l in stats.splitlines() if l.startswith("ladder_rung_"))
    rub = lambda x: f"{x:,.2f}".replace(",", " ").replace(".", ",")
    text = TEMPLATE.format(
        total=rub(F["total"]), calib=rub(F["calib"]), main=rub(F["main"]), est_rows=F["est_rows"],
        est_cost=rub(F["est_cost"]), stalled=len(F["stalled"]), stalled_cost=rub(sum(r["cost"] for r in F["stalled"])),
        emb_usd=f"{F['emb_usd']:.3f}".replace(".", ","), emb_tok=f"{F['emb_tok']:,}".replace(",", " "),
        n_calib=F["n_calib"], n_main=F["n_main"], over=F["over"], over_max=f"{F['over_max']:,}".replace(",", " "),
        over_stop=F["over_stop"], no_answer=F["no_answer_line"], empty_visible=F["empty_visible"],
        tok_s=round(F["tok_s"]), final=final.strip(), stats_sum=stats_sum, stats_table=stats_table,
        plan=plan.strip(), calib_table=calib_table.strip(), selftest=selftest)
    (LADDER / "REPORT_ladder.md").write_text(text)
    print(f"-> {(LADDER / 'REPORT_ladder.md').relative_to(ROOT)}")


TEMPLATE = open(Path(__file__).with_name("report_template.md")).read() if Path(__file__).with_name(
    "report_template.md").exists() else "{final}"

if __name__ == "__main__":
    sys.exit(main())
