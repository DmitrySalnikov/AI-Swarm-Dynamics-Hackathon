"""Synthetic agents from isolated solutions, in the layout of a baseline_sim run, for build_baseline_report.py.

Each task of --items (in file order) was solved in separate requests with no history (embedding_eval/gen_runs.py,
--run-id 1..N, output --runs). Agent i is solution i of every task joined in that order; each solution contributes
its reasoning and then its visible answer, as one call of a simulated agent. Written to
<out>/generated/agent_<i>.generated.{txt,index.json}, plus state.json (orders, extracted answers), manifest.json
(tasks) and result.json, so the report scripts treat it like a regime run.

  python scripts/build_isolated_run.py --items bench/ladder/candidates/v41_isolated_rung1_same_order.jsonl \\
      --runs embedding_eval/runs/ladder/deepseek-v4.1-flash/isolated --agents 10 \\
      --out artifacts/baseline-v41/rung_1/isolated_k6/seed_1
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embedding_eval.gen_runs import extract

SEP = "\n\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--agents", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    tasks = [json.loads(l) for l in a.items.read_text().splitlines() if l.strip()]
    order = [t["id"] for t in tasks]
    (a.out / "generated").mkdir(parents=True, exist_ok=True)
    agents = {}
    for i in range(1, a.agents + 1):
        text, index, answers = "", [], {}
        for call, t in enumerate(tasks):
            stem = a.runs / f"run_{i}" / t["set"] / t["domain"]
            if not Path(f"{stem}.json").exists():
                sys.exit(f"missing or failed solution: {stem}.json")
            rec = json.loads(Path(f"{stem}.json").read_text())
            reasoning = Path(f"{stem}.reasoning.txt").read_text().strip()
            for part, piece in (("reasoning", reasoning), ("response", rec["response"].strip())):
                if not piece:
                    continue
                if text:
                    text += SEP
                index.append({"start": len(text), "end": len(text) + len(piece), "part": part, "call": call,
                              "phase": "solving", "task_id": t["id"], "source": str(stem), "usage": rec.get("usage")})
                text += piece
            answers[t["id"]] = extract(rec["response"], fallback=True)
        (a.out / "generated" / f"agent_{i}.generated.txt").write_text(text, encoding="utf-8")
        (a.out / "generated" / f"agent_{i}.generated.index.json").write_text(json.dumps(index, indent=1))
        agents[str(i)] = {"order": order, "answers": answers}
    (a.out / "state.json").write_text(json.dumps({"status": "completed", "agents": agents}, indent=1))
    (a.out / "manifest.json").write_text(json.dumps({"kind": "isolated-solutions-joined", "tasks": tasks,
                                                     "runs": str(a.runs)}, ensure_ascii=False, indent=1))
    (a.out / "result.json").write_text(json.dumps({"status": "completed", "kind": "isolated-solutions-joined"}))
    print(f"{a.agents} agents x {len(tasks)} tasks -> {a.out}")


if __name__ == "__main__":
    main()
