# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tokenizers", "huggingface_hub"]
# ///
"""Extended solve phase for every chat agent: own solve continued by new independent solutions of the same question.

    uv run embedding_eval/embed_solve_ext.py [--budget-usd 1]

For each question, its 10 chat agents (one per chat run, in run order) take extra solutions
runs/deepseek-v4-flash-0731/extra/<set>/<domain>/sol_<j> in order j = 1, 2, ... ; each solution is used by one agent
only. An agent's stream = own solve (reasoning + response), then whole extra solutions (reasoning + response),
separated by a blank line, until it has >= TARGET tokens. The first K windows (2048 tokens, every 1024) are embedded,
so every agent has exactly K full windows of real text.
Output: concat/<set>/phases/run_k/agent_i_ext.npz (solve) + agent_i_ext.json (window spans, which solutions are used).
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_concat as E

HERE = Path(__file__).resolve().parent
CHAT, CONCAT = HERE / "runs" / "chat", HERE / "concat"
EXTRA = HERE / "runs" / "deepseek-v4-flash-0731" / "extra"
K, TARGET = 7, 8192


def join(*parts):
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def solution_text(stem):
    rec = json.loads(Path(f"{stem}.json").read_text())
    return join(Path(f"{stem}.reasoning.txt").read_text(), rec["response"])


def ntok(text):
    return len(E.tokenizer().encode(text, add_special_tokens=False).ids)


def plan(bench):
    """[(k, i, task, [(source, text), ...])] with the extra solutions assigned to each agent."""
    agents = []
    for k in range(1, 11):
        for i in range(1, 11):
            s = json.loads((CHAT / bench / "merge" / f"run_{k}" / "solve" / f"agent_{i}.json").read_text())
            agents.append((k, i, s["task_id"], [(f"own: merge/run_{k}/solve/agent_{i}", join(s["reasoning"], s["response"]))]))
    used = {}
    for k, i, task, parts in agents:
        pool = sorted((EXTRA / task).glob("sol_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        pool = [p for p in pool if not p.name.endswith(".failed.json")]
        n = ntok(parts[0][1])
        while n < TARGET:
            j = used.get(task, 0)
            if j >= len(pool):
                raise SystemExit(f"not enough extra solutions for {task} (run {k}, agent {i}): have {len(pool)}")
            used[task] = j + 1
            stem = pool[j].with_suffix("")
            text = solution_text(stem)
            parts.append((f"extra: {stem.relative_to(EXTRA)}", text))
            n = ntok(join(*[t for _, t in parts]))
    return agents, used


def do_agent(bench, k, i, task, parts, key, budget):
    base = CONCAT / bench / "phases" / f"run_{k}" / f"agent_{i}_ext"
    if base.with_suffix(".npz").exists():
        return 0
    text = join(*[t for _, t in parts])
    ws, n = E.windows(text)
    ws = ws[:K]
    assert len(ws) == K and all(w["tok_end"] - w["tok_start"] == 2048 for w in ws), (bench, k, i, n)
    X = E.embed(key, [text[w["char_start"]:w["char_end"]] for w in ws], budget)
    np.savez(base.with_suffix(".npz"), solve=X)
    base.with_suffix(".json").write_text(json.dumps({"task": task, "n_tokens": n, "sources": [s for s, _ in parts],
                                                     "windows": ws}, indent=1))
    return K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, default=1.0)
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    jobs = []
    for bench in ("set_3", "set_4"):
        agents, used = plan(bench)
        print(f"{bench}: extra solutions used per question: {dict(sorted(used.items()))}")
        jobs += [(bench, *x) for x in agents]
    if a.plan_only:
        return
    key = E.key()
    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(lambda j: do_agent(*j, key, a.budget_usd), jobs))
    print(f"agents {len(jobs)}, new vectors {n}, spent ${E.spent['usd']:.4f}")


if __name__ == "__main__":
    main()
