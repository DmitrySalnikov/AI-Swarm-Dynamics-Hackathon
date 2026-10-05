# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "numpy", "tokenizers", "huggingface_hub"]
# ///
"""Extra independent solo solutions per question, to extend short chat solves (see build_phases / dynamics_phases).

    uv run embedding_eval/gen_extra.py --plan            # print how many per question
    uv run embedding_eval/gen_extra.py --budget 3        # generate (resumable)

For each question, the deficit = sum over its 10 chat agents of max(0, TARGET - tokens of own solve);
needed = ceil(deficit / mean solo length * 1.4) + number of short agents.
Settings as the chat solve step: Together, DeepSeek-V4-Flash-0731, reasoning high, max_tokens 8000, temperature 1.0.
Output: runs/deepseek-v4-flash-0731/extra/<set>/<domain>/sol_<j>.{json,reasoning.txt}; cut at 8000 tokens is kept
as a normal solution; other failures stay .failed.* and are retried on the next run. Runs 1-20 are not touched.
"""
import argparse, glob, json, math, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_runs as G

TARGET, MARGIN = 8192, 1.4
OUT = HERE / "runs" / "deepseek-v4-flash-0731" / "extra"


def plan():
    need = {}
    for b in ("set_3", "set_4"):
        deficit, L = {}, {}
        for f in glob.glob(str(HERE / f"concat/{b}/phases/run_*/agent_*.json")):
            s = json.load(open(f))["solve"][0]
            deficit.setdefault(s["task"], []).append(max(0, TARGET - s["n_tokens"]))
        files = glob.glob(str(HERE / f"runs/deepseek-v4-flash-0731/run_*/{b}/*.json")) + \
            glob.glob(str(HERE.parent / f"results/deepseek-v4-flash-0731/{b}/*.json"))
        for f in files:
            if ".failed." in f:
                continue
            r = json.load(open(f))
            L.setdefault(r["id"], []).append(min(8000, (r.get("reasoning_tokens") or 0) + len(r["response"]) // 3))
        for q, d in deficit.items():
            need[q] = math.ceil(sum(d) / np.mean(L[q]) * MARGIN) + sum(x > 0 for x in d)
    return need


def accept_cut(stem):
    """A solve cut at max_tokens is a valid (long) solution: move .failed.* to normal names."""
    fj, ft = Path(f"{stem}.failed.json"), Path(f"{stem}.failed.reasoning.txt")
    if fj.exists() and json.loads(fj.read_text()).get("finish_reason") == "length" and ft.exists() and ft.stat().st_size:
        fj.rename(f"{stem}.json"); ft.rename(f"{stem}.reasoning.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--budget", type=float, default=3.0, help="hard limit, USD (ledger extra/spend.jsonl)")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    need = plan()
    items = {json.loads(l)["id"]: json.loads(l) for b in ("set_3", "set_4")
             for l in open(G.DATA / "bench" / f"{b}.jsonl") if l.strip()}
    todo = [(q, j) for q, n in sorted(need.items()) for j in range(1, n + 1)
            if not (OUT / q / f"sol_{j}.json").exists()]
    print(f"needed {sum(need.values())} solutions, still to generate {len(todo)}", flush=True)
    if a.plan:
        for q, n in sorted(need.items()):
            print(f"  {q:32s} {n}")
        return
    prov = G.PROVIDERS["together"]
    args = SimpleNamespace(provider="together", model=prov["model"], base_url=prov["base_url"], max_tokens=8000, temperature=1.0,
                           effort="high", extra_body="", seed=True)
    from openai import OpenAI
    client = OpenAI(api_key=G.get_key(prov), base_url=prov["base_url"], timeout=3600, max_retries=0)
    OUT.mkdir(parents=True, exist_ok=True)
    budget = G.Budget(a.budget, OUT / "spend.jsonl", prov["currency"])
    stop, done = threading.Event(), [0]

    def job(x):
        q, j = x
        if stop.is_set():
            return
        stem = OUT / q / f"sol_{j}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        rec = G.call(client, args, prov, items[q], f"extra_{j}", stem, budget)
        if rec is None:
            stop.set(); print(f"[budget] limit ${a.budget} reached, stopping", flush=True); return
        accept_cut(stem)
        done[0] += 1
        if rec["error"] or done[0] % 50 == 0:
            print(f"[{done[0]}/{len(todo)}] {q} sol_{j} finish={rec['finish_reason']} err={rec['error']} "
                  f"spent=${budget.spent:.3f}", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(job, todo))
    left = [x for x in todo if not (OUT / x[0] / f"sol_{x[1]}.json").exists()]
    print(f"done. spent ${budget.spent:.3f}; missing {len(left)}" + (" [STOPPED BY BUDGET]" if stop.is_set() else ""),
          flush=True)


if __name__ == "__main__":
    main()
