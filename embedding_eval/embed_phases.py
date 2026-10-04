# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tokenizers", "huggingface_hub"]
# ///
"""Per-phase embeddings of every chat agent, for phase-aligned dynamics.

    uv run embedding_eval/embed_phases.py [--budget-usd 1]

For agent i of chat run k (set_3, set_4), from runs/chat/<set>/{negotiation,merge}/run_k:
  negotiation  one vector per own turn (round 1, round 2): reasoning + message of that turn, embedded whole
  solve        windows over the agent's solve (reasoning + response): 2048 tokens every 1024, kept while
               >= 1024 tokens remain (as embed_concat.py); a solve shorter than 1024 tokens is one window
  merge        one vector per own turn in rounds 1 and 2 (Agent 1's extra FINAL turn is not included)
Model, provider, normalisation as embed_concat.py (Qwen3-Embedding-4B via OpenRouter, no instruction, L2).
Output: concat/<set>/phases/run_k/agent_i.npz (neg, solve, merge arrays) + agent_i.json (what each row is).
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import embed_concat as E

HERE = Path(__file__).resolve().parent
CHAT, OUT = HERE / "runs" / "chat", HERE / "concat"


def join(*parts):
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def agent_texts(bench, k, i):
    meta, texts = {"neg": [], "solve": [], "merge": []}, {"neg": [], "solve": [], "merge": []}
    for phase, kind in (("neg", "negotiation"), ("merge", "merge")):
        rows = [json.loads(l) for l in open(CHAT / bench / kind / f"run_{k}" / f"agent_{i}.jsonl") if l.strip()]
        for r in sorted(rows, key=lambda r: r["turn"]):
            if r["round"] >= 2:                         # Agent 1's FINAL turn
                continue
            texts[phase].append(join(r["reasoning"], r["message"]))
            meta[phase].append({"turn": r["turn"], "round": r["round"], "reasoning_tokens": r["reasoning_tokens"],
                                "n_tokens": len(E.tokenizer().encode(texts[phase][-1], add_special_tokens=False).ids),
                                "empty_message": not r["message"].strip()})
    s = json.loads((CHAT / bench / "merge" / f"run_{k}" / "solve" / f"agent_{i}.json").read_text())
    text = join(s["reasoning"], s["response"])
    ws, n = E.windows(text)
    if not ws:                                          # shorter than 1024 tokens: the whole solve
        ws = [{"idx": 0, "tok_start": 0, "tok_end": n, "char_start": 0, "char_end": len(text)}]
    texts["solve"] = [text[w["char_start"]:w["char_end"]] for w in ws]
    meta["solve"] = [{**w, "task": s["task_id"], "finish_reason": s["finish_reason"], "n_tokens": n} for w in ws]
    return texts, meta


def do_agent(bench, k, i, key, budget):
    base = OUT / bench / "phases" / f"run_{k}" / f"agent_{i}"
    if base.with_suffix(".npz").exists():
        return 0
    texts, meta = agent_texts(bench, k, i)
    flat = texts["neg"] + texts["solve"] + texts["merge"]
    X = np.concatenate([E.embed(key, flat[j:j + E.BATCH], budget) for j in range(0, len(flat), E.BATCH)])
    a, b = len(texts["neg"]), len(texts["neg"]) + len(texts["solve"])
    base.parent.mkdir(parents=True, exist_ok=True)
    np.savez(base.with_suffix(".npz"), neg=X[:a], solve=X[a:b], merge=X[b:])
    base.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return len(flat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, default=1.0)
    a = ap.parse_args()
    key = E.key()
    jobs = [(b, k, i) for b in ("set_3", "set_4") for k in range(1, 11) for i in range(1, 11)]
    with ThreadPoolExecutor(8) as ex:
        n = sum(ex.map(lambda j: do_agent(*j, key, a.budget_usd), jobs))
    print(f"agents {len(jobs)}, new vectors {n}, tokens {E.spent['tokens']}, spent ${E.spent['usd']:.4f}, "
          f"providers {sorted(filter(None, E.spent['providers']))}")


if __name__ == "__main__":
    main()
