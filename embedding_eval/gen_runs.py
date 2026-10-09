# /// script
# requires-python = ">=3.10"
# dependencies = ["openai>=1.40", "tokenizers", "huggingface_hub"]
# ///
"""Repeated benchmark runs: a copy of scripts/run.py with --run-id, providers and budget tracking.

Logic:
- output: runs/<model>/run_<k>/<set>/<domain>.{json,reasoning.txt}; one file per (task, run) pair;
- finished pairs (.json, finish_reason=stop) are skipped, so a launch can be resumed; interrupted streams
  are saved as .failed.json/.failed.reasoning.txt and recomputed on the next launch;
- explicit temperature, seed (deterministic from run_id and task id), full usage, system_fingerprint, provider;
- spend log runs/<model>/spend.jsonl (in the provider's tariff currency) and a hard --budget limit:
  before each request the worst case (max_tokens) is reserved; the request does not start if
  spent + reserved would exceed the limit.
The source dataset is read-only (writing there is forbidden by a path check).

  uv run gen_runs.py --provider cloudru --probe set_3/music --temperature T --budget 300
  uv run gen_runs.py --provider cloudru --models-info
  uv run gen_runs.py --provider cloudru --run-id 1 2 3 4 --temperature T --budget 300
"""
import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
# dataset: in the benchmark repo it is the root (embedding_eval/ lives inside), for us it is a sibling folder
DATA = HERE.parent if (HERE.parent / "bench").is_dir() else HERE.parent / "AI-Swarm-Dynamics-Hackathon"

PROVIDERS = {
    # prices: per 1 token, in the provider's currency
    "together": dict(base_url="https://api.together.xyz/v1", model="deepseek-ai/DeepSeek-V4-Flash-0731",
                     key_env="TOGETHER_API_KEY", key_file=".together_key", keychain="together-api", currency="USD",
                     price_in=0.14 / 1e6, price_out=0.28 / 1e6,
                     price_src="together.ai/pricing, 2026-10-04"),
    "cloudru": dict(base_url="https://foundation-models.api.cloud.ru/v1", model="deepseek-ai/DeepSeek-V4-Flash",
                    key_env="CLOUDRU_API_KEY", key_file=".cloudru_key", keychain="cloudru-foundation-models", currency="RUB",
                    price_in=43.2978 / 1e6, price_out=86.5834 / 1e6,
                    price_src="cdn.cloud.ru/docs/legal/tariffs/evolution/future-version/foundation-models.pdf, "
                              "version 260915, items 47–48, price incl. 22 % VAT"),
}


# extract — as in scripts/run.py; fallback=True (ladder runs) also accepts a bare last line when there is
# neither an ANSWER line nor \boxed (the model sometimes replies with just the answer)
def extract(text, fallback=False):
    m = re.findall(r"^\s*\**ANSWER\**\s*:\s*(.+?)\s*$", text, re.I | re.M)
    if m:
        return m[-1].strip("`* ")
    i = text.rfind("\\boxed{")
    if i >= 0:
        return text[i + 7:text.find("}", i)]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1].strip("`* ") if fallback and lines else None


def parse_num(s, thousands=True):
    """First number in s (after the last '=', so 'x_1 = 3/2' gives 3/2) as an exact Fraction; accepts a/b,
    \\frac{a}{b}, unicode minus. None if there is none."""
    s = s.rsplit("=", 1)[-1].replace("−", "-").replace("$", "").replace("\\!", "")
    s = re.sub(r"(-?)\\[dt]?frac\{\s*(-?\d+)\s*\}\{\s*(\d+)\s*\}", r"\1\2/\3", s).replace("--", "")
    digits = r"\d[\d,]*" if thousands else r"\d+"
    m = re.search(rf"-?{digits}\.?\d*(?:[eE][-+]?\d+)?(?:\s*/\s*\d+)?", s)
    if not m:
        return None
    try:
        return Fraction(re.sub(r"[\s,]", "", m[0]))
    except (ValueError, ZeroDivisionError):
        return None


def grade(item, got):
    """Old check types behave as in scripts/run.py; added: fractions in "num", optional rel_tol/abs_tol for "list"
    and abs_tol for "num" (a value passes if it is within either tolerance)."""
    if got is None:
        return False
    norm = lambda s: re.sub(r"\s+", "", s).lower().strip("$.")
    kind, ref, tol, atol = item["check"]["type"], item["answer"], item["check"].get("rel_tol"), item["check"].get("abs_tol")
    close = lambda g, r: g is not None and abs(g - r) <= max(Fraction(tol or 0) * abs(r), Fraction(atol or 0))
    if kind == "list":
        g, r = got.strip("[]() ").split(","), ref.split(",")
        if tol is None and atol is None:
            return [norm(x) for x in g] == [norm(x) for x in r]
        return len(g) == len(r) and all(close(parse_num(a, False), parse_num(b, False)) for a, b in zip(g, r))
    if kind == "num":
        return close(parse_num(got), parse_num(ref))
    return norm(got) == norm(ref)


def seed_for(run_id, item_id):
    return int(hashlib.sha256(f"{run_id}|{item_id}".encode()).hexdigest()[:8], 16) % (2**31 - 1)


class Budget:
    def __init__(self, limit, ledger, currency):
        self.limit, self.ledger, self.currency, self.lock = limit, ledger, currency, threading.Lock()
        rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()] if ledger.exists() else []
        assert all(r.get("currency", currency) == currency for r in rows), "the ledger uses a different currency"
        self.spent = sum(r["cost"] for r in rows)
        self.reserved = 0.0

    def reserve(self, amount):
        with self.lock:
            if self.spent + self.reserved + amount > self.limit:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved, rec):
        with self.lock:
            self.reserved -= reserved
            self.spent += rec["cost"]
            with open(self.ledger, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


_TOK = None


def count_tokens(text):
    """DeepSeek-V3 tokenizer (matched the API reasoning_tokens for V4-Flash-0731 on all 20 solutions)."""
    global _TOK
    try:
        from tokenizers import Tokenizer
        _TOK = _TOK or Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
        return len(_TOK.encode(text, add_special_tokens=False).ids)
    except Exception:  # noqa: BLE001
        return int(len(text) / 3.5)


def get_key(prov):
    """API key: environment variable -> macOS Keychain -> file. The key is never printed or written to logs."""
    key = os.environ.get(prov["key_env"])
    if not key and sys.platform == "darwin":
        import subprocess
        p = subprocess.run(["security", "find-generic-password", "-s", prov["keychain"], "-a", os.environ.get("USER", ""), "-w"],
                           capture_output=True, text=True)
        key = p.stdout.strip() if p.returncode == 0 else None
    kf = HERE / prov["key_file"]
    if not key and kf.exists():
        key = kf.read_text().strip()
    return key or None


def call(client, args, prov, item, run_id, stem, budget):
    """One streaming request. Returns the .json record (and writes files). None if the budget did not allow it."""
    worst = args.max_tokens * prov["price_out"] + max(4000, 2 * len(item["prompt"])) * prov["price_in"]
    if not budget.reserve(worst):
        return None
    seed = seed_for(run_id, item["id"])
    params = dict(model=args.model, messages=[{"role": "user", "content": item["prompt"]}], max_tokens=args.max_tokens,
                  temperature=args.temperature, stream=True, stream_options={"include_usage": True})
    extra = {}
    if args.effort:
        extra["reasoning_effort"] = args.effort
    extra.update(json.loads(args.extra_body) if args.extra_body else {})
    if extra:
        params["extra_body"] = extra
    if args.seed:
        params["seed"] = seed
    t, text, reasoning, usage, finish, fingerprint, resp_id, err = time.time(), [], [], {}, None, None, None, None
    delta_fields, reasoning_field = set(), None
    part = Path(f"{stem}.reasoning.txt.part")
    try:
        with open(part, "w") as f:
            stream = client.chat.completions.create(**params)
            for chunk in stream:
                usage = chunk.usage.model_dump() if chunk.usage else usage
                fingerprint = getattr(chunk, "system_fingerprint", None) or fingerprint
                resp_id = chunk.id or resp_id
                for ch in chunk.choices:
                    ex = ch.delta.model_extra or {}
                    delta_fields |= {k for k, v in ex.items() if v}
                    r = ""
                    for fld in ("reasoning_content", "reasoning"):  # vLLM/SGLang/DeepSeek — reasoning_content; Together — reasoning
                        if ex.get(fld):
                            r, reasoning_field = ex[fld], fld
                            break
                    f.write(r)
                    reasoning.append(r)
                    text.append(ch.delta.content or "")
                    finish = ch.finish_reason or finish
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    response, reasoning_text = "".join(text), "".join(reasoning)
    in_tok, out_tok = usage.get("prompt_tokens"), usage.get("completion_tokens")
    usage_estimated = out_tok is None
    if usage_estimated:  # cut off without usage — estimate from text so the budget is not understated
        in_tok = count_tokens(item["prompt"])
        out_tok = count_tokens(reasoning_text) + count_tokens(response)
    cost = in_tok * prov["price_in"] + out_tok * prov["price_out"]
    rtok_api = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    rec = {"id": item["id"], "run_id": run_id, "provider": args.provider, "base_url": args.base_url, "model": args.model,
           "reasoning_effort": args.effort, "extra_body": extra, "temperature": args.temperature, "seed": params.get("seed"),
           "max_tokens": args.max_tokens, "finish_reason": finish, "seconds": round(time.time() - t),
           "response_id": resp_id, "system_fingerprint": fingerprint, "usage": usage, "usage_estimated": usage_estimated,
           "reasoning_field": reasoning_field, "delta_extra_fields": sorted(delta_fields),
           "reasoning_tokens": rtok_api, "reasoning_tokens_counted": count_tokens(reasoning_text),
           "completion_tokens": usage.get("completion_tokens"), "cost": round(cost, 6), "currency": prov["currency"],
           "error": err, "answer": item["answer"], "gen": item.get("gen"), "answer_fallback": args.answer_fallback,
           "extracted": extract(response, args.answer_fallback),
           "response": response}
    rec["correct"] = grade(item, rec["extracted"])
    budget.settle(worst, {"id": item["id"], "run_id": run_id, "in": in_tok, "out": out_tok, "cost": cost,
                          "currency": prov["currency"], "estimated": usage_estimated, "ok": finish == "stop", "t": time.time()})
    ok = finish == "stop" and err is None and response.strip() and reasoning_text.strip()
    if ok:
        part.rename(f"{stem}.reasoning.txt")
        Path(f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
        Path(f"{stem}.failed.json").unlink(missing_ok=True)
    else:
        part.rename(f"{stem}.failed.reasoning.txt")
        Path(f"{stem}.failed.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=list(PROVIDERS), required=True)
    ap.add_argument("--model", default=None, help="default: the model from the provider preset")
    ap.add_argument("--run-id", type=int, nargs="+")
    ap.add_argument("--probe", metavar="TASK_ID", help="a single probe request (runs/<model>/probe/)")
    ap.add_argument("--models-info", action="store_true", help="GET /models: model metadata")
    ap.add_argument("--set", action="append", default=None)
    ap.add_argument("--items", action="append", default=None, help="task jsonl file(s) instead of bench/<set>.jsonl")
    ap.add_argument("--out", help="output folder (default runs/<model>)")
    ap.add_argument("--ledger", help="spend log (default <out>/spend.jsonl)")
    ap.add_argument("--read-timeout", type=float, default=300, help="seconds without stream data before failing")
    ap.add_argument("--answer-fallback", action="store_true", help="no ANSWER line and no \\boxed: take the last line")
    ap.add_argument("--effort", default="high", help="reasoning_effort ('' — do not send)")
    ap.add_argument("--extra-body", default="", help="extra JSON for extra_body")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--max-tokens", type=int, default=100000)
    ap.add_argument("--no-seed", dest="seed", action="store_false")
    ap.add_argument("--budget", type=float, help="hard limit in the provider's currency (USD / RUB)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    prov = PROVIDERS[args.provider]
    args.model = args.model or prov["model"]
    args.base_url = prov["base_url"]

    key = get_key(prov)
    if not key and not args.dry_run:
        sys.exit(f"no key: env {prov['key_env']}, Keychain service '{prov['keychain']}' or embedding_eval/{prov['key_file']}")

    from openai import OpenAI
    import httpx
    # read timeout: a stream that sends nothing for this long (network drop) fails and is redone on the next launch
    client = OpenAI(api_key=key, base_url=args.base_url, timeout=httpx.Timeout(3600, read=args.read_timeout),
                    max_retries=0) if key else None

    if args.models_info:
        import urllib.request
        req = urllib.request.Request(args.base_url + "/models", headers={"Authorization": f"Bearer {key}"})
        data = json.loads(urllib.request.urlopen(req, timeout=60).read())
        items = data.get("data", data)
        hit = [m for m in items if args.model.lower() in json.dumps(m).lower()]
        print(json.dumps(hit or items, ensure_ascii=False, indent=1)[:6000])
        return

    out = Path(args.out).resolve() if args.out else HERE / "runs" / args.model.split("/")[-1].lower()
    assert not any(d.resolve() in out.resolve().parents for d in (DATA / "bench", DATA / "results")), \
        "writing to the source dataset (bench/, results/) is forbidden"
    out.mkdir(parents=True, exist_ok=True)
    if args.temperature is None or args.budget is None:
        sys.exit("--temperature and --budget are required")
    budget = Budget(args.budget, Path(args.ledger) if args.ledger else out / "spend.jsonl", prov["currency"])
    files = args.items or [DATA / "bench" / f"{s}.jsonl" for s in (args.set or ["set_3", "set_4"])]
    items = {json.loads(l)["id"]: json.loads(l) for f in files for l in open(f) if l.strip()}
    cur = prov["currency"]

    if args.probe:
        item = items[args.probe]
        stem = out / "probe" / item["set"] / item["domain"]
        stem.parent.mkdir(parents=True, exist_ok=True)
        rec = call(client, args, prov, item, "probe", stem, budget)
        if rec is None:
            sys.exit(f"[budget] limit {args.budget} {cur} does not allow even a probe request")
        show = {k: rec[k] for k in ("finish_reason", "error", "seconds", "reasoning_field", "delta_extra_fields",
                                    "reasoning_tokens", "reasoning_tokens_counted", "completion_tokens", "usage",
                                    "system_fingerprint", "extracted", "correct", "cost", "currency")}
        print(json.dumps(show, ensure_ascii=False, indent=1))
        rt = Path(f"{stem}.reasoning.txt" if Path(f"{stem}.reasoning.txt").exists() else f"{stem}.failed.reasoning.txt").read_text()
        print(f"\nreasoning: {len(rt)} chars\n--- start ---\n{rt[:400]}\n--- end ---\n{rt[-300:]}")
        print(f"\nanswer:\n{rec['response'][-300:]}")
        print(f"\ntotal spent: {budget.spent:.4f} {cur}")
        return

    assert args.run_id, "--run-id is required"
    if out.name == "deepseek-v4-flash-0731":
        assert 1 not in args.run_id, "for 0731, run 1 is the original results/; it is not regenerated"
    todo = [(r, it) for r in sorted(args.run_id) for it in items.values()
            if not (out / f"run_{r}" / it["set"] / f"{it['domain']}.json").exists()]
    print(f"{args.provider} {args.model} -> {out}\nto generate: {len(todo)} pairs; already spent "
          f"{budget.spent:.4f} of {args.budget} {cur}", flush=True)
    if args.dry_run:
        for r, it in todo:
            print(" ", r, it["id"], "seed", seed_for(r, it["id"]))
        return
    stop_flag = threading.Event()

    def job(x):
        r, it = x
        if stop_flag.is_set():
            return
        stem = out / f"run_{r}" / it["set"] / it["domain"]
        stem.parent.mkdir(parents=True, exist_ok=True)
        rec = call(client, args, prov, it, r, stem, budget)
        if rec is None:
            stop_flag.set()
            print(f"[budget] limit {args.budget} {cur} does not allow starting {it['id']} run {r} "
                  f"(spent {budget.spent:.4f}, in flight {budget.reserved:.4f}); stopping", flush=True)
            return
        print(f"{it['id']:24s} run {r} finish={rec['finish_reason']} correct={rec['correct']} "
              f"reas_tok={rec['reasoning_tokens']}/{rec['reasoning_tokens_counted']} {rec['cost']:.4f} {cur} "
              f"total={budget.spent:.4f} err={rec['error']}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(job, todo))
    left = [(r, it["id"]) for r, it in todo if not (out / f"run_{r}" / it["set"] / f"{it['domain']}.json").exists()]
    print(f"done. spent {budget.spent:.4f} {cur}; not finished {len(left)}: {left}"
          + ("  [STOPPED BY LIMIT]" if stop_flag.is_set() else ""), flush=True)


if __name__ == "__main__":
    main()
