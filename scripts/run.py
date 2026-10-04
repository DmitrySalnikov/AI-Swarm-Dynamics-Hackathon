# /// script
# dependencies = ["openai>=1.40"]
# ///
"""Прогон наборов на модели через OpenAI-совместимый API (по умолчанию Together AI).

    TOGETHER_API_KEY=... uv run scripts/run.py --set set_1 [--model ...] [--effort high]

Пишет results/<модель>/<set>/<domain>.json и <domain>.reasoning.txt; посчитанные вопросы пропускает.
"""
import argparse, json, os, re, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent


def extract(text):
    m = re.findall(r"^\s*\**ANSWER\**\s*:\s*(.+?)\s*$", text, re.I | re.M)
    if m:
        return m[-1].strip("`* ")
    i = text.rfind("\\boxed{")  # запасной вариант для моделей, не следующих формату
    return text[i + 7:text.find("}", i)] if i >= 0 else None


def grade(item, got):
    if got is None:
        return False
    norm = lambda s: re.sub(r"\s+", "", s).lower().strip("$.")
    kind, ref = item["check"]["type"], item["answer"]
    if kind == "list":
        return [norm(x) for x in got.strip("[]").split(",")] == [norm(x) for x in ref.split(",")]
    if kind == "num":
        m = re.search(r"-?\d[\d,]*\.?\d*(?:[eE]-?\d+)?", got.replace("$", ""))
        return bool(m) and abs(float(m[0].replace(",", "")) - float(ref)) <= item["check"]["rel_tol"] * abs(float(ref))
    return norm(got) == norm(ref)


def run(client, args, item, out):
    stem = out / item["set"] / item["domain"]
    stem.parent.mkdir(parents=True, exist_ok=True)
    t, text, usage, finish = time.time(), [], {}, None
    with open(f"{stem}.reasoning.txt", "w") as f:
        stream = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": item["prompt"]}], max_tokens=args.max_tokens,
            stream=True, stream_options={"include_usage": True}, extra_body={"reasoning_effort": args.effort})
        for chunk in stream:
            usage = chunk.usage.model_dump() if chunk.usage else usage
            for ch in chunk.choices:
                f.write((ch.delta.model_extra or {}).get("reasoning") or "")
                text.append(ch.delta.content or "")
                finish = ch.finish_reason or finish
    response = "".join(text)
    rec = {"id": item["id"], "model": args.model, "reasoning_effort": args.effort, "finish_reason": finish,
           "seconds": round(time.time() - t), "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
           "completion_tokens": usage.get("completion_tokens"), "extracted": extract(response), "response": response}
    rec["correct"] = grade(item, rec["extracted"])
    Path(f"{stem}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    print(item["id"], rec["correct"], rec["reasoning_tokens"], flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--set", action="append", default=None, help="set_1 … set_4 (по умолчанию все)")
ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
ap.add_argument("--base-url", default="https://api.together.xyz/v1")
ap.add_argument("--effort", default="high")
ap.add_argument("--max-tokens", type=int, default=100000)
args = ap.parse_args()

out = ROOT / "results" / args.model.split("/")[-1].lower()
items = [json.loads(l) for s in (args.set or ["set_1", "set_2", "set_3", "set_4"]) for l in open(ROOT / "bench" / f"{s}.jsonl")]
items = [it for it in items if not (out / it["set"] / f"{it['domain']}.json").exists()]
client = OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url=args.base_url, timeout=3600, max_retries=0)
with ThreadPoolExecutor(5) as pool:
    list(pool.map(lambda it: run(client, args, it, out), items))
