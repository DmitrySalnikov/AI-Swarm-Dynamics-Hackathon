# /// script
# dependencies = ["openai>=1.40"]
# ///
"""Симуляция командного чата N=10 агентов на DeepSeek-V4-Flash (Together).

    TOGETHER_API_KEY=... uv run embedding_eval/chat_runs.py --bench set_3 --kind negotiation --runs 10
    TOGETHER_API_KEY=... uv run embedding_eval/chat_runs.py --bench set_3 --kind merge --runs 10

Пул — все 10 задач бенча (set_3 или set_4), результаты в runs/chat/<bench>/<kind>/run_<k>/.
Каждый ход — отдельный вызов LLM; рассуждение
и сообщение агента пишутся в agent_<i>.jsonl. Невалидный прогон (нет ASSIGNMENT/FINAL) сохраняется
как run_<k>_invalid_<n> и перезапускается.

merge run_<k> берёт распределение из negotiation run_<k>: каждый агент сначала решает свою задачу
(отдельный вызов, solve/agent_<i>.json), затем идёт чат по полученным ответам.
"""
import argparse, json, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "embedding_eval" / "runs" / "chat"            # + /<bench>, задаётся в main()
MODEL, PRICE_IN, PRICE_OUT = "deepseek-ai/DeepSeek-V4-Flash-0731", 0.14e-6, 0.28e-6

NEG_SYSTEM = """You are Agent {i} of {n} in a team. The team must solve the {n} problems listed below.
Rules:
- Each agent solves exactly one problem; each problem is taken by exactly one agent.
- Agree on who takes which problem in the team chat. Consider which problem fits you, avoid conflicts,
  and resolve them if they appear.
- Do NOT start solving any problem. Only coordinate.
- Keep messages short (2-5 sentences).
- When the assignment is fully agreed, end your message with one line:
  ASSIGNMENT: {{"agent_1": "<problem_id>", "agent_2": "<problem_id>", ...}}

Problems:
{problems}"""

MERGE_SYSTEM = """You are Agent {i} of {n} in a team. The team split {n} problems; each agent solved one.
You solved problem [{task_id}] ({domain}).
Your final answer: {answer}
Key steps of your solution (summary): {steps}

Now in the team chat:
- Report your answer and how you got it, in 2-4 sentences.
- Look at the other agents' reported answers; point out anything that looks inconsistent or doubtful.
- Help assemble the team's final result.
Keep messages short. Do not re-solve problems from scratch."""

FINAL_NOTE = """

This is the final turn. Write the team's final result as one line:
FINAL: {"<problem_id>": "<answer>", ...}"""

spent = {"usd": 0.0}


BENCH = {}                                                  # id -> задача, заполняется в main()
SOLVE_SLOTS = threading.Semaphore(8)                       # одновременных решений на все прогоны


def load_pool(rng):
    return [{"id": it["id"], "domain": it["domain"], "statement": it["prompt"].split("\n\nThink it through")[0]}
            for it in BENCH.values()]


def solve(client, task_id, seed, max_tokens):
    """Агент решает свою задачу: поток, чтобы не упираться в тайм-ауты на длинных рассуждениях."""
    with SOLVE_SLOTS:
        for attempt in range(4):
            try:
                stream = client.chat.completions.create(
                    model=MODEL, seed=seed, max_tokens=max_tokens, stream=True, stream_options={"include_usage": True},
                    extra_body={"reasoning_effort": "high"}, messages=[{"role": "user", "content": BENCH[task_id]["prompt"]}])
                reasoning, content, usage, finish = [], [], None, None
                for chunk in stream:
                    usage = chunk.usage or usage
                    for ch in chunk.choices:
                        extra = ch.delta.model_extra or {}                # поле зависит от провайдера и даты
                        reasoning.append(extra.get("reasoning") or extra.get("reasoning_content") or "")
                        content.append(ch.delta.content or "")
                        finish = ch.finish_reason or finish
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
    if usage:
        spent["usd"] += usage.prompt_tokens * PRICE_IN + usage.completion_tokens * PRICE_OUT
    content, reasoning = "".join(content), "".join(reasoning)
    found = re.findall(r"^\s*\**ANSWER\**\s*:\s*(.+?)\s*$", content, re.I | re.M)
    return {"task_id": task_id, "seed": seed, "finish_reason": finish, "answer": found[-1].strip("`* ") if found else None,
            "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None),
            "response": content, "reasoning": reasoning}


def call(client, system, history, i, temperature, seed):
    hist = "\n".join(f"Agent {a}: {m}" for a, m in history) or "(empty — you write first)"
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=temperature, seed=seed, max_tokens=8000,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": f"Team chat so far:\n{hist}\n\nWrite your next message as Agent {i}."}])
            break
        except Exception as e:  # 503 и обрывы у провайдера
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    m = r.choices[0].message
    extra = m.model_extra or {}
    u = r.usage
    spent["usd"] += u.prompt_tokens * PRICE_IN + u.completion_tokens * PRICE_OUT
    rt = getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None)
    return extra.get("reasoning") or extra.get("reasoning_content") or "", (m.content or "").strip(), rt


def last_json(prefix, transcript):
    for _, msg in reversed(transcript):
        found = re.findall(rf"{prefix}:\s*(\{{.*\}})", msg, re.S)
        if found:
            try:
                return json.loads(found[-1])
            except json.JSONDecodeError:
                return None
    return None


def run_one(client, kind, k, n, temperature, rounds, solve_tokens):
    rng = random.Random(f"{kind}-{k}-{time.time()}")
    order = list(range(1, n + 1)); rng.shuffle(order)        # порядок ходов агентов
    d = OUT / kind / f"run_{k}"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("agent_*.jsonl"):
        f.unlink()
    if kind == "negotiation":
        pool = load_pool(rng)[:n]
        rng.shuffle(pool)                                    # порядок задач в списке
        problems = "\n".join(f"[{t['id']}] ({t['domain']}) {t['statement']}" for t in pool)
        systems = {i: NEG_SYSTEM.format(i=i, n=n, problems=problems) for i in order}
        owner = None
    else:                                                    # распределение — из согласования с тем же k
        neg = json.loads((OUT / "negotiation" / f"run_{k}" / "result.json").read_text())
        owner = {i: neg[f"agent_{i}"] for i in range(1, n + 1)}
        pool = [{"id": t, "domain": BENCH[t]["domain"]} for t in owner.values()]
        (d / "solve").mkdir(exist_ok=True)
        with ThreadPoolExecutor(n) as ex:
            solved = dict(zip(owner, ex.map(lambda i: solve(client, owner[i], rng.randrange(2**31), solve_tokens), owner)))
        for i, sol in solved.items():
            (d / "solve" / f"agent_{i}.json").write_text(json.dumps(sol, ensure_ascii=False, indent=2))
        steps = {i: (s["response"] if len(s["response"]) >= 600 else s["reasoning"])[-1500:] for i, s in solved.items()}
        systems = {i: MERGE_SYSTEM.format(i=i, n=n, task_id=owner[i], domain=BENCH[owner[i]]["domain"],
                                          answer=solved[i]["answer"] or "(no final answer: ran out of time)",
                                          steps=steps[i]) for i in order}
    transcript, turns = [], [(r, i) for r in range(rounds) for i in order]
    if kind == "merge":
        turns.append((rounds, 1))                            # Agent 1 пишет итог
    for t, (rnd, i) in enumerate(turns):
        system = systems[i] + (FINAL_NOTE if kind == "merge" and rnd == rounds else "")
        seed = rng.randrange(2**31)
        reasoning, message, rt = call(client, system, transcript, i, temperature, seed)
        with open(d / f"agent_{i}.jsonl", "a") as f:
            f.write(json.dumps({"turn": t, "round": rnd, "agent": i, "reasoning": reasoning, "message": message,
                                "reasoning_tokens": rt, "seed": seed}, ensure_ascii=False) + "\n")
        transcript.append((i, message))

    if kind == "negotiation":
        a = last_json("ASSIGNMENT", transcript) or {}
        ids = {t["id"] for t in pool}
        valid = sorted(a) == sorted(f"agent_{i}" for i in range(1, n + 1)) and sorted(a.values()) == sorted(ids)
        result = {"valid": valid, **a}
    else:
        final = last_json("FINAL", transcript[-1:])
        result = {"valid": bool(final), "final": final,
                  "solved": {t: solved[i]["answer"] for i, t in owner.items()},
                  "reference": {t: BENCH[t]["answer"] for t in owner.values()}}
    (d / "transcript.json").write_text(json.dumps({
        "kind": kind, "model": MODEL, "N": n, "rounds": rounds, "temperature": temperature,
        "tasks_in_listed_order": [t["id"] for t in pool], "agent_turn_order": order,
        "agent_tasks": {f"agent_{i}": owner[i] for i in owner} if owner else None,
        "messages": [{"turn": t, "agent": a, "message": m} for t, (a, m) in enumerate(transcript)]},
        ensure_ascii=False, indent=2))
    (d / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result["valid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["set_3", "set_4"], required=True)
    ap.add_argument("--kind", choices=["negotiation", "merge"], required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--retries", type=int, default=3, help="перезапусков невалидного прогона")
    ap.add_argument("--budget", type=float, default=2.0, help="лимит в USD")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--solve-tokens", type=int, default=8000, help="max_tokens на решение задачи (merge)")
    args = ap.parse_args()
    global OUT
    OUT = OUT / args.bench
    BENCH.update({json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "bench" / f"{args.bench}.jsonl")})
    client = OpenAI(api_key=os.environ["TOGETHER_API_KEY"], base_url="https://api.together.xyz/v1", timeout=600, max_retries=0)

    def job(k):
        for attempt in range(args.retries + 1):
            if spent["usd"] > args.budget:
                print(f"run_{k}: budget exceeded"); return
            neg = OUT / "negotiation" / f"run_{k}" / "result.json"
            if args.kind == "merge" and not (neg.exists() and json.loads(neg.read_text()).get("valid")):
                print(f"merge run_{k}: no valid negotiation run_{k}"); return
            ok = run_one(client, args.kind, k, args.n, args.temperature, args.rounds, args.solve_tokens)
            print(f"{args.kind} run_{k} attempt {attempt}: valid={ok}  spent=${spent['usd']:.3f}", flush=True)
            if ok:
                return
            d = OUT / args.kind / f"run_{k}"                 # невалидный прогон сохраняем помеченным
            d.rename(d.with_name(f"run_{k}_invalid_{attempt}"))

    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(job, range(1, args.runs + 1)))


if __name__ == "__main__":
    main()
