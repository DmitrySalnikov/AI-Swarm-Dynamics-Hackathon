from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import urllib.error
import urllib.request

from embedding_eval.gen_runs import grade

CONDITIONS = ("fixed", "shuffled", "disjoint", "cooperative")
API_URL = "https://foundation-models.api.cloud.ru/v1"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROTOCOL = """Return exactly one JSON object, without markdown fences.
Actions:
claim: {"action":"claim","task_id":"..."} (negotiation only).
work: {"action":"work","notes":"private working notes to retain"}.
send: {"action":"send","to":"all" or an agent number,"message":"..."} (team only).
submit: {"action":"submit","task_id":"...","answer":"...","notes":"..."} (solving only).
finish: {"action":"finish","answers":{"task_id":"answer",...}} (exchange only).
Your own prior JSON responses and private notes stay in your history. Peers see
only messages explicitly sent to them. Preserve useful intermediate work in notes;
provider-internal reasoning is logged but is not automatically replayed as memory.
"""
RUNNER_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)


@contextmanager
def output_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a+b") as stream:
        stream.seek(0)
        stream.write(b"0")
        stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


@dataclass(frozen=True)
class Config:
    condition: str
    agents: int
    seed: int = 1
    model: str = MODEL
    temperature: float = 1.0
    max_tokens: int = 32768
    max_calls: int = 300
    max_context_bytes: int = 400000
    budget_rub: float = 0.0
    price_in: float = 43.2978 / 1e6
    price_out: float = 86.5834 / 1e6
    offline: bool = False


class CloudClient:
    def __init__(self, key, timeout=300):
        self.key, self.timeout = key, timeout

    def models(self):
        request = urllib.request.Request(API_URL + "/models", headers={"Authorization": "Bearer " + self.key})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return [m["id"] for m in json.load(response).get("data", [])]

    def complete(self, request):
        payload = {k: request[k] for k in ("model", "messages", "temperature", "max_tokens", "seed")}
        payload.update(stream=True, stream_options={"include_usage": True}, reasoning_effort="high")
        req = urllib.request.Request(API_URL + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        content, reasoning, usage, finish, response_id, fingerprint = [], [], None, None, None, None
        actual_model, reasoning_field = None, None
        # Byte count is a conservative client-side token proxy. A provider may keep
        # billing after cancellation; no client can promise an account-side hard cap.
        output_bytes = 0
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                response_id = chunk.get("id") or response_id
                fingerprint = chunk.get("system_fingerprint") or fingerprint
                actual_model = chunk.get("model") or actual_model
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    text = delta.get("content") or ""
                    thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if thought:
                        reasoning_field = "reasoning_content" if delta.get("reasoning_content") else "reasoning"
                    content.append(text)
                    reasoning.append(thought)
                    output_bytes += len((text + thought).encode("utf-8"))
                    finish = choice.get("finish_reason") or finish
                if output_bytes > request["max_tokens"] * 4:
                    finish = "client_output_limit"
                    break
        return {"content": "".join(content), "reasoning": "".join(reasoning), "usage": usage,
                "finish_reason": finish, "response_id": response_id, "system_fingerprint": fingerprint,
                "actual_model": actual_model, "reasoning_field": reasoning_field}


class OfflineClient:
    """Scripted protocol exerciser; answers are placeholders, not model results."""
    def complete(self, request):
        context = request["context"]
        agent, phase = context["agent"], context["phase"]
        history = request["messages"]
        own = [json.loads(m["content"]) for m in history if m["role"] == "assistant"]
        if phase == "negotiation":
            action = ({"action": "claim", "task_id": context["available_tasks"][0]} if context["negotiated"] else
                      {"action": "send", "to": "all", "message": f"Agent {agent}: I prefer {context['available_tasks'][0]}; we can divide the remaining tasks."})
        elif phase == "solving":
            task = context["current_task"]
            last = own[-1] if own else {}
            if task is None:
                action = {"action": "work", "notes": "Available to help peers with completed solution"}
            elif last.get("action") == "work" and last.get("notes", "").startswith(task):
                if context["team"]:
                    action = {"action": "send", "to": "all", "message": f"Agent {agent} working on {task}"}
                else:
                    action = {"action": "submit", "task_id": task, "answer": "offline-placeholder", "notes": task}
            elif last.get("action") == "send":
                action = {"action": "submit", "task_id": task, "answer": "offline-placeholder", "notes": task}
            else:
                action = {"action": "work", "notes": f"{task}: private memory for agent {agent}"}
        elif not context["reported"]:
            action = {"action": "send", "to": "all", "message": json.dumps(context["own_answers"])}
        elif context["all_reported"]:
            answers = dict(context["own_answers"])
            for message in context["inbox"]:
                try:
                    answers.update(json.loads(message["message"]))
                except (ValueError, TypeError):
                    pass
            action = {"action": "finish", "answers": answers}
        else:
            action = {"action": "work", "notes": "Waiting for team reports"}
        return {"content": json.dumps(action), "reasoning": "scripted offline reasoning", "usage": None,
                "finish_reason": "stop", "response_id": "offline", "system_fingerprint": "offline"}


def load_tasks(path):
    tasks = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [t["id"] for t in tasks]
    if not tasks or len(ids) != len(set(ids)):
        raise ValueError("Tasks must have unique IDs and cannot be empty")
    for task in tasks:
        if not isinstance(task.get("prompt"), str) or not task["prompt"]:
            raise ValueError("Each task needs a non-empty prompt")
        if "answer" not in task or "check" not in task:
            raise ValueError("Each task needs a private reference answer and grader")
    return tasks


class Runner:
    def __init__(self, tasks, config, root, client):
        if config.condition not in CONDITIONS or not 1 <= config.agents <= len(tasks):
            raise ValueError("Invalid condition or agent count")
        if config.condition == "cooperative" and config.agents != len(tasks):
            raise ValueError("Cooperative MVP needs one task per agent")
        if config.max_tokens < 1 or config.max_calls < 1 or config.budget_rub < 0:
            raise ValueError("Limits must be positive; budget cannot be negative")
        if not all(math.isfinite(v) for v in (config.budget_rub, config.price_in, config.price_out, config.temperature)):
            raise ValueError("Budget, prices, and temperature must be finite")
        if config.price_in < 0 or config.price_out < 0 or config.max_context_bytes < 1:
            raise ValueError("Prices cannot be negative")
        self.tasks = {t["id"]: t for t in tasks}
        self.config, self.root, self.client = config, Path(root), client
        self.directory = self.root / config.condition / f"seed_{config.seed}"
        self.manifest = {"schema_version": 1, "config": asdict(config), "task_hash": digest(tasks), "tasks": tasks,
                         "protocol_hash": digest(PROTOCOL), "runner_hash": RUNNER_HASH}
        self.state = None

    def initial_state(self):
        ids = list(self.tasks)
        agents = {}
        for i in range(1, self.config.agents + 1):
            order = list(ids)
            if self.config.condition == "shuffled":
                random.Random(f"{self.config.seed}:{i}").shuffle(order)
            elif self.config.condition == "disjoint":
                offset = i - 1
                order = order[offset:] + order[:offset]
            if self.config.condition == "cooperative":
                order = []
            system = f"You are Agent {i}. " + ("Work together as a team." if self.team else "Solve independently. No peer communication is available.")
            agents[str(i)] = {"history": [{"role": "system", "content": system + "\n" + PROTOCOL}],
                              "order": order, "answers": {}, "inbox": [], "seen_inbox": 0,
                              "negotiated": False, "reported": False, "finished": False}
        return {"phase": "negotiation" if self.team else "solving", "agents": agents, "calls": 0,
                "next_agent": 1, "pending": None, "status": "running"}

    @property
    def team(self):
        return self.config.condition == "cooperative"

    def context(self, agent_id):
        agent = self.state["agents"][str(agent_id)]
        phase = self.state["phase"]
        assigned = [t for a in self.state["agents"].values() for t in a["order"]] if self.team else []
        current = next((t for t in agent["order"] if t not in agent["answers"]), None)
        context = {"agent": agent_id, "phase": phase, "team": self.team, "current_task": current,
                   "own_answers": agent["answers"], "inbox": agent["inbox"], "reported": agent["reported"],
                   "all_reported": all(a["reported"] for a in self.state["agents"].values())}
        if phase == "negotiation":
            context["available_tasks"] = [t for t in self.tasks if t not in assigned]
            context["negotiated"] = agent["negotiated"]
            context["tasks"] = [{"id": t["id"], "prompt": t["prompt"]} for t in self.tasks.values()]
            context["instruction"] = "First send your task preference and a brief explanation to all agents. After reading the team discussion, claim one available task. If your preference is taken, choose another. Do not solve yet."
        elif phase == "solving":
            context["task"] = {"id": current, "prompt": self.tasks[current]["prompt"]} if current else None
            context["instruction"] = ("Your task is complete. Respond to peers needing help using your retained work."
                                      if current is None else "Solve your current task. Retain intermediate notes; ask peers for help when useful."
                                      if self.team else "Solve your current task, retaining previous solutions and notes.")
        else:
            context["task_ids"] = list(self.tasks)
            context["instruction"] = "Send your answer and key steps to the team. After all agents report, review their answers and finish with a complete answer map. You may discuss or challenge answers first."
        # Freeze the call-time snapshot. Later deliveries and submissions must
        # not retroactively change the recorded context or its seen-inbox count.
        return json.loads(json.dumps(context))

    def spent(self):
        total = 0.0
        for path in self.root.glob("*/seed_*/calls/*/request.json"):
            response = path.with_name("response.json")
            if response.exists():
                total += json.loads(response.read_text(encoding="utf-8"))["cost_rub"]
            else:
                total += json.loads(path.read_text(encoding="utf-8"))["reserved_rub"]
        return total

    def apply(self, request, response):
        agent = self.state["agents"][str(request["agent"])]
        agent["history"].append({"role": "user", "content": json.dumps(request["context"], ensure_ascii=False)})
        agent["history"].append({"role": "assistant", "content": response["content"]})
        agent["seen_inbox"] = len(request["context"]["inbox"])
        deliveries, error = [], None
        try:
            if response["finish_reason"] != "stop":
                raise ValueError("Incomplete provider response: " + str(response["finish_reason"]))
            text = response["content"].strip()
            if text.startswith("```") and text.endswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            action = json.loads(text)
            kind = action["action"]
            if kind == "claim":
                task = action["task_id"]
                if self.state["phase"] != "negotiation" or not agent["negotiated"] or agent["order"] or task not in request["context"]["available_tasks"]:
                    raise ValueError("Claim must name an available task during negotiation")
                agent["order"] = [task]
                for recipient in self.state["agents"]:
                    if int(recipient) == request["agent"]:
                        continue
                    message = {"from": request["agent"], "to": int(recipient), "message": f"Agent {request['agent']} claimed {task}.",
                               "call": request["call"], "kind": "claim_receipt"}
                    self.state["agents"][recipient]["inbox"].append(message)
                    deliveries.append(message)
            elif kind == "work":
                if not isinstance(action.get("notes"), str) or not action["notes"].strip():
                    raise ValueError("Work action must retain private notes")
            elif kind == "send":
                if not self.team or not isinstance(action.get("message"), str) or not action["message"].strip():
                    raise ValueError("Only cooperative agents may send non-empty messages")
                target = action.get("to")
                recipients = [int(i) for i in self.state["agents"] if int(i) != request["agent"]] if target == "all" else [target]
                if any(type(i) is not int or str(i) not in self.state["agents"] or i == request["agent"] for i in recipients):
                    raise ValueError("Invalid recipient")
                for recipient in recipients:
                    message = {"from": request["agent"], "to": recipient, "message": action["message"], "call": request["call"]}
                    self.state["agents"][str(recipient)]["inbox"].append(message)
                    deliveries.append(message)
                if self.state["phase"] == "exchange" and target == "all":
                    agent["reported"] = True
                if self.state["phase"] == "negotiation" and target == "all":
                    agent["negotiated"] = True
            elif kind == "submit":
                if self.state["phase"] != "solving" or request["context"]["current_task"] is None or action.get("task_id") != request["context"]["current_task"] or not isinstance(action.get("answer"), str):
                    raise ValueError("Submit must answer the current task during solving")
                agent["answers"][action["task_id"]] = action["answer"]
            elif kind == "finish":
                answers = action.get("answers")
                if self.state["phase"] != "exchange" or not request["context"]["all_reported"] or not isinstance(answers, dict) or set(answers) != set(self.tasks) or not all(isinstance(v, str) for v in answers.values()):
                    raise ValueError("Finish requires all team reports and an answer for every task")
                agent["final"] = answers
                agent["finished"] = True
            else:
                raise ValueError("Unknown action")
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            error = str(exc)
            agent["history"].append({"role": "user", "content": "Action rejected: " + error + ". Follow the JSON action protocol."})
        if response["finish_reason"] != "stop":
            self.state["status"] = "provider_incomplete"
        agents = list(self.state["agents"].values())
        if self.team:
            if self.state["phase"] == "negotiation" and all(a["order"] for a in agents):
                self.state["phase"] = "solving"
            elif self.state["phase"] == "solving" and all(len(a["answers"]) == len(a["order"]) for a in agents):
                self.state["phase"] = "exchange"
            elif all(a["finished"] for a in agents):
                self.state["status"] = "completed"
        elif all(len(a["answers"]) == len(a["order"]) for a in agents):
            self.state["status"] = "completed"
        self.state["calls"] += 1
        self.state["next_agent"] = request["agent"] % self.config.agents + 1
        self.state["pending"] = None
        return {"call": request["call"], "phase": request["context"]["phase"], "agent": request["agent"], "deliveries": deliveries, "action_error": error}

    def next_agent(self):
        for offset in range(self.config.agents):
            i = (self.state["next_agent"] - 1 + offset) % self.config.agents + 1
            agent = self.state["agents"][str(i)]
            if self.state["phase"] == "negotiation" and agent["order"]:
                continue
            if self.state["phase"] == "solving" and len(agent["answers"]) == len(agent["order"]):
                if not self.team or len(agent["inbox"]) <= agent["seen_inbox"]:
                    continue
            if agent["finished"]:
                continue
            return i
        raise RuntimeError("No runnable agent")

    def settle_response(self, request, response):
        usage = response.get("usage")
        if not isinstance(usage, dict) or not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
            usage = None
        response["cost_estimated"] = not bool(usage)
        response["cost_rub"] = 0.0 if self.config.offline else (
            usage["prompt_tokens"] * self.config.price_in + usage["completion_tokens"] * self.config.price_out
            if usage else request["reserved_rub"])
        return response

    def recover(self):
        pending = self.state["pending"]
        if pending is None:
            return
        directory = self.directory / "calls" / f"{pending:06d}"
        if not (directory / "response.json").exists():
            raise RuntimeError("Interrupted request has unknown billing/outcome. It will not be automatically replayed: " + str(directory))
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        response = json.loads((directory / "response.json").read_text(encoding="utf-8"))
        write_json(directory / "event.json", self.apply(request, response))
        write_json(self.directory / "state.json", self.state)

    def run(self):
        with output_lock(self.root):
            manifest_path, state_path = self.directory / "manifest.json", self.directory / "state.json"
            if manifest_path.exists():
                if json.loads(manifest_path.read_text(encoding="utf-8")) != self.manifest:
                    raise ValueError("Manifest differs; use a new output directory or seed")
            else:
                write_json(manifest_path, self.manifest)
            self.state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else self.initial_state()
            self.recover()
            if self.state["status"] not in {"running", "completed"}:
                raise RuntimeError("Run is halted: " + self.state["status"])
            while self.state["status"] == "running" and self.state["calls"] < self.config.max_calls:
                i = self.next_agent()
                context = self.context(i)
                messages = self.state["agents"][str(i)]["history"] + [{"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
                input_bytes = len(json.dumps(messages, ensure_ascii=False).encode())
                if input_bytes > self.config.max_context_bytes:
                    raise RuntimeError("Context limit reached; no history was silently truncated")
                reserved = input_bytes * self.config.price_in + self.config.max_tokens * self.config.price_out
                if not self.config.offline and self.spent() + reserved > self.config.budget_rub:
                    write_json(state_path, self.state)
                    raise RuntimeError("Budget reservation would exceed the configured RUB limit")
                call = self.state["calls"]
                request = {"call": call, "agent": i, "context": context, "messages": messages,
                           "model": self.config.model, "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
                           "seed": int(digest([self.config.seed, self.config.condition, i, call])[:8], 16) % (2**31-1),
                           "reserved_rub": reserved, "started_at": time.time()}
                directory = self.directory / "calls" / f"{call:06d}"
                write_json(directory / "request.json", request)
                self.state["pending"] = call
                write_json(state_path, self.state)
                try:
                    response = self.client.complete(request)
                except Exception as exc:
                    # Persist the conservative reservation. Do not log provider error
                    # bodies, which can contain sensitive request material.
                    response = {"content": "", "reasoning": "", "usage": None, "finish_reason": "transport_error",
                                "error_type": type(exc).__name__}
                response["completed_at"] = time.time()
                response = self.settle_response(request, response)
                write_json(directory / "response.json", response)
                write_json(directory / "event.json", self.apply(request, response))
                write_json(state_path, self.state)
                print(f"{self.config.condition}: call {call}, agent {i}, phase {context['phase']}, status {self.state['status']}", flush=True)
            result = self.results()
            write_json(self.directory / "result.json", result)
            self.export()
            return result

    def export(self):
        """Only generated reasoning and content; never replayed prompts or inboxes."""
        outputs = {i: {"parts": [], "index": [], "length": 0} for i in self.state["agents"]}
        for directory in sorted((self.directory / "calls").glob("*")):
            if not (directory / "response.json").exists() or not (directory / "event.json").exists():
                continue
            request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
            response = json.loads((directory / "response.json").read_text(encoding="utf-8"))
            out = outputs[str(request["agent"])]
            for field in ("reasoning", "content"):
                text = response.get(field, "")
                if not text:
                    continue
                start = out["length"]
                out["parts"].append(text + "\n\n")
                out["length"] += len(text) + 2
                out["index"].append({"start": start, "end": start + len(text), "part": field,
                                     "call": request["call"], "phase": request["context"]["phase"],
                                     "task_id": request["context"].get("current_task"),
                                     "finish_reason": response["finish_reason"], "usage": response.get("usage")})
        for i, out in outputs.items():
            path = self.directory / "generated" / f"agent_{i}.generated.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            part = path.with_suffix(".part")
            part.write_text("".join(out["parts"]), encoding="utf-8")
            os.replace(part, path)
            write_json(path.with_suffix(".index.json"), out["index"])

    def results(self):
        agents = {}
        for i, agent in self.state["agents"].items():
            answers = agent.get("final", agent["answers"])
            agents[i] = {"order": agent["order"], "answers": answers,
                         "correct": {task: grade(self.tasks[task], answer) for task, answer in answers.items()}}
        return {"status": self.state["status"] if self.state["status"] != "running" else "call_limit",
                "offline": self.config.offline, "calls": self.state["calls"], "spent_root_rub": self.spent(), "agents": agents}


def main():
    parser = argparse.ArgumentParser(description="Stateful Cloud.ru experiments; offline mode never calls providers.")
    parser.add_argument("--items", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, default="cooperative")
    parser.add_argument("--agents", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", type=Path, default=Path("artifacts/pipeline"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--check-api", action="store_true", help="read model catalogue, no generation")
    parser.add_argument("--budget-rub", type=float, default=0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-calls", type=int, default=300)
    parser.add_argument("--max-context-bytes", type=int, default=400000)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--input-rub-per-million", type=float, default=43.2978)
    parser.add_argument("--output-rub-per-million", type=float, default=86.5834)
    args = parser.parse_args()
    if args.offline and args.check_api:
        parser.error("--offline and --check-api are mutually exclusive")
    key = os.environ.get("CLOUDRU_API_KEY")
    if args.check_api:
        if not key:
            parser.error("CLOUDRU_API_KEY is missing")
        try:
            models = CloudClient(key).models()
        except urllib.error.HTTPError as exc:
            parser.exit(1, f"Cloud.ru catalogue HTTP {exc.code}\n")
        print(f"Cloud.ru authentication OK; {len(models)} models")
        print("Flash models:", [m for m in models if "deepseek" in m.lower() and "flash" in m.lower()])
        return
    if not args.items:
        parser.error("--items is required")
    if not args.offline and (not key or args.budget_rub <= 0):
        parser.error("Live runs require CLOUDRU_API_KEY and a positive --budget-rub")
    root = args.out.resolve()
    repo = Path(__file__).resolve().parents[1]
    protected = [repo / name for name in ("bench", "results", "embedding_eval", ".git", ".venv")]
    if root == repo or any(root == p or p in root.parents for p in protected):
        parser.error("Use a separate artifact directory outside source and baseline data")
    tasks = load_tasks(args.items)
    config = Config(condition=args.condition, agents=len(tasks) if args.agents is None else args.agents, seed=args.seed, model=args.model,
                    temperature=args.temperature, max_tokens=args.max_tokens, max_calls=args.max_calls,
                    max_context_bytes=args.max_context_bytes, budget_rub=args.budget_rub,
                    price_in=args.input_rub_per_million / 1e6, price_out=args.output_rub_per_million / 1e6, offline=args.offline)
    result = Runner(tasks, config, root, OfflineClient() if args.offline else CloudClient(key)).run()
    print(json.dumps({k: result[k] for k in ("status", "offline", "calls", "spent_root_rub")}))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
