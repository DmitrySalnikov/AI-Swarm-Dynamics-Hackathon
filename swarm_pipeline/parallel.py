"""Bounded concurrent solving with deterministic, journaled round snapshots.

Only the coordinator writes state. Each agent has at most one outstanding turn;
negotiation and exchange remain causal serial turns. Never retry unknown requests.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import swarm_pipeline.runner as base
from swarm_pipeline.stable_live import RetainingClient, RecoveringRunner
from swarm_pipeline.supervision import read_json, write_json


class ParallelRunner(RecoveringRunner):
    # User removed the artificial per-failure 100 RUB admission buffer.
    # Missing usage still uses its original request estimate, never zero billing.
    exposure_per_unreceipted_request = 0.0
    max_continuations = 300

    def __init__(self, *args, workers=10, partial_continuation=False, **kwargs):
        super().__init__(*args, **kwargs)
        if not 1 <= workers <= 10:
            raise ValueError("Workers must be between 1 and 10")
        self.workers = workers
        self.partial_continuation = partial_continuation
        if self.config.budget_rub > 3000:
            raise ValueError("Admission cap exceeds the human-authorized ceiling")
        self.manifest["execution"] = {
            "scheduler": "parallel-solving-rounds-v1", "workers": workers,
            "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "partial_continuation": partial_continuation,
            "transport_hash": hashlib.sha256(Path(sys.modules[RetainingClient.__module__].__file__).read_bytes()).hexdigest(),
            "additional_unreceipted_buffer_rub": self.exposure_per_unreceipted_request,
        }

    def repair_round(self):
        if not self.partial_continuation or self.state["status"] != "provider_incomplete":
            return
        candidates = []
        http_retries = []
        prior = [read_json(p) for p in (self.directory / "operational_events").glob("continuation_*.json")]
        for path in sorted((self.directory / "calls").glob("*/response.json")):
            response = read_json(path)
            call = int(path.parent.name)
            if response["finish_reason"] == "stop" or (self.directory / "operational_events" / f"continuation_{call:06d}.json").exists():
                continue
            request = read_json(path.with_name("request.json"))
            if ((response.get("http_status") in {429, 500, 502, 503, 504}
                    or response.get("error_type") in {"TimeoutError", "ConnectionResetError"})
                    and not response.get("content") and not response.get("reasoning")
                    and request["context"]["phase"] == "solving" and request["context"].get("current_task")):
                attempt = 1 + sum(r.get("kind") in {"automatic_http_error_retry", "automatic_transport_error_retry"}
                                  and r["agent"] == request["agent"] and r.get("task_id") == request["context"]["current_task"] for r in prior)
                if attempt > 3 or len(prior) + len(candidates) + len(http_retries) >= self.max_continuations:
                    return
                http_retries.append((request, response, attempt))
                continue
            if (response["finish_reason"] not in {None, "stream_incomplete", "transport_error"}
                    or not response.get("response_id") or not (response.get("reasoning") or response.get("content"))
                    or request["context"]["phase"] != "solving" or not request["context"].get("current_task")):
                return
            candidates.append(call)
        for call in candidates:
            if not self.continue_partial(call, locked=True):
                self.state = read_json(self.directory / "state.json")
                self.state["status"] = "provider_incomplete"
                write_json(self.directory / "state.json", self.state)
                return
        for request, response, attempt in http_retries:
            state_path = self.directory / "state.json"
            state = read_json(state_path)
            failure = f"HTTP {response['http_status']}" if response.get("http_status") else response.get("error_type")
            message = {"role": "user", "content":
                       f"The previous request failed with {failure} before any output was received. "
                       "Its billing is unresolved. This is a logged recovery turn, not a recovered response. "
                       "Continue your current task using your retained completed work. Return one valid JSON action."}
            agent = str(request["agent"])
            history = state["agents"][agent]["history"] + [message]
            if len(json.dumps(history, ensure_ascii=False).encode()) > self.config.max_context_bytes:
                return
            call = request["call"]
            events = self.directory / "operational_events"
            write_json(events / f"before_http_retry_{call:06d}.state.json", state)
            state["agents"][agent]["history"] = history
            delay = 10 * 2 ** (attempt - 1)
            try:
                delay = min(60, max(delay, float(response.get("retry_after") or 0)))
            except (ValueError, TypeError):
                pass
            record = {"kind": "automatic_http_error_retry" if response.get("http_status") else "automatic_transport_error_retry", "before_call": state["calls"],
                      "source_call": call, "agent": request["agent"], "task_id": request["context"]["current_task"],
                      "message": message, "source_response_hash": base.digest(response), "attempt": attempt,
                      "delay_seconds": delay, "http_status": response.get("http_status"),
                      "error_type": response.get("error_type"), "at": time.time(),
                      "additional_unreceipted_buffer_rub": self.exposure_per_unreceipted_request,
                      "original_request_replayed": False}
            write_json(events / f"continuation_{call:06d}.json", record)
            state["status"] = "running"
            write_json(state_path, state)
            print(f"{failure}: logged new turn for agent {agent}; backoff {delay}s (attempt {attempt}/3)", flush=True)
            time.sleep(delay)
        self.state = read_json(self.directory / "state.json")

    def runnable(self):
        if self.state["phase"] != "solving":
            return [self.next_agent()]
        selected = []
        for offset in range(self.config.agents):
            i = (self.state["next_agent"] - 1 + offset) % self.config.agents + 1
            agent = self.state["agents"][str(i)]
            if agent["finished"]:
                continue
            if len(agent["answers"]) == len(agent["order"]):
                if not self.team or len(agent["inbox"]) <= agent["seen_inbox"]:
                    continue
            selected.append(i)
        return selected[:self.workers]

    def recover(self):
        pending = self.state.get("pending_batch")
        if not pending:
            if self.state["pending"] is not None:
                return super().recover()
            # A crash can occur after request journals but before the round
            # checkpoint. Such requests also have an unknown outcome; never
            # overwrite their journals or resubmit them.
            pending = sorted(int(p.parent.name) for p in (self.directory / "calls").glob("*/request.json")
                             if int(p.parent.name) >= self.state["calls"])
            if not pending:
                return
            if pending != list(range(self.state["calls"], self.state["calls"] + len(pending))):
                raise RuntimeError("Nonconsecutive orphan request journals")
            self.state["pending_batch"] = pending
        # Check the whole outstanding round before applying any response.
        for call in pending:
            folder = self.directory / "calls" / f"{call:06d}"
            if not (folder / "response.json").exists():
                raise RuntimeError("Interrupted parallel request has unknown billing/outcome; will not replay: " + str(folder))
        for call in list(pending):
            folder = self.directory / "calls" / f"{call:06d}"
            phase = self.state["phase"]
            event = self.apply(read_json(folder / "request.json"), read_json(folder / "response.json"))
            # All actions in this round were generated for the same phase.
            # Defer the solving/exchange barrier until the final receipt.
            if len(self.state["pending_batch"]) > 1:
                self.state["phase"] = phase
            write_json(folder / "event.json", event)
            self.state["pending_batch"].remove(call)
            self.state["pending"] = self.state["pending_batch"][0] if self.state["pending_batch"] else None
            write_json(self.directory / "state.json", self.state)

    def complete(self, request):
        try:
            response = self.client.complete(request)
        except Exception as exc:
            response = {"content": "", "reasoning": "", "usage": None,
                        "finish_reason": "transport_error", "error_type": type(exc).__name__}
        response["completed_at"] = time.time()
        return self.settle_response(request, response)

    def run(self):
        with base.output_lock(self.root), ThreadPoolExecutor(max_workers=self.workers) as pool:
            manifest_path = self.directory / "manifest.json"
            state_path = self.directory / "state.json"
            if manifest_path.exists():
                if read_json(manifest_path) != self.manifest:
                    raise ValueError("Manifest differs; use a new output directory or seed")
            else:
                write_json(manifest_path, self.manifest)
            self.state = read_json(state_path) if state_path.exists() else self.initial_state()
            self.recover()
            self.repair_round()
            if self.state["status"] not in {"running", "completed"}:
                raise RuntimeError("Run is halted: " + self.state["status"])
            while self.state["status"] == "running" and self.state["calls"] < self.config.max_calls:
                start = self.state["calls"]
                requests = []
                blocked = None
                for i in self.runnable()[:self.config.max_calls - start]:
                    context = self.context(i)
                    messages = self.state["agents"][str(i)]["history"] + [{"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
                    input_bytes = len(json.dumps(messages, ensure_ascii=False).encode())
                    reserved = input_bytes * self.config.price_in + self.config.max_tokens * self.config.price_out
                    if input_bytes > self.config.max_context_bytes:
                        blocked = "Context limit reached; no history was silently truncated"
                        break
                    if not self.config.offline and self.spent() + reserved > self.config.budget_rub:
                        blocked = "Budget reservation would exceed the configured RUB limit"
                        break
                    call = start + len(requests)
                    request = {"call": call, "agent": i, "context": context, "messages": messages,
                               "model": self.config.model, "temperature": self.config.temperature,
                               "max_tokens": self.config.max_tokens,
                               "seed": int(base.digest([self.config.seed, self.config.condition, i, call])[:8], 16) % (2**31-1),
                               "reserved_rub": reserved, "started_at": time.time(), "batch_start": start}
                    write_json(self.directory / "calls" / f"{call:06d}" / "request.json", request)
                    requests.append(request)
                if not requests:
                    write_json(state_path, self.state)
                    raise RuntimeError(blocked or "No runnable agent")
                self.state["pending_batch"] = [r["call"] for r in requests]
                self.state["pending"] = start
                write_json(state_path, self.state)
                futures = [pool.submit(self.complete, request) for request in requests]
                # Persist receipts immediately, even when an earlier call is slow.
                from concurrent.futures import as_completed
                receipts = {future: request for future, request in zip(futures, requests)}
                for future in as_completed(futures):
                    request = receipts[future]
                    write_json(self.directory / "calls" / f"{request['call']:06d}" / "response.json", future.result())
                self.recover()
                self.repair_round()
                print(f"{self.config.condition}: {len(requests)} concurrent turns; {self.state['calls']} calls, {self.state['status']}", flush=True)
            result = self.results()
            result["execution"] = self.manifest["execution"]
            write_json(self.directory / "result.json", result)
            self.export()
            return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--partial-continuation", action="store_true")
    args, rest = parser.parse_known_args()
    sys.argv[1:] = rest
    base.Runner = lambda *a, **kw: ParallelRunner(*a, workers=args.workers, partial_continuation=args.partial_continuation, **kw)
    base.CloudClient = RetainingClient
    base.main()


if __name__ == "__main__":
    main()
