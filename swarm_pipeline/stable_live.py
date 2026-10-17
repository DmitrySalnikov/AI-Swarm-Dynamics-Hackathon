"""Retain interrupted streams and continue observed private work, never replay a request."""
from __future__ import annotations

from dataclasses import replace
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

import swarm_pipeline.runner as base
from swarm_pipeline.supervision import read_json, write_json

OriginalRunner = base.Runner
OriginalClient = base.CloudClient


class RetainingClient(OriginalClient):
    def __init__(self, key, timeout=900):
        super().__init__(key, timeout=timeout)

    def complete(self, request):
        payload = {k: request[k] for k in ("model", "messages", "temperature", "max_tokens", "seed")}
        payload.update(stream=True, stream_options={"include_usage": True}, reasoning_effort="high")
        req = urllib.request.Request(base.API_URL + "/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        parts, thoughts = [], []
        usage, finish, response_id, model, fingerprint, field = None, None, None, None, None, None
        chunks, done, output_bytes, error = 0, False, 0, None
        status_code, retry_after = None, None
        first_chunk, last_chunk, max_gap, previous = None, None, 0, None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as stream:
                http_status = stream.status
                for raw in stream:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    now = time.time()
                    first_chunk = first_chunk or now
                    if previous is not None:
                        max_gap = max(max_gap, now - previous)
                    previous = last_chunk = now
                    chunks += 1
                    usage = chunk.get("usage") or usage
                    response_id = chunk.get("id") or response_id
                    model = chunk.get("model") or model
                    fingerprint = chunk.get("system_fingerprint") or fingerprint
                    for choice in chunk.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content") or ""
                        thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        if thought:
                            field = "reasoning_content" if delta.get("reasoning_content") else "reasoning"
                        parts.append(content)
                        thoughts.append(thought)
                        output_bytes += len((content + thought).encode())
                        finish = choice.get("finish_reason") or finish
                    if output_bytes > request["max_tokens"] * 4:
                        finish = "client_output_limit"
                        break
        except urllib.error.HTTPError as exc:
            error = type(exc).__name__
            status_code = exc.code
            retry_after = exc.headers.get("Retry-After")
            finish = "transport_error"
        except Exception as exc:
            error = type(exc).__name__
            finish = "transport_error"
        if finish is None:
            finish = "stream_incomplete"
        return {"content": "".join(parts), "reasoning": "".join(thoughts), "usage": usage,
                "finish_reason": finish, "response_id": response_id, "actual_model": model,
                "system_fingerprint": fingerprint, "reasoning_field": field, "error_type": error,
                "http_status": status_code, "retry_after": retry_after,
                "stream_diagnostics": {"chunks": chunks, "done_seen": done, "first_chunk_at": first_chunk,
                                       "last_chunk_at": last_chunk, "max_gap_seconds": max_gap,
                                       "received_output_bytes": output_bytes}}


class RecoveringRunner(OriginalRunner):
    max_continuations = 6
    exposure_per_unreceipted_request = 100.0

    def spent(self):
        total = super().spent()
        for path in self.root.glob("*/seed_*/calls/*/response.json"):
            response = read_json(path)
            if response.get("cost_estimated") and response.get("finish_reason") != "stop":
                total += max(0, self.exposure_per_unreceipted_request - response["cost_rub"])
        return total

    def results(self):
        result = super().results()
        result["admission_exposure_rub"] = result["spent_root_rub"]
        result["spent_root_rub"] = OriginalRunner.spent(self)
        records = [read_json(p) for p in self.root.glob("*/seed_*/calls/*/response.json")]
        result["usage_backed_spend_rub"] = sum(r["cost_rub"] for r in records if not r["cost_estimated"])
        result["unreceipted_estimates_rub"] = sum(r["cost_rub"] for r in records if r["cost_estimated"])
        result["continuations"] = len(list((self.directory / "operational_events").glob("continuation_*.json")))
        return result

    def prepare_existing(self):
        """Revise only financial admission limits, preserving the science config."""
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.exists():
            return
        with base.output_lock(self.root):
            old = read_json(manifest_path)
            check = json.loads(json.dumps(old))
            check["config"]["budget_rub"] = self.manifest["config"]["budget_rub"]
            if check != self.manifest:
                raise ValueError("Recovery cannot change tasks, agent settings, protocol, or frozen runner")
            if old != self.manifest:
                record = {"kind": "budget_revision", "before": old["config"]["budget_rub"],
                          "after": self.config.budget_rub, "human_ceiling_rub": 3000, "at": time.time()}
                write_json(self.directory / "operational_events" / "budget_revision.json", record)
                write_json(self.directory / "operational_events" / "original_manifest.json", old)
                write_json(manifest_path, self.manifest)

    def continue_partial(self, source_call=None, locked=False):
        with nullcontext() if locked else base.output_lock(self.root):
            state_path = self.directory / "state.json"
            if not state_path.exists():
                return False
            state = read_json(state_path)
            if (state["status"] != "provider_incomplete" and source_call is None) or state["pending"] is not None:
                return False
            events = self.directory / "operational_events"
            if len(list(events.glob("continuation_*.json"))) >= self.max_continuations:
                return False
            call = state["calls"] - 1 if source_call is None else source_call
            if (events / f"continuation_{call:06d}.json").exists():
                return False
            folder = self.directory / "calls" / f"{call:06d}"
            request, response = read_json(folder / "request.json"), read_json(folder / "response.json")
            if (response["finish_reason"] not in {None, "stream_incomplete", "transport_error"}
                    or request["context"]["phase"] != "solving" or not request["context"].get("current_task")
                    or not response.get("response_id") or not (response.get("reasoning") or response.get("content"))):
                return False
            # This is observed private generated text, never a reference answer or
            # a fabricated assistant message. The new request has a new seed and
            # context, and cannot be mistaken for a replay of the missing receipt.
            note = ("The last provider stream was interrupted. Its billing is unresolved and its request will not be replayed. "
                    "Continue your current task from your own observed private working fragment below. Treat it as incomplete, "
                    "check the arithmetic, and avoid repeating finished work unnecessarily. Return one valid JSON action.\n\n"
                    + response.get("reasoning", "") + "\n\n" + response.get("content", ""))
            agent = str(request["agent"])
            message = {"role": "user", "content": note}
            before = base.digest(state)
            state["agents"][agent]["history"].append(message)
            if len(json.dumps(state["agents"][agent]["history"], ensure_ascii=False).encode()) > self.config.max_context_bytes:
                return False
            state["status"] = "running"
            record = {"kind": "private_fragment_continuation", "before_call": state["calls"], "source_call": call,
                      "agent": request["agent"], "message": message, "source_response_hash": base.digest(response),
                      "before_state_hash": before, "after_state_hash": base.digest(state),
                      "uncertain_exposure_rub": self.exposure_per_unreceipted_request, "at": time.time(),
                      "original_request_replayed": False}
            write_json(events / f"before_continuation_{call:06d}.state.json", read_json(state_path))
            write_json(events / f"continuation_{call:06d}.json", record)
            write_json(state_path, state)
            return True

    def run(self):
        if self.config.budget_rub > 3000:
            raise ValueError("Recovery admission cap exceeds the human-authorized ceiling")
        self.prepare_existing()
        self.continue_partial()
        write_json(self.directory / "transport_policy.json", {
            "version": "observed-private-fragment-continuation-v1", "max_continuations": self.max_continuations,
            "unknown_billing_exposure_per_call_rub": self.exposure_per_unreceipted_request,
            "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "changes_memory_on_interruption": True, "replays_original_requests": False})
        while True:
            result = super().run()
            if result["status"] != "provider_incomplete" or not self.continue_partial():
                return result


def main():
    # Use the frozen CLI/config/scheduler/JSON protocol. Only the transport and
    # explicitly logged fault-recovery controller differ from the reference.
    base.CloudClient = RetainingClient
    base.Runner = RecoveringRunner
    base.main()


if __name__ == "__main__":
    main()
