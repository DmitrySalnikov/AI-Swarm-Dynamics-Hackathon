"""Explicit recovery of a halted parallel trial, retaining its journal and history."""
import argparse
import json
from pathlib import Path
import time

from swarm_pipeline.parallel import ParallelRunner
from swarm_pipeline.runner import Config, digest, output_lock
from swarm_pipeline.supervision import read_json, write_json


def prepare(root, condition, seed, workers, budget):
    root = Path(root)
    directory = root / condition / f"seed_{seed}"
    with output_lock(root):
        old = read_json(directory / "manifest.json")
        state = read_json(directory / "state.json")
        if state["pending"] is not None or state.get("pending_batch"):
            raise RuntimeError("Outstanding requests must be resolved before explicit recovery")
        if state["status"] != "provider_incomplete":
            raise ValueError("Explicit recovery requires a halted incomplete trial")
        if not 0 < budget <= 3000:
            raise ValueError("Budget must stay within the approved 3000 RUB ceiling")
        cfg = Config(**{**old["config"], "budget_rub": budget})
        runner = ParallelRunner(old["tasks"], cfg, root, None, workers=workers, partial_continuation=True)
        check = json.loads(json.dumps(old))
        check["config"]["budget_rub"] = budget
        check["execution"] = runner.manifest["execution"]
        if check != runner.manifest:
            raise ValueError("Recovery cannot alter task set, model, protocol or per-agent memory policy")
        events = directory / "operational_events"
        covered = {read_json(p)["source_call"] for p in events.glob("continuation_*.json")}
        repairs = []
        for path in sorted((directory / "calls").glob("*/response.json")):
            response = read_json(path)
            call = int(path.parent.name)
            if response["finish_reason"] == "stop" or call in covered:
                continue
            request = read_json(path.with_name("request.json"))
            if (request["context"]["phase"] != "solving" or not request["context"].get("current_task")
                    or response["finish_reason"] not in {"transport_error", "stream_incomplete", None}):
                raise ValueError("Unsupported interrupted action; retain halted checkpoint")
            # Explicitly authorized new turn, not a receipt reconstruction or
            # hidden replay. Unknown billing remains reserved by admission accounting.
            note = ("The previous attempt was interrupted and may have been billed. Its outcome is unknown. "
                    "This is an explicit recovery turn. Continue solving your current task from your retained "
                    "completed notes and answers. No missing output has been fabricated. Return one JSON action.")
            if response.get("reasoning") or response.get("content"):
                note += "\nYour observed incomplete private fragment follows:\n" + response.get("reasoning", "") + "\n" + response.get("content", "")
            message = {"role": "user", "content": note}
            agent = str(request["agent"])
            history = state["agents"][agent]["history"] + [message]
            if len(json.dumps(history, ensure_ascii=False).encode()) > cfg.max_context_bytes:
                raise ValueError("Recovery note exceeds context limit")
            state["agents"][agent]["history"] = history
            repairs.append({"kind": "explicit_interrupted_turn_restart", "before_call": state["calls"],
                            "source_call": call, "agent": request["agent"], "message": message,
                            "source_response_hash": digest(response), "at": time.time(),
                            "unknown_billing_exposure_rub": 100, "original_request_replayed": False})
        if not repairs:
            raise ValueError("No unresolved interruptions to recover")
        if len(repairs) > 10:
            raise ValueError("Explicit recovery is bounded to one round")
        write_json(events / f"before_restart_{state['calls']:06d}.state.json", read_json(directory / "state.json"))
        revision = {"kind": "execution_revision", "before_call": state["calls"], "before": old,
                    "after": runner.manifest, "at": time.time(), "reason": "Human authorized completion; reduce simultaneous HTTP requests after interruption"}
        write_json(events / f"execution_revision_{state['calls']:06d}.json", revision)
        for repair in repairs:
            write_json(events / f"continuation_{repair['source_call']:06d}.json", repair)
        state["status"] = "running"
        write_json(directory / "manifest.json", runner.manifest)
        write_json(directory / "state.json", state)
        return {"resumed_at_call": state["calls"], "retained_answers": sum(len(a["answers"]) for a in state["agents"].values()),
                "restarted_turns": len(repairs), "workers": workers, "budget_rub": budget,
                "admission_exposure_rub": runner.spent()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--condition", default="fixed")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-rub", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.condition, args.seed, args.workers, args.budget_rub)))


if __name__ == "__main__":
    main()
