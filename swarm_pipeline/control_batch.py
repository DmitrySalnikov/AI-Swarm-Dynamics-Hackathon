"""Supervise the three authorized independent control pilots sequentially."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from swarm_pipeline.runner import RUNNER_HASH
from swarm_pipeline.supervision import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/pipeline-live"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget-rub", type=float, default=1000)
    parser.add_argument("--reference-seed", type=int, help="cooperative reference seed, if replacing a failed control trial")
    parser.add_argument("--partial-continuation", action="store_true")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error("Workers must be 1..10")
    root = args.out.resolve()
    reference_seed = args.seed if args.reference_seed is None else args.reference_seed
    coop = root / "cooperative" / f"seed_{reference_seed}"
    manifest = json.loads((coop / "manifest.json").read_text(encoding="utf-8"))
    audit = json.loads((coop / "audit.json").read_text(encoding="utf-8"))
    if audit.get("integrity") != "passed" or audit.get("task_hash") != manifest["task_hash"]:
        raise ValueError("Cooperative audit and frozen task hash must match")
    if manifest["runner_hash"] != RUNNER_HASH:
        raise ValueError("Runner changed since the cooperative reference; preserve matched experimental versions")
    cfg = manifest["config"]
    if cfg["offline"]:
        raise ValueError("Live controls need a live cooperative reference")
    status = root / ("controls.status.json" if args.seed == 1 else f"controls-seed-{args.seed}.status.json")
    details = {"supervisor_pid": os.getpid(), "seed": args.seed, "budget_rub": args.budget_rub,
               "workers": args.workers,
               "conditions": ["fixed", "shuffled", "disjoint"], "completed": [], "active": None,
               "task_hash": manifest["task_hash"], "reference_seed": reference_seed}
    input_path = root / "inputs" / f"{manifest['task_hash']}.jsonl"

    def publish(state, **extra):
        try:
            write_json(status, {"state": state, "event_key": f"controls-seed-{args.seed}:{state}",
                                "details": {**details, **extra}, "updated_at": time.time()})
            return True
        except PermissionError:
            print("Batch status file busy; existing pilot continues.", file=sys.stderr, flush=True)
            return False

    publish("running")
    try:
        for condition in details["conditions"]:
            details["active"] = condition
            run_directory = root / condition / f"seed_{args.seed}"
            existing_result = run_directory / "result.json"
            if existing_result.exists() and read_json(existing_result).get("status") == "completed":
                existing_manifest = read_json(run_directory / "manifest.json")
                if existing_manifest["task_hash"] != manifest["task_hash"] or existing_manifest["runner_hash"] != RUNNER_HASH:
                    raise ValueError("Completed control differs from frozen reference")
                for key in ("model", "temperature", "max_tokens", "max_context_bytes", "agents", "price_in", "price_out", "offline"):
                    if existing_manifest["config"][key] != cfg[key]:
                        raise ValueError("Completed control has different scientific settings: " + key)
                if read_json(run_directory / "state.json")["status"] != "completed":
                    raise ValueError("Completed result and checkpoint disagree")
                from swarm_pipeline.audit import audit as audit_run
                audit_run(run_directory)
                saved = read_json(existing_result)
                details["completed"].append({"condition": condition, "calls": saved["calls"],
                    "spent_root_rub": saved["spent_root_rub"], "result_file": str(existing_result), "reused_completed_run": True})
                publish("running")
                continue
            command = [sys.executable, "-X", "utf8", "-m", "swarm_pipeline.pilot", "--items", str(input_path),
                       "--out", str(root), "--condition", condition, "--seed", str(args.seed),
                       "--budget-rub", str(args.budget_rub), "--max-tokens", str(cfg["max_tokens"]), "--max-calls", "300",
                       "--model", cfg["model"], "--temperature", str(cfg["temperature"]), "--agents", str(cfg["agents"]),
                       "--max-context-bytes", str(cfg["max_context_bytes"]),
                       "--workers", str(args.workers),
                       "--input-rub-per-million", str(cfg["price_in"] * 1e6), "--output-rub-per-million", str(cfg["price_out"] * 1e6)]
            if args.partial_continuation:
                command.append("--partial-continuation")
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            with (root / f"controls-{condition}.supervisor.stdout.log").open("a", encoding="utf-8") as stdout, (root / f"controls-{condition}.supervisor.stderr.log").open("a", encoding="utf-8") as stderr:
                child = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=flags)
                details["active_supervisor_pid"] = child.pid
                while child.poll() is None:
                    progress_path = root / f"{condition}-seed-{args.seed}.status.json"
                    progress = read_json(progress_path) if progress_path.exists() else {}
                    publish("running", progress=progress.get("details", {}))
                    time.sleep(5)
            if child.returncode:
                publish("failed", exit_code=child.returncode)
                raise SystemExit(child.returncode)
            result_path = root / condition / f"seed_{args.seed}" / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            from swarm_pipeline.audit import audit as audit_run
            checked = audit_run(result_path.parent)
            details["completed"].append({"condition": condition, "calls": result["calls"], "spent_root_rub": result["spent_root_rub"], "result_file": str(result_path)})
        details["active"] = None
        for _ in range(60):
            if publish("complete", spent_root_rub=details["completed"][-1]["spent_root_rub"]):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Controls finished but terminal status remains blocked")
    except Exception as exc:
        publish("failed", error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
