"""Run one bounded pilot condition and publish a nonsecret monitor status file."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from swarm_pipeline.runner import CONDITIONS, MODEL, digest, load_tasks
from swarm_pipeline.supervision import progress_from_calls, write_json


def main():
    # User-requested direct transport. Also applies to children launched by an
    # already-running batch supervisor with an older inherited environment.
    os.environ["NO_PROXY"] = "*"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, default="cooperative")
    parser.add_argument("--out", type=Path, default=Path("artifacts/pipeline-live"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--budget-rub", type=float, default=1000)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--max-calls", type=int, default=300)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-context-bytes", type=int, default=400000)
    parser.add_argument("--agents", type=int)
    parser.add_argument("--input-rub-per-million", type=float, default=43.2978)
    parser.add_argument("--output-rub-per-million", type=float, default=86.5834)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--protocol", choices=["v1", "v2"], default="v1")
    parser.add_argument("--partial-continuation", action="store_true")
    parser.add_argument("--workers", type=int, default=10, help="Concurrent solving turns (1 uses the serial runner)")
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error("--workers must be between 1 and 10")
    root = args.out.resolve()
    overrides_path = root / "runtime_overrides.json"
    if overrides_path.exists():
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
        workers = overrides.get("workers_by_condition", {}).get(args.condition, args.workers)
        if type(workers) is not int or not 1 <= workers <= 10:
            parser.error("Runtime worker override must be an integer between 1 and 10")
        args.workers = workers
    repo = Path(__file__).resolve().parents[1]
    protected = [repo / p for p in ("bench", "results", "embedding_eval", ".git", ".venv")]
    if root == repo or any(root == p or p in root.parents for p in protected):
        parser.error("Pilot output must be separate from source and baseline artifacts")
    tasks = load_tasks(args.items)
    job = f"{args.condition}-seed-{args.seed}"
    status_path = root / f"{job}.status.json"
    input_path = root / "inputs" / f"{digest(tasks)}.jsonl"
    root.mkdir(parents=True, exist_ok=True)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    frozen = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    if input_path.exists() and input_path.read_text(encoding="utf-8") != frozen:
        raise ValueError("Frozen task input differs")
    if not input_path.exists():
        input_path.write_text(frozen, encoding="utf-8")
    module = ("swarm_pipeline.parallel" if args.workers > 1 else
              "swarm_pipeline.stable_live" if args.partial_continuation else "swarm_pipeline.runner")
    if args.protocol == "v2":
        module = "swarm_pipeline.protocol_v2"
    command = [sys.executable, "-X", "utf8", "-m", module, "--items", str(input_path),
               "--out", str(root), "--condition", args.condition, "--seed", str(args.seed),
               "--budget-rub", str(args.budget_rub), "--max-tokens", str(args.max_tokens),
               "--max-calls", str(args.max_calls), "--model", args.model, "--temperature", str(args.temperature),
               "--max-context-bytes", str(args.max_context_bytes),
               "--input-rub-per-million", str(args.input_rub_per_million), "--output-rub-per-million", str(args.output_rub_per_million)]
    if args.agents is not None:
        command += ["--agents", str(args.agents)]
    if args.offline:
        command.append("--offline")
    if args.workers > 1 or args.protocol == "v2":
        command += ["--workers", str(args.workers)]
        if args.partial_continuation:
            command.append("--partial-continuation")
    directory = root / args.condition / f"seed_{args.seed}"
    details = {"condition": args.condition, "seed": args.seed, "tasks": len(tasks), "offline": args.offline,
               "workers": args.workers,
               "protocol": args.protocol,
               "budget_rub": args.budget_rub, "result_file": str(directory / "result.json"),
               "calls_directory": str(directory / "calls"), "supervisor_pid": os.getpid(),
               "stdout_file": str(root / f"{job}.stdout.log"), "stderr_file": str(root / f"{job}.stderr.log")}

    def publish(state, **extra):
        try:
            write_json(status_path, {"state": state, "event_key": job + ":" + state,
                                     "details": {**details, **extra}, "updated_at": time.time()})
            return True
        except PermissionError:
            print("Status file busy; supervisor continues tracking the existing worker.", file=sys.stderr, flush=True)
            return False

    publish("running", completed_calls=0)
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        with Path(details["stdout_file"]).open("a", encoding="utf-8") as stdout, Path(details["stderr_file"]).open("a", encoding="utf-8") as stderr:
            worker_started_at = time.time()
            child = subprocess.Popen(command, cwd=repo, stdout=stdout, stderr=stderr, creationflags=flags)
            details["worker_pid"] = child.pid
            while child.poll() is None:
                progress = progress_from_calls(directory)
                publish("running", completed_calls=progress.get("calls", 0), phase=progress.get("phase"), pending_call=progress.get("pending"))
                time.sleep(3)
        result_path = directory / "result.json"
        progress = progress_from_calls(directory)
        result = (json.loads(result_path.read_text(encoding="utf-8"))
                  if result_path.exists() and result_path.stat().st_mtime >= worker_started_at else {})
        failure_reason = None
        if child.returncode and not result:
            tail = Path(details["stderr_file"]).read_text(encoding="utf-8").splitlines()[-5:]
            failure_reason = ("budget_admission_limit" if any("Budget reservation would exceed" in line for line in tail)
                              else "worker_error_before_result")
        successful = child.returncode == 0 and result.get("status") == "completed"
        for _ in range(60):
            if publish("complete" if successful else "failed", exit_code=child.returncode,
                       completed_calls=result.get("calls", progress.get("calls")),
                       result_status=result.get("status"), failure_reason=failure_reason,
                       spent_root_rub=result.get("spent_root_rub")):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Worker finished but terminal status remains blocked")
        if not successful:
            raise SystemExit(child.returncode or 1)
    except Exception as exc:
        publish("failed", error_type=type(exc).__name__)
        raise


if __name__ == "__main__":
    main()
