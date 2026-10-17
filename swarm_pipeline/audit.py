"""Verify persisted context/delivery/export integrity without model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from embedding_eval.gen_runs import grade
from swarm_pipeline.runner import write_json


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    tasks = {t["id"]: t for t in manifest["tasks"]}
    histories, inboxes, exports = {}, {}, {}
    repairs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory / "operational_events").glob("continuation_*.json"))]
    covered = {r["source_call"] for r in repairs}
    revisions = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory / "operational_events").glob("execution_revision_*.json"))]
    phases, errors, calls = {}, [], []
    batch_inboxes = None
    batch_start = None
    batch_agents = set()
    batch_phase = None
    for number, folder in enumerate(sorted((directory / "calls").iterdir())):
        if not folder.is_dir():
            continue
        request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        response = json.loads((folder / "response.json").read_text(encoding="utf-8"))
        event = json.loads((folder / "event.json").read_text(encoding="utf-8"))
        agent = str(request["agent"])
        execution = manifest.get("execution", {})
        for revision in reversed(revisions):
            if number < revision["before_call"]:
                execution = revision["before"].get("execution", {})
        if "batch_start" in request:
            assert execution.get("scheduler") == "parallel-solving-rounds-v1", "Unrecognized parallel scheduler"
            if request["batch_start"] != batch_start:
                assert request["batch_start"] == number, "Invalid round boundary"
                batch_start = number
                batch_inboxes = json.loads(json.dumps(inboxes))
                batch_agents = set()
                batch_phase = request["context"]["phase"]
            assert agent not in batch_agents, "Agent has multiple turns in one round"
            batch_agents.add(agent)
            assert len(batch_agents) <= execution["workers"], "Concurrency bound exceeded"
            assert batch_phase == request["context"]["phase"], "Phase changed within round"
            assert batch_phase == "solving" or (batch_phase == "exchange" and manifest.get("protocol_version") == "v2") or len(batch_agents) == 1, "Discussion must remain serial"
        for repair in repairs:
            if repair["before_call"] == number:
                target = str(repair["agent"])
                histories[target].append(repair["message"])
                source = json.loads((directory / "calls" / f"{repair['source_call']:06d}" / "response.json").read_text(encoding="utf-8"))
                from swarm_pipeline.runner import digest
                assert digest(source) == repair["source_response_hash"], "Recovery source changed"
        assert request["call"] == number, "Nonconsecutive calls"
        assert json.loads(request["messages"][-1]["content"]) == request["context"], "Input/context mismatch"
        history = histories.setdefault(agent, [request["messages"][0]])
        assert request["messages"][:-1] == history, "History did not carry forward"
        inboxes.setdefault(agent, [])
        expected_inbox = batch_inboxes.get(agent, []) if "batch_start" in request else inboxes[agent]
        assert request["context"]["inbox"] == expected_inbox, "Delivery/input mismatch"
        assert response["finish_reason"] == "stop" or number in covered, "Unresolved incomplete provider response"
        assert response.get("usage") or number in covered, "Missing provider usage"
        for t in request["context"].get("tasks", []):
            assert set(t) == {"id", "prompt"}, "Private task metadata in prompt"
        if request["context"].get("task"):
            assert set(request["context"]["task"]) == {"id", "prompt"}, "Private task metadata in prompt"
        history += [request["messages"][-1], {"role": "assistant", "content": response["content"]}]
        if event["action_error"]:
            errors.append({"call": number, "agent": request["agent"], "phase": event["phase"], "error": event["action_error"]})
            history.append({"role": "user", "content": "Action rejected: " + event["action_error"] + ". Follow the JSON action protocol."})
        for message in event["deliveries"]:
            inboxes.setdefault(str(message["to"]), []).append(message)
        outputs = exports.setdefault(agent, [])
        outputs += [response[field] + "\n\n" for field in ("reasoning", "content") if response.get(field)]
        phase = phases.setdefault(event["phase"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_rub": 0.0, "delivered_messages": 0})
        phase["calls"] += 1
        phase["input_tokens"] += (response.get("usage") or {}).get("prompt_tokens", 0)
        phase["output_tokens"] += (response.get("usage") or {}).get("completion_tokens", 0)
        phase["cost_rub"] += response["cost_rub"]
        phase["delivered_messages"] += len(event["deliveries"])
        calls.append((request, response, event))
    assert state["pending"] is None and len(calls) == state["calls"], "Pending or missing calls"
    assert state["status"] == result["status"] == "completed", "Run not complete"
    for agent, record in state["agents"].items():
        assert record["history"] == histories[agent], "Checkpoint history differs"
        assert record["inbox"] == inboxes[agent], "Checkpoint inbox differs"
        actual = (directory / "generated" / f"agent_{agent}.generated.txt").read_text(encoding="utf-8")
        assert actual == "".join(exports[agent]), "Generated export differs from own outputs"
    own = {i: {t: grade(tasks[t], answer) for t, answer in a["answers"].items()} for i, a in state["agents"].items()}
    final = {i: {t: grade(tasks[t], answer) for t, answer in a.get("final", a["answers"]).items()} for i, a in state["agents"].items()}
    disagreement = {}
    for task in tasks:
        answers = {i: a["final"][task] for i, a in state["agents"].items() if "final" in a}
        normalized = {v.replace(" ", "").replace("%", "") for v in answers.values()}
        if len(normalized) > 1:
            disagreement[task] = answers
    phase_order = [e["phase"] for _, _, e in calls]
    rank = {"negotiation": 0, "solving": 1, "exchange": 2}
    assert [rank[p] for p in phase_order] == sorted(rank[p] for p in phase_order), "Phase regression"
    report = {"integrity": "passed", "condition": manifest["config"]["condition"], "seed": manifest["config"]["seed"],
              "calls": len(calls), "phases": phases, "action_errors": errors,
              "own_correct": sum(sum(scores.values()) for scores in own.values()),
              "own_total": sum(len(scores) for scores in own.values()),
              "final_per_agent_correct": {i: sum(scores.values()) for i, scores in final.items()},
              "final_per_agent_total": {i: len(scores) for i, scores in final.items()},
              "final_disagreements": disagreement, "pilot_cost_rub": sum(r["cost_rub"] for _, r, _ in calls),
              "elapsed_minutes": (max(r["completed_at"] for _, r, _ in calls) - min(q["started_at"] for q, _, _ in calls)) / 60,
              "provider_models": dict(Counter(r.get("actual_model", "unknown") for _, r, _ in calls)),
              "task_hash": manifest["task_hash"], "runner_hash": manifest["runner_hash"]}
    report["partial_continuations"] = sum(r["kind"] == "private_fragment_continuation" for r in repairs)
    report["explicit_turn_restarts"] = sum(r["kind"] == "explicit_interrupted_turn_restart" for r in repairs)
    report["execution_revisions"] = len(revisions)
    report["automatic_http_retries"] = sum(r["kind"] == "automatic_http_error_retry" for r in repairs)
    report["automatic_transport_retries"] = sum(r["kind"] == "automatic_transport_error_retry" for r in repairs)
    if manifest.get("protocol_version") == "v2":
        from swarm_pipeline.runner import digest
        packet=json.loads((directory/"shared-packet.json").read_text(encoding="utf-8"))
        expected={"schema":"shared-results-v2","results":[]}
        for task in tasks:
            owner=next(i for i,a in state["agents"].items() if task in a["answers"])
            a=state["agents"][owner]
            expected["results"].append({"task_id":task,"agent":int(owner),"answer":a["answers"][task],"notes":a["solution_notes"][task]})
        assert packet == state["shared_packet"] == expected, "Shared packet differs from published solutions"
        commits=Counter()
        submits=Counter()
        for q,r,e in calls:
            if q["context"]["phase"] != "negotiation":
                assert not e["deliveries"], "Unexpected solving or commit discussion"
            if q["context"]["phase"] == "exchange":
                assert q["context"]["shared_packet"] == packet, "Commit input packet differs"
                assert q["context"]["shared_packet_hash"] == digest(packet), "Commit packet hash differs"
            if not e["action_error"]:
                if e["phase"] == "solving": submits[q["agent"]]+=1
                if e["phase"] == "exchange": commits[q["agent"]]+=1
        assert set(submits)==set(commits)==set(range(1,len(state["agents"])+1))
        assert all(v==1 for v in submits.values()) and all(v==1 for v in commits.values()), "Multiple successful solves or commits"
        answers={r["task_id"]:r["answer"] for r in packet["results"]}
        assert all(a["final"]==answers for a in state["agents"].values()), "Published values changed at commit"
        report["protocol_version"]="v2"
        report["shared_packet_hash"]=digest(packet)
        report["single_successful_submit_and_commit_per_agent"]=True
        report["commit_retries"]=sum(r["kind"]=="v2_commit_retry" for r in repairs)
    report["usage_incomplete_calls"] = sorted(covered)
    write_json(directory / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = audit(args.directory)
    print(json.dumps({k: report[k] for k in ("integrity", "calls", "own_correct", "own_total", "final_per_agent_correct", "pilot_cost_rub", "elapsed_minutes", "action_errors")}, indent=2))


if __name__ == "__main__":
    main()
