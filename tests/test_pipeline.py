from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from swarm_pipeline.runner import CONDITIONS, CloudClient, Config, OfflineClient, Runner, write_json

TASKS = [
    {"id": "demo/a", "prompt": "Solve task A", "answer": "PRIVATE_REFERENCE_A", "check": {"type": "exact"}},
    {"id": "demo/b", "prompt": "Solve task B", "answer": "PRIVATE_REFERENCE_B", "check": {"type": "exact"}},
]


class RecordingClient(OfflineClient):
    def __init__(self):
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        result = super().complete(request)
        result["reasoning"] = f"PRIVATE_REASONING_{request['agent']}"
        return result


class ProofClient(RecordingClient):
    """Submit changes causally when a peer's message is delivered."""
    def complete(self, request):
        result = super().complete(request)
        action = json.loads(result["content"])
        if request["context"]["phase"] == "negotiation" and request["agent"] == 1 and not any(r["agent"] == 1 for r in self.requests[:-1]):
            action = {"action": "send", "to": "all", "message": "NEGOTIATION_SENTINEL"}
        if request["context"]["phase"] == "solving" and action["action"] == "send":
            action["message"] = "SOLVING_SENTINEL"
        if action["action"] == "submit":
            action["answer"] = "with_peer" if any(m["message"] == "SOLVING_SENTINEL" for m in request["context"]["inbox"]) else "without_peer"
        result["content"] = json.dumps(action)
        return result


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def run_case(self, condition="cooperative", client=None, **kwargs):
        config = Config(condition=condition, agents=2, offline=True, **kwargs)
        client = client or RecordingClient()
        runner = Runner(TASKS, config, self.root / condition, client)
        with redirect_stdout(io.StringIO()):
            result = runner.run()
        return runner, client, result

    def test_all_conditions_complete_and_controls_have_no_peers(self):
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                runner, client, result = self.run_case(condition)
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["spent_root_rub"], 0)
                for request in client.requests:
                    text = json.dumps(request["messages"])
                    self.assertNotIn("PRIVATE_REFERENCE_", text)
                    self.assertNotIn("PRIVATE_REASONING_", text)
                    if condition != "cooperative":
                        self.assertEqual(request["context"]["inbox"], [])
                if condition == "disjoint":
                    self.assertEqual(result["agents"]["1"]["order"], ["demo/a", "demo/b"])
                    self.assertEqual(result["agents"]["2"]["order"], ["demo/b", "demo/a"])
                for agent in result["agents"]:
                    own = (runner.directory / "generated" / f"agent_{agent}.generated.txt").read_text(encoding="utf-8")
                    other = "2" if agent == "1" else "1"
                    self.assertIn(f"PRIVATE_REASONING_{agent}", own)
                    self.assertNotIn(f"PRIVATE_REASONING_{other}", own)

    def test_messages_affect_later_solves_and_negotiation_is_retained(self):
        runner, client, result = self.run_case(client=ProofClient())
        submits = [r for r in client.requests if r["context"]["phase"] == "solving" and any(m["message"] == "SOLVING_SENTINEL" for m in r["context"]["inbox"])]
        self.assertTrue(submits)
        for agent in runner.state["agents"].values():
            self.assertTrue(all(answer == "with_peer" for answer in agent["answers"].values()))
        solve_requests = [r for r in client.requests if r["context"]["phase"] == "solving"]
        self.assertTrue(all("NEGOTIATION_SENTINEL" in json.dumps(r["messages"]) for r in solve_requests))
        exchange_requests = [r for r in client.requests if r["context"]["phase"] == "exchange"]
        self.assertTrue(all("private memory" in json.dumps(r["messages"]) for r in exchange_requests))

    def test_previous_solution_enters_next_task_context(self):
        runner, client, result = self.run_case("fixed")
        second_task = [r for r in client.requests if r["context"]["current_task"] == "demo/b"]
        self.assertTrue(second_task)
        self.assertTrue(all(any(m["role"] == "assistant" and json.loads(m["content"]).get("action") == "submit" and json.loads(m["content"]).get("task_id") == "demo/a" for m in r["messages"]) for r in second_task))

    def test_completed_resume_does_not_call_provider(self):
        runner, client, first = self.run_case()
        before = len(client.requests)
        with redirect_stdout(io.StringIO()):
            again = runner.run()
        self.assertEqual(len(client.requests), before)
        self.assertEqual(first, again)

    def test_completed_response_recovers_without_duplicate_call(self):
        class Crash(BaseException):
            pass
        client = RecordingClient()
        runner = Runner(TASKS, Config(condition="cooperative", agents=2, offline=True), self.root, client)
        original = runner.apply
        runner.apply = lambda *args: (_ for _ in ()).throw(Crash())
        with self.assertRaises(Crash), redirect_stdout(io.StringIO()):
            runner.run()
        self.assertEqual(len(client.requests), 1)
        runner.apply = original
        with redirect_stdout(io.StringIO()):
            result = runner.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(sum(r["call"] == 0 for r in client.requests), 1)

    def test_interrupted_unknown_request_is_not_replayed(self):
        class CrashClient:
            def complete(self, request):
                raise KeyboardInterrupt()
        runner = Runner(TASKS, Config(condition="cooperative", agents=2, offline=True), self.root, CrashClient())
        with self.assertRaises(KeyboardInterrupt):
            runner.run()
        runner.client = RecordingClient()
        with self.assertRaisesRegex(RuntimeError, "unknown billing"):
            runner.run()
        self.assertEqual(runner.client.requests, [])

    def test_budget_and_context_limits_prevent_requests(self):
        client = RecordingClient()
        runner = Runner(TASKS, Config(condition="fixed", agents=2, budget_rub=0.001), self.root / "budget", client)
        with self.assertRaisesRegex(RuntimeError, "Budget"):
            runner.run()
        self.assertEqual(client.requests, [])
        runner = Runner(TASKS, Config(condition="fixed", agents=2, offline=True, max_context_bytes=1), self.root / "context", client)
        with self.assertRaisesRegex(RuntimeError, "Context"):
            runner.run()
        self.assertEqual(client.requests, [])

    def test_task_or_config_change_cannot_reuse_manifest(self):
        runner, client, _ = self.run_case()
        runner.config = replace(runner.config, model="different-model")
        runner.manifest["config"] = {**runner.manifest["config"], "model": "different-model"}
        with self.assertRaisesRegex(ValueError, "Manifest differs"):
            runner.run()

    def test_illegal_peer_message_in_independent_condition_is_rejected(self):
        class BadClient(RecordingClient):
            def complete(self, request):
                result = super().complete(request)
                result["content"] = json.dumps({"action": "send", "to": "all", "message": "forbidden"})
                return result
        runner, client, result = self.run_case("fixed", client=BadClient(), max_calls=2)
        self.assertEqual(result["status"], "call_limit")
        self.assertTrue(all(not a["inbox"] for a in runner.state["agents"].values()))
        events = [json.loads(p.read_text()) for p in runner.directory.glob("calls/*/event.json")]
        self.assertTrue(all(e["action_error"] for e in events))

    def test_completed_solver_can_receive_and_answer_new_requests(self):
        runner = Runner(TASKS, Config(condition="cooperative", agents=2, offline=True), self.root, RecordingClient())
        runner.state = runner.initial_state()
        runner.state["phase"] = "solving"
        a, b = runner.state["agents"]["1"], runner.state["agents"]["2"]
        a["order"], b["order"] = ["demo/a"], ["demo/b"]
        a["answers"] = {"demo/a": "my-completed-answer"}
        a["inbox"].append({"from": 2, "to": 1, "message": "Please help", "call": 0})
        self.assertEqual(runner.next_agent(), 1)
        context = runner.context(1)
        self.assertIsNone(context["current_task"])
        self.assertIn("Respond to peers", context["instruction"])
        a["inbox"].append({"from": 2, "to": 1, "message": "More detail", "call": 1})
        self.assertEqual(len(context["inbox"]), 1)
        request = {"call": 0, "agent": 1, "context": context}
        response = {"content": json.dumps({"action": "send", "to": 2, "message": "Here is help"}), "finish_reason": "stop"}
        event = runner.apply(request, response)
        self.assertEqual(event["deliveries"][0]["message"], "Here is help")
        self.assertEqual(a["seen_inbox"], 1)
        invalid = {"content": json.dumps({"action": "submit", "task_id": None, "answer": "invalid"}), "finish_reason": "stop"}
        self.assertIsNotNone(runner.apply(request, invalid)["action_error"])
        self.assertEqual(a["answers"], {"demo/a": "my-completed-answer"})

    def test_streaming_adapter_preserves_reasoning_content_and_usage(self):
        chunks = [
            {"id": "req-1", "choices": [{"delta": {"reasoning_content": "private notes"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": '{"action":"work","notes":"kept"}'}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 10}},
        ]
        stream = io.BytesIO(b"".join(("data: " + json.dumps(c) + "\n\n").encode() for c in chunks) + b"data: [DONE]\n")
        request = {"model": "test", "messages": [], "temperature": 1, "max_tokens": 100, "seed": 1}
        with patch("urllib.request.urlopen", return_value=stream):
            response = CloudClient("fake-test-key").complete(request)
        self.assertEqual(response["reasoning"], "private notes")
        self.assertEqual(response["finish_reason"], "stop")
        self.assertEqual(response["usage"]["completion_tokens"], 10)
        self.assertEqual(response["response_id"], "req-1")

    def test_audit_checks_recorded_history_and_detects_tampering(self):
        from swarm_pipeline.audit import audit
        class UsageClient(RecordingClient):
            def complete(self, request):
                response = super().complete(request)
                response["usage"] = {"prompt_tokens": 20, "completion_tokens": 10}
                return response
        runner, _, _ = self.run_case(client=UsageClient())
        self.assertEqual(audit(runner.directory)["integrity"], "passed")
        request_path = runner.directory / "calls" / "000004" / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["messages"].pop(1)
        write_json(request_path, request)
        with self.assertRaisesRegex(AssertionError, "History did not carry"):
            audit(runner.directory)


if __name__ == "__main__":
    unittest.main()
