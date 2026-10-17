from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from swarm_pipeline.audit import audit
from swarm_pipeline.parallel import ParallelRunner
from swarm_pipeline.runner import Config, OfflineClient

TASKS = [{"id": f"demo/{i}", "prompt": f"Solve {i}", "answer": "unused", "check": {"type": "exact"}} for i in range(4)]


class OverlapClient(OfflineClient):
    def __init__(self):
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.requests = []
        self.busy_agents = set()
        self.agent_overlap = False

    def complete(self, request):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            if request["agent"] in self.busy_agents:
                self.agent_overlap = True
            self.busy_agents.add(request["agent"])
            self.requests.append(request)
        time.sleep(.025)
        result = super().complete(request)
        result["usage"] = {"prompt_tokens": 1, "completion_tokens": 1}
        with self.lock:
            self.active -= 1
            self.busy_agents.remove(request["agent"])
        return result


class ParallelTests(unittest.TestCase):
    def test_empty_timeout_continues_as_logged_new_turn(self):
        class TimedOut(OverlapClient):
            def complete(self, request):
                if request["call"] == 0:
                    return {"content": "", "reasoning": "", "usage": None, "response_id": None,
                            "finish_reason": "transport_error", "error_type": "TimeoutError"}
                copied = dict(request)
                copied["messages"] = [m for m in request["messages"] if m["role"] != "assistant" or m["content"]]
                return super().complete(copied)
        with tempfile.TemporaryDirectory() as temp:
            runner = ParallelRunner(TASKS[:1], Config(condition="fixed", agents=1, offline=True), Path(temp), TimedOut(),
                                    workers=2, partial_continuation=True)
            with redirect_stdout(io.StringIO()), patch("swarm_pipeline.parallel.time.sleep"):
                self.assertEqual(runner.run()["status"], "completed")
            self.assertEqual(audit(runner.directory)["automatic_transport_retries"], 1)
        from swarm_pipeline.stable_live import RetainingClient
        self.assertEqual(RetainingClient("fake").timeout, 900)

    def test_missing_usage_uses_request_estimate_without_flat_buffer(self):
        class Empty:
            def complete(self, request):
                return {"content": "", "reasoning": "", "usage": None, "finish_reason": "transport_error",
                        "http_status": 500, "response_id": None}
        with tempfile.TemporaryDirectory() as temp:
            cfg = Config(condition="fixed", agents=1, budget_rub=1, max_tokens=10, price_in=0, price_out=.01)
            runner = ParallelRunner(TASKS[:1], cfg, Path(temp), Empty(), workers=1)
            with redirect_stdout(io.StringIO()):
                result = runner.run()
            self.assertEqual(result["status"], "provider_incomplete")
            self.assertAlmostEqual(result["admission_exposure_rub"], .1)
            self.assertAlmostEqual(result["unreceipted_estimates_rub"], .1)
            self.assertEqual(result["usage_backed_spend_rub"], 0)

    def test_http_retry_is_logged_bounded_and_auditable(self):
        class Flaky(OverlapClient):
            def __init__(self, always=False):
                super().__init__()
                self.always = always
                self.count = 0
            def complete(self, request):
                self.count += 1
                if self.always or self.count == 1:
                    return {"content": "", "reasoning": "", "usage": None, "response_id": None,
                            "finish_reason": "transport_error", "error_type": "HTTPError", "http_status": 500}
                copied = dict(request)
                copied["messages"] = [m for m in request["messages"] if m["role"] != "assistant" or m["content"]]
                return super().complete(copied)
        for always in (False, True):
            with self.subTest(always=always), tempfile.TemporaryDirectory() as temp:
                client = Flaky(always)
                runner = ParallelRunner(TASKS[:1], Config(condition="fixed", agents=1, offline=True), Path(temp), client,
                                        workers=2, partial_continuation=True)
                with redirect_stdout(io.StringIO()), patch("swarm_pipeline.parallel.time.sleep"):
                    result = runner.run()
                if always:
                    self.assertEqual(result["status"], "provider_incomplete")
                    self.assertEqual(client.count, 4)
                else:
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(audit(runner.directory)["automatic_http_retries"], 1)

    def test_explicit_restart_preserves_answers_and_revision_audits(self):
        from swarm_pipeline.resume_parallel import prepare
        class Interrupted(OverlapClient):
            def complete(self, request):
                if request["call"] == 4:
                    return {"content": "", "reasoning": "", "usage": None, "response_id": None,
                            "finish_reason": "transport_error", "error_type": "HTTPError"}
                copied = dict(request)
                copied["messages"] = [m for m in request["messages"] if m["role"] != "assistant" or m["content"]]
                return super().complete(copied)
        with tempfile.TemporaryDirectory() as temp:
            client = Interrupted()
            cfg = Config(condition="fixed", agents=4, offline=True)
            runner = ParallelRunner(TASKS, cfg, Path(temp), client, workers=4)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(runner.run()["status"], "provider_incomplete")
            old_requests = [p.read_bytes() for p in sorted((runner.directory / "calls").glob("*/request.json"))]
            recovery = prepare(Path(temp), "fixed", 1, 2, 2000)
            self.assertEqual(recovery["restarted_turns"], 1)
            resumed = ParallelRunner(TASKS, Config(condition="fixed", agents=4, offline=True, budget_rub=2000), Path(temp), client,
                                     workers=2, partial_continuation=True)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(resumed.run()["status"], "completed")
            for number, contents in enumerate(old_requests):
                self.assertEqual((runner.directory / "calls" / f"{number:06d}" / "request.json").read_bytes(), contents)
            report = audit(resumed.directory)
            self.assertEqual(report["explicit_turn_restarts"], 1)
            self.assertEqual(report["execution_revisions"], 1)

    def test_multiple_partial_streams_continue_privately_and_audit(self):
        class PartialClient(OverlapClient):
            def complete(self, request):
                if request["call"] in (0, 1):
                    return {"content": "", "reasoning": f"PRIVATE_FRAGMENT_{request['agent']}",
                            "response_id": f"partial-{request['call']}", "usage": None,
                            "finish_reason": "stream_incomplete"}
                copied = dict(request)
                copied["messages"] = [m for m in request["messages"] if m["role"] != "assistant" or m["content"]]
                return super().complete(copied)
        with tempfile.TemporaryDirectory() as temp:
            client = PartialClient()
            runner = ParallelRunner(TASKS, Config(condition="fixed", agents=4, offline=True), Path(temp), client,
                                    workers=4, partial_continuation=True)
            with redirect_stdout(io.StringIO()):
                result = runner.run()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["continuations"], 2)
            self.assertEqual(audit(runner.directory)["usage_incomplete_calls"], [0, 1])
            for request in client.requests:
                for other in (1, 2):
                    if other != request["agent"]:
                        self.assertNotIn(f"PRIVATE_FRAGMENT_{other}", json.dumps(request["messages"]))

    def test_all_modes_overlap_keep_histories_and_pass_integrity(self):
        for condition in ("fixed", "shuffled", "disjoint", "cooperative"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as temp:
                client = OverlapClient()
                runner = ParallelRunner(TASKS, Config(condition=condition, agents=4, offline=True), Path(temp), client, workers=3)
                with redirect_stdout(io.StringIO()):
                    result = runner.run()
                self.assertEqual(result["status"], "completed")
                self.assertEqual(client.peak, 3)
                self.assertFalse(client.agent_overlap)
                self.assertEqual(audit(runner.directory)["integrity"], "passed")
                for request in client.requests:
                    if condition != "cooperative":
                        self.assertEqual(request["context"]["inbox"], [])
                with redirect_stdout(io.StringIO()):
                    runner.run()
                self.assertEqual(len(client.requests), result["calls"])

    def test_round_messages_only_enter_next_round(self):
        with tempfile.TemporaryDirectory() as temp:
            client = OverlapClient()
            runner = ParallelRunner(TASKS, Config(condition="cooperative", agents=4, offline=True), Path(temp), client, workers=4)
            with redirect_stdout(io.StringIO()):
                runner.run()
            for request in client.requests:
                self.assertTrue(all(m["call"] < request["batch_start"] for m in request["context"]["inbox"]))
            self.assertTrue(any(any(m["message"].endswith("working on demo/0") for m in r["context"]["inbox"]) for r in client.requests))

    def test_budget_is_reserved_before_parallel_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            client = OverlapClient()
            config = Config(condition="fixed", agents=4, budget_rub=1, max_tokens=1, price_in=0, price_out=1)
            runner = ParallelRunner(TASKS, config, Path(temp), client, workers=4)
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Budget"):
                runner.run()
            self.assertEqual(len(client.requests), 1)
            self.assertEqual(runner.spent(), 1)

    def test_receipted_round_recovers_without_replay(self):
        class Crash(BaseException):
            pass
        with tempfile.TemporaryDirectory() as temp:
            client = OverlapClient()
            runner = ParallelRunner(TASKS, Config(condition="fixed", agents=4, offline=True), Path(temp), client, workers=4)
            with patch.object(runner, "apply", side_effect=Crash), self.assertRaises(Crash):
                runner.run()
            self.assertEqual(len(client.requests), 4)
            with redirect_stdout(io.StringIO()):
                result = runner.run()
            self.assertEqual(len(client.requests), result["calls"])
            self.assertEqual(audit(runner.directory)["integrity"], "passed")

    def test_unknown_round_is_not_replayed(self):
        class Crash(BaseException):
            pass
        class CrashClient:
            def complete(self, request):
                raise Crash()
        with tempfile.TemporaryDirectory() as temp:
            runner = ParallelRunner(TASKS, Config(condition="fixed", agents=4, offline=True), Path(temp), CrashClient(), workers=4)
            with self.assertRaises(Crash):
                runner.run()
            with self.assertRaisesRegex(RuntimeError, "unknown billing/outcome"):
                runner.run()

    def test_crash_before_checkpoint_does_not_overwrite_requests(self):
        from swarm_pipeline.parallel import write_json
        class Crash(BaseException):
            pass
        def crashing_write(path, value):
            if path.name == "state.json":
                raise Crash()
            write_json(path, value)
        with tempfile.TemporaryDirectory() as temp:
            client = OverlapClient()
            runner = ParallelRunner(TASKS, Config(condition="fixed", agents=4, offline=True), Path(temp), client, workers=4)
            with patch("swarm_pipeline.parallel.write_json", side_effect=crashing_write), self.assertRaises(Crash):
                runner.run()
            original = (runner.directory / "calls" / "000000" / "request.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "unknown billing/outcome"):
                runner.run()
            self.assertEqual((runner.directory / "calls" / "000000" / "request.json").read_bytes(), original)
            self.assertEqual(client.requests, [])


if __name__ == "__main__":
    unittest.main()
