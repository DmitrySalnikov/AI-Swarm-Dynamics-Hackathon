import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from swarm_pipeline.runner import Config, OfflineClient
from swarm_pipeline.stable_live import RecoveringRunner, RetainingClient


class PartialContinuationTests(unittest.TestCase):
    def test_transport_keeps_observed_fragment_when_stream_breaks(self):
        class BrokenStream:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def __iter__(self):
                yield b'data: {"id":"partial-id","choices":[{"delta":{"reasoning_content":"observed work"},"finish_reason":null}]}\n'
                raise ConnectionResetError()
        request = {"model": "test", "messages": [], "temperature": 1, "max_tokens": 100, "seed": 1}
        with patch("urllib.request.urlopen", return_value=BrokenStream()):
            response = RetainingClient("fake-test-key").complete(request)
        self.assertEqual(response["reasoning"], "observed work")
        self.assertEqual(response["response_id"], "partial-id")
        self.assertEqual(response["finish_reason"], "transport_error")
        self.assertIsNone(response["usage"])

    def test_continuation_uses_private_fragment_in_distinct_request(self):
        tasks = [{"id": "test/a", "prompt": "Solve A", "answer": "unused", "check": {"type": "exact"}}]
        class PartialClient(OfflineClient):
            def __init__(self):
                self.requests = []
            def complete(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    return {"content": "", "reasoning": "OBSERVED_PRIVATE_FRAGMENT", "usage": None,
                            "finish_reason": "stream_incomplete", "response_id": "known-partial"}
                # The scripted fixture parses assistant JSON; unlike a real
                # model it cannot read the empty interrupted assistant turn.
                scripted = dict(request)
                scripted["messages"] = [m for m in request["messages"] if m["role"] != "assistant" or m["content"]]
                result = super().complete(scripted)
                result["usage"] = {"prompt_tokens": 10, "completion_tokens": 10}
                return result
        with tempfile.TemporaryDirectory() as temp:
            client = PartialClient()
            runner = RecoveringRunner(tasks, Config(condition="fixed", agents=1, offline=True), Path(temp), client)
            with redirect_stdout(io.StringIO()):
                result = runner.run()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["continuations"], 1)
            self.assertIn("OBSERVED_PRIVATE_FRAGMENT", json.dumps(client.requests[1]["messages"]))
            self.assertNotEqual(client.requests[0]["seed"], client.requests[1]["seed"])
            self.assertEqual(sum(r["call"] == 0 for r in client.requests), 1)
            from swarm_pipeline.audit import audit
            self.assertEqual(audit(runner.directory)["partial_continuations"], 1)

    def test_empty_unreceipted_response_is_never_replayed(self):
        tasks = [{"id": "test/a", "prompt": "Solve A", "answer": "unused", "check": {"type": "exact"}}]
        class EmptyClient:
            def __init__(self):
                self.calls = 0
            def complete(self, request):
                self.calls += 1
                return {"content": "", "reasoning": "", "usage": None, "finish_reason": "transport_error", "response_id": None}
        with tempfile.TemporaryDirectory() as temp:
            client = EmptyClient()
            runner = RecoveringRunner(tasks, Config(condition="fixed", agents=1, offline=True), Path(temp), client)
            with redirect_stdout(io.StringIO()):
                result = runner.run()
            self.assertEqual(result["status"], "provider_incomplete")
            self.assertEqual(client.calls, 1)
            self.assertEqual(result["continuations"], 0)


if __name__ == "__main__":
    unittest.main()
