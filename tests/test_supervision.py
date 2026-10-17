import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from swarm_pipeline.runner import write_json as runner_write_json
from swarm_pipeline.supervision import progress_from_calls, read_json, write_json


class SupervisionTests(unittest.TestCase):
    def test_lock_retry_preserves_old_status_until_atomic_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            write_json(path, {"state": "old"})
            replace = os.replace
            attempts = []
            def busy_then_replace(source, target):
                attempts.append(1)
                if len(attempts) <= 2:
                    self.assertEqual(read_json(path)["state"], "old")
                    raise PermissionError("simulated reader lock")
                return replace(source, target)
            with patch("swarm_pipeline.supervision.os.replace", side_effect=busy_then_replace), patch("swarm_pipeline.supervision.time.sleep"):
                write_json(path, {"state": "new"})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(read_json(path)["state"], "new")
            self.assertFalse(list(Path(directory).glob("*.part")))

    def test_supervisor_observes_journals_without_opening_mutable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "state.json").write_text("not read by supervision")
            runner_write_json(root / "calls/000000/event.json", {"call": 0})
            runner_write_json(root / "calls/000001/request.json", {"call": 1, "context": {"phase": "solving"}})
            self.assertEqual(progress_from_calls(root), {"calls": 1, "phase": "solving", "pending": 1})


if __name__ == "__main__":
    unittest.main()
