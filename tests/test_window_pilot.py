import json
from pathlib import Path
import tempfile
import time
import unittest

from swarm_pipeline.runner import Config
from swarm_pipeline.window_pilot import DistributionRunner, GradualRunner, GradualOfflineClient

TASKS = [{'id': f't{i}', 'prompt': f'Task {i}', 'answer': 'PRIVATE_REFERENCE',
          'check': {'type': 'exact'}} for i in range(3)]


class WindowPilotTests(unittest.TestCase):
    def test_temporary_reservations_wait_for_receipts(self):
        from swarm_pipeline.gradual_scheduler import BudgetAwareGradualRunner
        class Delayed(GradualOfflineClient):
            def complete(self, q):
                if q['context']['phase'] == 'solving':
                    time.sleep(.02 if q['agent'] == 1 else .15)
                return super().complete(q)
        with tempfile.TemporaryDirectory() as tmp:
            r = BudgetAwareGradualRunner(TASKS[:2], Config('cooperative', 2, offline=False,
                max_tokens=10, budget_rub=24, price_in=0, price_out=1), Path(tmp), Delayed(), workers=2)
            self.assertEqual(r.run()['status'], 'completed')
            self.assertEqual(r.spent(), 10)
            self.assertFalse(r.state['pending_calls'])

    def test_distribution_stops_before_solving(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = DistributionRunner(TASKS, Config('cooperative', 3, offline=True), Path(tmp), GradualOfflineClient(), workers=3)
            self.assertEqual(r.run()['status'], 'completed')
            self.assertEqual(r.state['calls'], 6)
            self.assertEqual(len({a['order'][0] for a in r.state['agents'].values()}), 3)
            self.assertFalse(any(a['answers'] for a in r.state['agents'].values()))

    def test_overlap_and_frozen_inflight_context(self):
        class Delayed(GradualOfflineClient):
            def complete(self, q):
                if q['context']['phase'] == 'solving':
                    time.sleep(.02 if q['agent'] == 1 else .25)
                return super().complete(q)
        with tempfile.TemporaryDirectory() as tmp:
            r = GradualRunner(TASKS, Config('cooperative', 3, offline=True), Path(tmp), Delayed(), workers=3)
            self.assertEqual(r.run()['status'], 'completed')
            events = [json.loads(p.read_text()) for p in sorted(r.directory.glob('calls/*/event.json'))]
            reports = [e for e in events if e['phase'] == 'report']
            solves = [e for e in events if e['phase'] == 'solving']
            self.assertLess(min(e['applied_at'] for e in reports), max(e['applied_at'] for e in solves))
            requests = [json.loads(p.read_text()) for p in sorted(r.directory.glob('calls/*/request.json'))]
            later = next(q for q in requests if q['agent'] == 2 and q['context']['phase'] == 'report')
            self.assertTrue(any(m['call'] == reports[0]['call'] for m in later['context']['inbox']))
            inflight = next(q for q in requests if q['agent'] == 2 and q['context']['phase'] == 'solving')
            self.assertFalse(any(m['call'] == reports[0]['call'] for m in inflight['context']['inbox']))
            self.assertNotIn('PRIVATE_REFERENCE', json.dumps(requests))
            self.assertEqual(len({a['reported'] for a in r.state['agents'].values()}), 1)
            self.assertTrue(all(a['reported'] for a in r.state['agents'].values()))
            from swarm_pipeline.audit_gradual import audit
            self.assertGreater(audit(r.directory)['reports_before_last_solve'], 0)


if __name__ == '__main__':
    unittest.main()
