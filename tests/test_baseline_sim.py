import random
import unittest

from swarm_pipeline.baseline_sim import REGIMES, orders, random_latin_rows

A = [f"a{i}" for i in range(10)]
B = [f"b{i}" for i in range(10)]


class OrderTest(unittest.TestCase):
    def test_every_agent_opens_with_a_then_solves_all_of_b(self):
        for regime in REGIMES:
            for seed in range(20):
                for o in orders(regime, A, B, 10, seed):
                    self.assertEqual(len(o), 11)
                    self.assertIn(o[0], A)
                    self.assertEqual(sorted(o[1:]), sorted(B))

    def test_same_regime_is_identical_for_all_agents(self):
        rows = orders("same", A, B, 10, 3)
        self.assertTrue(all(r == rows[0] for r in rows))

    def test_different_regimes_hold_different_tasks_at_every_position(self):
        for regime in ("cyclic", "latin"):
            for seed in range(20):
                rows = orders(regime, A, B, 10, seed)
                for p in range(11):
                    self.assertEqual(len({r[p] for r in rows}), 10, (regime, seed, p))

    def test_latin_square_is_not_a_cyclic_shift(self):
        # in a cyclic shift every agent has the same successor of each task; a random square should not
        shared = 0
        for seed in range(20):
            rows = orders("latin", A, B, 10, seed)
            succ = [{r[p]: r[p + 1] for p in range(1, 10)} for r in rows]
            pairs = {(t, s[t]) for s in succ for t in s}
            shared += len(pairs) == 9            # cyclic shift gives exactly 9 distinct (task, successor) pairs
        self.assertEqual(shared, 0)

    def test_orders_are_reproducible_and_depend_on_seed(self):
        for regime in REGIMES:
            self.assertEqual(orders(regime, A, B, 10, 5), orders(regime, A, B, 10, 5))
            self.assertNotEqual(orders(regime, A, B, 10, 5), orders(regime, A, B, 10, 6))

    def test_random_latin_rows(self):
        for seed in range(50):
            rows = random_latin_rows(B, random.Random(seed))
            self.assertTrue(all(sorted(r) == sorted(B) for r in rows))
            self.assertTrue(all(len({r[c] for r in rows}) == 10 for c in range(10)))


if __name__ == "__main__":
    unittest.main()
