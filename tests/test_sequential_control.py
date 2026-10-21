import json,tempfile,unittest
from pathlib import Path
from swarm_pipeline.runner import Config
from swarm_pipeline.sequential_control import SequentialRunner,SequentialOfflineClient,INSTRUCTION
from swarm_pipeline.audit import audit

TASKS=[{'id':f't{i}','prompt':'Give one integer.','answer':'1','check':{'type':'exact'}} for i in range(4)]
WIDE=[{'id':f't{i}','prompt':'Give one integer.','answer':'1','check':{'type':'exact'}} for i in range(6)]

class SequentialControlTests(unittest.TestCase):
    def run_condition(self,tmp,condition):
        run=SequentialRunner(TASKS,Config(condition=condition,agents=3,offline=True,seed=73),Path(tmp)/condition,SequentialOfflineClient(),workers=3,partial_continuation=True)
        run.run();return run

    def test_every_agent_solves_all_tasks_in_its_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            for condition in ('fixed','shuffled'):
                run=self.run_condition(tmp,condition)
                self.assertEqual(run.state['status'],'completed');self.assertEqual(audit(run.directory)['integrity'],'passed')
                orders=[a['order'] for a in run.state['agents'].values()]
                for a in run.state['agents'].values():
                    self.assertEqual(sorted(a['order']),[t['id'] for t in TASKS]);self.assertEqual(set(a['answers']),set(a['order']))
                    self.assertEqual(a['inbox'],[])
                if condition=='fixed':self.assertTrue(all(o==[t['id'] for t in TASKS] for o in orders))
                seen={}
                for p in sorted((run.directory/'calls').glob('*/request.json')):
                    q=json.loads(p.read_text());seen.setdefault(q['agent'],[]).append(q['context']['current_task'])
                    self.assertNotIn('shared_packet',q['context']);self.assertEqual(q['context']['inbox'],[])
                for agent,tasks in seen.items():
                    self.assertEqual(tasks,run.state['agents'][str(agent)]['order'])

    def test_prompt_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=self.run_condition(tmp,'shuffled')
            for p in (run.directory/'calls').glob('*/request.json'):
                q=json.loads(p.read_text())
                self.assertIn('one by one',q['messages'][0]['content']);self.assertEqual(q['context']['instruction'],INSTRUCTION)

    def test_unique_start_then_own_shuffle(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=SequentialRunner(WIDE,Config(condition='shuffled',agents=6,offline=True,seed=84),Path(tmp),SequentialOfflineClient(),workers=6,partial_continuation=True,unique_start=True)
            run.run()
            self.assertEqual(run.directory.name,'seed_84');self.assertEqual(run.directory.parent.name,'shuffled_unique_start')
            self.assertEqual(run.state['status'],'completed');self.assertEqual(audit(run.directory)['integrity'],'passed')
            orders=[a['order'] for a in run.state['agents'].values()]
            self.assertEqual(sorted(o[0] for o in orders),[t['id'] for t in WIDE])
            for o in orders:self.assertEqual(sorted(o),[t['id'] for t in WIDE])
            self.assertGreater(len({tuple(o[1:]) for o in orders}),1)
            self.assertEqual(run.manifest['order_mode'],'unique-start-then-own-shuffle-v1')
            first={}
            for p in sorted((run.directory/'calls').glob('*/request.json')):
                q=json.loads(p.read_text());first.setdefault(q['agent'],q['context']['current_task'])
            self.assertEqual(sorted(first.values()),[t['id'] for t in WIDE])
            again=SequentialRunner(WIDE,Config(condition='shuffled',agents=6,offline=True,seed=84),Path(tmp)/'b',SequentialOfflineClient(),workers=6,unique_start=True)
            self.assertEqual([a['order'] for a in again.initial_state()['agents'].values()],orders)
            with self.assertRaises(ValueError):
                SequentialRunner(WIDE,Config(condition='fixed',agents=6,offline=True),Path(tmp)/'c',SequentialOfflineClient(),workers=6,unique_start=True)

    def test_team_condition_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                SequentialRunner(TASKS,Config(condition='cooperative',agents=3,offline=True),Path(tmp),SequentialOfflineClient(),workers=3)

    def test_shared_random_order_and_task_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            orders={}
            for seed in (3,1003):
                run=SequentialRunner(WIDE,Config(condition='fixed',agents=6,offline=True,seed=seed),Path(tmp),SequentialOfflineClient(),workers=6,partial_continuation=True,shared_random_order=True,tasks_per_agent=4)
                run.run()
                self.assertEqual(run.directory.parent.name,'fixed_random_order_k4');self.assertEqual(run.state['status'],'completed')
                self.assertEqual(audit(run.directory)['integrity'],'passed')
                o=[a['order'] for a in run.state['agents'].values()]
                self.assertTrue(all(x==o[0] for x in o));self.assertEqual(len(o[0]),4);self.assertEqual(len(set(o[0])),4)
                self.assertTrue(all(len(a['answers'])==4 for a in run.state['agents'].values()))
                orders[seed]=o[0]
            self.assertNotEqual(orders[3],orders[1003])
            with self.assertRaises(ValueError):
                SequentialRunner(WIDE,Config(condition='shuffled',agents=6,offline=True),Path(tmp)/'c',SequentialOfflineClient(),workers=6,shared_random_order=True)

if __name__=='__main__':unittest.main()
