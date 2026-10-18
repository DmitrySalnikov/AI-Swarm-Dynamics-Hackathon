import tempfile,unittest,json
from pathlib import Path
from swarm_pipeline.runner import Config
from swarm_pipeline.private_control import PrivateRunner,PrivateOfflineClient
from swarm_pipeline.audit import audit

class PrivateControlTests(unittest.TestCase):
    def test_step_style_keeps_private_solving_identical(self):
        from unittest.mock import patch
        snapshots=[]
        with tempfile.TemporaryDirectory() as tmp:
            for style in ('v2','simple','simple_steps','other'):
                with patch.dict('os.environ',{'SWARM_PROMPT_STYLE':style}):
                    tasks=[{'id':'t0','prompt':'Give one integer.','answer':'1','check':{'type':'exact'}}]
                    run=PrivateRunner(tasks,Config(condition='fixed',agents=1,offline=True),Path(tmp)/style,
                                      PrivateOfflineClient(),assignments={'1':'t0'},workers=1)
                    run.state=run.initial_state()
                    snapshots.append((run.state['agents']['1']['history'],run.context(1)))
            self.assertTrue(all(s==snapshots[0] for s in snapshots))

    def test_one_task_each_no_peer_inputs_and_no_exchange(self):
        tasks=[{'id':f't{i}','prompt':'Give one integer.','answer':'1','check':{'type':'exact'}} for i in range(3)]
        mapping={'1':'t2','2':'t0','3':'t1'}
        with tempfile.TemporaryDirectory() as tmp:
            cfg=Config(condition='fixed',agents=3,offline=True,seed=73)
            run=PrivateRunner(tasks,cfg,Path(tmp),PrivateOfflineClient(),assignments=mapping,workers=3)
            run.run();report=audit(run.directory)
            self.assertEqual(report['integrity'],'passed');self.assertEqual(report['calls'],3)
            self.assertEqual(set(report['phases']),{'solving'})
            for i,a in run.state['agents'].items():
                self.assertEqual(a['order'],[mapping[i]]);self.assertEqual(a['inbox'],[])
            for p in (run.directory/'calls').glob('*/request.json'):
                q=json.loads(p.read_text());self.assertEqual(q['context']['task']['id'],mapping[str(q['agent'])])
                self.assertNotIn('shared_packet',q['context']);self.assertEqual(q['context']['inbox'],[])

    def test_default_solving_prompts(self):
        from swarm_pipeline.private_control import PROTOCOL,INSTRUCTION
        tasks=[{'id':f't{i}','prompt':'Give one integer.','answer':'1','check':{'type':'exact'}} for i in range(3)]
        mapping={'1':'t2','2':'t0','3':'t1'}
        with tempfile.TemporaryDirectory() as tmp:
            run=PrivateRunner(tasks,Config(condition='fixed',agents=3,offline=True,seed=73),Path(tmp),PrivateOfflineClient(),assignments=mapping,workers=3)
            run.run();self.assertEqual(audit(run.directory)['integrity'],'passed')
            self.assertNotIn('prompt_style',run.manifest)
            for p in (run.directory/'calls').glob('*/request.json'):
                q=json.loads(p.read_text())
                self.assertIn(PROTOCOL,q['messages'][0]['content']);self.assertEqual(q['context']['instruction'],INSTRUCTION)

if __name__=='__main__':unittest.main()
