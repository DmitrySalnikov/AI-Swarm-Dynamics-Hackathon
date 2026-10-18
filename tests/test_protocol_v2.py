import io,json,tempfile,unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
from swarm_pipeline.runner import Config
from swarm_pipeline.protocol_v2 import V2Runner,V2OfflineClient
from swarm_pipeline.audit import audit

TASKS=[{"id":f"task/{i}","prompt":f"Solve task {i}","answer":"PRIVATE_REFERENCE","check":{"type":"exact"}} for i in range(3)]

class V2Tests(unittest.TestCase):
    def test_barriers_common_packet_and_single_commits(self):
        with tempfile.TemporaryDirectory() as temp:
            r=V2Runner(TASKS,Config(condition="cooperative",agents=3,offline=True),Path(temp),V2OfflineClient(),workers=3)
            with redirect_stdout(io.StringIO()): result=r.run()
            self.assertEqual(result['status'],'completed')
            self.assertTrue(audit(r.directory)['single_successful_submit_and_commit_per_agent'])
            packets=[]
            for p in sorted(r.directory.glob('calls/*/request.json')):
                q=json.loads(p.read_text(encoding='utf-8'))
                self.assertNotIn('PRIVATE_REFERENCE',json.dumps(q['messages']))
                if q['context']['phase']=='exchange': packets.append(q['context']['shared_packet'])
            self.assertEqual(len(packets),3)
            self.assertTrue(all(p==packets[0] for p in packets))
            self.assertNotIn('Scripted private reasoning',json.dumps(packets))

    def test_peer_send_during_solving_is_rejected(self):
        class Bad(V2OfflineClient):
            def complete(self,q):
                if q['context']['phase']=='solving':
                    return {'content':json.dumps({'action':'send','to':'all','message':'Forbidden'}),'reasoning':'','finish_reason':'stop','usage':{'prompt_tokens':1,'completion_tokens':1}}
                return super().complete(q)
        with tempfile.TemporaryDirectory() as temp:
            r=V2Runner(TASKS,Config(condition='cooperative',agents=3,offline=True,max_calls=9),Path(temp),Bad(),workers=3)
            with redirect_stdout(io.StringIO()):result=r.run()
            self.assertEqual(result['status'],'call_limit')
            self.assertIsNone(r.state['shared_packet'])
            self.assertEqual(sum(len(a['answers']) for a in r.state['agents'].values()),0)

    def test_failed_commit_recovers_same_packet(self):
        class Flaky(V2OfflineClient):
            failed=False
            def complete(self,q):
                if q['context']['phase']=='exchange' and not self.failed:
                    self.failed=True
                    return {'content':'','reasoning':'','finish_reason':'transport_error','usage':None,'error_type':'TimeoutError'}
                return super().complete(q)
        with tempfile.TemporaryDirectory() as temp:
            r=V2Runner(TASKS,Config(condition='cooperative',agents=3,offline=True),Path(temp),Flaky(),workers=3,partial_continuation=True)
            with redirect_stdout(io.StringIO()),patch('time.sleep'):
                self.assertEqual(r.run()['status'],'completed')
            self.assertEqual(audit(r.directory)['commit_retries'],1)

    def test_step_prompts_follow_accepted_actions_and_keep_full_tasks(self):
        from swarm_pipeline.protocol_v2 import STEP_INSTRUCTIONS, INSTRUCTIONS
        with tempfile.TemporaryDirectory() as temp:
            r=V2Runner(TASKS,Config(condition='cooperative',agents=3,offline=True),Path(temp),V2OfflineClient(),workers=3)
            r.state=r.initial_state()
            def act(action, finish='stop'):
                q={'agent':1,'call':r.state['calls'],'context':r.context(1)}
                return r.apply(q,{'content':json.dumps(action),'finish_reason':finish})
            c=r.context(1)
            self.assertEqual(c['instruction'],STEP_INSTRUCTIONS['send'])
            self.assertEqual({t['id']:t['prompt'] for t in c['tasks']},{t['id']:t['prompt'] for t in TASKS})
            self.assertIsNotNone(act({'actions':[]})['action_error'])
            self.assertEqual(r.context(1)['instruction'],STEP_INSTRUCTIONS['send'])
            self.assertIsNone(act({'action':'send','to':'all','message':'I prefer task/0.'})['action_error'])
            self.assertEqual(r.context(1)['instruction'],STEP_INSTRUCTIONS['claim'])
            self.assertIsNotNone(act({'action':'claim','task_id':'unavailable'})['action_error'])
            self.assertEqual(r.context(1)['instruction'],STEP_INSTRUCTIONS['claim'])
            self.assertIsNone(act({'action':'claim','task_id':'task/0'})['action_error'])
            r.state['phase']='solving'
            self.assertEqual(r.context(1)['instruction'],INSTRUCTIONS['solving'])

    def test_step_prompt_full_offline_run(self):
        from swarm_pipeline.protocol_v2 import STEP_INSTRUCTIONS, PROTOCOL
        with tempfile.TemporaryDirectory() as temp:
            r=V2Runner(TASKS,Config(condition='cooperative',agents=3,offline=True),Path(temp),V2OfflineClient(),workers=3)
            with redirect_stdout(io.StringIO()):self.assertEqual(r.run()['status'],'completed')
            counts={'send':0,'claim':0}
            for p in r.directory.glob('calls/*/request.json'):
                q=json.loads(p.read_text())
                self.assertIn(PROTOCOL,q['messages'][0]['content'])
                if q['context']['phase']=='negotiation':
                    step='claim' if q['context']['negotiated'] else 'send'
                    self.assertEqual(q['context']['instruction'],STEP_INSTRUCTIONS[step])
                    counts[step]+=1
            self.assertEqual(counts,{'send':3,'claim':3})
            self.assertTrue(audit(r.directory)['single_successful_submit_and_commit_per_agent'])

    def test_legacy_environment_cannot_select_old_prompts(self):
        from swarm_pipeline.protocol_v2 import PROTOCOL, STEP_INSTRUCTIONS
        snapshots=[]
        with tempfile.TemporaryDirectory() as temp:
            for value in ('v2','simple','simple_steps','other'):
                with patch.dict('os.environ',{'SWARM_PROMPT_STYLE':value}):
                    r=V2Runner(TASKS,Config(condition='cooperative',agents=3,offline=True),Path(temp)/value,V2OfflineClient(),workers=3)
                    r.state=r.initial_state()
                    self.assertIn(PROTOCOL,r.state['agents']['1']['history'][0]['content'])
                    self.assertEqual(r.context(1)['instruction'],STEP_INSTRUCTIONS['send'])
                    self.assertNotIn('prompt_style',r.manifest)
                    snapshots.append((r.manifest['protocol_hash'],r.state,r.context(1)))
            self.assertTrue(all(s==snapshots[0] for s in snapshots))

if __name__=='__main__':unittest.main()
