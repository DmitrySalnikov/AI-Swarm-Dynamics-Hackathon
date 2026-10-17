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

if __name__=='__main__':unittest.main()
