import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from swarm_pipeline.phase_windows import load_channels, make_rows, payload_text


class CharacterTokenizer:
    def encode(self,text,**kwargs):
        return SimpleNamespace(offsets=[(i,i+1) for i in range(len(text))])


class PhaseWindowTests(unittest.TestCase):
    def test_trim_and_available_diverge_only_after_common_prefix(self):
        streams={i:'abcdefghij' for i in range(1,11)}
        streams[1]='abcde'
        available,trim,info=make_rows(streams,CharacterTokenizer(),4,'swarm')
        self.assertEqual(available[0]['windows'],trim[0]['windows'])
        self.assertEqual(len(available[1]['windows']),9)
        self.assertEqual(len(trim[1]['windows']),0)
        self.assertEqual(info['discarded_by_trim_tokens'],45)
        self.assertTrue(all(len(r['windows']) in (0,10) for r in trim))

    def test_semantic_payload_excludes_claim_and_canonicalizes_commit(self):
        self.assertEqual(payload_text({'action':'claim','task_id':'task'}),'')
        self.assertEqual(payload_text({'action':'send','message':'Actual message','to':'all'}),'Actual message')
        self.assertEqual(payload_text({'action':'finish','answers':{'b':'2','a':'1'}}),
                         payload_text({'action':'finish','answers':{'a':'1','b':'2'}}))

    def test_rejected_fragments_are_observed_output_but_not_accepted_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'state.json').write_text(json.dumps({'status':'completed','pending':None,'agents':{'1':{}}}))
            for call,(reason,content,error,finish) in enumerate([
                ('OWN_REASONING',json.dumps({'action':'send','message':'hello','to':'all'}),None,'stop'),
                ('INTERRUPTED','partial','Incomplete response','transport_error'),
                ('CLAIM_REASONING',json.dumps({'action':'claim','task_id':'task'}),None,'stop')]):
                folder=root/'calls'/f'{call:06d}';folder.mkdir(parents=True)
                (folder/'request.json').write_text(json.dumps({'call':call,'agent':1,'context':{'phase':'negotiation','inbox':['PEER_SECRET']}}))
                (folder/'response.json').write_text(json.dumps({'reasoning':reason,'content':content,'finish_reason':finish}))
                (folder/'event.json').write_text(json.dumps({'action_error':error}))
            streams,receipts=load_channels(root)
            self.assertEqual(streams[('negotiation','payload')][1],'hello')
            self.assertIn('INTERRUPTED',streams[('negotiation','reasoning')][1])
            self.assertIn('partial',streams[('negotiation','visible_output')][1])
            self.assertNotIn('PEER_SECRET',str(streams))
            self.assertEqual(len(receipts),3)
            self.assertTrue(all(len(r['response_sha256'])==64 for r in receipts))


if __name__=='__main__':unittest.main()
