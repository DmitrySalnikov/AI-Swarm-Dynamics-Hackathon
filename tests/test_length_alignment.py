import json,tempfile,unittest
from pathlib import Path
from swarm_pipeline.length_alignment import SEP,donor_pool,solving_streams,supplement,trim,trim_facts


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):
        return type('Encoding',(),{'offsets':[(i,i+1) for i in range(len(text))]})()


def make_run(root,pieces_by_agent,prompts):
    """pieces_by_agent: agent -> [(call, phase, task, text)]"""
    root.mkdir(parents=True);(root/'generated').mkdir()
    (root/'state.json').write_text(json.dumps({'status':'completed','agents':{str(a):{} for a in pieces_by_agent}}))
    (root/'manifest.json').write_text(json.dumps({'tasks':[{'id':t,'prompt':p} for t,p in prompts.items()]}))
    for a,pieces in pieces_by_agent.items():
        text,index='',[]
        for call,phase,task,chunk in pieces:
            index.append({'start':len(text),'end':len(text)+len(chunk),'call':call,'phase':phase,'task_id':task});text+=chunk
        (root/'generated'/f'agent_{a}.generated.txt').write_text(text)
        (root/'generated'/f'agent_{a}.generated.index.json').write_text(json.dumps(index))
    return root


PROMPTS={'a':'Solve a.','b':'Solve b.'}


class LengthAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.tok=CharacterTokenizer()
        self.recipient=make_run(self.root/'swarm',{1:[(0,'negotiation',None,'NEG'),(2,'solving','a','aaaa'),(5,'solving','a','AA')],
                                                    2:[(1,'negotiation',None,'N'),(3,'solving','b','bbbbbbbbbbbbbbbbbbbb')]},PROMPTS)

    def tearDown(self):self.tmp.cleanup()

    def test_solving_streams_keep_solving_only_in_call_order(self):
        s={x['agent']:x for x in solving_streams(self.recipient,self.tok)}
        self.assertEqual(s[1]['text'],'aaaa'+SEP+'AA');self.assertEqual(s[1]['original_tokens'],8)
        self.assertEqual(set(s[1]['token_task']),{'a'});self.assertEqual(s[2]['text'],'b'*20)

    def test_trim_cuts_to_the_shortest_stream(self):
        streams=solving_streams(self.recipient,self.tok);cut=trim(streams)
        self.assertEqual({len(x['offsets']) for x in cut},{8});self.assertEqual(cut[1]['text'],'b'*20)
        f=trim_facts(streams)
        self.assertEqual(f['kept_tokens_per_agent'],8);self.assertEqual(f['cut_tokens_per_agent'],[0,12]);self.assertAlmostEqual(f['cut_share'],12/28)

    def test_supplement_extends_with_unused_same_task_donors(self):
        donors1=make_run(self.root/'seq1',{1:[(0,'solving','a','x1'*3),(2,'solving','b','y'*5)],2:[(1,'solving','a','x2'*3)]},PROMPTS)
        donors2=make_run(self.root/'seq2',{1:[(0,'solving','a','z'*9)]},{'a':'Different prompt.','b':'Solve b.'})
        pool=donor_pool([donors1,donors2],self.tok)
        self.assertEqual(len(pool['a']),3);self.assertEqual(len(pool['b']),1)
        streams=solving_streams(self.recipient,self.tok);used=set()
        out,prov=supplement(streams,{1:'a',2:'b'},PROMPTS,pool,used,self.tok,'swarm')
        self.assertEqual({len(x['offsets']) for x in out},{20})
        one=out[0];self.assertEqual(one['synthetic_from'],8);self.assertTrue(one['text'].startswith('aaaa'+SEP+'AA'+SEP))
        self.assertNotIn('z',one['text'][:one['offsets'][-1][1]])  # the donor with another prompt is never used
        self.assertEqual(len(prov),1);self.assertEqual(prov[0]['synthetic_tokens'],12);self.assertEqual(len(prov[0]['donors']),2)
        self.assertEqual(len(used),2);self.assertIs(out[1],streams[1])
        again=solving_streams(self.recipient,self.tok)
        with self.assertRaises(RuntimeError):  # both valid donors of task a are used up
            supplement(again,{1:'a',2:'b'},PROMPTS,pool,used,self.tok,'swarm-2')


if __name__=='__main__':unittest.main()
