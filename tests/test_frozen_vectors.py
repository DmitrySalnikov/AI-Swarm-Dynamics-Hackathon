import json,tempfile,unittest
from pathlib import Path
import numpy as np
from swarm_pipeline.analyze import vector_key
from swarm_pipeline.frozen_vectors import (agent_streams,canonical_points,stage_changes,swarm_clock,swarm_tokens,timeline,
                                           updates,window_records)


class CharacterTokenizer:
    def encode(self,text,add_special_tokens=False):
        return type('Encoding',(),{'offsets':[(i,i+1) for i in range(len(text))]})()


def make_run(root,calls,started_at,twin_started_at=None,clock_shift=None,journal=True):
    """calls: agent -> [(call, phase, task, [(channel, text, at), ...] deltas, imported)]"""
    (root/'generated').mkdir(parents=True)
    state={'started_at':started_at,'agents':{str(a):{} for a in calls}}
    if twin_started_at is not None:state.update(twin_started_at=twin_started_at,clock_shift=clock_shift)
    (root/'state.json').write_text(json.dumps(state))
    for a,cs in calls.items():
        text,index='',[]
        for call,phase,task,deltas,imported in cs:
            segs=[]
            for channel,chunk,at in deltas:
                if not segs or segs[-1][0]!=channel:segs.append([channel,'',[]])
                segs[-1][1]+=chunk;segs[-1][2].append([len(segs[-1][1]),at])
            for channel,chunk,marks in segs:
                index.append({'start':len(text),'end':len(text)+len(chunk),'part':channel,'call':call,'phase':phase,'task_id':task,
                              'imported':imported,'timing':'journal' if journal else 'interpolated','marks':marks if journal else None,
                              'started_at':deltas[0][2]-1,'first_chunk_at':deltas[0][2],'last_chunk_at':deltas[-1][2],'completed_at':deltas[-1][2]+1})
                text+=chunk+'\n\n'
        (root/'generated'/f'agent_{a}.generated.txt').write_text(text)
        (root/'generated'/f'agent_{a}.generated.index.json').write_text(json.dumps(index))


def vectors_for(ups):
    return {vector_key(u['text']):np.eye(8)[k%8] for k,u in enumerate(u for a in ups for u in a)}


class FrozenVectorTests(unittest.TestCase):
    def test_journal_times_token_finalization_and_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            make_run(root,{1:[(0,'prelude','a',[('reasoning','x'*30,110),('reasoning','x'*30,120),('content','{}',121)],False)]},started_at=100)
            s=agent_streams(root,CharacterTokenizer())[0]
            self.assertEqual(s['text'],'x'*60+'\n\n{}\n\n')
            self.assertEqual(s['times'][0],10);self.assertEqual(s['times'][29],20)  # the char after token 29 came with the 2nd delta
            self.assertEqual(s['times'][59],20);self.assertEqual(s['times'][62],21)
            self.assertEqual(s['channel'][0],'reasoning');self.assertEqual(s['channel'][62],'content')
            u=updates(s,first=20,step=20,width=20)
            self.assertEqual([x['end'] for x in u],[20,40,60]);self.assertEqual([x['t'] for x in u],[10,20,20])
            self.assertEqual(u[0]['channels'],['reasoning'])
            rows=window_records([u])
            self.assertEqual(rows[0]['chars'],[0,20]);self.assertEqual(len(rows[0]['sha256']),64)

    def test_interrupted_call_and_continuation_have_no_gaps_or_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            make_run(root,{1:[(0,'solving','b',[('reasoning','frag',10),('reasoning','ment',11)],False),
                              (1,'solving','b',[('reasoning','cont',30),('content','{}',31)],False)]},started_at=0)
            s=agent_streams(root,CharacterTokenizer())[0]
            self.assertEqual(s['text'],'fragment\n\ncont\n\n{}\n\n')
            self.assertEqual(len(s['times']),len(s['text']))
            self.assertTrue(all(np.diff(s['times'])>=0))
            self.assertEqual(s['times'][7],11);self.assertEqual(s['times'][10],30)  # the continuation follows the fragment
            self.assertEqual([x['call'] for x in s['spans']],[0,1])

    def test_visible_view_keeps_only_replies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            make_run(root,{1:[(0,'prelude','a',[('reasoning','r'*10,5),('content','c'*4,6)],False),
                              (1,'chat_1',None,[('reasoning','q'*10,8),('content','m'*3,9)],False)]},started_at=0)
            v=agent_streams(root,CharacterTokenizer(),channels=('content',))[0]
            self.assertEqual(v['text'],'cccc\n\nmmm\n\n');self.assertEqual(set(v['channel']),{'content'})
            self.assertEqual(v['times'][0],6);self.assertEqual(v['times'][6],9)

    def test_interpolation_without_a_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            make_run(root,{1:[(0,'prelude','a',[('reasoning','x'*99,10),('reasoning','',20)],False)]},started_at=0,journal=False)
            s=agent_streams(root,CharacterTokenizer())[0]
            self.assertEqual(s['timing'],['interpolated'])
            self.assertAlmostEqual(s['times'][0],10+10/100,places=6);self.assertAlmostEqual(s['times'][-1],20)

    def test_frozen_while_waiting_and_swarm_axis_ignores_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            make_run(root,{1:[(0,'prelude','a',[('reasoning','x'*98,200)],False),(1,'chat_1',None,[('reasoning','m'*98,1100)],False)],
                           2:[(2,'prelude','b',[('reasoning','y'*98,200)],False)]},started_at=100)
            streams=agent_streams(root,CharacterTokenizer())
            clock=swarm_clock(streams)
            self.assertEqual(len(clock),300);self.assertEqual(swarm_tokens(clock,-1),0)
            self.assertEqual(swarm_tokens(clock,500),200);self.assertEqual(swarm_tokens(clock,999),200)  # waiting: the axis stands still
            ups=[updates(x,first=50,step=50,width=50) for x in streams]
            for u in (u for a in ups for u in a):u['g']=swarm_tokens(clock,u['t'])
            rows=timeline(ups,vectors_for(ups),[0,200,250,300],key='g')
            self.assertEqual([r['N'] for r in rows],[0,2,2,2])
            self.assertEqual(rows[1]['phases'],{'prelude':2});self.assertEqual(rows[3]['phases'],{'chat_1':1,'prelude':1})
            self.assertEqual(rows[3]['age_max'],100)  # agent 2 is frozen since swarm token 200
            self.assertEqual([r['updated'] for r in rows],[0,2,0,1])
            self.assertEqual(canonical_points(ups),[200,300])

    def test_baseline_own_calls_move_onto_the_twin_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'run'
            make_run(root,{1:[(0,'prelude','a',[('reasoning','x'*10,120)],True),(0,'solving','b',[('reasoning','z'*10,5030)],False)]},
                     started_at=5000,twin_started_at=100,clock_shift=-4850)  # twin barrier at 150
            (root/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in [
                {'kind':'stage','agent':1,'before':'prelude','after':'barrier','twin_at':120,'imported':True,'at':5000},
                {'kind':'stage','agent':1,'before':'barrier','after':'solving','at':5000},
                {'kind':'stage','agent':1,'before':'solving','after':'done','at':5030}]))
            s=agent_streams(root,CharacterTokenizer())[0]
            self.assertEqual(s['times'][0],20);self.assertEqual(s['times'][12],80)  # 5030-4850-100
            self.assertEqual(stage_changes(root)[1],[(0.0,'prelude'),(20,'barrier'),(50,'solving'),(80,'done')])


    def test_geometry_matches_the_dynamics_metrics(self):
        from embedding_eval.dynamics import metrics_at
        from swarm_pipeline.frozen_vectors import geometry
        rng=np.random.default_rng(0)
        V=rng.normal(size=(10,32))+2;V/=np.linalg.norm(V,axis=1,keepdims=True)
        g,m=geometry(V),metrics_at(V)
        for a,b in (('R','R'),('var_perp','var_perp_std'),('var_par','var_par_std'),('eff_dim','eff_dim')):
            self.assertAlmostEqual(g[a],m[b],places=10)
        line=np.array([[1,0],[1,0],[0,1],[0,1]],float)  # agents split along one line: one direction of spread
        self.assertAlmostEqual(geometry(line)['eff_dim'],1.0)
        self.assertIsNone(geometry(np.array([[1.,0],[1,0]]))['eff_dim'])  # identical vectors: no spread


if __name__=='__main__':unittest.main()
