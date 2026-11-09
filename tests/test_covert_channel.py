import json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from swarm_pipeline import covert_channel as cc
from swarm_pipeline.runner import Config
from swarm_pipeline.stream_journal import DeltaJournal,check_deltas,read_deltas
from swarm_pipeline.supervision import read_json

A=[{'id':f'a{i}','prompt':f'Problem a{i}.','answer':'1','check':{'type':'exact'}} for i in range(3)]
B=[{'id':f'b{i}','prompt':f'Problem b{i}.','answer':'1','check':{'type':'exact'}} for i in range(3)]


class Scripted(cc.CovertOfflineClient):
    """Offline agents with fixed lengths per agent and phase (words) and optional faults (agent, phase, kind)."""
    def __init__(self,prelude=(40,400,420),faults=()):
        super().__init__(seconds_per_word=.0005)
        self.prelude,self.faults=prelude,list(faults)
    def words(self,request):
        return {'prelude':self.prelude[request['agent']-1],'solving':200,'chat_1':150,'chat_2':150}[request['phase']]
    def complete(self,request):
        c,phase=request['context'],request['phase']
        for f in list(self.faults):
            if f[0]==request['agent'] and f[1]==phase:
                self.faults.remove(f)
                if f[2]=='fragment':
                    j=DeltaJournal(request['delta_journal']);j.add('reasoning','partial ');j.add('reasoning','thought');j.close()
                    return {'content':'','reasoning':'partial thought','finish_reason':'stream_incomplete','usage':None}
                if f[2]=='empty':
                    return {'content':'','reasoning':'','finish_reason':'transport_error','http_status':429,'usage':None}
                if f[2]=='taken':
                    taken=next(iter(c['claimed']))
                    return {'content':json.dumps({'action':'post','message':'mine','claim':taken}),'reasoning':'r','finish_reason':'stop','usage':{'prompt_tokens':1,'completion_tokens':1}}
        return super().complete(request)


def run(tmp,arm,client):
    r=cc.CovertRunner(A,B,Config(condition='fixed',agents=3,offline=True,seed=5),Path(tmp),client,arm,workers=3)
    return r,r.run()


def events(d):
    return [json.loads(l) for l in (d/'events.jsonl').read_text().splitlines()]


def requests(d):
    return [read_json(p) for p in sorted(d.glob('calls/*/request.json'))]


class CovertChannelTests(unittest.TestCase):
    def test_swarm_queue_barrier_and_end_on_the_last_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            sw,res=run(tmp,'swarm',Scripted())
            self.assertEqual(res['status'],'completed');self.assertIsNone(res['halt'])
            s=read_json(sw.directory/'state.json')
            self.assertEqual(sorted(s['claims']),['b0','b1','b2']);self.assertEqual(sorted(s['claims'].values()),[1,2,3])
            self.assertEqual([m['chat_phase'] for m in s['chat']],[1,1,1,1,2,2,2]);self.assertEqual(s['chat'][3]['from'],'coordinator')
            ev,reqs=events(sw.directory),requests(sw.directory)
            self.assertEqual({q['phase'] for q in reqs},{'prelude','chat_1','solving','chat_2'})
            self.assertEqual(len(reqs),12)  # no commit: three calls per phase
            dispatch={e['call']:e for e in ev if e['kind']=='dispatch'};applied={e['call']:e for e in ev if e['kind']=='applied'}
            published={e['call']:e for e in ev if e['kind']=='published' and e['call'] is not None}
            for phase in ('chat_1','chat_2'):  # one writer at a time: the next turn is dispatched after the previous publication
                chat=sorted((q for q in reqs if q['phase']==phase),key=lambda q:dispatch[q['call']]['seq'])
                for prev,nxt in zip(chat,chat[1:]):
                    self.assertLess(published[prev['call']]['seq'],dispatch[nxt['call']]['seq'])
            for q in reqs:  # the inbox holds exactly the messages published before the dispatch, never later ones
                before=[e['message'] for e in ev if e['kind']=='published' and e['seq']<dispatch[q['call']]['seq']]
                self.assertTrue(set(q['inbox'])<=set(before))
                self.assertEqual([m['id'] for m in q['context'].get('chat',[])],q['inbox'])
            for a in (1,2,3):  # one outstanding request per agent
                mine=sorted([(dispatch[c]['seq'],applied[c]['seq']) for c in dispatch if dispatch[c]['agent']==a])
                self.assertTrue(all(x[1]<y[0] for x,y in zip(mine,mine[1:])))
            barrier=next(e['seq'] for e in ev if e['kind']=='barrier')
            solves=[dispatch[q['call']]['seq'] for q in reqs if q['phase']=='solving']
            self.assertTrue(all(x>barrier for x in solves))
            self.assertTrue(all(dispatch[q['call']]['seq']<barrier for q in reqs if q['phase']=='chat_1'))
            last=max(e['seq'] for e in ev if e['kind']=='published' and e['chat_phase']==2)
            end=[e['seq'] for e in ev if e['kind']=='end'];self.assertEqual(len(end),1);self.assertLess(last,end[0])
            self.assertFalse([e for e in ev if e['kind']=='dispatch' and e['seq']>end[0]])
            start=[q for q in reqs if q['phase']=='solving']
            self.assertTrue(all(len(q['context']['assignments'])==3 and q['context']['instruction']==cc.START_INSTRUCTION for q in start))
            chat1=[q for q in reqs if q['phase']=='chat_1']
            self.assertTrue(all('new_problems' in q['context'] and q['context']['notice']==cc.CHAT_NOTICE for q in chat1))
            for a in s['agents'].values():  # own reasoning is fed back in <think> tags
                turns=[m['content'] for m in a['history'] if m['role']=='assistant']
                self.assertEqual(len(turns),4);self.assertTrue(all(t.startswith('<think>\n') and '\n</think>\n\n{' in t for t in turns))
            for d in sw.directory.glob('calls/*'):
                self.assertTrue(check_deltas(read_deltas(d/'deltas.jsonl'),read_json(d/'response.json')))

    def test_retried_solving_call_does_not_see_results_posted_after_the_barrier(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cc,'BACKOFF_SECONDS',1.0):  # the others post while agent 3 backs off
            sw,res=run(tmp,'swarm',Scripted(faults=[(3,'solving','empty')]))
            self.assertEqual(res['status'],'completed')
            chat1=[m['id'] for m in read_json(sw.directory/'state.json')['chat'] if m['chat_phase']==1]
            ev=events(sw.directory);dispatch={e['call']:e['seq'] for e in ev if e['kind']=='dispatch'}
            retry=[q for q in requests(sw.directory) if q['phase']=='solving' and q['agent']==3][-1]
            self.assertLess(min(e['seq'] for e in ev if e['kind']=='published' and e['chat_phase']==2),dispatch[retry['call']])
            self.assertEqual(retry['chat_len'],len(chat1));self.assertTrue(set(retry['inbox'])<=set(chat1))
            self.assertEqual([m['id'] for m in retry['context']['chat']],retry['inbox'])

    def test_early_join_and_waiting_for_the_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            sw,_=run(tmp,'swarm',Scripted(prelude=(40,400,420)))
            ev=events(sw.directory)
            pub1=next(e for e in ev if e['kind']=='published' and e['agent']==1)
            done3=next(e for e in ev if e['kind']=='stage' and e['agent']==3 and e['after']=='queued_1')
            self.assertLess(pub1['seq'],done3['seq'])  # agent 1 wrote while agent 3 was still on its first problem
            q=next(e for e in ev if e['kind']=='stage' and e['agent']==3 and e['after']=='queued_1')
            w=next(e for e in ev if e['kind']=='stage' and e['agent']==3 and e['after']=='writing_1')
            self.assertGreater(w['at']-q['at'],.01)  # agent 3 finished during agent 2's turn and waited

    def test_baseline_waits_for_the_twin_barrier_and_solves_the_same_problem(self):
        with tempfile.TemporaryDirectory() as tmp:
            sw,_=run(tmp,'swarm',Scripted())
            s=read_json(sw.directory/'state.json')
            bl,res=run(tmp,'baseline',Scripted())
            self.assertEqual(res['status'],'completed')
            b=read_json(bl.directory/'state.json')
            breqs=requests(bl.directory)
            self.assertEqual([q['phase'] for q in breqs],['solving']*3)
            self.assertAlmostEqual(b['clock_shift'],s['barrier_at']-b['started_at'],places=6)
            for i,a in b['agents'].items():
                t=s['agents'][i]
                self.assertEqual(a['history'][:t['fork_history_len']],t['history'][:t['fork_history_len']])
                q=next(q for q in breqs if q['agent']==int(i))
                self.assertEqual(q['context']['current_task'],t['task'])
                self.assertNotIn('chat',q['context']);self.assertEqual(q['context']['instruction'],cc.SOLO_INSTRUCTION)
                self.assertEqual(sorted(a['answers']),sorted([t['solo_task'],t['task']]))
            self.assertEqual(len(b['chat']),0)
            ix=read_json(bl.directory/'generated'/'agent_2.generated.index.json')
            self.assertTrue(ix[0]['imported']);self.assertEqual(ix[0]['phase'],'prelude');self.assertFalse(ix[-1]['imported'])
            self.assertTrue(all(p['timing']=='journal' for p in ix))
            ev=events(bl.directory)
            self.assertEqual([e['kind'] for e in ev if e['kind'] in ('barrier','end')],['barrier','end'])
            self.assertTrue(all(e['twin_at']==s['agents'][str(e['agent'])]['fork_at'] for e in ev if e.get('imported')))

    def test_independent_generates_its_prelude_and_solves_latin_rows_without_waiting(self):
        with tempfile.TemporaryDirectory() as tmp:
            sw,_=run(tmp,'swarm',Scripted())
            s=read_json(sw.directory/'state.json')
            with patch.object(cc,'run_tokens',return_value=10**9):  # no token cut: every agent solves all its problems
                ind,res=run(tmp,'independent',Scripted())
            self.assertEqual((res['status'],res['end_reason']),('completed','all_problems_solved'))
            b=read_json(ind.directory/'state.json')
            orders=[b['agents'][str(i)]['order'] for i in (1,2,3)]
            self.assertTrue(all(sorted(o)==['b0','b1','b2'] for o in orders))  # latin_tasks = min(4, |B|) = 3
            self.assertTrue(all(len({o[c] for o in orders})==3 for c in range(3)))  # columns differ: a Latin square
            reqs=requests(ind.directory)
            self.assertEqual(sum(q['phase']=='prelude' for q in reqs),3)  # own prelude, not imported
            for i,a in b['agents'].items():
                self.assertEqual(a['solo_task'],s['agents'][i]['solo_task'])
                mine=[q for q in sorted(reqs,key=lambda q:q['call']) if q['agent']==int(i)]
                self.assertEqual([q['context']['current_task'] for q in mine],[a['solo_task']]+a['order'])
                self.assertTrue(all(q['context']['instruction']==cc.SOLO_INSTRUCTION and 'chat' not in q['context'] for q in mine))
                turns=[m['content'] for m in a['history'] if m['role']=='assistant']
                self.assertEqual(len(turns),4);self.assertTrue(all(t.startswith('<think>\n') for t in turns))
            ev=events(ind.directory)
            self.assertFalse([e for e in ev if e['kind'] in ('barrier','queued','published')])
            applied={e['call']:e['seq'] for e in ev if e['kind']=='applied'}
            dispatch={e['call']:e for e in ev if e['kind']=='dispatch'}
            pre={q['agent']:applied[q['call']] for q in reqs if q['phase']=='prelude'}
            first={}
            for q in sorted(reqs,key=lambda q:q['call']):
                if q['phase']=='solving':first.setdefault(q['agent'],dispatch[q['call']]['seq'])
            self.assertLess(first[1],max(pre.values()))  # agent 1 starts set B while others are still on the prelude
            ix=read_json(ind.directory/'generated'/'agent_1.generated.index.json')
            self.assertFalse(any(p['imported'] for p in ix))

    def test_independent_stops_at_the_swarm_token_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            sw,_=run(tmp,'swarm',Scripted())
            target=cc.run_tokens(sw.directory)
            ind,res=run(tmp,'independent',Scripted())
            self.assertEqual((res['status'],res['end_reason'],res['token_target']),('completed','token_target',target))
            self.assertGreaterEqual(res['generated_tokens'],target)
            ev=events(ind.directory);end=next(e['seq'] for e in ev if e['kind']=='end')
            self.assertFalse([e for e in ev if e['kind']=='dispatch' and e['seq']>end])
            self.assertEqual(res['generated_tokens'],cc.run_tokens(ind.directory))  # cancelled fragments are kept and counted
            cancelled=[read_json(p) for p in ind.directory.glob('calls/*/response.json')]
            cancelled=[r for r in cancelled if r['finish_reason']=='cancelled']
            self.assertTrue(cancelled)  # calls in flight at the target are cut, not run to the end
            self.assertTrue(all(r['cost_estimated'] and r['usage_estimate']['completion_tokens']>0 for r in cancelled))
            self.assertLess(res['generated_tokens'],target*1.3)

    def test_latin_rows(self):
        import random
        for n in (3,10):
            rows=cc.random_latin_rows(list(range(n)),random.Random(n))
            self.assertTrue(all(sorted(r)==list(range(n)) for r in rows))
            self.assertTrue(all(sorted(r[c] for r in rows)==list(range(n)) for c in range(n)))

    def test_baseline_needs_a_completed_swarm(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                run(tmp,'baseline',Scripted())

    def test_interruptions_retries_and_rejections_keep_the_turn(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cc,'BACKOFF_SECONDS',.01):
            sw,res=run(tmp,'swarm',Scripted(faults=[(1,'solving','fragment'),(2,'chat_1','empty'),(3,'chat_1','taken')]))
            self.assertEqual(res['status'],'completed')
            ops=[read_json(p) for p in sorted(sw.directory.glob('operational_events/continuation_*.json'))]
            self.assertEqual(len(ops),2)
            notes=[o['message']['content'] for o in ops]
            self.assertIn(cc.INTERRUPTED_NOTE,notes);self.assertTrue(any('HTTP 429' in n for n in notes))
            s=read_json(sw.directory/'state.json')
            h=s['agents']['1']['history']
            k=next(j for j,m in enumerate(h) if m['role']=='assistant' and 'partial thought' in m['content'])
            self.assertEqual(h[k]['content'],'<think>\npartial thought\n</think>\n\n');self.assertEqual(h[k+1]['content'],cc.INTERRUPTED_NOTE)
            text=(sw.directory/'generated'/'agent_1.generated.txt').read_text()
            self.assertEqual(text.count('partial thought'),1)  # the fragment once, then the continuation
            ix=read_json(sw.directory/'generated'/'agent_1.generated.index.json')
            frag=next(j for j,p in enumerate(ix) if p['finish_reason']=='stream_incomplete')
            self.assertEqual(ix[frag+1]['phase'],'solving');self.assertGreater(ix[frag+1]['call'],ix[frag]['call'])
            self.assertGreaterEqual(ix[frag+1]['marks'][0][1],ix[frag]['marks'][-1][1])
            ev=events(sw.directory)
            self.assertTrue(any('already taken' in (e['error'] or '') for e in ev if e['kind']=='applied'))
            reqs=sorted((q for q in requests(sw.directory) if q['phase']=='chat_1'),key=lambda q:q['call'])
            order=[q['agent'] for q in reqs]
            for a in (2,3):  # failed and rejected turns are retried by the same writer before anyone else writes
                first=order.index(a);self.assertEqual(order[first+1],a)
            self.assertEqual(sorted(s['claims']),['b0','b1','b2'])

    def test_a_stuck_writer_halts_the_run_with_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(cc,'BACKOFF_SECONDS',0), patch.object(cc,'MAX_ATTEMPTS',2):
            t0=time.time()
            sw,res=run(tmp,'swarm',Scripted(faults=[(1,'chat_1','empty')]*3))
            self.assertLess(time.time()-t0,30)
            self.assertEqual(res['status'],'retry_limit')
            h=res['halt'];self.assertEqual((h['reason'],h['agent'],h['stage']),('retry_limit',1,'writing_1'))
            self.assertIn('2 failed attempts at chat_1',h['detail']);self.assertEqual(h['queue'][0],1)
            self.assertFalse([q for q in requests(sw.directory) if q['phase']=='chat_1' and q['agent']!=1])
            ev=events(sw.directory);halt=next(e['seq'] for e in ev if e['kind']=='halt')
            self.assertFalse([e for e in ev if e['kind']=='dispatch' and e['seq']>halt])

    def test_identical_prelude_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            sw,_=run(tmp,'swarm',Scripted())
            pre=[q for q in requests(sw.directory) if q['phase']=='prelude']
            self.assertEqual({q['context']['instruction'] for q in pre},{cc.SOLO_INSTRUCTION})
            self.assertTrue(all(set(q['context'])=={'agent','current_task','task','instruction'} for q in pre))
            self.assertTrue(all('team' not in q['messages'][0]['content'] and 'chat' not in q['messages'][0]['content'] for q in pre))


if __name__=='__main__':unittest.main()
