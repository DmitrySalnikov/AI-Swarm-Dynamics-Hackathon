import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from swarm_pipeline.ladder_private import random_assignments
from swarm_pipeline.supervision import read_json
from swarm_pipeline.audit import audit

TASKS=[{'id':f't{i}','prompt':'Give one integer.','answer':'1','check':{'type':'exact'}} for i in range(6)]
IDS=[t['id'] for t in TASKS]


class LadderPrivateTests(unittest.TestCase):
    def test_random_assignments(self):
        a=random_assignments(IDS,3)
        self.assertEqual(sorted(a),[str(i) for i in range(1,7)]);self.assertEqual(sorted(a.values()),IDS)
        self.assertEqual(a,random_assignments(IDS,3))
        self.assertEqual(len({tuple(random_assignments(IDS,s).values()) for s in (3,1003,2003,3003)}),4)

    def launch(self,tmp,*extra):
        return subprocess.run([sys.executable,'-m','swarm_pipeline.ladder_private','--root',str(tmp/'root'),'--items-dir',str(tmp/'items'),
                               '--offline','--rungs','1','2','--run-offsets','0','1000',*extra],capture_output=True,text=True)

    def test_offline_runs_follow_their_assignments_and_rerun_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'items').mkdir()
            for rung in (1,2):(tmp/'items'/f'rung_{rung}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in TASKS))
            self.assertEqual(self.launch(tmp).returncode,0)
            orders=set()
            for rung,seed in ((1,3),(1,1003),(2,82),(2,1082)):
                private=tmp/'root'/f'rung_{rung}'/'private'
                run=private/'fixed'/f'seed_{seed}';mapping=read_json(private/f'assignments_seed_{seed}.json')
                self.assertEqual(mapping,random_assignments(IDS,seed))
                state=read_json(run/'state.json');self.assertEqual(state['status'],'completed')
                self.assertEqual(audit(run)['integrity'],'passed');self.assertEqual(read_json(run/'manifest.json')['assignments'],mapping)
                for i,a in state['agents'].items():
                    self.assertEqual(a['order'],[mapping[i]]);self.assertEqual(list(a['answers']),[mapping[i]]);self.assertEqual(a['inbox'],[])
                orders.add(tuple(mapping.values()))
            self.assertEqual(len(orders),4)
            self.assertEqual(read_json(tmp/'root'/'private-replicates.status.json')['state'],'complete')
            calls=sorted((tmp/'root').glob('rung_*/private/fixed/seed_*/calls/*'))
            self.assertEqual(self.launch(tmp).returncode,0)
            self.assertEqual(sorted((tmp/'root').glob('rung_*/private/fixed/seed_*/calls/*')),calls)
            path=tmp/'root'/'rung_1'/'private'/'assignments_seed_2003.json'
            path.write_text(json.dumps({str(i):t for i,t in enumerate(IDS,1)}))
            failed=self.launch(tmp,'--run-offsets','2000')
            self.assertNotEqual(failed.returncode,0);self.assertFalse((path.parent/'fixed'/'seed_2003').exists())

    def test_stagger_and_hold_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'items').mkdir()
            for rung in (1,2):(tmp/'items'/f'rung_{rung}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in TASKS))
            self.assertEqual(self.launch(tmp,'--run-offsets','0','--stagger-seconds','3').returncode,0)
            first={rung:min(read_json(p)['started_at'] for p in (tmp/'root'/f'rung_{rung}'/'private'/'fixed'/f'seed_{seed}').glob('calls/*/request.json'))
                   for rung,seed in ((1,3),(2,82))}
            self.assertGreaterEqual(first[2]-first[1],3)
            (tmp/'hold').touch()
            self.assertEqual(self.launch(tmp,'--run-offsets','0','1000','--hold-file',str(tmp/'hold')).returncode,0)
            status=read_json(tmp/'root'/'private-replicates.status.json')
            self.assertEqual(status['state'],'held');self.assertEqual(status['details']['held'],[1,2])
            self.assertFalse(list((tmp/'root').glob('rung_*/private/fixed/seed_10*')))


if __name__=='__main__':unittest.main()
