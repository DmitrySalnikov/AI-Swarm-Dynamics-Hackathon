import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from swarm_pipeline.supervision import read_json
from swarm_pipeline.audit import audit

TASKS=[{'id':f't{i}','prompt':'Give one integer.','answer':'1','check':{'type':'exact'}} for i in range(3)]


class LadderSwarmTests(unittest.TestCase):
    def launch(self,tmp):
        return subprocess.run([sys.executable,'-m','swarm_pipeline.ladder_swarm','--root',str(tmp/'root'),'--items-dir',str(tmp/'items'),
                               '--offline','--rungs','1','2','--run-offsets','0','1000'],capture_output=True,text=True)

    def test_offline_swarms_complete_and_rerun_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'items').mkdir()
            for rung in (1,2):(tmp/'items'/f'rung_{rung}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in TASKS))
            self.assertEqual(self.launch(tmp).returncode,0)
            for rung,seed in ((1,3),(1,1003),(2,82),(2,1082)):
                run=tmp/'root'/f'rung_{rung}'/'swarm'/'cooperative'/f'seed_{seed}'
                state=read_json(run/'state.json');self.assertEqual(state['status'],'completed')
                self.assertEqual(audit(run)['integrity'],'passed');self.assertEqual(read_json(run/'manifest.json')['config']['seed'],seed)
                self.assertEqual(sorted(a['order'][0] for a in state['agents'].values()),[t['id'] for t in TASKS])
            self.assertEqual(read_json(tmp/'root'/'swarm-replicates.status.json')['state'],'complete')
            calls=sorted((tmp/'root').glob('rung_*/swarm/cooperative/seed_*/calls/*'))
            self.assertEqual(self.launch(tmp).returncode,0)
            self.assertEqual(sorted((tmp/'root').glob('rung_*/swarm/cooperative/seed_*/calls/*')),calls)


if __name__=='__main__':unittest.main()
