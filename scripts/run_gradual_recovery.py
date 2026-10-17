"""Start a corrected timing pilot from the already observed assignment history.

This is an explicitly separate solving replicate, not a replay of unknown calls.
Only completed distribution turns are imported; all new solves have a new seed.
"""
import os
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from swarm_pipeline.runner import Config
from swarm_pipeline.stable_live import RetainingClient
from swarm_pipeline.window_pilot import GradualOfflineClient
from swarm_pipeline.gradual_scheduler import BudgetAwareGradualRunner
from swarm_pipeline.supervision import read_json, write_json


def main():
    source = Path('artifacts/pipeline-windows/live/gradual-601/cooperative/seed_601')
    root = Path('artifacts/pipeline-windows/live/gradual-602')
    manifest = read_json(source/'manifest.json')
    config = Config(**{**manifest['config'],'seed':602,'budget_rub':125})
    runner = BudgetAwareGradualRunner(manifest['tasks'],config,root,
                                     RetainingClient(os.environ['CLOUDRU_API_KEY']),workers=10,partial_continuation=True)
    runner.manifest['imported_distribution'] = {'source':str(source),'new_solving_seed':602,
        'reason':'Initial pilot stopped before reporting because temporary in-flight reservations were treated as a permanent budget stop.'}
    if not (runner.directory/'state.json').exists():
        runner.state = runner.initial_state()
        cost = 0
        for path in sorted(source.glob('calls/*/request.json')):
            q = read_json(path)
            if q['context']['phase'] != 'negotiation':
                break
            r, e = read_json(path.with_name('response.json')), read_json(path.with_name('event.json'))
            assert r['finish_reason']=='stop' and not e['action_error']
            assert q['context']==runner.context(q['agent'])
            target = runner.directory/'calls'/path.parent.name
            target.mkdir(parents=True,exist_ok=True)
            for name in ('request.json','response.json','event.json'):
                shutil.copy2(path.with_name(name),target/name)
            runner.apply(q,r)
            runner.state['applied_order'].append(q['call'])
            cost += r['cost_rub']
        assert runner.state['phase']=='solving' and runner.state['calls']==20
        runner.state['next_call']=20
        runner.state['pending_calls']=[]
        write_json(runner.directory/'state.json',runner.state)
        write_json(runner.directory/'imported-distribution.json',{'cost_rub':cost,'source':str(source),'calls':20})
    result = runner.run()
    imported = read_json(runner.directory/'imported-distribution.json')['cost_rub']
    write_json(Path('artifacts/pipeline-windows/recovered-gradual-602.json'),
               {'kind':'gradual','seed':602,'directory':str(runner.directory),'status':result['status'],
                'calls':result['calls'],'cost_rub':result['spent_root_rub']-imported,
                'imported_cost_excluded_rub':imported,'recovered_at':time.time()})


if __name__=='__main__':
    main()
