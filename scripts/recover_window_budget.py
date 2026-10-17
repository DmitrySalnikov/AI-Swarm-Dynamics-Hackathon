"""Revise only a completed worker's admission budget and resume saved state."""
import argparse
from dataclasses import replace
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_pipeline.runner import Config
from swarm_pipeline.stable_live import RetainingClient
from swarm_pipeline.window_pilot import DistributionRunner, GradualRunner
from swarm_pipeline.supervision import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('distribution','gradual'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--budget', type=float, required=True)
    args = parser.parse_args()
    root = Path('artifacts/pipeline-windows/live') / f'{args.kind}-{args.seed}'
    directory = root/'cooperative'/f'seed_{args.seed}'
    old = read_json(directory/'manifest.json')
    config = replace(Config(**old['config']), budget_rub=args.budget)
    cls = DistributionRunner if args.kind=='distribution' else GradualRunner
    runner = cls(old['tasks'], config, root, RetainingClient(os.environ['CLOUDRU_API_KEY']),workers=10,partial_continuation=True)
    runner.prepare_existing()
    if args.kind=='gradual':
        state = read_json(directory/'state.json')
        if state['pending_calls'] or state['status'] != 'budget_limit':
            raise ValueError('Only a drained budget-limited gradual worker may resume')
        write_json(directory/'operational_events'/'before_budget_resume.state.json',state)
        state['status']='running'; write_json(directory/'state.json',state)
    result = runner.run()
    record = {'kind':args.kind,'seed':args.seed,'directory':str(directory),'status':result['status'],
              'calls':result['calls'],'cost_rub':result['spent_root_rub'],'recovered_at':time.time()}
    write_json(Path('artifacts/pipeline-windows')/f'recovered-{args.kind}-{args.seed}.json',record)


if __name__=='__main__':
    main()
