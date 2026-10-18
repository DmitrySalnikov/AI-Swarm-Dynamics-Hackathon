"""Run distribution only on four saved task sets using explicit per-request actions."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_pipeline.runner import Config
from swarm_pipeline.stable_live import RetainingClient
from swarm_pipeline.supervision import read_json, write_json
from swarm_pipeline.window_pilot import DistributionRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('artifacts/pipeline-step-prompts'))
    parser.add_argument('--budget-per-level', type=float, default=25)
    parser.add_argument('--repeat-count', type=int, default=1)
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--parallel-groups', type=int, default=4)
    args = parser.parse_args()
    if args.repeat_count < 1 or args.parallel_groups < 1:
        parser.error('Repeat count and parallel groups must be positive')
    os.environ['SWARM_PROMPT_STYLE'] = 'simple_steps'

    def run_level(level, seed, repeat):
        source = Path(f'artifacts/pipeline-ladder-v2-simple/rung_{level}/swarm/cooperative/seed_{seed}/manifest.json')
        baseline = read_json(source)
        actual_seed = seed + args.seed_offset + repeat*1000
        cfg = replace(Config(**baseline['config']), seed=actual_seed,
                      budget_rub=args.budget_per_level, max_calls=40)
        group_root = args.out/f'repeat_{repeat+1}' if args.repeat_count > 1 else args.out
        runner = DistributionRunner(baseline['tasks'], cfg, group_root/f'rung_{level}',
                                    RetainingClient(os.environ['CLOUDRU_API_KEY']), workers=10,
                                    partial_continuation=True)
        assert runner.manifest['task_hash'] == baseline['task_hash']
        try:
            result = runner.run()
            return {'level':level,'repeat':repeat+1,'seed':actual_seed,'status':result['status'],'directory':str(runner.directory),
                    'calls':result['calls'],'spent_root_rub':result['spent_root_rub']}
        except Exception as exc:
            return {'level':level,'repeat':repeat+1,'seed':actual_seed,'status':'failed','directory':str(runner.directory),
                    'error_type':type(exc).__name__,'error':str(exc)}

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    maximum_total = 4*args.repeat_count*args.budget_per_level
    write_json(args.out/'status.json', {'state':'running','maximum_total_rub':maximum_total})
    with ThreadPoolExecutor(max_workers=args.parallel_groups) as pool:
        futures = [pool.submit(run_level, level, seed, repeat) for repeat in range(args.repeat_count)
                   for level, seed in ((1,3),(2,82),(3,83),(4,84))]
        for future in as_completed(futures):
            records.append(future.result())
            write_json(args.out/'results.json', records)
            print(json.dumps(records[-1]), flush=True)
    success = all(r['status']=='completed' for r in records)
    write_json(args.out/'status.json', {'state':'completed' if success else 'failed','results':records,
                                     'maximum_total_rub':maximum_total})
    if not success:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
