"""Run the approved small live window/overlap study, preserving independent journals."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_pipeline.runner import Config, load_tasks
from swarm_pipeline.stable_live import RetainingClient
from swarm_pipeline.window_pilot import DistributionRunner, GradualRunner
from swarm_pipeline.supervision import write_json

ROOT = Path('artifacts/pipeline-windows')


def run_one(kind, seed):
    root = ROOT / 'live' / f'{kind}-{seed}'
    config = Config(condition='cooperative', agents=10, seed=seed,
                    budget_rub=75 if kind == 'gradual' else 25, max_calls=100)
    cls = GradualRunner if kind == 'gradual' else DistributionRunner
    runner = cls(load_tasks('bench/ladder/rung_1.jsonl'), config, root,
                 RetainingClient(os.environ['CLOUDRU_API_KEY']), workers=10, partial_continuation=True)
    try:
        result = runner.run()
        return {'kind': kind, 'seed': seed, 'directory': str(runner.directory),
                'status': result['status'], 'calls': result['calls'], 'cost_rub': result['spent_root_rub']}
    except Exception as exc:
        return {'kind': kind, 'seed': seed, 'directory': str(runner.directory),
                'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)}


def main():
    records = []
    write_json(ROOT / 'live.status.json', {'state': 'running', 'updated_at': time.time()})
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(run_one, 'distribution', seed) for seed in range(501, 506)]
        futures.append(pool.submit(run_one, 'gradual', 601))
        for future in as_completed(futures):
            records.append(future.result())
            write_json(ROOT / 'live-results.json', records)
            print(json.dumps(records[-1]), flush=True)
    success = all(r['status'] == 'completed' for r in records)
    write_json(ROOT / 'live.status.json', {'state': 'complete' if success else 'failed',
        'event_key': 'window-live:complete' if success else 'window-live:failed',
        'updated_at': time.time(), 'details': records})


if __name__ == '__main__':
    main()
