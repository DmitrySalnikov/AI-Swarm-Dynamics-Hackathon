"""Swarm repeats on the four ladder rungs with the current v2 prompts (distribution, solving, exchange, commit).

Each run is the cooperative v2 protocol launched through swarm_pipeline.pilot, exactly as set4_experiment launches
the ladder swarm, but without the paired control and without embedding analysis (no OpenRouter calls). Run seed =
rung seed (3, 82, 83, 84) + offset. Use a root separate from earlier swarms whose prompts differ (protocol_hash in
each manifest). Rungs run in parallel (one process each, ten workers); within a rung the runs go one after the other.
The budget applies to all swarm runs of the rung. Distribution and commit interruptions are retried by the v2 runner
itself; a run halted in solving is continued with scripts/restart_halted_run.py, as in ladder_sequential.
Status: <root>/swarm-replicates.status.json.

  CLOUDRU_API_KEY=... python -m swarm_pipeline.ladder_swarm --root artifacts/pipeline-ladder-v2-swarm-repeats \\
      --items-dir artifacts/pipeline-ladder-v2/items --run-offsets 0 1000 2000 3000 --budget-per-rung 600
"""
import argparse,os,sys,time
from pathlib import Path
from swarm_pipeline.supervision import read_json,write_json
from swarm_pipeline.ladder_sequential import SEEDS,add_launch_options,completed,drive,may_start,run_with_restarts


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--items-dir',type=Path,required=True,help='Directory with rung_1..4.jsonl')
    parser.add_argument('--budget-per-rung',type=float,default=600)
    parser.add_argument('--offline',action='store_true',help='scripted agents, no provider calls (pipeline check)')
    parser.add_argument('--run-offsets',nargs='+',type=int,default=[0,1000,2000,3000],help='run seed = rung seed + offset')
    parser.add_argument('--max-restarts',type=int,default=30,help='explicit restarts per run after provider interruptions')
    parser.add_argument('--rungs',nargs='+',type=int,default=list(SEEDS),choices=list(SEEDS))
    add_launch_options(parser)
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    status=args.root/'swarm-replicates.status.json'

    def directory(rung,offset):
        return args.root/f'rung_{rung}'/'swarm'/'cooperative'/f'seed_{SEEDS[rung]+offset}'

    def publish(state,**details):
        write_json(status,{'state':state,'details':{'pid':os.getpid(),'run_offsets':args.run_offsets,**details},'updated_at':time.time()})

    def run(rung):
        root=args.root/f'rung_{rung}'/'swarm'
        for offset in args.run_offsets:
            done,seed=directory(rung,offset),SEEDS[rung]+offset
            if completed(done):
                continue
            if not may_start(args):
                return 'held'
            cmd=[sys.executable,'-X','utf8','-m','swarm_pipeline.pilot','--items',str(args.items_dir/f'rung_{rung}.jsonl'),
                 '--out',str(root),'--condition','cooperative','--protocol','v2','--seed',str(seed),'--workers','10',
                 '--budget-rub',str(args.budget_per_rung),'--partial-continuation']+(['--offline'] if args.offline else [])
            root.mkdir(parents=True,exist_ok=True)
            run_with_restarts(cmd,done,root/f'supervisor-seed_{seed}',args.max_restarts)
        return rung

    def progress():
        out={}
        for rung in args.rungs:
            for offset in args.run_offsets:
                p=directory(rung,offset)/'state.json'
                if p.exists():
                    s=read_json(p);out[f'{rung}/{p.parent.name}']=(s['status'],s['phase'],s['calls'])
        return out

    drive(args,run,progress,publish)

if __name__=='__main__':main()
