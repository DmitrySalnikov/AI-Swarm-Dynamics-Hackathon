"""One-task private controls on the four ladder rungs with a random task-to-agent assignment per run.

Every agent solves one task on its own (swarm_pipeline.private_control, unchanged prompts), with no negotiation and
no peer exchange. The ten tasks of a rung go to the ten agents by a random permutation drawn from the run seed, so the
control does not need a swarm run first. R and the same-task share do not depend on which agent got which task
(agents are interchangeable); only that the ten tasks differ matters.

Run seed = rung seed (3, 82, 83, 84, as in the swarm runs) + offset. The assignment of each run is written to
<root>/rung_k/private/assignments_seed_S.json and stored in the run manifest; an existing file is never replaced.
Runs land in <root>/rung_k/private/fixed/seed_S, next to the swarm-matched control seed_S of the ladder experiment,
which a rerun with offset 0 skips as completed. Rungs run in parallel (one process each, ten workers); within a rung
the runs go one after the other. The budget applies to all private runs of the rung. Halted runs are continued with
scripts/restart_halted_run.py, as in ladder_sequential. Status: <root>/private-replicates.status.json.

  CLOUDRU_API_KEY=... python -m swarm_pipeline.ladder_private --root artifacts/pipeline-ladder-v2-simple \\
      --items-dir artifacts/pipeline-ladder-v2/items --run-offsets 1000 2000 3000 --budget-per-rung 150
"""
import argparse,os,random,sys,time
from pathlib import Path
from swarm_pipeline.runner import load_tasks
from swarm_pipeline.supervision import read_json,write_json
from swarm_pipeline.ladder_sequential import SEEDS,add_launch_options,completed,drive,may_start,run_with_restarts


def random_assignments(task_ids,seed):
    """Agent '1'..'N' -> task id, a deterministic random permutation of the tasks for this run seed."""
    order=list(task_ids);random.Random(f'{seed}:assignments').shuffle(order)
    return {str(i+1):t for i,t in enumerate(order)}


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--items-dir',type=Path,required=True,help='Directory with rung_1..4.jsonl')
    parser.add_argument('--budget-per-rung',type=float,default=150)
    parser.add_argument('--offline',action='store_true',help='scripted agents, no provider calls (pipeline check)')
    parser.add_argument('--run-offsets',nargs='+',type=int,default=[1000,2000,3000],help='run seed = rung seed + offset')
    parser.add_argument('--max-restarts',type=int,default=30,help='explicit restarts per run after provider interruptions')
    parser.add_argument('--rungs',nargs='+',type=int,default=list(SEEDS),choices=list(SEEDS))
    add_launch_options(parser)
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    status=args.root/'private-replicates.status.json'

    def directory(rung,offset):
        return args.root/f'rung_{rung}'/'private'/'fixed'/f'seed_{SEEDS[rung]+offset}'

    def publish(state,**details):
        write_json(status,{'state':state,'details':{'pid':os.getpid(),'run_offsets':args.run_offsets,**details},'updated_at':time.time()})

    def run(rung):
        root=args.root/f'rung_{rung}'/'private';items=args.items_dir/f'rung_{rung}.jsonl'
        for offset in args.run_offsets:
            done,seed=directory(rung,offset),SEEDS[rung]+offset
            if completed(done):
                continue
            if not may_start(args):
                return 'held'
            mapping=random_assignments([t['id'] for t in load_tasks(items)],seed)
            path=root/f'assignments_seed_{seed}.json'
            if path.exists() and read_json(path)!=mapping:
                raise RuntimeError(f'{path} differs from the assignment drawn for seed {seed}')
            root.mkdir(parents=True,exist_ok=True);write_json(path,mapping)
            cmd=[sys.executable,'-X','utf8','-m','swarm_pipeline.private_control','--items',str(items),'--out',str(root),
                 '--condition','fixed','--assignments',str(path),'--seed',str(seed),'--workers','10',
                 '--budget-rub',str(args.budget_per_rung)]+(['--offline'] if args.offline else [])
            run_with_restarts(cmd,done,root/f'seed_{seed}',args.max_restarts)
        return rung

    def progress():
        out={}
        for rung in args.rungs:
            for offset in args.run_offsets:
                p=directory(rung,offset)/'state.json'
                if p.exists():
                    s=read_json(p);out[f'{rung}/{p.parent.name}']=(s['status'],sum(len(a['answers']) for a in s['agents'].values()))
        return out

    drive(args,run,progress,publish)

if __name__=='__main__':main()
