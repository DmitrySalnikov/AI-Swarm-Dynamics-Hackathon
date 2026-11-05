"""Covert-channel experiment (covert-channel-v2) on the ladder rungs (swarm_pipeline.covert_channel).

For every run seed (rung seed 3, 82, 83, 84 + offset) the swarm arm runs first and then its baseline twin, which
imports the swarm's prelude (--arms picks the arms; the independent baseline also needs its completed swarm twin,
whose token count ends it). Rungs run in parallel (one process each, ten workers); within a rung the runs go one
after the other. The budget is the ledger ceiling of the whole rung root: every arm and run in it, earlier ones
included. Completed runs are skipped on a rerun.
Interruptions are continued inside the runner; a run that halts (retry limit, budget, context, deadlock) stops its
rung with the reason in its state.json and result.json. Status: <root>/covert.status.json.

  # first probe: one rung, one seed
  CLOUDRU_API_KEY=... python -m swarm_pipeline.ladder_covert --root artifacts/pipeline-ladder-v2-covert \\
      --items-a artifacts/pipeline-ladder-v2/items --items-b artifacts/pipeline-ladder-v2/items-b \\
      --rungs 1 --run-offsets 0 --budget-per-rung 1000
  # full set
  ... --run-offsets 0 1000 2000 3000 --budget-per-rung 3000 --stagger-seconds 25
  # independent baseline next to the completed swarm runs
  ... --arms swarm independent --run-offsets 0 1000 2000 3000 --budget-per-rung <spent so far + new> --stagger-seconds 25
"""
import argparse,os,sys,time
from pathlib import Path
from swarm_pipeline.supervision import read_json,write_json
from swarm_pipeline.covert_channel import ARMS as ALL_ARMS
from swarm_pipeline.ladder_sequential import SEEDS,add_launch_options,completed,drive,may_start,run_with_restarts



def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--items-a',type=Path,required=True,help='Directory with set-A rung_1..4.jsonl')
    parser.add_argument('--items-b',type=Path,required=True,help='Directory with set-B rung_1..4.jsonl')
    parser.add_argument('--budget-per-rung',type=float,default=3000)
    parser.add_argument('--offline',action='store_true',help='scripted agents, no provider calls (pipeline check)')
    parser.add_argument('--run-offsets',nargs='+',type=int,default=[0,1000,2000,3000],help='run seed = rung seed + offset')
    parser.add_argument('--arms',nargs='+',default=['swarm','baseline'],choices=ALL_ARMS,help='in this order per seed; swarm first')
    parser.add_argument('--rungs',nargs='+',type=int,default=list(SEEDS),choices=list(SEEDS))
    add_launch_options(parser)
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    status=args.root/'covert.status.json'

    def directory(rung,arm,offset):
        return args.root/f'rung_{rung}'/arm/f'seed_{SEEDS[rung]+offset}'

    def publish(state,**details):
        write_json(status,{'state':state,'details':{'pid':os.getpid(),'run_offsets':args.run_offsets,**details},'updated_at':time.time()})

    def run(rung):
        root=args.root/f'rung_{rung}'
        for offset in args.run_offsets:
            for arm in args.arms:
                done=directory(rung,arm,offset)
                if completed(done):
                    continue
                if not may_start(args):
                    return 'held'
                cmd=[sys.executable,'-X','utf8','-m','swarm_pipeline.covert_channel','--arm',arm,
                     '--items-a',str(args.items_a/f'rung_{rung}.jsonl'),'--items-b',str(args.items_b/f'rung_{rung}.jsonl'),
                     '--out',str(root),'--seed',str(SEEDS[rung]+offset),'--workers','10',
                     '--budget-rub',str(args.budget_per_rung)]+(['--offline'] if args.offline else [])
                root.mkdir(parents=True,exist_ok=True)
                run_with_restarts(cmd,done,root/f'{arm}-seed_{SEEDS[rung]+offset}',0)
        return rung

    def progress():
        out={}
        for rung in args.rungs:
            for offset in args.run_offsets:
                for arm in args.arms:
                    p=directory(rung,arm,offset)/'state.json'
                    if p.exists():
                        s=read_json(p);out[f'{rung}/{arm}/{p.parent.name}']=(s['status'],sum(len(a['answers']) for a in s['agents'].values()))
        return out

    drive(args,run,progress,publish)


if __name__=='__main__':main()
