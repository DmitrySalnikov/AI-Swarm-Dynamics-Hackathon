"""Sequential independent controls on the four ladder rungs: every agent solves its tasks one after another,
with no negotiation and no peer exchange. Conditions:
  fixed                  every agent takes the supplied order 1..N (one order shared by all agents);
  fixed_random_order     one random order per run, shared by all agents; differs between runs (run seeds);
  shuffled               every agent takes its own random order;
  shuffled_unique_start  ten different first tasks (random permutation), then each agent's own random order.

Rungs run in parallel (one process each, ten workers); within a rung the runs (conditions x run offsets) run one
after the other. Run seed = rung seed (3, 82, 83, 84, as in the swarm runs) + offset, so --run-offsets 0 1000 2000 3000
gives four reproducible runs per rung. The budget applies to the whole rung root (all its runs). Completed runs are
reused on a rerun; nothing is regenerated. A run halted by provider stream interruptions is continued with
scripts/restart_halted_run.py (explicit recovery turns, unchanged manifest), up to --max-restarts times.

  CLOUDRU_API_KEY=... python -m swarm_pipeline.ladder_sequential --conditions shuffled_unique_start \\
      --run-offsets 0 1000 2000 3000 --budget-per-rung 2000 \\
      --root artifacts/pipeline-ladder-v2-simple-sequential --items-dir artifacts/pipeline-ladder-v2/items
"""
import argparse,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from swarm_pipeline.supervision import read_json,write_json
from swarm_pipeline.sequential_control import condition_dir

SEEDS={1:3,2:82,3:83,4:84}
RESTART=Path(__file__).resolve().parents[1]/'scripts'/'restart_halted_run.py'
CONDITIONS={'fixed':('fixed',[]),'fixed_random_order':('fixed',['--shared-random-order']),
            'shuffled':('shuffled',[]),'shuffled_unique_start':('shuffled',['--unique-start'])}


def completed(run):
    return (run/'result.json').exists() and read_json(run/'state.json')['status']=='completed'


def run_with_restarts(cmd,run,logs,max_restarts):
    """Run one generation command for run directory `run`, appending its output to <logs>.stdout.log/.stderr.log."""
    with Path(f'{logs}.stdout.log').open('a') as out,Path(f'{logs}.stderr.log').open('a') as err:
        for attempt in range(max_restarts+1):
            if not subprocess.Popen(cmd,stdout=out,stderr=err).wait():
                return
            # A run halted by provider stream interruptions gets explicit recovery turns (manifest unchanged)
            # and the same command is rerun; anything else stops with its journals retained.
            halted=(run/'state.json').exists() and read_json(run/'state.json')['status']=='provider_incomplete'
            if not halted or attempt==max_restarts or subprocess.run(
                    [sys.executable,str(RESTART),str(run)],stdout=out,stderr=err).returncode:
                raise RuntimeError(f'{run} failed; journals retained')
            time.sleep(30)


def add_launch_options(parser):
    parser.add_argument('--stagger-seconds',type=float,default=0,help='rung i of --rungs starts i x this delay after the first')
    parser.add_argument('--hold-file',type=Path,help='once this file exists no further run starts; runs in progress finish')


def may_start(args):
    return not (args.hold_file and args.hold_file.exists())


def drive(args,run,progress,publish):
    """Run every rung in parallel with a staggered start; run(rung) returns 'held' when the hold file stopped it."""
    def staggered(rung):
        time.sleep(args.rungs.index(rung)*args.stagger_seconds)
        return run(rung)
    publish('running')
    errors,held=[],[]
    with ThreadPoolExecutor(max_workers=len(args.rungs)) as pool:
        futures={pool.submit(staggered,r):r for r in args.rungs}
        while futures:
            for f in [f for f in futures if f.done()]:
                rung=futures.pop(f)
                try:
                    if f.result()=='held':held.append(rung)
                except Exception as e:errors.append(str(e))
            publish('running',progress=progress(),errors=errors,held=sorted(held))
            if futures:time.sleep(10)
    publish('failed' if errors else 'held' if held else 'complete',progress=progress(),errors=errors,held=sorted(held))
    if errors:raise SystemExit('; '.join(errors))


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--items-dir',type=Path,required=True,help='Directory with rung_1..4.jsonl')
    parser.add_argument('--budget-per-rung',type=float,default=600)
    parser.add_argument('--offline',action='store_true',help='scripted agents, no provider calls (pipeline check)')
    parser.add_argument('--conditions',nargs='+',default=['fixed','shuffled_unique_start'],choices=list(CONDITIONS))
    parser.add_argument('--run-offsets',nargs='+',type=int,default=[0],help='run seed = rung seed + offset')
    parser.add_argument('--tasks-per-agent',type=int,help='keep the first K tasks of every order')
    parser.add_argument('--max-restarts',type=int,default=30,help='explicit restarts per run after provider interruptions')
    parser.add_argument('--rungs',nargs='+',type=int,default=list(SEEDS),choices=list(SEEDS))
    add_launch_options(parser)
    args=parser.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    status=args.root/'experiment.status.json'
    k=['--tasks-per-agent',str(args.tasks_per_agent)] if args.tasks_per_agent else []

    def directory(rung,condition,offset):
        base,flags=CONDITIONS[condition]
        name=condition_dir(base,'--unique-start' in flags,'--shared-random-order' in flags,args.tasks_per_agent)
        return args.root/f'rung_{rung}'/name/f'seed_{SEEDS[rung]+offset}'

    def publish(state,**details):
        write_json(status,{'state':state,'details':{'pid':os.getpid(),'conditions':args.conditions,'run_offsets':args.run_offsets,
            'tasks_per_agent':args.tasks_per_agent,**details},'updated_at':time.time()})

    def run(rung):
        root=args.root/f'rung_{rung}'
        for condition in args.conditions:
            for offset in args.run_offsets:
                done=directory(rung,condition,offset)
                if completed(done):
                    continue
                if not may_start(args):
                    return 'held'
                base,flags=CONDITIONS[condition]
                cmd=[sys.executable,'-X','utf8','-m','swarm_pipeline.sequential_control','--items',str(args.items_dir/f'rung_{rung}.jsonl'),
                     '--out',str(root),'--condition',base,'--seed',str(SEEDS[rung]+offset),'--workers','10',
                     '--budget-rub',str(args.budget_per_rung)]+flags+k+(['--offline'] if args.offline else [])
                root.mkdir(parents=True,exist_ok=True)
                run_with_restarts(cmd,done,root/done.parent.name,args.max_restarts)
        return rung

    def progress():
        out={}
        for rung in args.rungs:
            for condition in args.conditions:
                for offset in args.run_offsets:
                    p=directory(rung,condition,offset)/'state.json'
                    if p.exists():
                        s=read_json(p);out[f'{rung}/{p.parent.parent.name}/{p.parent.name}']=(s['status'],sum(len(a['answers']) for a in s['agents'].values()))
        return out

    drive(args,run,progress,publish)

if __name__=='__main__':main()
