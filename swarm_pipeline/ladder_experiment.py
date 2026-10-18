"""Complete full-arc matched comparisons for all four ladder rungs."""
import argparse,os,shutil,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from swarm_pipeline.supervision import read_json,write_json
from swarm_pipeline.audit import audit

ROOT=Path('artifacts/pipeline-ladder')
SEEDS={1:3,2:82,3:83,4:84}
LABELS={1:'Unrelated domains',2:'Physics subfields',3:'Linear algebra',4:'Same determinant template'}

def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--items-dir',type=Path,default=Path('bench/ladder'),help='Directory with rung_1..4.jsonl')
    parser.add_argument('--reuse-rung1',default='artifacts/pipeline-v2/cooperative/seed_3',help='Completed rung-1 swarm to reuse; empty string generates a new one')
    parser.add_argument('--swarm-budget',type=float,default=100)
    parser.add_argument('--private-budget',type=float,default=60)
    parser.add_argument('--skip-checkpoint-html',action='store_true')
    args=parser.parse_args();ROOT=args.root;reuse=bool(args.reuse_rung1)
    ROOT.mkdir(parents=True,exist_ok=True)
    def publish(state,**details):write_json(ROOT/'experiment.status.json',{'state':state,'event_key':'ladder-full-arcs:'+state,'details':{'pid':os.getpid(),'rungs':4,'parallel_runs':4,'workers_each':10,**details},'updated_at':time.time()})
    publish('running',stage='generation-and-analysis')
    try:
        if reuse:
            source=Path(args.reuse_rung1)
            destination=ROOT/'rung_1/swarm/cooperative/seed_3'
            reference=audit(source)
            if not destination.exists():
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source,destination)
                write_json(ROOT/'rung_1/reused-swarm.json',{'source':str(source.resolve()),'task_hash':reference['task_hash'],'additional_generation_cost_rub':0})
        def run(rung):
            root=ROOT/f'rung_{rung}';root.mkdir(parents=True,exist_ok=True)
            command=[sys.executable,'-X','utf8','-m','swarm_pipeline.set4_experiment','--root',str(root),'--seed',str(SEEDS[rung]),'--items',str(args.items_dir/f'rung_{rung}.jsonl'),'--label',f'Ladder rung {rung}: {LABELS[rung]}','--swarm-budget',str(args.swarm_budget),'--private-budget',str(args.private_budget)]
            env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1'
            with (root/'supervisor.stdout.log').open('a') as stdout,(root/'supervisor.stderr.log').open('a') as stderr:
                child=subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if child.wait():raise RuntimeError(f'Ladder rung {rung} failed; saved journals and logs retained')
            return rung
        done=[];errors=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(run,r):r for r in SEEDS}
            while futures:
                for future in list(futures):
                    if future.done():
                        r=futures.pop(future)
                        try:done.append(future.result())
                        except Exception as e:errors.append(str(e))
                progress={str(r):read_json(ROOT/f'rung_{r}/experiment.status.json') if (ROOT/f'rung_{r}/experiment.status.json').exists() else {'state':'queued'} for r in SEEDS}
                publish('running',completed=sorted(done),errors=errors,progress=progress)
                if futures:time.sleep(5)
        if errors:raise RuntimeError('; '.join(errors))
        summaries=[];new_cost=0
        lines=['# Four-rung full-arc comparison','','One swarm and one matched one-task-per-agent private control per rung. '+('Rung1 swarm is reused; other runs are new.' if reuse else 'All runs are new.')+f' Tasks: {args.items_dir}. Model/settings and finite-scalar normalization retained. Correctness never filters trajectories.','','| Rung | Dataset | Swarm R | Private R | Swarm dimension | Private dimension |','|---|---|---:|---:|---:|---:|']
        for r in SEEDS:
            directory=ROOT/f'rung_{r}';seed=SEEDS[r]
            sa=audit(directory/'swarm/cooperative'/f'seed_{seed}')
            pa=audit(directory/'private/fixed'/f'seed_{seed}')
            mapping=read_json(directory/'assignments.json');private=read_json(directory/'private/fixed'/f'seed_{seed}'/'state.json')
            assert all(a['order']==[mapping[i]] and not a['inbox'] for i,a in private['agents'].items())
            assert sa['task_hash']==pa['task_hash']
            new_cost+=pa['pilot_cost_rub']+(sa['pilot_cost_rub'] if r!=1 or not reuse else 0)
            values=read_json(directory/'analysis/summary.json');summaries.append({'rung':r,'seed':seed,'metrics':values,'swarm_action_errors':len(sa['action_errors']),'private_action_errors':len(pa['action_errors'])})
            sw=next(x for x in values if x['width']==2048 and x['condition']=='Swarm v2');pr=next(x for x in values if x['width']==2048 and x['condition'].startswith('Private'))
            lines.append(f"| {r} | {LABELS[r]} | {sw['R']:.4f} | {pr['R']:.4f} | {sw['eff_dim']:.4f} | {pr['eff_dim']:.4f} |")
        lines.extend(['','Full-arc blue curves retain real distribution/solving/commit observations. Orange has solving only, mapped by relative progress: no independent chat endpoints are invented. Paired and active-count figures at400/2048/4096 are saved per rung. One pair per rung; no uncertainty bands or causal claim from cross-rung differences. Lengths, held windows, short windows, final output format and recovery remain interpretation constraints. Inspect all errors/lengths; no desired-effect selection.'])
        (ROOT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write_json(ROOT/'summary.json',summaries)
        subprocess.run([sys.executable,'scripts/build_ladder_overview.py',str(ROOT)],check=True)
        html=None
        if not args.skip_checkpoint_html:
            subprocess.run([sys.executable,'scripts/build_checkpoint_html.py'],check=True);html='artifacts/pipeline-checkpoint/swarm-team-checkpoint.html'
        publish('complete',stage='complete',audited_pairs=4,new_generation_local_ledger_rub=new_cost,html=html)
    except Exception as e:publish('failed',error=str(e));raise

if __name__=='__main__':main()
