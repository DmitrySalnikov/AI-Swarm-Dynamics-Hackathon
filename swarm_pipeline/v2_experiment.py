"""Finish a v2 pilot, audit it and build cached baseline and v1 comparisons."""
import argparse,csv,json,os,subprocess,sys,time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from swarm_pipeline.supervision import read_json,write_json,progress_from_calls
from swarm_pipeline.audit import audit


def compare(out,baseline):
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    metrics=[('R','Mean vector length'),('var_perp_std','Perpendicular spread'),('eff_dim','Effective dimension')]
    for name,path,color in [('v1',baseline/'rung1_w2048_raw.csv','#7a94b8'),('v2',out/'rung1_v2_w2048_raw.csv','#2268d0')]:
        rows=[r for r in csv.DictReader(path.open(encoding='utf-8')) if r['condition']=='cooperative']
        all_x,all_rows=[],[]
        for phase,start in [('negotiation',0),('solving',1),('exchange',2)]:
            selected=[r for r in rows if r['phase']==phase]
            xs=[start+.9] if len(selected)==1 else [start+.1+.8*i/(len(selected)-1) for i in range(len(selected))]
            all_x.extend(xs);all_rows.extend(selected)
        for ax,(key,title) in zip(axes,metrics):
            ax.plot(all_x,[float(r[key]) for r in all_rows],color=color,marker='o',ms=4,label=name)
            ax.set_title(title,loc='left');ax.grid(alpha=.2)
    for ax in axes:
        ax.axvspan(0,1,color='#eeede9');ax.axvspan(2,3,color='#eeede9');ax.set_ylim(bottom=0)
    axes[0].set_ylim(0,1);axes[0].legend()
    axes[-1].set_xticks([.5,1.5,2.5],['Distribution','Parallel solving','Exchange / commit'])
    axes[-1].set_xlim(0,3)
    fig.suptitle('Protocol v1 vs v2 | Qwen3-Embedding-4B, 2048-token solve windows\nEach phase uses relative progress; one pilot each, no uncertainty bands',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'protocol_comparison.png',dpi=160);fig.savefig(out/'protocol_comparison.pdf');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--items',type=Path,default=Path('bench/ladder/rung_1.jsonl'))
    p.add_argument('--root',type=Path,default=Path('artifacts/pipeline-v2'))
    p.add_argument('--baseline',type=Path,default=Path('artifacts/pipeline-live'))
    p.add_argument('--analysis',type=Path,default=Path('artifacts/pipeline-analysis-v2'))
    p.add_argument('--seed',type=int,default=3)
    p.add_argument('--workers',type=int,default=10)
    p.add_argument('--budget-rub',type=float,default=500)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True);args.analysis.mkdir(parents=True,exist_ok=True)
    status=args.root/'experiment.status.json'
    details={'supervisor_pid':os.getpid(),'protocol':'v2','workers':args.workers,'seed':args.seed,'budget_rub':args.budget_rub,
             'root':str(args.root.resolve()),'analysis':str(args.analysis.resolve())}
    def publish(state,**extra):write_json(status,{'state':state,'event_key':'protocol-v2:'+state,'details':{**details,**extra},'updated_at':time.time()})
    def run(command,stage):
        publish('running',stage=stage)
        with (args.root/f'{stage}.stdout.log').open('a',encoding='utf-8') as stdout,(args.root/f'{stage}.stderr.log').open('a',encoding='utf-8') as stderr:
            child=subprocess.Popen(command,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            while child.poll() is None:
                progress=progress_from_calls(args.root/'cooperative'/f'seed_{args.seed}')
                publish('running',stage=stage,child_pid=child.pid,progress=progress)
                time.sleep(3)
        if child.returncode:raise RuntimeError(f'{stage} exited with code {child.returncode}; see saved stderr')
    try:
        run([sys.executable,'-X','utf8','-m','swarm_pipeline.pilot','--items',str(args.items),'--out',str(args.root),
             '--condition','cooperative','--protocol','v2','--seed',str(args.seed),'--workers',str(args.workers),
             '--budget-rub',str(args.budget_rub),'--partial-continuation'],'generation')
        report=audit(args.root/'cooperative'/f'seed_{args.seed}')
        assert report['single_successful_submit_and_commit_per_agent']
        publish('running',stage='analysis',audit='passed')
        os.environ['OPENBLAS_NUM_THREADS']='1'
        run([sys.executable,'-X','utf8','-m','swarm_pipeline.analyze','--root',str(args.baseline),
             '--swarm-root',str(args.root),'--swarm-seed',str(args.seed),'--out',str(args.analysis),
             '--cache-dir','artifacts/pipeline-analysis/embedding-cache','--prefix','rung1_v2'],'analysis')
        compare(args.analysis,Path('artifacts/pipeline-analysis'))
        publish('complete',stage='complete',audit='passed',calls=report['calls'],
                generation_ledger_rub=report['pilot_cost_rub'],figure=str((args.analysis/'protocol_comparison.png').resolve()))
    except Exception as e:
        publish('failed',error_type=type(e).__name__,error=str(e));raise


if __name__=='__main__':main()
