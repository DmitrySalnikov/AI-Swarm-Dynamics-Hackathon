"""Original set_4: stateful v2 and matched one-task-per-agent private control."""
import argparse,csv,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from swarm_pipeline.audit import audit
from swarm_pipeline.supervision import read_json,write_json,progress_from_calls
from swarm_pipeline.analyze import load_agents,windows,embed,attach,chat_endpoints,phase_rows,metrics

ROOT=Path('artifacts/pipeline-set4');SEED=73
LABEL='Original set_4'

def analysis():
    out=ROOT/'analysis';out.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    swarm,sm=load_agents(ROOT/'swarm','cooperative',SEED,tok)
    private,pm=load_agents(ROOT/'private','fixed',SEED,tok)
    assert sm['task_hash']==pm['task_hash']
    texts=[];prepared={}
    for width in [400,2048,4096]:
        sw=[windows(a['phases']['solving']['text'],a['phases']['solving']['offsets'],width) for a in swarm]
        pr=[windows(a['text'],a['offsets'],width) for a in private]
        prepared[width]=(sw,pr)
        texts.extend(w['text'] for per in [sw,pr] for ws in per for w in ws)
    texts.extend(w['text'] for a in swarm for ph in ['negotiation','exchange'] for w in chat_endpoints(a,ph))
    # Separate per-experiment writes let ladder analyses run concurrently.
    # Reuse matching shared vectors without mutating a common cache in parallel.
    from swarm_pipeline.analyze import vector_key
    cache=out/'embedding-cache';cache.mkdir(parents=True,exist_ok=True)
    for key in {vector_key(t) for t in texts if t.strip()}:
        source=Path('artifacts/pipeline-analysis/embedding-cache')/f'{key}.npy'
        if source.exists() and not (cache/source.name).exists():shutil.copyfile(source,cache/source.name)
    vectors=embed(texts,out,workers=6,cache_dir=cache)
    summaries=[]
    panels=[('R','Mean vector length'),('pair_cosine','Mean pairwise cosine'),('eff_dim','Effective dimension'),('var_perp_std','Perpendicular spread'),('n_active','Vectors included')]
    for width,(sw,pr) in prepared.items():
        for ws in sw+pr:attach(ws,vectors)
        count=max(len(ws) for ws in sw+pr);series={};natural={}
        for label,per in [('Swarm v2',sw),('Private: same assigned tasks',pr)]:
            snapshots=[]
            for step in range(count):
                chosen=[ws[round(step*(len(ws)-1)/max(count-1,1))] for ws in per]
                snapshots.append({'x':step/max(count-1,1),'phase':'solving','X':np.stack([w['vector'] for w in chosen])})
            series[label]=metrics(snapshots)
            active=[]
            for i in range(max(map(len,per))):
                chosen=[ws[i] for ws in per if i<len(ws)]
                if len(chosen)>=4:active.append({'x':float(np.mean([w['x'] for w in chosen])),'phase':'solving','X':np.stack([w['vector'] for w in chosen])})
            natural[label]=metrics(active)
            summaries.append({'width':width,'condition':label,**{k:float(np.mean([r[k] for r in series[label]])) for k,_ in panels}})
        fig,axs=plt.subplots(5,2,figsize=(13,13))
        for col,(data,clock) in enumerate([(series,'Relative solving progress; windows repeated to retain all agents'),(natural,'Own solving tokens; finished agents removed (N >= 4)')]):
            for ax,(k,title) in zip(axs[:,col],panels):
                for (label,rows),color in zip(data.items(),['#2563eb','#ea580c']):
                    ax.plot([r['x'] if col==0 else r['x']/1000 for r in rows],[r[k] for r in rows],color=color,lw=2,marker='o',ms=3,label=label)
                ax.set_ylabel(title);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            axs[0,col].legend(fontsize=8);axs[-1,col].set_xlabel(clock,fontsize=8);axs[-1,col].set_ylim(0,11)
        fig.suptitle(f'{LABEL}: same tasks and allocation, with / without interaction\n{width}-token windows, stride {width//2}; one group each, no uncertainty bands',fontsize=12)
        fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/f'paired_w{width}.png',dpi=160);fig.savefig(out/f'paired_w{width}.pdf');plt.close(fig)
        rows=[{'condition':label,'clock':clock,**r} for clock,data in [('progress',series),('active_token',natural)] for label,rr in data.items() for r in rr]
        with (out/f'paired_w{width}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        full,spans=phase_rows(swarm,width,vectors)
        full_metrics=metrics(full)
        fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
        solving=[r for r in full_metrics if r['phase']=='solving']
        private_rows=series['Private: same assigned tasks']
        mapped_x=np.interp([r['x'] for r in private_rows],np.linspace(0,1,len(solving)),[r['x'] for r in solving])
        for ax,(key,label) in zip(axs,[('R','Mean vector length'),('pair_cosine','Mean pairwise cosine'),('eff_dim','Effective dimension')]):
            ax.plot([r['x']/1000 for r in full_metrics],[r[key] for r in full_metrics],color='#2563eb',marker='o',ms=3,label='Swarm v2: distribution / solve / commit')
            ax.plot(mapped_x/1000,[r[key] for r in private_rows],color='#ea580c',lw=2,label='Private control: solving only')
            for phase in ['negotiation','exchange']:ax.axvspan(spans[phase][0]/1000,spans[phase][1]/1000,color='#dbeafe',alpha=.3)
            ax.set_ylabel(label);ax.grid(alpha=.2)
        axs[0].legend(fontsize=8);axs[-1].set_xlabel('Mean generated tokens (thousands); private solving mapped by relative progress')
        fig.suptitle(f'{LABEL}: full swarm arc and matched private solving | {width}-token windows\nNo private chat endpoints invented; line segments between phases are not additional observations',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/f'full_arc_w{width}.png',dpi=160);fig.savefig(out/f'full_arc_w{width}.pdf');plt.close(fig)
        with (out/f'swarm_full_arc_w{width}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(full_metrics[0]));w.writeheader();w.writerows(full_metrics)
    write_json(out/'summary.json',summaries)
    lengths={'swarm':[len(a['phases']['solving']['offsets']) for a in swarm],'private':[len(a['offsets']) for a in private]}
    receipts=[read_json(p) for p in (out/'embedding-requests').glob('*.response.json')]
    write_json(out/'manifest.json',{'model':'qwen/qwen3-embedding-4b','dimensions':2560,'task_hash':sm['task_hash'],'seed':SEED,'windows':[400,2048,4096],'solving_tokens':lengths,'replicates':1,'assignment_source':'completed swarm; identical tasks assigned to identical agent IDs in private control','embedding_receipt_usd':sum((r.get('usage')or{}).get('cost',0) for r in receipts),'limitations':['Negotiation history exists only in swarm','No private final common packet or fake endpoint','Relative-progress windows may repeat','Natural-clock dropout changes task composition','One group per condition; not a validated interaction detector','No solution extension by additional independent samples','Correctness never filters texts']})
    lines=[f'# {LABEL} paired pilot','','Same generator/settings, tasks and agent-task mapping. Private control gets only its own task and own history: no distribution chat, peer results, or common final commit. One group per condition; this isolates task allocation more closely but does not match prior context or final output.','','| Window | Condition | R | Pairwise cosine | Effective dimension |','|---|---|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['width']} | {r['condition']} | {r['R']:.4f} | {r['pair_cosine']:.4f} | {r['eff_dim']:.4f} |")
    lines.extend(['','Means use private-solving relative progress, ten retained vectors. See active-token panels for true dropout. No fake private distribution or exchange is plotted. Swarm full-arc CSVs retain real phase observations. Compare actual solution lengths in analysis/manifest.json before interpreting small-window/large-window differences. Short text windows may contain whole solutions. Original legacy set_4 used independently generated same-task extensions; this run does not. No desired-effect selection, confidence intervals or detector accuracy claim.'])
    (ROOT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    global ROOT,SEED,LABEL
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--items',type=Path,default=Path('bench/set_4.jsonl'))
    parser.add_argument('--label',default=LABEL)
    parser.add_argument('--swarm-budget',type=float,default=180)
    parser.add_argument('--private-budget',type=float,default=120)
    args=parser.parse_args();ROOT=args.root;SEED=args.seed;LABEL=args.label
    ROOT.mkdir(parents=True,exist_ok=True);status=ROOT/'experiment.status.json'
    def publish(state,**details):write_json(status,{'state':state,'event_key':'set4-paired:'+state,'details':{'pid':os.getpid(),'seed':SEED,**details},'updated_at':time.time()})
    def run(cmd,name,directory):
        publish('running',stage=name)
        with (ROOT/f'{name}.stdout.log').open('a') as stdout,(ROOT/f'{name}.stderr.log').open('a') as stderr:
            child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            while child.poll() is None:
                publish('running',stage=name,child_pid=child.pid,progress=progress_from_calls(directory));time.sleep(4)
        if child.returncode:raise RuntimeError(f'{name} failed; inspect logs and preserve journals')
    try:
        sw=ROOT/'swarm'/'cooperative'/f'seed_{SEED}'
        if not (sw/'result.json').exists() or read_json(sw/'state.json')['status']!='completed':
            run([sys.executable,'-X','utf8','-m','swarm_pipeline.pilot','--items',str(args.items),'--out',str(ROOT/'swarm'),'--condition','cooperative','--protocol','v2','--seed',str(SEED),'--workers','10','--budget-rub',str(args.swarm_budget),'--partial-continuation'],'swarm',sw)
        sa=audit(sw);state=read_json(sw/'state.json')
        assignments={i:a['order'][0] for i,a in state['agents'].items()}
        write_json(ROOT/'assignments.json',assignments)
        private=ROOT/'private'/'fixed'/f'seed_{SEED}'
        if not (private/'result.json').exists() or read_json(private/'state.json')['status']!='completed':
            run([sys.executable,'-X','utf8','-m','swarm_pipeline.private_control','--items',str(args.items),'--out',str(ROOT/'private'),'--condition','fixed','--assignments',str(ROOT/'assignments.json'),'--seed',str(SEED),'--workers','10','--budget-rub',str(args.private_budget)],'private',private)
        pa=audit(private);ps=read_json(private/'state.json')
        assert set(pa['phases'])=={'solving'} and pa['phases']['solving']['delivered_messages']==0
        assert all(a['order']==[assignments[i]] and not a['inbox'] and len(a['answers'])==1 for i,a in ps['agents'].items())
        assert read_json(private/'manifest.json')['task_hash']==read_json(sw/'manifest.json')['task_hash']
        publish('running',stage='analysis');analysis()
        publish('complete',stage='complete',audit='passed',generation_local_ledger_rub=sa['pilot_cost_rub']+pa['pilot_cost_rub'],findings=str((ROOT/'FINDINGS.md').resolve()))
    except Exception as e:publish('failed',error=str(e));raise

if __name__=='__main__':main()
