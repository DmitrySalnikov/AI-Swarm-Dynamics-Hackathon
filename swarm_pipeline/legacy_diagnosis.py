"""Audit legacy construction and isolate task-allocation geometry offline."""
from pathlib import Path
import csv,json,random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from embedding_eval.dynamics import metrics_at,unit

OUT=Path('artifacts/pipeline-legacy-diagnosis')
BASE=Path('embedding_eval/concat/set_4')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pool={};extensions=[]
    for run in range(1,11):
        for agent in range(1,11):
            stem=BASE/'phases'/f'run_{run}'/f'agent_{agent}_ext'
            meta=json.loads(stem.with_suffix('.json').read_text())
            pool[(run,meta['task'])]=unit(np.load(stem.with_suffix('.npz'))['solve'])
            extensions.append({'run':run,'agent':agent,'task':meta['task'],'sources':len(meta['sources']),'tokens':meta['n_tokens']})
    tasks=sorted({t for _,t in pool})
    assert len(tasks)==10 and len(pool)==100
    rng=np.random.default_rng(731)
    scenarios={'one_each':[],'random_tasks':[],'same_task':[]}
    # Independent texts in the same fixed pool. Reassembly, not live communication.
    for repeat in range(200):
        runs=rng.permutation(np.arange(1,11))
        assignments={'one_each':rng.permutation(10),'random_tasks':rng.integers(0,10,10),'same_task':np.repeat(repeat%10,10)}
        for name,ids in assignments.items():
            chosen=[pool[(int(run),tasks[int(i)])] for run,i in zip(runs,ids)]
            scenarios[name].append([metrics_at(np.stack([a[s] for a in chosen])) for s in range(7)])
    labels={'one_each':'One distinct task per agent','random_tasks':'Random tasks, collisions allowed','same_task':'Same task for every agent'}
    colors={'one_each':'#2563eb','random_tasks':'#059669','same_task':'#ea580c'}
    metrics=[('R','Mean vector length'),('var_perp_std','Perpendicular spread'),('eff_dim','Effective dimension')]
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    summary={}
    for name,data in scenarios.items():
        summary[name]={}
        for ax,(key,label) in zip(axs,metrics):
            arr=np.array([[m[key] for m in repeat] for repeat in data]);mean=arr.mean(0);sd=arr.std(0)
            ax.plot(np.arange(1,8),mean,color=colors[name],lw=2.4,label=labels[name])
            ax.fill_between(np.arange(1,8),mean-sd,mean+sd,color=colors[name],alpha=.13)
            ax.set_ylabel(label);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            summary[name][key]=float(mean.mean())
    axs[0].legend(frameon=False,ncol=1);axs[-1].set_xlabel('2048-token window index within extended same-task solution')
    fig.suptitle('Legacy pool: task allocation alone changes geometry\nNo peer information supplied to these solves; bands = reassembly SD, not new experimental repeats',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'allocation_without_communication.png',dpi=170);fig.savefig(OUT/'allocation_without_communication.pdf');plt.close(fig)
    # Reproduce the original control grouping and inspect task labels per window.
    files=sorted((BASE/'shuffled').glob('run_*.vectors.npy'))
    subsets=random.Random('set_4-mixed')
    groups=[subsets.sample(range(len(files)),10) for _ in range(10)]
    labels_by_file=[]
    for p in files:
        stem=p.with_name(p.name.replace('.vectors.npy',''))
        pieces=json.loads(stem.with_suffix('.index.json').read_text())
        windows=json.loads(stem.with_suffix('.windows.json').read_text())['windows']
        dominant=[]
        for w in windows[:9]:
            overlap={}
            for i in w['pieces']:
                piece=pieces[i];n=max(0,min(w['char_end'],piece['end'])-max(w['char_start'],piece['start']))
                overlap[piece['task']]=overlap.get(piece['task'],0)+n
            dominant.append(max(overlap,key=overlap.get))
        labels_by_file.append(dominant)
    coverage=[{'token_center':1024*(i+1),'mean_distinct_dominant_tasks':float(np.mean([len({labels_by_file[j][i] for j in group}) for group in groups]))} for i in range(9)]
    legacy=list(csv.DictReader(Path('embedding_eval/dynamics/set_4_mixed.csv').open()))
    old={}
    for name in ['chat','shuffled','sequential']:
        rows=[r for r in legacy if r['line']==name and r['variant']=='raw' and 1500<=float(r['x'])<=8000]
        if name=='chat':rows=[r for r in rows if r.get('phase')=='solve']
        old[name]={k:float(np.mean([float(r[k]) for r in rows])) for k,_ in metrics}
    result={'paid_requests':0,'pool':'legacy set_4 extended solve vectors','synthetic_reassemblies':200,'allocation_means':summary,'original_plot_solving_interval_means':old,'legacy_shuffled_task_coverage':coverage,'extension_sources_min':min(r['sources'] for r in extensions),'extension_sources_max':max(r['sources'] for r in extensions),'agents_with_extra_solutions':sum(r['sources']>1 for r in extensions),'limitations':['Allocation ablation uses extended-solve pool, not the original shuffled streams','Dominant task uses character overlap, not token weights','Original control groups overlap: ten subsets of twenty streams','Different dataset/model/provider/history/length changed together between legacy and current runs']}
    (OUT/'diagnosis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
