"""Complete three groups per condition with bounded parallel jobs and analysis."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from swarm_pipeline.audit import audit
from swarm_pipeline.supervision import read_json, write_json, progress_from_calls

ROOT=Path('artifacts/pipeline-replicates')
CONDITIONS=('cooperative','fixed','shuffled','disjoint')
SEEDS=(3,41,42)
COLORS=dict(zip(CONDITIONS,['#2563eb','#ea580c','#059669','#9333ea']))


def sources(seed):
    if seed==3:
        return {c:str(Path('artifacts/pipeline-v2' if c=='cooperative' else 'artifacts/pipeline-live').resolve()) for c in CONDITIONS}
    return {c:str((ROOT/'runs'/f'{seed}-{c}').resolve()) for c in CONDITIONS}


def aggregate():
    summaries=[]
    metrics=[('R','Mean vector length'),('pair_cosine','Mean pairwise cosine'),('var_perp_std','Perpendicular spread'),('eff_dim','Effective dimension'),('n_active','Vectors retained in progress alignment')]
    grid=np.arange(33)/32
    series={c:[] for c in CONDITIONS}
    for seed in SEEDS:
        directory=ROOT/'analysis'/f'seed_{seed}'
        manifest=read_json(directory/'analysis-manifest.json')
        rows=list(csv.DictReader((directory/'replicate_w2048_raw.csv').open()))
        swarm=[r for r in rows if r['condition']=='cooperative' and r['phase']=='solving']
        xs=np.array([float(r['x']) for r in swarm])
        target=xs[0]+grid*(xs[-1]-xs[0])
        for c in CONDITIONS:
            selected=[r for r in rows if r['condition']==c and (c!='cooperative' or r['phase']=='solving')]
            sx=[float(r['x']) for r in selected]
            interpolated={k:np.interp(target,sx,[float(r[k]) for r in selected],left=np.nan,right=np.nan) for k,_ in metrics}
            series[c].append(interpolated)
            summaries.append({'seed':seed,'condition':c,**{k:float(np.nanmean(v)) for k,v in interpolated.items()}})
    common=np.logical_and.reduce([np.isfinite(r['pair_cosine']) for c in CONDITIONS for r in series[c]])
    if common.sum()<2:
        raise ValueError('Insufficient shared token coverage across all groups')
    # No extrapolation, and no changing from three groups to two in the band.
    # Summaries must use the same progress points for every condition/group.
    for r in summaries:
        values=series[r['condition']][SEEDS.index(r['seed'])]
        for k,_ in metrics:r[k]=float(values[k][common].mean())
    write_json(ROOT/'comparison_coverage.json',{'grid_points':len(grid),'shared_points':int(common.sum()),'progress_min':float(grid[common].min()),'progress_max':float(grid[common].max()),'reason':'Controls use 32000-token prefixes; no extrapolation beyond available observations'})
    fig,axs=plt.subplots(5,1,figsize=(12,14),sharex=True)
    for ax,(k,label) in zip(axs,metrics):
        for c in CONDITIONS:
            vals=np.stack([r[k][common] for r in series[c]])
            mean=vals.mean(0);sd=vals.std(0,ddof=1)
            for row in vals:ax.plot(grid[common],row,color=COLORS[c],alpha=.23,lw=1)
            ax.plot(grid[common],mean,color=COLORS[c],lw=2,label=c)
            ax.fill_between(grid[common],mean-sd,mean+sd,color=COLORS[c],alpha=.12)
        ax.set_ylabel(label);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axs[-1].set_xlabel('Relative swarm solving progress; controls sampled at corresponding token centres')
    axs[0].legend(ncol=4,frameon=False)
    axs[-1].set_ylim(0,11)
    fig.suptitle('Three groups per condition: shared-coverage solving geometry\nThin lines = groups; thick = mean; bands = sample SD across three groups, not confidence intervals',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(ROOT/'solving_three_groups.png',dpi=180);fig.savefig(ROOT/'solving_three_groups.pdf');plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for seed,color in zip(SEEDS,['#2563eb','#059669','#9333ea']):
        rows=list(csv.DictReader((ROOT/'analysis'/f'seed_{seed}'/'replicate_w2048_active_only.csv').open()))
        for ax,k in zip(axs,['n_active','pair_cosine','eff_dim']):
            ax.plot([float(r['x'])/1000 for r in rows],[float(r[k]) for r in rows],label=f'Group seed {seed}',color=color,lw=2)
    for ax,label in zip(axs,['Active vectors (N >= 4)','Mean pairwise cosine','Effective dimension']):ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[0].legend();axs[-1].set_xlabel('Own solving tokens (thousands); natural token clock, no holding')
    fig.suptitle('Actual vector dropout in three swarm groups');fig.tight_layout();fig.savefig(ROOT/'active_vectors_three_groups.png',dpi=180);fig.savefig(ROOT/'active_vectors_three_groups.pdf');plt.close(fig)
    write_json(ROOT/'group_summaries.json',summaries)
    lines=['# Three-group experiment findings','', 'Three groups per condition, seeds 3, 41 and 42. Seed 3 reuses completed pilots; its controls have historical recovery policies. Seeds 41 and 42 use ten workers each, identical scientific settings and the current recovery policy. This is not a fully uniform execution-policy replication.', '', '## Solving-phase descriptive summaries', '', '| Condition | R mean / SD | Pairwise cosine mean / SD | Effective dimension mean / SD |','|---|---:|---:|---:|']
    for c in CONDITIONS:
        rr=[r for r in summaries if r['condition']==c]
        values=[]
        for k in ['R','pair_cosine','eff_dim']:
            v=np.array([r[k] for r in rr]);values.append(f'{v.mean():.4f} / {v.std(ddof=1):.4f}')
        lines.append('| '+c+' | '+' | '.join(values)+' |')
    for c in CONDITIONS[1:]:
        differences=np.array([series['cooperative'][i]['pair_cosine'][common].mean()-series[c][i]['pair_cosine'][common].mean() for i in range(3)])
        lines.extend(['',f'Swarm minus {c}, solving mean pairwise-cosine differences by group: '+', '.join(f'{v:+.4f}' for v in differences)+'.'])
    lines.extend(['',f'All summaries and bands use the same {int(common.sum())}/{len(grid)} progress points, from {grid[common].min():.3f} to {grid[common].max():.3f}. The 32000-token control prefix does not cover the final part of the longest swarm trajectory. Missing observations are not extrapolated or averaged as if three groups remained.','','## Interpretation limits','','Three groups permit a first descriptive variability estimate, not validated detection accuracy. Overlapping windows are dependent. Seed pairing is not deterministic provider pairing. Vector dropout changes both N and task composition. Progress alignment retains ten vectors and repeats windows; natural token-clock active-only plots drop finished agents. The two clocks answer different questions. Common final packet and JSON format can create convergence. Existing controls solve ten tasks per agent, while swarm solves one. Answer correctness never filters observations. No t-test or classifier accuracy is reported.','', 'Inspect solving_three_groups.png and active_vectors_three_groups.png along with each group full-arc 400/2048/4096-token figures. Completion includes audits and figures; the owning chat must verify figure layout and scientific conclusions.'])
    (ROOT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    status=ROOT/'experiment.status.json'
    jobs=[(s,c) for s in (41,42) for c in CONDITIONS]
    def publish(state,**details):
        write_json(status,{'state':state,'event_key':'three-groups:'+state,'details':{'pid':os.getpid(),'seeds':list(SEEDS),'parallel_jobs':4,'workers_per_job':10,**details},'updated_at':time.time()})
    publish('running',stage='generation')
    try:
        for c,root in sources(3).items():audit(Path(root)/c/'seed_3')
        def generate(job):
            seed,c=job;root=Path(sources(seed)[c]);root.mkdir(parents=True,exist_ok=True)
            directory=root/c/f'seed_{seed}'
            if (directory/'result.json').exists() and read_json(directory/'state.json')['status']=='completed':return audit(directory)
            command=[sys.executable,'-X','utf8','-m','swarm_pipeline.pilot','--items','bench/ladder/rung_1.jsonl','--out',str(root),'--condition',c,'--seed',str(seed),'--workers','10','--budget-rub','240','--partial-continuation']
            if c=='cooperative':command.extend(['--protocol','v2'])
            with (root/'supervisor.stdout.log').open('a') as stdout,(root/'supervisor.stderr.log').open('a') as stderr:
                child=subprocess.Popen(command,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                if child.wait():raise RuntimeError(f'Generation failed: seed {seed}, {c}; see {root}')
            return audit(directory)
        completed=[];failures=[]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(generate,job):job for job in jobs}
            while futures:
                for future in list(futures):
                    if future.done():
                        job=futures.pop(future)
                        try:report=future.result();completed.append({'seed':job[0],'condition':job[1],'cost_rub':report['pilot_cost_rub']})
                        except Exception as e:failures.append(str(e))
                progress={f'{s}-{c}':progress_from_calls(Path(sources(s)[c])/c/f'seed_{s}') for s,c in jobs}
                publish('running',stage='generation',completed=completed,failures=failures,progress=progress)
                if futures:time.sleep(5)
        if failures:raise RuntimeError('; '.join(failures))
        for seed in SEEDS:
            out=ROOT/'analysis'/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
            source_file=out/'sources.json';write_json(source_file,sources(seed))
            shutil.copyfile('artifacts/pipeline-analysis/deepseek-tokenizer.json',out/'deepseek-tokenizer.json')
            publish('running',stage='analysis',seed=seed,completed=completed)
            cmd=[sys.executable,'-X','utf8','-m','swarm_pipeline.analyze','--sources',str(source_file),'--swarm-seed',str(seed),'--control-seed',str(seed),'--out',str(out),'--prefix','replicate','--cache-dir','artifacts/pipeline-analysis/embedding-cache']
            env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1'
            with (out/'stdout.log').open('a') as stdout,(out/'stderr.log').open('a') as stderr:
                result=subprocess.run(cmd,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            if result.returncode:raise RuntimeError(f'Analysis failed for seed {seed}')
        aggregate()
        publish('complete',stage='complete',new_runs=completed,new_generation_cost_rub=sum(r['cost_rub'] for r in completed),findings=str((ROOT/'FINDINGS.md').resolve()))
    except Exception as e:
        publish('failed',error=str(e));raise


if __name__=='__main__':main()
