"""Cache-only scientific diagnostics for the completed v2 pilot."""
from pathlib import Path
import csv
import json
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from swarm_pipeline.analyze import load_agents, windows, vector_key
from embedding_eval.dynamics import metrics_at

OUT = Path('artifacts/pipeline-diagnostics')
COLORS = {'cooperative':'#2563eb','fixed':'#ea580c','shuffled':'#059669','disjoint':'#9333ea'}
LABELS = {'cooperative':'Swarm v2','fixed':'Independent: same order','shuffled':'Independent: shuffled','disjoint':'Independent: cyclic offsets'}


def save(fig, name):
    fig.savefig(OUT/f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def tidy(axes):
    for ax in np.asarray(axes).flat:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.18)


def read(width):
    with Path(f'artifacts/pipeline-analysis-v2/rung1_v2_w{width}_raw.csv').open() as f:
        return [{k:(v if k in ('phase','condition') else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]


def extended(X):
    m=metrics_at(X)
    n=len(X)
    m['pair_cosine']=(n*m['R']**2-1)/(n-1) if n>1 else float('nan')
    m['dimension_fraction']=m['eff_dim']/(n-1) if n>1 else float('nan')
    return m


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','figure.facecolor':'#fafafa','axes.facecolor':'#fafafa'})
    rows=read(2048)
    coop=[r for r in rows if r['condition']=='cooperative']
    solve=[r for r in coop if r['phase']=='solving']
    lo,hi=min(r['x'] for r in solve),max(r['x'] for r in solve)
    panels=[('R','Alignment R (higher = more similar)'),('var_perp_std','Perpendicular spread'),('eff_dim','Effective dimension')]
    fig,axs=plt.subplots(3,3,figsize=(15,9),sharex=True,sharey='row')
    for col,control in enumerate(['fixed','shuffled','disjoint']):
        for row,(metric,label) in enumerate(panels):
            ax=axs[row,col]
            for c in ['cooperative',control]:
                rr=[r for r in rows if r['condition']==c and r['x']<=coop[-1]['x']]
                ax.plot([r['x']/1000 for r in rr],[r[metric] for r in rr],color=COLORS[c],lw=2.2,label=LABELS[c])
            ax.axvspan(0,lo/1000,color='#dbeafe',alpha=.35)
            ax.axvspan(hi/1000,coop[-1]['x']/1000,color='#dbeafe',alpha=.35)
            if row==0: ax.set_title(LABELS[control]);ax.legend(fontsize=8,loc='lower left')
            if col==0: ax.set_ylabel(label)
            if row==2: ax.set_xlabel('Generated tokens (thousands)')
    tidy(axs)
    fig.suptitle('Interaction versus independence: compare blue with each control\nOne pilot per condition; shaded ends = distribution / final commit; approximate token alignment',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93));save(fig,'controls_side_by_side')

    fig,axs=plt.subplots(1,3,figsize=(15,4.5))
    for ax,(metric,label) in zip(axs,panels):
        for c in COLORS:
            rr=[r for r in rows if r['condition']==c and lo<=r['x']<=hi and (c!='cooperative' or r['phase']=='solving')]
            ax.plot([r['x']/1000 for r in rr],[r[metric] for r in rr],lw=2,color=COLORS[c],label=LABELS[c])
        ax.set_title(label);ax.set_xlabel('Generated tokens (thousands)')
    tidy(axs);fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.suptitle('Solving only: overlap is visible without the high-alignment endpoints',fontsize=14)
    fig.tight_layout(rect=(0,.14,1,.91));save(fig,'solving_zoom')

    tokenizer=Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    agents,_=load_agents(Path('artifacts/pipeline-v2'),'cooperative',3,tokenizer)
    cache=Path('artifacts/pipeline-analysis/embedding-cache')
    lengths=[len(a['phases']['solving']['offsets']) for a in agents]
    wins=[windows(a['phases']['solving']['text'],a['phases']['solving']['offsets'],2048) for a in agents]
    vecs=[np.stack([np.load(cache/f'{vector_key(w["text"])}.npy') for w in ws]) for ws in wins]
    count=max(map(len,vecs))
    diagnostics=[]
    for i in range(count):
        active=[v[i] for v in vecs if i<len(v)]
        held=np.stack([v[min(i,len(v)-1)] for v in vecs])
        progress=np.stack([v[round(i*(len(v)-1)/max(count-1,1))] for v in vecs])
        for name,X in [('Active only',np.stack(active)),('Hold final window',held),('Relative progress (existing)',progress)]:
            diagnostics.append({'step':i,'token_center':1024*(i+1),'method':name,**extended(X)})
    fig,axs=plt.subplots(2,2,figsize=(13,9))
    for ax,(metric,label) in zip(axs.flat,[('n_active','Number of vectors'),('R','Alignment R'),('pair_cosine','Mean pairwise cosine (removes R count bias)'),('eff_dim','Effective dimension (bounded by N - 1)')]):
        for name,color in [('Active only','#dc2626'),('Hold final window','#d97706'),('Relative progress (existing)','#2563eb')]:
            rr=[r for r in diagnostics if r['method']==name and (metric=='n_active' or r['n_active']>=4)]
            ax.plot([r['token_center']/1000 for r in rr],[r[metric] for r in rr],color=color,lw=2,label=name)
        ax.set_title(label);ax.set_xlabel('Nominal 2048-token window centre (thousands)')
    tidy(axs);fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.suptitle('Alignment sensitivity: dropping agents, holding windows, or stretching progress\nRelative progress uses a different clock; active-only metrics omitted below four vectors',fontsize=14)
    fig.tight_layout(rect=(0,.06,1,.92));save(fig,'alignment_sensitivity')

    # Exhaustive subsets of ONE fixed snapshot isolate vector-count effects.
    mid=count//2
    X=np.stack([v[round(mid*(len(v)-1)/max(count-1,1))] for v in vecs])
    subset=[]
    for n in range(2,11):
        mm=[extended(X[list(ids)]) for ids in itertools.combinations(range(10),n)]
        subset.append({'n':n,'subsets':len(mm),**{k:{'mean':float(np.mean([m[k] for m in mm])),'q10':float(np.quantile([m[k] for m in mm],.1)),'q90':float(np.quantile([m[k] for m in mm],.9))} for k in ['R','pair_cosine','eff_dim','dimension_fraction']}})
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    for ax,(metric,label) in zip(axs.flat,[('R','R rises when fewer vectors remain'),('pair_cosine','Pairwise cosine: subset mean is count-invariant'),('eff_dim','Raw effective dimension depends on vector count'),('dimension_fraction','Effective dimension / (N - 1): diagnostic only')]):
        ns=[s['n'] for s in subset]
        ax.fill_between(ns,[s[metric]['q10'] for s in subset],[s[metric]['q90'] for s in subset],color='#93c5fd',alpha=.6)
        ax.plot(ns,[s[metric]['mean'] for s in subset],color='#2563eb',marker='o',lw=2)
        ax.set_title(label);ax.set_xlabel('Number of vectors in subset');ax.set_xticks(ns)
    tidy(axs);fig.suptitle('Same texts, fewer vectors: exhaustive subsets of one solving snapshot\nBands = 10th–90th subset percentiles, NOT uncertainty across independent runs',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.91));save(fig,'vector_count_effect')

    fig,ax=plt.subplots(figsize=(10,5))
    ax.barh(range(1,11),np.array(lengths)/1000,color='#2563eb')
    ax.set_yticks(range(1,11));ax.set_ylabel('Agent');ax.set_xlabel('Own solving tokens (thousands)')
    ax.set_title('Different finish lengths change which vectors remain')
    tidy([ax]);fig.tight_layout();save(fig,'solution_lengths')
    with (OUT/'alignment_sensitivity.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(diagnostics[0]));w.writeheader();w.writerows(diagnostics)
    report={'source':'completed v2 seed 3; cached Qwen3-Embedding-4B only','new_paid_requests':0,'solving_tokens':lengths,'length_ratio_max_min':max(lengths)/min(lengths),'subsets':subset,'limitations':['One pilot per condition','Subset bands are not replicate confidence intervals','Non-random agent dropout also changes task composition','Pairwise cosine does not remove informative missingness','Relative progress and token-index clocks answer different questions']}
    (OUT/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'solving_tokens':lengths,'length_ratio_max_min':report['length_ratio_max_min'],'new_paid_requests':0}))


if __name__=='__main__':
    main()
