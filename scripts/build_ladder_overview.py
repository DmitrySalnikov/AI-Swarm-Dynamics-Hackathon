"""Show all ladder arcs on a common scale without new model requests."""
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]/'artifacts/pipeline-ladder'
LABELS=['Unrelated domains','Physics subfields','Linear algebra','Same determinant template']

def main():
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for rung,(ax,label) in enumerate(zip(axes.flat,LABELS),1):
        root=ROOT/f'rung_{rung}/analysis'
        full=list(csv.DictReader((root/'swarm_full_arc_w2048.csv').open()))
        xx=[];yy=[]
        for phase,start in [('negotiation',0),('solving',1),('exchange',2)]:
            rr=[r for r in full if r['phase']==phase]
            xx.extend([start+.9] if len(rr)==1 else np.linspace(start+.1,start+.9,len(rr)))
            yy.extend(float(r['R']) for r in rr)
        private=[r for r in csv.DictReader((root/'paired_w2048.csv').open()) if r['clock']=='progress' and r['condition'].startswith('Private')]
        ax.plot(xx,yy,color='#2563eb',lw=2.2,marker='o',ms=3,label='Swarm v2')
        ax.plot([1.1+.8*float(r['x']) for r in private],[float(r['R']) for r in private],color='#ea580c',lw=2,label='Matched private solving')
        ax.axvspan(0,1,color='#eff5ff');ax.axvspan(2,3,color='#eff5ff')
        ax.set_ylim(0,1);ax.set_xlim(0,3);ax.set_title(f'Rung {rung}: {label}',loc='left',fontsize=11)
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.set_xticks([.5,1.5,2.5],['Distribution','Solving','Commit'])
    axes[0,0].legend(loc='lower left',fontsize=8,frameon=False)
    axes[0,0].set_ylabel('Mean vector length R');axes[1,0].set_ylabel('Mean vector length R')
    fig.suptitle('All four ladder rungs · same vertical scale\nWithin-phase progress, 2048-token windows; one pair per rung, no uncertainty bands',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(ROOT/'full_arcs_overview.png',dpi=180);fig.savefig(ROOT/'full_arcs_overview.pdf');plt.close(fig)

if __name__=='__main__':main()
