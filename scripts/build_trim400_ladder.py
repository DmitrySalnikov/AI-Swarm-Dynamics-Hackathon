"""Compare every saved ladder pair with full 400-token, ten-author windows."""
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from swarm_pipeline.uniform_windows import ROOT, load_streams, layouts, segment, snapshot_rows, compute, draw
from swarm_pipeline.analyze import embed
from swarm_pipeline.supervision import read_json, write_json


def main():
    tokenizer = Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    plans, texts = [], []
    for rung, seed in ((1,3),(2,82),(3,83),(4,84)):
        source = Path(f'artifacts/pipeline-ladder/rung_{rung}')
        swarm = load_streams(source/'swarm/cooperative'/f'seed_{seed}',tokenizer)
        private = load_streams(source/'private/fixed'/f'seed_{seed}',tokenizer)
        layout, spans, _ = layouts(swarm,tokenizer,'trim')
        control = {a['agent']:[segment(a['text'],0,tokenizer)] for a in private}
        private_min = min(s['end'] for parts in control.values() for s in parts)
        for parts in control.values():
            parts[0]['end'] = private_min
            parts[0]['offsets'] = parts[0]['offsets'][:private_min]
        end = max([s['end'] for parts in layout.values() for s in parts]+[private_min])
        starts = list(range(0,max(0,end-400+1),200))
        rows = snapshot_rows(layout,starts,400,'swarm')+snapshot_rows(control,starts,400,'private')
        assert all(len(r['windows']) in (0,10) for r in rows)
        texts.extend(t for row in rows for t in row['windows'].values())
        plans.append((rung,seed,spans,layout,private_min,rows))
    vectors = embed(texts,ROOT/'trim400-analysis',workers=12,cache_dir=Path('artifacts/pipeline-analysis/embedding-cache'))
    summary, all_records = [], []
    fig, axes = plt.subplots(2,2,figsize=(14,8))
    for (rung,seed,spans,layout,private_min,rows),ax in zip(plans,axes.flat):
        half = len(rows)//2
        records = compute(rows[:half],vectors)+compute(rows[half:],vectors)
        assert all(r['n_active'] in (0,10) for r in records)
        stem = ROOT/'trim400-analysis'/f'rung_{rung}_trim400_n10'
        with stem.with_suffix('.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        draw(records,spans,400,f'Ladder {rung}: trimmed phases; every populated point N=10',stem.with_suffix('.png'),min_authors=10)
        phases = {}
        for phase in ('negotiation','solving','exchange'):
            start = spans[phase][0]
            end = min(s['end'] for parts in layout.values() for s in parts if s['phase']==phase)
            selected = [r for r in records[:half] if r['n_active']==10 and r['start']>=start and r['start']+400<=end]
            phases[phase] = {'retained_tokens_per_author':end-start,'points':len(selected),
                'R_mean':float(np.mean([r['R'] for r in selected])) if selected else None,
                'R_min':min((r['R'] for r in selected),default=None),
                'R_max':max((r['R'] for r in selected),default=None),
                'eff_dim_mean':float(np.mean([r['eff_dim'] for r in selected])) if selected else None}
        means = [phases[p]['R_mean'] for p in ('negotiation','solving','exchange')]
        full = all(m is not None for m in means)
        summary.append({'rung':rung,'seed':seed,'width':400,'stride':200,'every_populated_point_n10':True,
                        'private_retained_tokens':private_min,'phases':phases,'all_three_phases_observed':full,
                        'descriptive_high_low_high_R':bool(means[0]>means[1] and means[2]>means[1]) if full else None})
        for condition,color in (('swarm','#2476d2'),('private','#e57532')):
            series=[r for r in records if r['condition']==condition]
            ax.plot([r['x'] for r in series],[r['R'] for r in series],'.-',ms=3,color=color,label=condition)
        for phase in ('negotiation','exchange'):
            ax.axvspan(*spans[phase],color='#689bce',alpha=.1)
        ax.set_ylim(0,1);ax.set_title(f'Ladder {rung}'+(' — exchange shorter than W' if not full else ''))
        ax.set_xlabel('Conditional token position');ax.set_ylabel('R (ten authors)');ax.grid(alpha=.2)
        all_records.extend(records)
    axes[0,0].legend()
    fig.suptitle('Uniform W=400, stride=200; shortest-phase trimming; N=10 at every populated point\nGaps stay blank. Private trimmed to its shortest author; no private chat phases.',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(ROOT/'trim400-analysis/ladder_trim400_overview.png',dpi=160);plt.close(fig)
    write_json(ROOT/'trim400-summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    # Rebuild only the HTML from persisted audited outputs; no live jobs or audits are restarted.
    from scripts.finalize_window_study import report
    report(read_json(ROOT/'final-run-ledger.json'),True,read_json(ROOT/'distribution-summary.json'),
           read_json(ROOT/'verification.json')['gradual'],[])


if __name__=='__main__':main()
