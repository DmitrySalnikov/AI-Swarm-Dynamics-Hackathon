"""Reanalyse saved outputs by phase, separating reasoning, output and payload.

This module makes embedding calls only. It cannot start a generation client.
"""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from swarm_pipeline.analyze import embed, token_data, vector_key
from swarm_pipeline.uniform_windows import segment, snapshot_rows, compute
from swarm_pipeline.supervision import read_json, write_json

ROOT = Path('artifacts/pipeline-windows')
OUT = ROOT/'phase-analysis'
PHASES = ('negotiation','solving','exchange')
CHANNELS = ('reasoning','visible_output','payload')
LABELS = {'reasoning':'Private reasoning only', 'visible_output':'Visible model output, including JSON protocol',
          'payload':'Accepted message / result payload only'}


def parse_action(text):
    text=text.strip()
    if text.startswith('```') and text.endswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0]
    return json.loads(text)


def payload_text(action):
    """Keep authored semantic payload; distinguish its role in the manifest."""
    kind=action.get('action')
    if kind=='send':
        return action['message']
    if kind=='submit':
        return '\n\n'.join(str(action.get(k,'')) for k in ('answer','notes') if str(action.get(k,'')).strip())
    if kind=='finish':
        # Canonical order removes key-order formatting as a source of differences.
        return json.dumps(action['answers'],sort_keys=True,ensure_ascii=False)
    return ''


def load_channels(directory):
    state=read_json(directory/'state.json')
    assert state['status']=='completed' and state['pending'] is None
    pieces=defaultdict(list)
    receipts=[]
    for path in sorted(directory.glob('calls/*/request.json')):
        q=read_json(path); r=read_json(path.with_name('response.json')); e=read_json(path.with_name('event.json'))
        phase=q['context']['phase']; aid=q['agent']
        assert phase in PHASES
        for channel,field in (('reasoning','reasoning'),('visible_output','content')):
            if r.get(field): pieces[(phase,channel,aid)].append(r[field])
        if r['finish_reason']=='stop' and not e.get('action_error'):
            payload=payload_text(parse_action(r['content']))
            if payload: pieces[(phase,'payload',aid)].append(payload)
        receipts.append({'call':q['call'],'agent':aid,'phase':phase,'finish_reason':r['finish_reason'],
                         'action_error':e.get('action_error'),
                         'response_sha256':hashlib.sha256(path.with_name('response.json').read_bytes()).hexdigest()})
    result={}
    for phase in PHASES:
        for channel in CHANNELS:
            result[(phase,channel)]={int(aid):'\n\n'.join(pieces[(phase,channel,int(aid))]) for aid in state['agents']}
    return result,receipts


def make_rows(streams,tokenizer,width,condition):
    layout={aid:[segment(text,0,tokenizer)] for aid,text in streams.items()}
    lengths=[parts[0]['end'] for parts in layout.values()]
    horizon=max(lengths,default=0)
    starts=list(range(0,max(0,horizon-width+1),width//2))
    available=snapshot_rows(layout,starts,width,condition)
    minimum=min(lengths,default=0)
    # The common prefix uses the exact same tokenization, not a re-tokenized string.
    trimmed={aid:[{**parts[0],'offsets':parts[0]['offsets'][:minimum],'end':minimum}] for aid,parts in layout.items()}
    trim=snapshot_rows(trimmed,starts,width,condition)
    assert all(len(r['windows']) in (0,10) for r in trim)
    return available,trim,{'min_tokens':minimum,'median_tokens':float(np.median(lengths)),
        'max_tokens':horizon,'total_tokens':sum(lengths),'discarded_by_trim_tokens':sum(lengths)-10*minimum,
        'lengths':dict(zip(sorted(streams),lengths))}


def draw_case(rung,channel,width,records,coverage,path):
    fig,axes=plt.subplots(3,3,figsize=(14,8.5),sharey='row',sharex='col')
    panels=(('R','Mean vector length R',(0,1.02)),('eff_dim','Effective dimension',(0,9.3)),('n_active','Active authors',(-.2,10.5)))
    for col,phase in enumerate(PHASES):
        phase_rows=[r for r in records if r['phase']==phase]
        for row,(metric,label,limits) in enumerate(panels):
            ax=axes[row,col]
            for condition,color in (('swarm','#2476d2'),('private','#e57532')):
                for mode in ('available','trim'):
                    chosen=[r for r in phase_rows if r['condition']==condition and r['mode']==mode]
                    if not chosen: continue
                    # Trim gaps have no point. Available-author counts remain visible down to zero.
                    y=[r[metric] if mode=='available' or r['n_active']==10 else np.nan for r in chosen]
                    ax.plot([r['x'] for r in chosen],y,color=color,lw=1.9 if mode=='trim' else .9,
                        ls='-' if mode=='trim' else '--',alpha=1 if mode=='trim' else .55,
                        marker='o' if mode=='trim' else '.',ms=3,
                        label=f'{condition}: '+('trimmed, N=10' if mode=='trim' else 'all available, N varies'))
            ax.set_ylim(*limits);ax.grid(alpha=.2)
            if col==0:ax.set_ylabel(label)
            if row==2:ax.set_xlabel('Local generated-token position')
        maximum=max((c['max_tokens'] for c in coverage if c['phase']==phase),default=0)
        measurable=[r['x']+width/2 for r in phase_rows if r['n_active']>=4]
        # Do not compress measurable curves against a long single-author tail.
        # Every original row and the full text-length horizon remain in CSV/coverage.
        display_end=max(measurable) if measurable else maximum
        axes[0,col].set_xlim(0,max(width,display_end))
        title={'negotiation':'Distribution','solving':'Private solving','exchange':'Common-result commit'}[phase]
        axes[0,col].set_title(title)
        valid=[r for r in phase_rows if np.isfinite(r['R'])]
        if not valid:axes[0,col].text(.5,.5,'No full windows with N ≥ 4',transform=axes[0,col].transAxes,ha='center',fontsize=10)
        elif not any(r['mode']=='trim' and r['n_active']==10 for r in phase_rows):
            axes[0,col].text(.5,.08,'No full ten-author window',transform=axes[0,col].transAxes,ha='center',fontsize=9)
        if phase!='solving':axes[2,col].text(.5,.12,'Private has no corresponding phase',transform=axes[2,col].transAxes,ha='center',fontsize=8,color='#6c7179')
    handles,labels=axes[0,1].get_legend_handles_labels()
    if not handles:handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,fontsize=9)
    fig.suptitle(f'Ladder {rung} | {LABELS[channel]} | W={width}, stride={width//2}\nLocal phase clocks; full windows; display ends after last N >= 4 window (complete tails in CSV)',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.86));fig.savefig(path,dpi=140);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/'status.json',{'state':'running','updated_at':time.time()})
    tokenizer=Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    plans,texts,coverage,source_records=[],[],[],[]
    for rung,seed in ((1,3),(2,82),(3,83),(4,84)):
        sources={}
        for condition,folder,storage in (('swarm','swarm','cooperative'),('private','private','fixed')):
            directory=Path(f'artifacts/pipeline-ladder/rung_{rung}')/folder/storage/f'seed_{seed}'
            streams,receipts=load_channels(directory)
            sources[condition]=streams
            source_records.append({'rung':rung,'condition':condition,'directory':str(directory),
                'manifest_sha256':hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest(),'receipts':receipts})
        for channel in CHANNELS:
            for width in (100,200,400):
                groups=[]
                for phase in PHASES:
                    for condition in ('swarm','private'):
                        if condition=='private' and phase!='solving':continue
                        available,trim,info=make_rows(sources[condition][(phase,channel)],tokenizer,width,condition)
                        if width==100:coverage.append({'rung':rung,'phase':phase,'channel':channel,'condition':condition,**info})
                        for mode,rows in (('available',available),('trim',trim)):
                            groups.append((phase,condition,mode,rows))
                            if mode=='available':texts.extend(t for r in rows for t in r['windows'].values())
                plans.append((rung,channel,width,groups))
    write_json(OUT/'manifest.json',{'source':'Saved completed ladder pairs only; no generation',
        'channels':{'reasoning':'All observed provider reasoning, including interrupted/rejected turns.',
            'visible_output':'All observed own response content, including JSON protocol and interrupted/rejected turns.',
            'payload':'Accepted send.message during distribution; answer+notes on private submission; canonical complete answer map on commit. Claim actions excluded. No reasoning.'},
        'commit_warning':'V2 final phase is a commit of the identical coordinator packet, not a free-form peer conversation.',
        'clock':'Each phase starts at local token zero independently for each condition and channel. No inferred wall time.',
        'windows':'Full windows of 100/200/400, stride W/2. No short tails, held vectors, or synthetic text.',
        'geometry_threshold':4,'trim':'Shortest author prefix separately in each phase, condition and channel; every populated trim point has ten authors.',
        'model':'qwen/qwen3-embedding-4b','vector_normalization':'L2; unchanged upstream geometry',
        'source_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'sources':source_records,'unique_windows':len({vector_key(t) for t in texts})})
    write_json(OUT/'coverage.json',coverage)
    vectors=embed(texts,OUT,workers=16,cache_dir=Path('artifacts/pipeline-analysis/embedding-cache'))
    summaries,records_all=[],[]
    for rung,channel,width,groups in plans:
        case=[]
        for phase,condition,mode,rows in groups:
            records=compute(rows,vectors)
            for r in records:r.update(rung=rung,channel=channel,width=width,phase=phase,mode=mode)
            case.extend(records)
            full=[r for r in records if r['n_active']==10]
            summaries.append({'rung':rung,'channel':channel,'width':width,'phase':phase,'condition':condition,'mode':mode,
                'points_N4':sum(r['n_active']>=4 for r in records),'points_N10':len(full),
                'mean_R_N10':float(np.mean([r['R'] for r in full])) if full else None,
                'mean_dimension_N10':float(np.mean([r['eff_dim'] for r in full])) if full else None,
                'max_active':max((r['n_active'] for r in records),default=0)})
        if case:
            with (OUT/f'rung_{rung}_{channel}_w{width}.csv').open('w',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(case[0]));writer.writeheader();writer.writerows(case)
        draw_case(rung,channel,width,case,[c for c in coverage if c['rung']==rung and c['channel']==channel],
                  OUT/f'rung_{rung}_{channel}_w{width}.png')
        records_all.extend(case)
        print(f'Phase plots: ladder {rung}, {channel}, W={width}',flush=True)
    contrasts=[]
    for rung in range(1,5):
        for channel in CHANNELS:
            for width in (100,200,400):
                phase_means={p:next(s for s in summaries if s['rung']==rung and s['channel']==channel and s['width']==width
                    and s['phase']==p and s['condition']=='swarm' and s['mode']=='trim') for p in PHASES}
                m=[phase_means[p]['mean_R_N10'] for p in PHASES]
                solve=[r for r in records_all if r['rung']==rung and r['channel']==channel and r['width']==width
                       and r['phase']=='solving' and r['mode']=='trim' and r['n_active']==10]
                sw={r['start']:r for r in solve if r['condition']=='swarm'}
                pr={r['start']:r for r in solve if r['condition']=='private'}
                common=sorted(set(sw)&set(pr))
                contrasts.append({'rung':rung,'channel':channel,'width':width,
                    'phase_means_R_N10':dict(zip(PHASES,m)),
                    'phase_points_N10':{p:phase_means[p]['points_N10'] for p in PHASES},
                    'three_phases_observed':all(x is not None for x in m),
                    'high_low_high_phase_means':bool(m[0]>m[1] and m[2]>m[1]) if all(x is not None for x in m) else None,
                    'matched_solving_points_N10':len(common),
                    'mean_swarm_minus_private_R_same_local_token':float(np.mean([sw[x]['R']-pr[x]['R'] for x in common])) if common else None})
    write_json(OUT/'summary.json',summaries)
    write_json(OUT/'contrasts.json',contrasts)
    write_json(OUT/'status.json',{'state':'complete','updated_at':time.time(),'figures':len(plans)})


if __name__=='__main__':main()
