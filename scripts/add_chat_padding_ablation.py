"""Literal synthetic chat-padding proposal, kept separate from natural output."""
import csv
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tokenizers import Tokenizer
from swarm_pipeline.uniform_windows import ROOT, load_streams, layouts, donor_pool, segment, snapshot_rows, compute, draw
from swarm_pipeline.analyze import embed, vector_key
from swarm_pipeline.supervision import read_json, write_json


def main():
    tokenizer=Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    root=Path('artifacts/pipeline-ladder/rung_1')
    swarm=load_streams(root/'swarm/cooperative/seed_3',tokenizer)
    private=load_streams(root/'private/fixed/seed_3',tokenizer)
    donors=donor_pool(tokenizer,{a['task'] for a in swarm})
    layout,spans,provenance=layouts(swarm,tokenizer,'chat_supplement',donors)
    control={a['agent']:[segment(a['text'],0,tokenizer)] for a in private}
    end=max(s['end'] for parts in list(layout.values())+list(control.values()) for s in parts)
    plans,texts=[],[]
    for width in (400,2048,4096):
        starts=list(range(0,end-width+1,width//2))
        rows=snapshot_rows(layout,starts,width,'swarm')+snapshot_rows(control,starts,width,'private')
        texts.extend(t for r in rows for t in r['windows'].values()); plans.append((width,rows))
    vectors=embed(texts,ROOT/'chat-padding-analysis',workers=12,cache_dir=Path('artifacts/pipeline-analysis/embedding-cache'))
    summary=read_json(ROOT/'analysis-summary.json')
    summary=[r for r in summary if r['mode']!='chat_supplement']
    import numpy as np
    for width,rows in plans:
        half=len(rows)//2
        records=compute(rows[:half],vectors)+compute(rows[half:],vectors)
        stem=ROOT/'analysis'/f'rung_1_chat_supplement_w{width}'
        with stem.with_suffix('.csv').open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        draw(records,spans,width,'Ladder 1: SYNTHETIC chat + independent same-task solution padding',stem.with_suffix('.png'))
        paired=[a['R']-b['R'] for a,b in zip(records[:half],records[half:]) if a['n_active']==b['n_active']==10]
        summary.append({'rung':1,'mode':'chat_supplement','width':width,'full10_paired_points':len(paired),
            'mean_R_difference_at_full10':float(np.mean(paired)) if paired else None,
            'swarm_points_N4':sum(r['n_active']>=4 for r in records[:half]),
            'private_points_N4':sum(r['n_active']>=4 for r in records[half:])})
    write_json(ROOT/'analysis-summary.json',summary)
    write_json(ROOT/'chat-padding-provenance.json',{'mode':'chat_supplement','phase_spans':spans,
        'warning':'Independent solutions appended to short chat phases. Deliberately synthetic; not observed communication.',
        'no_donor_reuse_per_author':True,'supplements':provenance})


if __name__=='__main__':main()
