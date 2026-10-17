"""Render saved metrics with readable phase horizons, without embedding calls."""
import csv
import hashlib
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from swarm_pipeline.phase_windows import OUT, draw_case
from swarm_pipeline.supervision import read_json, write_json


def main():
    coverage=read_json(OUT/'coverage.json')
    for rung in range(1,5):
        for channel in ('reasoning','visible_output','payload'):
            for width in (100,200,400):
                stem=OUT/f'rung_{rung}_{channel}_w{width}'
                with stem.with_suffix('.csv').open(encoding='utf-8') as f:records=list(csv.DictReader(f))
                for r in records:
                    for key in ('n_active','n_matched','rung','width'):r[key]=int(r[key])
                    for key in ('start','x','R','eff_dim','pair_cosine','matched_drift'):r[key]=float(r[key])
                draw_case(rung,channel,width,records,[c for c in coverage if c['rung']==rung and c['channel']==channel],stem.with_suffix('.png'))
    manifest=read_json(OUT/'manifest.json')
    snapshot=OUT/'source_phase_windows.py'
    assert hashlib.sha256(snapshot.read_bytes()).hexdigest()==manifest['source_code_sha256']
    write_json(OUT/'rendering-revision.json',{'metric_source_snapshot':str(snapshot),
        'metric_source_hash':manifest['source_code_sha256'],
        'renderer_source_hash':hashlib.sha256(Path('swarm_pipeline/phase_windows.py').read_bytes()).hexdigest(),
        'change':'Display each phase through the final N>=4 window instead of a long unmeasurable N<4 tail. Raw CSVs and metrics unchanged.',
        'new_embeddings':False,'new_generation':False})


if __name__=='__main__':main()
