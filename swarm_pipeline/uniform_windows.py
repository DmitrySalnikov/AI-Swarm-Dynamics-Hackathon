"""Uniform author-preserving windows on an explicit conditional token clock.

The clock is not wall time. Missing text stays missing; no sample-and-hold,
progress interpolation, short tails, or prompt/inbox text enter embeddings.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from embedding_eval.dynamics import metrics_at
from swarm_pipeline.analyze import embed, token_data, vector_key
from swarm_pipeline.supervision import read_json, write_json

PHASES = ('negotiation', 'solving', 'exchange')
ROOT = Path('artifacts/pipeline-windows')


def load_streams(directory, tokenizer):
    state = read_json(directory / 'state.json')
    assert state['status'] == 'completed'
    streams = []
    for aid, agent in state['agents'].items():
        path = directory / 'generated' / f'agent_{aid}.generated.txt'
        text = path.read_text(encoding='utf-8')
        index = read_json(path.with_suffix('.index.json'))
        phases = {}
        for phase in PHASES:
            pieces = [text[p['start']:p['end']] for p in index if p['phase'] == phase]
            phases[phase] = '\n\n'.join(pieces)
        streams.append({'agent': int(aid), 'task': agent['order'][0], 'text': text,
                        'phases': phases, 'source': str(path)})
    return streams


def segment(text, start, tokenizer, **metadata):
    offsets = token_data(tokenizer, text)
    return {'text': text, 'offsets': offsets, 'start': start, 'end': start + len(offsets), **metadata}


def extract(segments, start, width):
    """Return a full window only when every clock token is covered exactly once."""
    cursor, parts = start, []
    for s in sorted(segments, key=lambda p: p['start']):
        if s['end'] <= cursor:
            continue
        if s['start'] > cursor:
            return None
        stop = min(start + width, s['end'])
        a, b = cursor - s['start'], stop - s['start']
        if b > a:
            parts.append(s['text'][int(s['offsets'][a, 0]):int(s['offsets'][b-1, 1])])
        cursor = stop
        if cursor == start + width:
            return '\n\n'.join(parts)
    return None


def donor_pool(tokenizer, tasks):
    """Independent same-task solutions, excluding the paired private control."""
    pool = {task: [] for task in tasks}
    reference = {t['id']: t['prompt'] for t in read_json(Path('artifacts/pipeline-ladder/rung_1/swarm/cooperative/seed_3/manifest.json'))['tasks']}
    for directory in sorted(Path('artifacts/pipeline-live').glob('*/seed_3')):
        if directory.parent.name not in {'fixed', 'shuffled', 'disjoint'}:
            continue
        manifest = read_json(directory / 'manifest.json')
        valid = {t['id'] for t in manifest['tasks'] if t['id'] in reference and t['prompt'] == reference[t['id']]}
        grouped = {}
        for path in sorted(directory.glob('calls/*/request.json')):
            q = read_json(path)
            task = q['context'].get('current_task')
            if q['context']['phase'] != 'solving' or task not in valid or task not in pool:
                continue
            if not path.with_name('event.json').exists():
                continue
            r, e = read_json(path.with_name('response.json')), read_json(path.with_name('event.json'))
            if r['finish_reason'] != 'stop' or e.get('action_error'):
                continue
            grouped.setdefault((q['agent'], task), []).append((str(path.parent), r))
        for (_, task), records in grouped.items():
            text = '\n\n'.join(r.get(field, '') for _, r in records for field in ('reasoning', 'content') if r.get(field))
            pool[task].append({'text': text, 'paths': [p for p, _ in records],
                               'sha256': hashlib.sha256(text.encode()).hexdigest()})
    return pool


def layouts(streams, tokenizer, mode, donors=None):
    lengths = {p: [len(token_data(tokenizer, a['phases'][p])) for a in streams] for p in PHASES}
    spans, cursor = {}, 0
    for p in PHASES:
        spans[p] = (cursor, cursor + max(lengths[p]))
        cursor = spans[p][1]
    result, provenance = {}, []
    for index, a in enumerate(streams):
        if mode == 'continuous':
            result[a['agent']] = [segment(a['text'], 0, tokenizer)]
            continue
        parts = []
        used_donors = set()
        for p in PHASES:
            original = segment(a['phases'][p], spans[p][0], tokenizer, phase=p)
            if mode == 'trim':
                n = min(lengths[p])
                original['offsets'] = original['offsets'][:n]
                original['end'] = original['start'] + n
            parts.append(original)
            if ((mode == 'supplement' and p == 'solving') or
                    (mode == 'chat_supplement' and p in {'negotiation','exchange'})):
                needed = max(lengths[p]) - lengths[p][index]
                donor_cursor = original['end']
                for donor in (donors or {}).get(a['task'], []):
                    if needed <= 0:
                        break
                    if donor['sha256'] in used_donors:
                        continue
                    s = segment(donor['text'], donor_cursor, tokenizer, phase=p, synthetic=True)
                    n = min(needed, len(s['offsets']))
                    s['offsets'] = s['offsets'][:n]; s['end'] = s['start'] + n
                    if n:
                        used_donors.add(donor['sha256'])
                        parts.append(s)
                        provenance.append({'agent': a['agent'], 'task': a['task'], 'tokens': n,
                                           'phase': p, 'start': s['start'], 'end': s['end'], 'donor': donor})
                    donor_cursor += n; needed -= n
        result[a['agent']] = parts
    return result, spans, provenance


def snapshot_rows(layout, starts, width, condition):
    return [{'condition': condition, 'start': start, 'x': start + width/2,
             'windows': {aid: text for aid, parts in layout.items()
                         if (text := extract(parts, start, width)) is not None}}
            for start in starts]


def compute(rows, vectors):
    result = []
    previous = None
    for row in rows:
        keys = sorted(row['windows'])
        n = len(keys)
        record = {k: row[k] for k in ('condition', 'start', 'x')}
        record.update(n_active=n, agents=','.join(map(str, keys)),
                      R=float('nan'), eff_dim=float('nan'), pair_cosine=float('nan'),
                      matched_drift=float('nan'), n_matched=0)
        X = np.stack([vectors[vector_key(row['windows'][i])] for i in keys]) if n else None
        if n >= 4:
            m = metrics_at(X)
            record.update(R=float(m['R']), eff_dim=float(m['eff_dim']),
                          pair_cosine=float((n*m['R']**2 - 1)/(n-1)))
        if previous:
            common = sorted(set(keys) & set(previous['windows']))
            record['n_matched'] = len(common)
            if len(common) >= 4:
                before = np.stack([vectors[vector_key(previous['windows'][i])] for i in common])
                after = np.stack([vectors[vector_key(row['windows'][i])] for i in common])
                record['matched_drift'] = float(np.mean(1-np.sum(before * after, axis=1)))
        result.append(record)
        previous = row
    return result


def draw(records, spans, width, title, path, min_authors=4):
    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True)
    panels = [('R', 'Mean vector length R'), ('pair_cosine', 'Mean pairwise cosine'),
              ('eff_dim', 'Effective dimension'), ('n_active', 'Active vectors (all available authors)'),
              ('matched_drift', 'Adjacent-window drift, same authors only')]
    for ax, (key, label) in zip(axes, panels):
        for condition, color in [('swarm', '#2476d2'), ('private', '#e57532')]:
            series = [r for r in records if r['condition'] == condition]
            ax.plot([r['x'] for r in series], [r[key] for r in series], '.-', ms=3,
                    lw=1.2, label=condition, color=color)
        for p in ('negotiation', 'exchange'):
            ax.axvspan(*spans[p], color='#689bce', alpha=.09)
        for start, _ in spans.values():
            ax.axvline(start, color='.65', lw=.6, ls=':')
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].set_ylim(0, 1); axes[3].set_ylim(-.3, 10.5)
    axes[0].legend(loc='best')
    axes[-1].set_xlabel('Conditional generated-token clock; fixed grid, no time rescaling')
    fig.suptitle(f'{title} | W={width}, stride={width//2}\nOwn reasoning + output; full windows only; gaps retained; geometry shown for N >= {min_authors}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    plans, texts, descriptions = [], [], []
    for rung, seed in ((1, 3), (2, 82), (3, 83), (4, 84)):
        root = Path(f'artifacts/pipeline-ladder/rung_{rung}')
        swarm = load_streams(root / 'swarm' / 'cooperative' / f'seed_{seed}', tokenizer)
        private = load_streams(root / 'private' / 'fixed' / f'seed_{seed}', tokenizer)
        donors = donor_pool(tokenizer, {a['task'] for a in swarm}) if rung == 1 else {}
        widths = (400, 2048, 4096) if rung == 1 else (2048,)
        modes = ('continuous', 'barrier', 'trim', 'supplement') if rung == 1 else ('continuous', 'barrier', 'trim')
        for mode in modes:
            layout, spans, provenance = layouts(swarm, tokenizer, mode, donors)
            control = {a['agent']: [segment(a['text'], 0, tokenizer)] for a in private}
            end = max([s['end'] for parts in layout.values() for s in parts] +
                      [s['end'] for parts in control.values() for s in parts])
            info = {'rung': rung, 'mode': mode, 'phase_spans': spans,
                    'phase_lengths': {a['agent']: {p: len(token_data(tokenizer, a['phases'][p])) for p in PHASES} for a in swarm},
                    'supplements': provenance, 'sources': [a['source'] for a in swarm + private]}
            descriptions.append(info)
            for width in widths:
                starts = list(range(0, max(0, end-width+1), width//2))
                rows = snapshot_rows(layout, starts, width, 'swarm') + snapshot_rows(control, starts, width, 'private')
                texts.extend(t for row in rows for t in row['windows'].values())
                plans.append((rung, mode, width, spans, rows))
    write_json(ROOT / 'window-methods.json', {'clock': 'Fixed token grid starting at private token zero; union horizon. Not wall time.',
        'coverage_rule': 'Full W logical tokenizer tokens; N>=4 for geometry; no padding, tails, holds, or interpolation.',
        'continuous_spans': 'Shading is reference barrier timing only, not the actual per-author phase boundaries.',
        'supplement_rule': 'Independent same-prompt solving output from old controls; never add solutions to chat. Synthetic ablation only.',
        'cases': descriptions, 'window_count': len(texts), 'unique_windows': len({vector_key(t) for t in texts})})
    vectors = embed(texts, ROOT / 'analysis', workers=args.workers,
                    cache_dir=Path('artifacts/pipeline-analysis/embedding-cache'))
    summary = []
    for rung, mode, width, spans, rows in plans:
        records = compute(rows[:len(rows)//2], vectors) + compute(rows[len(rows)//2:], vectors)
        stem = ROOT / 'analysis' / f'rung_{rung}_{mode}_w{width}'
        with stem.with_suffix('.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        draw(records, spans, width, f'Ladder {rung}: {mode}', stem.with_suffix('.png'))
        paired = []
        half = len(records)//2
        for a, b in zip(records[:half], records[half:]):
            if a['n_active'] == b['n_active'] == 10:
                paired.append(a['R']-b['R'])
        summary.append({'rung': rung, 'mode': mode, 'width': width, 'full10_paired_points': len(paired),
                        'mean_R_difference_at_full10': float(np.mean(paired)) if paired else None,
                        'swarm_points_N4': sum(r['n_active'] >= 4 for r in records[:half]),
                        'private_points_N4': sum(r['n_active'] >= 4 for r in records[half:])})
    write_json(ROOT / 'analysis-summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
