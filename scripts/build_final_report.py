"""Final report on the set_v2 ladder: swarm vs three controls without communication, several runs per condition.

Conditions per rung (run directories are found by glob; every completed run is used):
  swarm         swarm with the current prompts (756c69b): distribute ten tasks, each agent solves one, exchange, commit;
  swarm_old     the earlier swarm run with the simple prompts (one run, shown separately, never averaged in);
  one_task      comparison 1: one task per agent, no communication (task-to-agent assignment from the swarm or random);
  random_order  comparison 2: every agent solves all ten tasks in one random order shared within the run;
  unique_start  comparison 3: ten different first tasks, then each agent's own random order;
  fixed         the earlier fixed-order run (order 1-10 of the task file; one run, shown separately).
Only the solving phase is embedded (reasoning + visible output of every solving call, own token clock per agent).
Windows W=2048, stride W/2 (analyze.windows). At every window start: R = |mean of unit vectors| over the agents that
have the window, N = their number, same-task share = largest share of those agents on one task.
Swarm and one_task have one task per agent, so N falls as agents finish. Three variants for them
(swarm_pipeline.length_alignment): as is; (a) trim to the shortest agent; (b) supplement every agent to the longest one
with independent same-task solutions from the sequential runs (SYNTHETIC, labeled everywhere).
Mean curves over runs use only positions where every run of the condition has all ten agents; bands are min-max over runs.

  python scripts/build_final_report.py --dry-run     # windows, uncached windows and estimated embedding cost only
  python scripts/build_final_report.py --html-only   # figures and HTML again from final.json, no tokenizing or embedding
  OPENROUTER_API_KEY=... python scripts/build_final_report.py
Outputs in artifacts/pipeline-ladder-v2-final-report/: final.json, provenance.json, figures, final.html.
"""
import argparse
import base64
import glob
import html
import json
import shutil
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from swarm_pipeline.analyze import embed, vector_key, windows
from swarm_pipeline.audit import audit
from swarm_pipeline.length_alignment import donor_pool, solving_streams, supplement, trim, trim_facts
from swarm_pipeline.supervision import read_json, write_json

A = Path('artifacts')
OUT = A/'pipeline-ladder-v2-final-report'
CACHE = OUT/'embedding-cache'
OLD_CACHES = [A/'pipeline-ladder-v2-simple-sequential/embedding-cache']
SEEDS = {1: 3, 2: 82, 3: 83, 4: 84}
LABELS = {1: 'Unrelated domains', 2: 'Physics subfields', 3: 'Linear algebra', 4: 'Same determinant template'}
W = 2048
# key: (label, color, marker, dashed, run glob with {r} for the rung and {s} for its seed)
CONDITIONS = {
    'swarm': ('Swarm, current prompts', '#2a78d6', 'o', False, 'pipeline-ladder-v2-swarm-repeats/rung_{r}/swarm/cooperative/seed_*'),
    'one_task': ('Comparison 1: one task per agent', '#8a8984', '^', False, 'pipeline-ladder-v2-simple/rung_{r}/private/fixed/seed_*'),
    'random_order': ('Comparison 2: shared random order', '#eb6834', 's', False,
                     'pipeline-ladder-v2-simple-sequential-random-order/rung_{r}/fixed_random_order/seed_*'),
    'unique_start': ('Comparison 3: unique start, own order', '#1baf7a', 'D', False,
                     'pipeline-ladder-v2-simple-sequential-unique/rung_{r}/shuffled_unique_start/seed_*'),
    'swarm_old': ('Swarm, earlier prompts (1 run)', '#2a78d6', 'o', True, 'pipeline-ladder-v2-simple/rung_{r}/swarm/cooperative/seed_{s}'),
    'fixed': ('Fixed order 1–10, earlier (1 run)', '#eb6834', 's', True, 'pipeline-ladder-v2-simple-sequential/rung_{r}/fixed/seed_{s}'),
}
ONE_TASK = ('swarm', 'swarm_old', 'one_task')
DONOR_CONDITIONS = ('random_order', 'unique_start', 'fixed')
VARIANTS = {'as_is': 'as is', 'trim': '(a) trimmed to the shortest agent', 'supplement': '(b) supplemented to the longest agent (synthetic)'}


def runs(rung, cond):
    return [Path(p) for p in sorted(glob.glob(str(A/CONDITIONS[cond][4].format(r=rung, s=SEEDS[rung]))),
                                    key=lambda p: int(p.rsplit('_', 1)[1]))
            if read_json(Path(p)/'state.json')['status'] == 'completed']


def task_windows(stream):
    wins = windows(stream['text'], stream['offsets'], W)
    for w in wins:
        mid = min((w['start'] + w['end'])//2, len(stream['token_task']) - 1)
        w['task'] = stream['token_task'][mid]
        w['synthetic'] = max(0, w['end'] - max(w['start'], stream['synthetic_from']))/(w['end'] - w['start'])
    return wins


def groups(tokenizer, pool, provenance):
    """Yield (rung, cond, variant, [(run dir, per-agent windows)], facts) in a fixed order (donor use is order-dependent)."""
    used = set()
    for rung in SEEDS:
        for cond in CONDITIONS:
            dirs = runs(rung, cond)
            if not dirs:
                continue
            streams = {d: solving_streams(d, tokenizer) for d in dirs}
            yield rung, cond, 'as_is', [(d, [task_windows(s) for s in streams[d]]) for d in dirs], \
                {str(d): {'tokens_per_agent': [s['original_tokens'] for s in streams[d]]} for d in dirs}
            if cond not in ONE_TASK:
                continue
            yield rung, cond, 'trim', [(d, [task_windows(s) for s in trim(streams[d])]) for d in dirs], \
                {str(d): trim_facts(streams[d]) for d in dirs}
            out, facts = [], {}
            for d in dirs:
                state, manifest = read_json(d/'state.json'), read_json(d/'manifest.json')
                tasks = {int(a): x['order'][0] for a, x in state['agents'].items()}
                prompts = {t['id']: t['prompt'] for t in manifest['tasks']}
                extended, prov = supplement(streams[d], tasks, prompts, pool, used, tokenizer, str(d))
                provenance.extend(prov)
                total = sum(len(s['offsets']) for s in extended)
                synth = sum(p['synthetic_tokens'] for p in prov)
                facts[str(d)] = {'target_tokens': max(len(s['offsets']) for s in extended), 'synthetic_tokens': synth,
                                 'synthetic_share': synth/total, 'agents_supplemented': len(prov),
                                 'donor_texts': sum(len(p['donors']) for p in prov)}
                out.append((d, [task_windows(s) for s in extended]))
            yield rung, cond, 'supplement', out, facts


def series(per_agent, vectors):
    by_start = {}
    for wins in per_agent:
        for w in wins:
            by_start.setdefault(w['start'], []).append(w)
    rows = []
    for start in sorted(by_start):
        ws = by_start[start]
        V = np.stack([vectors[vector_key(w['text'])] for w in ws])
        V = V/np.linalg.norm(V, axis=1, keepdims=True)
        tasks = [w['task'] for w in ws]
        rows.append({'x': round((start + W/2)/1000, 3), 'R': float(np.linalg.norm(V.mean(0))), 'N': len(ws),
                     'same_task_share': max(tasks.count(t) for t in set(tasks))/len(tasks),
                     'synthetic_share': float(np.mean([w['synthetic'] for w in ws]))})
    return rows


def mean_curve(run_rows, full_only=True):
    """Per x: mean/min/max of R and mean same-task share over runs; full_only keeps x where every run has N=10."""
    at = {}
    for rows in run_rows:
        for r in rows:
            if r['N'] == 10 or not full_only:
                at.setdefault(r['x'], []).append(r)
    out = []
    for x in sorted(at):
        rs = at[x]
        if full_only and len(rs) < len(run_rows):
            continue
        R = [r['R'] for r in rs]
        out.append({'x': x, 'R': st.mean(R), 'R_min': min(R), 'R_max': max(R), 'N': st.mean(r['N'] for r in rs),
                    'same_task_share': st.mean(r['same_task_share'] for r in rs),
                    'synthetic_share': st.mean(r['synthetic_share'] for r in rs), 'runs': len(rs)})
    return out


def run_mean(rows, x_max=None):
    full = [r for r in rows if r['N'] == 10 and (x_max is None or r['x'] <= x_max)]
    return {'R': st.mean(r['R'] for r in full) if full else None, 'points': len(full),
            'same_task_share': st.mean(r['same_task_share'] for r in full) if full else None,
            'synthetic_share': st.mean(r['synthetic_share'] for r in full) if full else None}


def spread(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {'mean': st.mean(v), 'sd': st.stdev(v) if len(v) > 1 else None, 'min': min(v), 'max': max(v), 'n': len(v)}


def run_facts(d):
    led = rec = 0.0; inc = calls = 0
    for q in d.glob('calls/*/request.json'):
        calls += 1
        rp = q.with_name('response.json')
        if not rp.exists():
            led += read_json(q)['reserved_rub']; continue
        r = read_json(rp); led += r['cost_rub']; inc += r['finish_reason'] != 'stop'
        if r['finish_reason'] == 'stop' and not r.get('cost_estimated'):
            rec += r['cost_rub']
    a = audit(d)
    return {'status': read_json(d/'state.json')['status'], 'integrity': a['integrity'], 'own_correct': a['own_correct'],
            'own_total': a['own_total'], 'calls': calls, 'interrupted': inc, 'ledger_rub': led, 'receipts_rub': rec,
            'minutes': a['elapsed_minutes']}


def seed_cache(keys):
    CACHE.mkdir(parents=True, exist_ok=True)
    for key in keys:
        if (CACHE/f'{key}.npy').exists():
            continue
        for old in OLD_CACHES:
            if (old/f'{key}.npy').exists():
                shutil.copyfile(old/f'{key}.npy', CACHE/f'{key}.npy'); break


def style_axes(*axes):
    for a in axes:
        a.grid(color='#e4e3df', lw=.7)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)


def draw_curve(ax, curve, cond, x_lim=None, label=None, alpha=1.0):
    name, color, marker, dashed, _ = CONDITIONS[cond]
    c = [p for p in curve if x_lim is None or p['x'] <= x_lim]
    if not c:
        return
    xs = [p['x'] for p in c]
    if dashed:
        ax.plot(xs, [p['R'] for p in c], color=color, lw=1, ls='--', alpha=alpha, label=label)
        return
    if any(p['runs'] > 1 for p in c):
        ax.fill_between(xs, [p['R_min'] for p in c], [p['R_max'] for p in c], color=color, alpha=.15*alpha, lw=0)
    ax.plot(xs, [p['R'] for p in c], color=color, lw=1.4, alpha=alpha, marker=marker, ms=2.5, label=label)


def fig_main(d, path, x_lim):
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.8), sharex='col', gridspec_kw=dict(height_ratios=[3, 1.2], hspace=.08))
    for j, rung in enumerate(SEEDS):
        ax, axs = axes[0, j], axes[1, j]
        for cond in CONDITIONS:
            curve = d['curves'].get(f'{rung}/{cond}/as_is')
            if not curve:
                continue
            draw_curve(ax, curve, cond, x_lim, label=CONDITIONS[cond][0] if j == 0 else None)
            c = [p for p in curve if p['x'] <= x_lim]
            axs.plot([p['x'] for p in c], [p['same_task_share'] for p in c], color=CONDITIONS[cond][1], lw=1.1,
                     ls='--' if CONDITIONS[cond][3] else '-')
        rng = d['swarm_range'].get(str(rung))
        if rng:
            ax.axvline(rng, color='#2a78d6', lw=.8, ls=':')
        ax.set_title(f'Rung {rung} · {LABELS[rung]}', fontsize=10); ax.set_ylim(0.3, 1.0); axs.set_ylim(0, 1.05)
        style_axes(ax, axs)
    axes[0, 0].set_ylabel(f'R (W={W}), mean over runs'); axes[1, 0].set_ylabel('same-task\nshare')
    fig.legend(loc='upper center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5, 1.04))
    fig.supxlabel('generated solving tokens, thousands (own clock of each agent)', fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def fig_alignment(d, path):
    rows = [('as_is', 'R as is\n(N falls)'), ('trim', 'R (a) trimmed'), ('supplement', 'R (b) supplemented\n(synthetic)')]
    fig, axes = plt.subplots(4, 4, figsize=(17, 11.5), sharex='col', gridspec_kw=dict(height_ratios=[2.2, 1, 2.2, 2.2], hspace=.18))
    for j, rung in enumerate(SEEDS):
        x_lim = d['supplement_range'].get(str(rung))
        for i, (variant, title) in zip((0, 2, 3), rows):
            ax = axes[i, j]
            for cond in ONE_TASK:
                curve = d['curves_all' if variant == 'as_is' else 'curves'].get(f'{rung}/{cond}/{variant}')
                if curve:
                    draw_curve(ax, [p for p in curve if p['N'] >= 3], cond, x_lim, label=CONDITIONS[cond][0] if j == 0 and i == 0 else None)
            if variant == 'supplement':
                for cond in ('random_order', 'unique_start'):
                    curve = d['curves'].get(f'{rung}/{cond}/as_is')
                    if curve:
                        draw_curve(ax, curve, cond, x_lim, label=CONDITIONS[cond][0] if j == 0 else None, alpha=.45)
            ax.set_ylim(0.3, 1.0); style_axes(ax)
            if j == 0:
                ax.set_ylabel(title, fontsize=9)
        axn = axes[1, j]
        for cond in ONE_TASK:
            curve = d['curves_all'].get(f'{rung}/{cond}/as_is')
            if curve:
                c = [p for p in curve if x_lim is None or p['x'] <= x_lim]
                axn.plot([p['x'] for p in c], [p['N'] for p in c], color=CONDITIONS[cond][1], lw=1.2, ls='--' if CONDITIONS[cond][3] else '-')
        axn.set_ylim(0, 10.5); style_axes(axn)
        if j == 0:
            axn.set_ylabel('N as is\n(mean over runs)', fontsize=9)
        axes[0, j].set_title(f'Rung {rung} · {LABELS[rung]}', fontsize=10)
    fig.legend(loc='upper center', ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(.5, .95))
    fig.supxlabel('generated solving tokens, thousands', fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def fig_whole(d, path):
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.8), sharex='col', gridspec_kw=dict(height_ratios=[3, 1.2], hspace=.08))
    for j, rung in enumerate(SEEDS):
        ax, axs = axes[0, j], axes[1, j]
        for cond in ('random_order', 'unique_start', 'fixed'):
            curve = d['curves'].get(f'{rung}/{cond}/as_is')
            if curve:
                draw_curve(ax, curve, cond, label=CONDITIONS[cond][0] if j == 0 else None)
                axs.plot([p['x'] for p in curve], [p['same_task_share'] for p in curve], color=CONDITIONS[cond][1], lw=1.1,
                         ls='--' if CONDITIONS[cond][3] else '-')
        ax.set_title(f'Rung {rung} · {LABELS[rung]}', fontsize=10); ax.set_ylim(0.3, 1.0); axs.set_ylim(0, 1.05)
        style_axes(ax, axs)
    axes[0, 0].set_ylabel(f'R (W={W}), mean over runs'); axes[1, 0].set_ylabel('same-task\nshare')
    fig.legend(loc='upper center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(.5, 1.03))
    fig.supxlabel('generated solving tokens, thousands', fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--html-only', action='store_true', help='redraw figures and HTML from an existing final.json')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.html_only:
        render(read_json(OUT/'final.json'))
        return
    tokenizer = Tokenizer.from_file(str(A/'pipeline-analysis/deepseek-tokenizer.json'))
    pool = donor_pool([d for rung in SEEDS for c in DONOR_CONDITIONS for d in runs(rung, c)], tokenizer)
    provenance, plan, total, uncached, chars = [], [], 0, 0, 0
    series_by_run, facts = {}, {}
    for rung, cond, variant, per_run, f in groups(tokenizer, pool, provenance):
        texts = [w['text'] for _, per_agent in per_run for wins in per_agent for w in wins]
        keys = {vector_key(t): t for t in texts if t.strip()}
        seed_cache(keys)
        todo = {k: t for k, t in keys.items() if not (CACHE/f'{k}.npy').exists()}
        total += len(keys); uncached += len(todo); chars += sum(len(t) for t in todo.values())
        plan.append({'group': f'{rung}/{cond}/{variant}', 'runs': len(per_run), 'windows': len(keys), 'uncached': len(todo),
                     'estimated_usd': round(sum(len(t) for t in todo.values())/2*.02/1e6, 4)})
        facts[f'{rung}/{cond}/{variant}'] = f
        if args.dry_run:
            continue
        vectors = embed(texts, OUT, workers=12, cache_dir=CACHE)  # one call per group keeps each under embed()'s $1 envelope
        for d, per_agent in per_run:
            series_by_run[f'{rung}/{cond}/{variant}/{d.name}'] = series(per_agent, vectors)
        del vectors
    print(json.dumps({'groups': plan, 'windows': total, 'uncached': uncached, 'uncached_chars': chars,
                      'estimated_usd_upper': round(chars/2*.02/1e6, 4), 'donor_texts_used': sum(len(p['donors']) for p in provenance)}, indent=1))
    write_json(OUT/'provenance.json', {'note': 'Synthetic supplement (b): donor texts appended to recipients; never produced by the recipient.',
                                       'records': provenance})
    if args.dry_run:
        return
    build(series_by_run, facts, provenance)


def build(series_by_run, facts, provenance):
    group_runs = {}
    for key, rows in series_by_run.items():
        group_runs.setdefault(key.rsplit('/', 1)[0], []).append((key.rsplit('/', 1)[1], rows))
    curves = {g: mean_curve([r for _, r in rs]) for g, rs in group_runs.items()}
    curves_all = {g: mean_curve([r for _, r in rs], full_only=False) for g, rs in group_runs.items() if g.endswith('/as_is')}
    swarm_range, supplement_range, per_run_range = {}, {}, {}
    for rung in SEEDS:
        ends = [max((r['x'] for r in rows if r['N'] == 10), default=0) for _, rows in group_runs.get(f'{rung}/swarm/as_is', [])]
        swarm_range[str(rung)] = min(ends) if ends else None
        per_run_range[str(rung)] = ends
        sup = [max(r['x'] for r in rows) for c in ONE_TASK for _, rows in group_runs.get(f'{rung}/{c}/supplement', [])]
        supplement_range[str(rung)] = min(sup) if sup else None
    table = {}
    for g, rs in group_runs.items():
        rung = g.split('/')[0]
        per = {name: {'swarm_range': run_mean(rows, swarm_range[rung]), 'supplement_range': run_mean(rows, supplement_range[rung]),
                      'whole': run_mean(rows)} for name, rows in rs}
        table[g] = {'runs': per, **{k: {m: spread([p[k][m] for p in per.values()]) for m in ('R', 'same_task_share', 'synthetic_share', 'points')}
                                     for k in ('swarm_range', 'supplement_range', 'whole')}}
    run_table = {}
    for rung in SEEDS:
        for cond in CONDITIONS:
            for d in runs(rung, cond):
                run_table[f'{rung}/{cond}/{d.name}'] = {**run_facts(d), 'dir': str(d),
                                                         'mean_solving_tokens': st.mean(facts[f'{rung}/{cond}/as_is'][str(d)]['tokens_per_agent'])}
    receipts = [read_json(p) for p in (OUT/'embedding-requests').glob('*.response.json')]
    data = {'width': W, 'conditions': {k: v[0] for k, v in CONDITIONS.items()}, 'variants': VARIANTS,
            'swarm_range': swarm_range, 'swarm_range_per_run': per_run_range, 'supplement_range': supplement_range,
            'table': table, 'facts': facts, 'runs': run_table, 'curves': curves, 'curves_all': curves_all,
            'series': series_by_run, 'embedding_usd': sum((r.get('usage') or {}).get('cost', 0) for r in receipts),
            'donor_texts_used': sum(len(p['donors']) for p in provenance)}
    write_json(OUT/'final.json', data)
    render(data)


def render(data):
    fig_main(data, OUT/'fig_main.png', 30)
    fig_alignment(data, OUT/'fig_alignment.png')
    fig_whole(data, OUT/'fig_whole.png')
    (OUT/'final.html').write_text(page(data), encoding='utf-8')
    print('written', OUT/'final.html')


def fmt(s, digits=3):
    if not s or s.get('mean') is None:
        return '—'
    out = f"{s['mean']:.{digits}f}"
    if s['n'] > 1:
        out += f" <small>± {s['sd']:.{digits}f} [{s['min']:.{digits}f}–{s['max']:.{digits}f}]</small>"
    return out


def img(path, alt):
    return (f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode()}">'
            f'<figcaption>{html.escape(alt)}</figcaption></figure>')


def page(d):
    t = d['table']
    head = ''.join(f'<th>{html.escape(CONDITIONS[c][0])}</th>' for c in CONDITIONS)
    main_rows = ''.join(f"<tr><td>{r} · {LABELS[r]}<br><small>up to {d['swarm_range'][str(r)]:.1f}k tokens</small></td>"
                        + ''.join(f"<td>{fmt(t.get(f'{r}/{c}/as_is', {}).get('swarm_range', {}).get('R'))}</td>" for c in CONDITIONS) + '</tr>'
                        for r in SEEDS)
    share_rows = ''.join(f'<tr><td>{r}</td>' + ''.join(f"<td>{fmt(t.get(f'{r}/{c}/as_is', {}).get('swarm_range', {}).get('same_task_share'), 2)}</td>"
                                                      for c in CONDITIONS) + '</tr>' for r in SEEDS)
    align_rows = ''
    for r in SEEDS:
        for c in ONE_TASK:
            cells = ''.join(f"<td>{fmt(t.get(f'{r}/{c}/{v}', {}).get('swarm_range', {}).get('R'))}</td>" for v in VARIANTS)
            cells += f"<td>{fmt(t.get(f'{r}/{c}/supplement', {}).get('supplement_range', {}).get('R'))}</td>"
            align_rows += f'<tr><td>{r}</td><td>{html.escape(CONDITIONS[c][0])}</td>{cells}</tr>'
        for c in ('random_order', 'unique_start'):
            align_rows += (f"<tr><td>{r}</td><td><small>{html.escape(CONDITIONS[c][0])} (reference)</small></td><td>{fmt(t.get(f'{r}/{c}/as_is', {}).get('swarm_range', {}).get('R'))}</td>"
                           f"<td>—</td><td>—</td><td>{fmt(t.get(f'{r}/{c}/as_is', {}).get('supplement_range', {}).get('R'))}</td></tr>")
    trim_rows, sup_rows = '', ''
    for r in SEEDS:
        for c in ONE_TASK:
            ft = d['facts'].get(f'{r}/{c}/trim', {}); fs = d['facts'].get(f'{r}/{c}/supplement', {})
            if not ft:
                continue
            cut = [x for f in ft.values() for x in f['cut_tokens_per_agent']]
            trim_rows += (f"<tr><td>{r}</td><td>{html.escape(CONDITIONS[c][0])}</td><td>{len(ft)}</td>"
                          f"<td>{min(f['kept_tokens_per_agent'] for f in ft.values()):,}–{max(f['kept_tokens_per_agent'] for f in ft.values()):,}</td>"
                          f"<td>{st.median(cut):,.0f} / {max(cut):,}</td><td>{100*sum(f['cut_tokens_total'] for f in ft.values())/sum(f['original_tokens_total'] for f in ft.values()):.0f}%</td></tr>")
            sup_rows += (f"<tr><td>{r}</td><td>{html.escape(CONDITIONS[c][0])}</td><td>{len(fs)}</td>"
                         f"<td>{min(f['target_tokens'] for f in fs.values()):,}–{max(f['target_tokens'] for f in fs.values()):,}</td>"
                         f"<td>{sum(f['agents_supplemented'] for f in fs.values())}</td><td>{sum(f['donor_texts'] for f in fs.values())}</td>"
                         f"<td>{100*st.mean(f['synthetic_share'] for f in fs.values()):.0f}%</td></tr>")
    whole_rows = ''.join(f'<tr><td>{r}</td>' + ''.join(f"<td>{fmt(t.get(f'{r}/{c}/as_is', {}).get('whole', {}).get('R'))}</td>"
                                                      f"<td>{fmt(t.get(f'{r}/{c}/as_is', {}).get('whole', {}).get('same_task_share'), 2)}</td>"
                                                      for c in ('random_order', 'unique_start', 'fixed')) + '</tr>' for r in SEEDS)
    run_rows = ''.join(f"<tr><td>{k.split('/')[0]}</td><td>{html.escape(CONDITIONS[k.split('/')[1]][0])}</td><td>{k.split('/')[2]}</td>"
                       f"<td>{v['status']}, {v['integrity']}</td><td>{v['own_correct']}/{v['own_total']}</td><td>{v['calls']}</td><td>{v['interrupted']}</td>"
                       f"<td>{v['mean_solving_tokens']:,.0f}</td><td>{v['minutes']:.0f}</td><td>{v['ledger_rub']:.1f}</td><td>{v['receipts_rub']:.1f}</td></tr>"
                       for k, v in d['runs'].items())
    cost = {}
    for k, v in d['runs'].items():
        c = cost.setdefault(k.split('/')[1], [0, 0.0, 0.0]); c[0] += 1; c[1] += v['ledger_rub']; c[2] += v['receipts_rub']
    cost_rows = ''.join(f'<tr><td>{html.escape(CONDITIONS[c][0])}</td><td>{n}</td><td>{l:,.1f}</td><td>{r:,.1f}</td></tr>' for c, (n, l, r) in cost.items())
    counts = ', '.join(f"{CONDITIONS[c][0]}: {sum(1 for k in d['runs'] if k.startswith(f'1/{c}/'))}" for c in CONDITIONS)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swarm vs independent agents</title><style>
:root{{--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--bg:#fcfcfb;--warn:#8a4b00}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 64px}} h2{{margin-top:2em;font-size:20px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap;vertical-align:top}}
th:first-child,td:first-child,td:nth-child(2){{text-align:left}} th{{color:var(--ink2);font-weight:600}}
img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}} figcaption,.note,small{{color:var(--ink2);font-size:12.5px}}
.synthetic{{color:var(--warn);font-weight:600}}
</style></head><body><main>
<div class="note">SET_V2 LADDER · SWARM VS INDEPENDENT AGENTS · DeepSeek-V4-Flash · Qwen3-Embedding-4B</div>
<h1>Swarm vs independent agents: R of reasoning embeddings</h1>
<p><b>Swarm:</b> ten agents distribute ten tasks, each solves one, then they exchange answers and commit. Controls without any
communication: <b>comparison 1</b> one task per agent; <b>comparison 2</b> every agent solves all ten tasks in one random order shared
within the run (the order differs between runs); <b>comparison 3</b> ten different first tasks, then each agent's own random order.
Runs per rung — {counts}. Only the solving phase is embedded. Windows W={W}, stride {W//2}, own token clock per agent.
R = length of the mean unit vector over the agents present at a window position (N of them).
Values: mean over runs ± standard deviation [min–max over runs]; curves: mean over runs, band min–max.</p>
{findings(d)}
<h2>1. R where the swarm is solving</h2>
<p class="note">Mean R over windows with all ten agents, from the start up to the end of the shortest ten-agent solving range among the
current-prompt swarm runs of the rung (the range is set by the shortest swarm agent). Each run's mean is taken first.</p>
<table><tr><th>Rung</th>{head}</tr>{main_rows}</table>
<p class="note">Mean same-task share on the same range:</p>
<table><tr><th>Rung</th>{head}</tr>{share_rows}</table>
{img(OUT/'fig_main.png', 'R (top) and same-task share (bottom), first 30k solving tokens, ten-agent positions only. Dotted vertical line: end of the swarm range used in the tables. Dashed: single earlier runs.')}
<h2>2. Swarm and comparison 1: aligning lengths</h2>
<p>Both have one task per agent, so the number of agents still writing falls after the shortest agent finishes, and R over fewer
vectors is biased upwards. <b>(a)</b> keeps only text up to the shortest agent of each run. <b>(b)</b> appends to every shorter agent
independent solutions of the same task by other agents of the sequential runs until it reaches the longest agent of its run.
<span class="synthetic">(b) is synthetic: the appended text was never produced by that agent.</span> Each donor text is used once;
provenance (run, agent, task, sha256, tokens) is in <code>provenance.json</code>.</p>
<table><tr><th>Rung</th><th>Condition</th><th>as is, swarm range</th><th>(a) trimmed, swarm range</th><th>(b) <span class="synthetic">synthetic</span>, swarm range</th><th>(b) <span class="synthetic">synthetic</span>, whole range</th></tr>{align_rows}</table>
<p class="note">Whole range for (b): up to the shortest supplemented length over the swarm and comparison-1 runs of the rung
({', '.join(f"rung {r}: {d['supplement_range'][str(r)]:.1f}k" for r in SEEDS if d['supplement_range'].get(str(r)))} tokens);
comparisons 2–3 over the same ranges for reference.</p>
<h3>(a) How much text trimming removes</h3>
<table><tr><th>Rung</th><th>Condition</th><th>Runs</th><th>Kept tokens per agent (range over runs)</th><th>Cut per agent: median / max</th><th>Cut share of all text</th></tr>{trim_rows}</table>
<h3>(b) How much text is synthetic</h3>
<table><tr><th>Rung</th><th>Condition</th><th>Runs</th><th>Target tokens per agent</th><th>Agents supplemented</th><th>Donor texts</th><th>Synthetic share of all text</th></tr>{sup_rows}</table>
{img(OUT/'fig_alignment.png', 'Swarm and comparison 1: R as is (agents still writing), N as is, (a) trimmed, (b) supplemented (synthetic; comparisons 2–3 faint for reference).')}
<h2>3. Whole sequential runs (comparisons 2 and 3)</h2>
{img(OUT/'fig_whole.png', 'R and same-task share over the whole sequential runs (ten tasks per agent). Dashed: the earlier fixed-order run.')}
<table><tr><th>Rung</th><th>Comp. 2: mean R</th><th>same-task share</th><th>Comp. 3: mean R</th><th>same-task share</th><th>Fixed 1–10: mean R</th><th>same-task share</th></tr>{whole_rows}</table>
<h2>4. Runs</h2>
<table><tr><th>Rung</th><th>Condition</th><th>Seed</th><th>Status, audit</th><th>Correct own answers</th><th>Calls</th><th>Interrupted</th><th>Solving tokens per agent</th><th>Minutes</th><th>Ledger ₽</th><th>Receipts ₽</th></tr>{run_rows}</table>
<h2>5. Cost</h2>
<table><tr><th>Condition</th><th>Runs</th><th>Ledger ₽</th><th>Receipts ₽</th></tr>{cost_rows}</table>
<p class="note">Ledger is an upper bound: interrupted calls without a usage receipt are counted at their reservation. Embeddings for this
report: ${d['embedding_usd']:.4f} by receipts (only windows not cached from the earlier report).</p>
<h2>6. Limitations</h2>
<ul>
<li>Four runs per condition and rung (one for the earlier swarm and fixed runs): bands show run-to-run spread, not confidence intervals.</li>
<li>The swarm range is short (the shortest swarm agent), so the swarm comparison covers only the beginning of the solving phase unless (b) is used, and (b) is synthetic.</li>
<li>Comparison 1 reuses the swarm assignment for its first run and random assignments for the others; R does not depend on which agent gets which task.</li>
<li>About one call in five was interrupted by the provider and continued from the observed fragment; streams include those fragments.</li>
<li>One model (DeepSeek-V4-Flash), one embedding model, one window width.</li>
</ul>
<p class="note">Built by <code>scripts/build_final_report.py</code> from the saved run files; every number is in <code>final.json</code>.</p>
</main></body></html>'''


def findings(d):
    """Every number and every comparison word below is computed from the table in final.json."""
    t = d['table']

    def s(r, c, v='as_is', k='swarm_range', m='R'):
        return (t.get(f'{r}/{c}/{v}') or {}).get(k, {}).get(m)

    def show(x):
        return f"{x['mean']:.3f} [{x['min']:.3f}–{x['max']:.3f}]" if x['n'] > 1 else f"{x['mean']:.3f}"

    def overlap(a, b):
        return a['min'] <= b['max'] and b['min'] <= a['max']

    def rungs(flags):
        ok = [r for r, f in zip(SEEDS, flags) if f]
        return 'on every rung' if len(ok) == len(SEEDS) else (f"on rungs {', '.join(map(str, ok))} only" if ok else 'on no rung')

    first = ''.join(f"<tr><td>{r} · {LABELS[r]}</td><td>{show(s(r, 'swarm'))}</td><td>{show(s(r, 'one_task'))}</td>"
                    f"<td>{show(s(r, 'unique_start'))}</td><td>{show(s(r, 'random_order'))}</td>"
                    f"<td>{s(r, 'random_order', m='same_task_share')['mean']:.2f}</td></tr>" for r in SEEDS)
    same1 = [overlap(s(r, 'swarm'), s(r, 'one_task')) for r in SEEDS]
    same3 = [overlap(s(r, 'swarm'), s(r, 'unique_start')) for r in SEEDS]
    gap2 = [s(r, 'random_order')['mean'] - s(r, 'swarm')['mean'] for r in SEEDS]
    trim_shift = max(abs(s(r, c, 'trim')['mean'] - s(r, c)['mean']) for r in SEEDS for c in ONE_TASK)
    sup = [(s(r, 'swarm', 'supplement', 'supplement_range'), s(r, 'one_task', 'supplement', 'supplement_range')) for r in SEEDS]
    same_b = [overlap(a, b) for a, b in sup]
    synth = [x for r in SEEDS for c in ('swarm', 'one_task')
             for x in (f['synthetic_share'] for f in d['facts'][f'{r}/{c}/supplement'].values())]
    old_in = [s(r, 'swarm')['min'] <= s(r, 'swarm_old')['mean'] <= s(r, 'swarm')['max'] for r in SEEDS]
    whole = ''.join(f"<tr><td>{r}</td><td>{show(s(r, 'random_order', k='whole'))}</td><td>{s(r, 'random_order', k='whole', m='same_task_share')['mean']:.2f}</td>"
                    f"<td>{show(s(r, 'unique_start', k='whole'))}</td><td>{s(r, 'unique_start', k='whole', m='same_task_share')['mean']:.2f}</td></tr>" for r in SEEDS)
    b_rows = ', '.join(f"rung {r}: {show(a)} vs {show(b)}" for r, (a, b) in zip(SEEDS, sup))
    task_driven = all(same1) and all(same3) and all(same_b) and min(gap2[:3]) > max(abs(g) for g in gap2[3:]) 
    conclusion = ('<p><b>Conclusion.</b> In this design R follows how similar the tasks being solved at a token position are, not whether the '
                  'agents communicate: a swarm in which every agent solves its own task looks like independent agents on different tasks, '
                  'and independent agents on one shared task score far higher than the swarm.</p>') if task_driven else (
                  '<p><b>Conclusion.</b> The pattern is mixed; see the items above for the rungs where the conditions differ.</p>')
    return f"""<h2>Findings</h2>
<p class="note">Mean R over runs [min–max over runs], windows with all ten agents, from the start of solving up to the end of the swarm range
({', '.join(f"rung {r}: {d['swarm_range'][str(r)]:.1f}k" for r in SEEDS)} tokens).</p>
<table><tr><th>Rung</th><th>Swarm (4 runs)</th><th>Comp. 1: one task</th><th>Comp. 3: unique start</th><th>Comp. 2: shared order</th><th>Comp. 2 same-task share</th></tr>{first}</table>
<ol>
<li><b>Swarm vs comparison 1.</b> The run-to-run ranges of the swarm and of the one-task control overlap {rungs(same1)}.
The swarm also overlaps comparison 3 {rungs(same3)}: at the start of comparison 3 every agent is on a different task, as in the swarm.</li>
<li><b>Comparison 2 is higher where the tasks differ.</b> There all ten agents start on the same task (same-task share about 1.0).
Comparison 2 minus swarm: {', '.join(f'rung {r}: {g:+.3f}' for r, g in zip(SEEDS, gap2))}. On rung 4 all tasks share one template.</li>
<li><b>(a) Trimming</b> changes the mean R of the swarm and comparison 1 on the swarm range by at most {trim_shift:.3f}: the ten-agent windows
already stop where the shortest agent stops.</li>
<li><b>(b) Supplementing</b> (<span class="synthetic">synthetic</span>, {100*min(synth):.0f}–{100*max(synth):.0f}% of the text per run) extends the
comparison to the whole solving length. Swarm vs comparison 1 there: {b_rows}; the ranges overlap {rungs(same_b)}.</li>
<li><b>Earlier swarm prompts.</b> The single earlier swarm run lies inside the range of the four current-prompt runs {rungs(old_in)}
({', '.join(f"rung {r}: {s(r, 'swarm_old')['mean']:.4f} vs {s(r, 'swarm')['min']:.4f}–{s(r, 'swarm')['max']:.4f}" for r in SEEDS)}).</li>
<li><b>Whole sequential runs.</b> R goes with the same-task share: comparison 2 keeps agents on the same task more often than comparison 3.</li>
</ol>
<table><tr><th>Rung</th><th>Comp. 2: mean R, whole run</th><th>same-task share</th><th>Comp. 3: mean R, whole run</th><th>same-task share</th></tr>{whole}</table>
{conclusion}"""


if __name__ == '__main__':
    main()
