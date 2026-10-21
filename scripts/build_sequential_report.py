"""Swarm vs sequential independent agents on the set_v2 ladder (simple prompts).

Conditions per rung: swarm solving phase (ladder run with simple prompts), the one-task private control of that run,
and the sequential controls (every agent solves all ten tasks one after another: one shared order 1-10, or a unique
random first task followed by the agent's own random order).
Each agent's solving stream (reasoning + visible output, in call order) is cut into windows of W tokens with stride
W/2 (analyze.windows, the original δ(t) rule); at every window position R = |mean of the unit vectors| over the agents
that have that window, N = their number. "Same-task share" = the largest share of those agents working on one task.

  python scripts/build_sequential_report.py --dry-run     # windows and estimated embedding cost only
  OPENROUTER_API_KEY=... python scripts/build_sequential_report.py
Outputs in artifacts/pipeline-ladder-v2-simple-sequential/report/: sequential.json, figures, sequential.html.
"""
import argparse
import base64
import html
import json
import os
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from swarm_pipeline.analyze import embed, token_data, vector_key, windows
from swarm_pipeline.supervision import read_json, write_json

SWARM_ROOT = Path('artifacts/pipeline-ladder-v2-simple')
SEQ_ROOT = Path(os.environ.get('SEQ_ROOT', 'artifacts/pipeline-ladder-v2-simple-sequential'))
OUT = SEQ_ROOT/'report'
SEEDS = {1: 3, 2: 82, 3: 83, 4: 84}
LABELS = {1: 'Unrelated domains', 2: 'Physics subfields', 3: 'Linear algebra', 4: 'Same determinant template'}
W = 2048
# palette: validated categorical slots (blue, orange, aqua) + neutral gray; every series also has its own marker
STYLE = {'swarm': ('#2a78d6', 'o', 'Swarm (one task per agent, after distribution)'),
         'fixed': ('#eb6834', 's', 'Independent, fixed order 1–10'),
         'shuffled_unique_start': ('#1baf7a', 'D', 'Independent, unique start, then own random order'),
         'private': ('#8a8984', '^', 'Independent, one task per agent (earlier control)')}


def run_dir(rung, condition):
    if condition == 'swarm':
        return SWARM_ROOT/f'rung_{rung}/swarm/cooperative/seed_{SEEDS[rung]}'
    if condition == 'private':
        return SWARM_ROOT/f'rung_{rung}/private/fixed/seed_{SEEDS[rung]}'
    return SEQ_ROOT/f'rung_{rung}/{condition}/seed_{SEEDS[rung]}'


def streams(directory, tokenizer):
    """Per agent: solving text in call order, token offsets, and the task id of every token."""
    state = read_json(directory/'state.json')
    assert state['status'] == 'completed', directory
    out = []
    for aid in sorted(state['agents'], key=int):
        path = directory/'generated'/f'agent_{aid}.generated.txt'
        text, index = path.read_text(encoding='utf-8'), read_json(path.with_suffix('.index.json'))
        pieces = sorted((p for p in index if p['phase'] == 'solving'), key=lambda p: (p['call'], p['start']))
        parts, tasks = [], []
        for p in pieces:
            chunk = text[p['start']:p['end']]
            parts.append(chunk)
            tasks.append((len('\n\n'.join(parts)), p.get('task_id')))
        joined = '\n\n'.join(parts)
        offsets = token_data(tokenizer, joined)
        ends = np.array([e for e, _ in tasks]); ids = [t for _, t in tasks]
        token_task = [ids[min(int(np.searchsorted(ends, o[0], side='right')), len(ids) - 1)] for o in offsets] if len(offsets) else []
        out.append({'agent': int(aid), 'text': joined, 'offsets': offsets, 'token_task': token_task})
    return out


def plan(tokenizer):
    plans = {}
    for rung in SEEDS:
        for cond in STYLE:
            d = run_dir(rung, cond)
            if not (d/'state.json').exists():
                continue
            per_agent = []
            for s in streams(d, tokenizer):
                wins = windows(s['text'], s['offsets'], W)
                for w in wins:
                    mid = min((w['start'] + w['end']) // 2, len(s['token_task']) - 1)
                    w['task'] = s['token_task'][mid] if s['token_task'] else None
                per_agent.append(wins)
            plans[(rung, cond)] = per_agent
    return plans


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
        top = max(tasks.count(t) for t in set(tasks)) / len(tasks)
        rows.append({'x': (start + W/2)/1000, 'R': float(np.linalg.norm(V.mean(0))), 'N': len(ws), 'same_task_share': top})
    return rows


def summary(rows, x_max=None):
    full = [r for r in rows if r['N'] == 10 and (x_max is None or r['x'] <= x_max)]
    return {'mean_R_N10': round(st.mean(r['R'] for r in full), 4) if full else None, 'points_N10': len(full),
            'mean_same_task_share_N10': round(st.mean(r['same_task_share'] for r in full), 3) if full else None}


def figure(results, path, x_lim=None, title=''):
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.6), sharex='col', gridspec_kw=dict(height_ratios=[3, 1.2], hspace=.08))
    for j, rung in enumerate(SEEDS):
        ax, axn = axes[0, j], axes[1, j]
        for cond, (color, marker, label) in STYLE.items():
            rows = results.get(f'{rung}/{cond}')
            if not rows:
                continue
            rows = [r for r in rows if r['N'] >= 3 and (x_lim is None or r['x'] <= x_lim)]
            full = [r for r in rows if r['N'] == 10]
            ax.plot([r['x'] for r in rows], [r['R'] for r in rows], color=color, lw=.8, alpha=.35)
            ax.plot([r['x'] for r in full], [r['R'] for r in full], marker, color=color, ms=3.5, ls='', label=label if j == 0 else None)
            axn.plot([r['x'] for r in rows], [r['same_task_share'] for r in rows], color=color, lw=1.2)
        ax.set_title(f'Rung {rung} · {LABELS[rung]}', fontsize=10); ax.set_ylim(0.3, 1.0)
        axn.set_ylim(0, 1.05)
        for a in (ax, axn):
            a.grid(color='#e4e3df', lw=.7)
            for s in ('top', 'right'):
                a.spines[s].set_visible(False)
    axes[0, 0].set_ylabel(f'R, window W={W}'); axes[1, 0].set_ylabel('same-task\nshare')
    fig.legend(loc='upper center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(.5, .985))
    fig.supxlabel('generated tokens, thousands (own clock of each agent)', fontsize=10)
    fig.suptitle(title, y=1.03, fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches='tight'); plt.close(fig)


def img(path, alt):
    return (f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode()}">'
            f'<figcaption>{html.escape(alt)}</figcaption></figure>')


def page(d):
    rows = ''
    for rung in SEEDS:
        cells = ''
        for cond in STYLE:
            s = d['summary'].get(f'{rung}/{cond}', {})
            v = s.get('mean_R_N10_in_swarm_range'); sh = s.get('same_task_share_in_swarm_range')
            cells += f"<td>{'—' if v is None else f'{v:.3f}'} <small>({'—' if sh is None else f'{sh:.2f}'})</small></td>"
        rows += f'<tr><td>{rung} · {LABELS[rung]}</td>{cells}</tr>'
    whole = ''
    for rung in SEEDS:
        whole += f'<tr><td>{rung}</td>' + ''.join(
            f"<td>{d['summary'].get(f'{rung}/{c}', {}).get('mean_R_N10') or '—'}</td><td>{d['summary'].get(f'{rung}/{c}', {}).get('points_N10', 0)}</td>"
            for c in ('fixed', 'shuffled_unique_start')) + '</tr>'
    runs = d['runs']
    run_rows = ''.join(f"<tr><td>{k}</td><td>{v['status']}</td><td>{v['answers']}</td><td>{v['correct']}</td><td>{v['calls']}</td>"
                       f"<td>{v['interrupted']}</td><td>{v['mean_solve_tokens']:,}</td></tr>" for k, v in runs.items())
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swarm vs sequential agents</title><style>
:root{{--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--bg:#fcfcfb;--card:#f3f1ea}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}}
main{{max-width:1150px;margin:0 auto;padding:24px 16px 64px}} h2{{margin-top:2em;font-size:20px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
th{{color:var(--ink2)}} img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}} figcaption,.note,small{{color:var(--ink2);font-size:13px}}
</style></head><body><main>
<div class="note">SET_V2 LADDER · SEQUENTIAL CONTROLS · simple prompts</div>
<h1>Swarm vs independent agents solving all tasks one after another</h1>
<p>Swarm: ten agents distribute ten tasks and each solves one. Controls without any communication: every agent solves all ten
tasks in order 1–10 (<b>fixed</b>) or starts on its own task (ten different first tasks) and then takes the rest in its own random
order (<b>unique start</b>); plus the earlier one-task-per-agent control.
Windows W={W}, stride {W//2}, reasoning + visible output, own token clock per agent. One run per condition and rung; no uncertainty bands.</p>
{d['findings_html']}
<h2>1. R where the swarm is solving</h2>
<table><tr><th>Rung</th>{''.join(f'<th>{STYLE[c][2]}</th>' for c in STYLE)}</tr>{rows}</table>
<p class="note">Mean R over ten-author windows up to the end of the swarm's ten-author solving range ({d['swarm_range_note']}); in brackets the mean same-task share.</p>
{img(OUT/'fig_sequential_zoom.png', 'R (top) and same-task share (bottom) over the first tokens; dots: all ten agents present, faint line: fewer.')}
<h2>2. Whole sequential runs</h2>
{img(OUT/'fig_sequential_full.png', 'R and same-task share over the whole sequential runs (about ten tasks per agent).')}
<table><tr><th>Rung</th><th>fixed: mean R (N=10)</th><th>points</th><th>unique start: mean R (N=10)</th><th>points</th></tr>{whole}</table>
<h2>3. Runs</h2>
<table><tr><th>Run</th><th>Status</th><th>Answers</th><th>Correct</th><th>Calls</th><th>Interrupted calls</th><th>Mean tokens per solve</th></tr>{run_rows}</table>
<p class="note">Sources: <code>{SWARM_ROOT}</code> (swarm, one-task control), <code>{SEQ_ROOT}</code> (fixed, unique start). Generation ledger of the sequential
runs: {d['ledger_rub']:.2f} ₽. Embeddings: qwen3-embedding-4b via OpenRouter, ${d['embedding_usd']:.4f} by receipts. Built by
<code>scripts/build_sequential_report.py</code> from saved files; numbers in <code>sequential.json</code>.</p>
</main></body></html>'''


def findings(d):
    s, runs = d['summary'], d['runs']
    get = lambda k, f: s.get(k, {}).get(f)
    rows = ''.join(f"<li>Rung {r}: swarm {get(f'{r}/swarm', 'mean_R_N10_in_swarm_range'):.3f}, one-task control {get(f'{r}/private', 'mean_R_N10_in_swarm_range'):.3f}, "
                   f"fixed order {get(f'{r}/fixed', 'mean_R_N10_in_swarm_range'):.3f} over the same token range.</li>" for r in SEEDS)
    bins = ''
    for r in SEEDS:
        full = [x for x in d['series'][f'{r}/fixed'] if x['N'] == 10]
        cells = []
        for lo, hi in ((0.95, 1.01), (0.5, 0.95), (0, 0.5)):
            v = [x['R'] for x in full if lo <= x['same_task_share'] < hi]
            cells.append(f'{st.mean(v):.3f} <small>(n={len(v)})</small>' if v else '—')
        bins += f'<tr><td>{r} · {LABELS[r]}</td>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>'
    correct = ', '.join(f"rung {k.split('/')[0]} {v['correct']}" for k, v in runs.items())
    return f'''<h2>Findings</h2><ol>
<li><b>The swarm solving phase is indistinguishable from the one-task independent control.</b> Mean R over ten-author windows:<ul>{rows}</ul></li>
<li><b>The fixed-order control is far above both on rungs 1–3</b> while all ten agents work on the same task (same-task share 1.0),
and the gap shrinks up the ladder; on rung 4 all tasks share one template and the three conditions coincide.</li>
<li><b>R in the fixed run follows the same-task share.</b> As agents drift apart in speed, the share falls and R falls with it.
Mean R over ten-author windows of the fixed run, by same-task share:</li></ol>
<table><tr><th>Rung</th><th>all on one task</th><th>50–90%</th><th>under 50%</th></tr>{bins}</table>
<p class="note">So R measures how similar the tasks being solved are, not whether the agents communicate. In this design a swarm in which
each agent solves its own task looks like independent agents on different tasks. Correct answers in the fixed runs: {correct}.</p>'''


def run_facts(tokenizer):
    out = {}
    for rung in SEEDS:
        for cond in ('fixed', 'shuffled_unique_start'):
            d = run_dir(rung, cond)
            if not (d/'state.json').exists():
                continue
            s, a = read_json(d/'state.json'), read_json(d/'audit.json') if (d/'audit.json').exists() else {}
            usage = []
            for f in d.glob('calls/*/response.json'):
                u = read_json(f).get('usage') or {}
                if u.get('completion_tokens'):
                    usage.append(u['completion_tokens'])
            out[f'{rung}/{cond}'] = {'status': s['status'], 'answers': sum(len(x['answers']) for x in s['agents'].values()),
                                     'correct': f"{a.get('own_correct', '?')}/{a.get('own_total', '?')}", 'calls': s['calls'],
                                     'interrupted': len(a.get('usage_incomplete_calls', [])),
                                     'mean_solve_tokens': round(st.mean(usage)) if usage else 0}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    plans = plan(tokenizer)
    texts = {k: [w['text'] for wins in v for w in wins] for k, v in plans.items()}
    n = sum(len(t) for t in texts.values()); chars = sum(len(x) for t in texts.values() for x in t)
    print(json.dumps({'conditions': sorted(f'{r}/{c}' for r, c in plans), 'windows': n, 'chars': chars,
                      'estimated_usd_upper': round(chars/2*.02/1e6, 4)}))
    if args.dry_run:
        return
    vectors = {}
    for key, t in texts.items():  # one call per rung/condition keeps every call under the $1 envelope of embed()
        vectors.update(embed(t, OUT, workers=12, cache_dir=SEQ_ROOT/'embedding-cache'))
    results = {f'{r}/{c}': series(v, vectors) for (r, c), v in plans.items()}
    summ = {}
    for rung in SEEDS:
        sw = [r for r in results.get(f'{rung}/swarm', []) if r['N'] == 10]
        x_max = max((r['x'] for r in sw), default=None)
        for cond in STYLE:
            rows = results.get(f'{rung}/{cond}')
            if rows is None:
                continue
            s = summary(rows); near = summary(rows, x_max)
            summ[f'{rung}/{cond}'] = {**s, 'mean_R_N10_in_swarm_range': near['mean_R_N10'],
                                      'same_task_share_in_swarm_range': near['mean_same_task_share_N10'], 'swarm_range_kt': x_max}
    figure(results, OUT/'fig_sequential_zoom.png', x_lim=30, title='First 30 thousand generated tokens per agent')
    figure(results, OUT/'fig_sequential_full.png', title='Whole runs')
    # the budget ledger is cumulative per rung root (both sequential conditions share it): take the largest spent_root_rub per rung
    ledger = sum(max((read_json(run_dir(r, c)/'result.json')['spent_root_rub'] for c in ('fixed', 'shuffled_unique_start')
                      if (run_dir(r, c)/'result.json').exists()), default=0) for r in SEEDS)
    receipts = [read_json(p) for p in (OUT/'embedding-requests').glob('*.response.json')] if (OUT/'embedding-requests').exists() else []
    data = {'width': W, 'series': results, 'summary': summ, 'runs': run_facts(tokenizer),
            'ledger_rub': ledger,
            'embedding_usd': sum((r.get('usage') or {}).get('cost', 0) for r in receipts)}
    write_json(OUT/'sequential.json', data)
    data['findings_html'] = findings(data)
    data['swarm_range_note'] = ', '.join(f"rung {r}: {summ[f'{r}/swarm']['swarm_range_kt']:.1f}k tokens" for r in SEEDS if f'{r}/swarm' in summ)
    (OUT/'sequential.html').write_text(page(data), encoding='utf-8')
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
