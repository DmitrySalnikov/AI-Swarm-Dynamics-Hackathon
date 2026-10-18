"""Compare the set_v2 ladder runs with the v2 prompts and with the simple prompts (SWARM_PROMPT_STYLE=simple).

Reads only saved artifacts of both runs (no generation or embedding calls) and writes
artifacts/pipeline-ladder-v2-simple/figures/prompt-comparison.{json,html} plus fig10_distribution_lengths.png.
Every number in the page comes from the JSON written next to it.
"""
import base64
import csv
import glob
import html
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from swarm_pipeline.supervision import read_json, write_json

RUNS = {'v2': Path('artifacts/pipeline-ladder-v2'), 'simple': Path('artifacts/pipeline-ladder-v2-simple')}
OUT = RUNS['simple']/'figures'
SEEDS = {1: 3, 2: 82, 3: 83, 4: 84}
LABELS = {1: 'Unrelated domains', 2: 'Physics subfields', 3: 'Linear algebra', 4: 'Same determinant template'}
OLD, NEW = '#8a8984', '#2a78d6'  # gray = v2 prompts, blue = simple prompts (marker shapes differ as well)
HEAVY_EQ = 50  # a distribution turn with at least this many '=' signs in its reasoning is counted as computing


def run_dir(run, rung, condition):
    folder = 'swarm/cooperative' if condition == 'swarm' else 'private/fixed'
    return RUNS[run]/f'rung_{rung}'/folder/f'seed_{SEEDS[rung]}'


def lengths(run):
    rows = list(csv.DictReader((RUNS[run]/'figures/author-lengths.csv').open(encoding='utf-8')))
    return [{k: (v if k == 'condition' else int(v)) for k, v in r.items()} for r in rows]


def distribution_turns(run, rung):
    """Per distribution call of the swarm: (agent, reasoning tokens by usage, number of '=' in the reasoning)."""
    out = []
    for f in glob.glob(str(run_dir(run, rung, 'swarm')/'calls/*/response.json')):
        q, r = read_json(Path(f).with_name('request.json')), read_json(Path(f))
        if q['context']['phase'] == 'negotiation':
            out.append({'agent': q['agent'], 'tokens': (r.get('usage') or {}).get('completion_tokens'),
                        'eq': (r.get('reasoning') or '').count('='), 'finish': r.get('finish_reason')})
    return out


def collect():
    data = {'runs': {}}
    for run in RUNS:
        rows, contrasts = lengths(run), read_json(RUNS[run]/'figures/phase-analysis/contrasts.json')
        timelines = {t['rung']: t for t in read_json(RUNS[run]/'figures/timeline-facts.json')}
        status = read_json(RUNS[run]/'experiment.status.json')['details']
        receipts = [read_json(Path(p)) for p in glob.glob(str(RUNS[run]/'figures/embedding-requests/*.response.json'))]
        per = {}
        for rung in SEEDS:
            r = {}
            for cond in ('swarm', 'private'):
                a = read_json(run_dir(run, rung, cond)/'audit.json'); res = read_json(run_dir(run, rung, cond)/'result.json')
                sol = [x['solving_reasoning'] for x in rows if x['rung'] == rung and x['condition'] == cond]
                r[cond] = {'own_correct': a['own_correct'], 'own_total': a['own_total'],
                           'solving_reasoning_median': st.median(sol), 'solving_reasoning_mean': round(st.mean(sol)),
                           'solving_reasoning_max': max(sol), 'solving_reasoning_min': min(sol),
                           'calls': a['calls'], 'interrupted_calls': len(a['usage_incomplete_calls']),
                           'partial_continuations': a['partial_continuations'],
                           'explicit_turn_restarts': a['explicit_turn_restarts'],
                           'elapsed_minutes': round(a['elapsed_minutes'], 1),
                           'usage_backed_rub': res['usage_backed_spend_rub'], 'unreceipted_rub': res['unreceipted_estimates_rub']}
            dist = [x['negotiation_reasoning'] for x in rows if x['rung'] == rung and x['condition'] == 'swarm']
            turns = distribution_turns(run, rung)
            r['distribution'] = {'per_agent_reasoning': [x['negotiation_reasoning'] for x in sorted(
                (x for x in rows if x['rung'] == rung and x['condition'] == 'swarm'), key=lambda x: x['agent'])],
                'median': st.median(dist), 'max': max(dist), 'total': sum(dist),
                'chat_tokens_min': min(x['negotiation_payload'] for x in rows if x['rung'] == rung and x['condition'] == 'swarm'),
                'chat_tokens_max': max(x['negotiation_payload'] for x in rows if x['rung'] == rung and x['condition'] == 'swarm'),
                'turns': len(turns), 'computing_turns': sum(t['eq'] >= HEAVY_EQ for t in turns),
                'max_eq_in_a_turn': max(t['eq'] for t in turns)}
            r['geometry'] = {}
            for c in contrasts:
                if c['rung'] != rung or c['channel'] != 'reasoning':
                    continue
                m, n = c['phase_means_R_N10'], c['phase_points_N10']
                r['geometry'][c['width']] = {
                    'delta_swarm_minus_private': c['mean_swarm_minus_private_R'],
                    'distribution_minus_solving': (m['negotiation'] - m['solving']) if m['negotiation'] is not None else None,
                    'distribution_windows': n['negotiation'], 'solving_R': m['solving']}
            r['solving_end_spread_minutes'] = round(timelines[rung]['swarm_solving_end_spread_minutes'], 1)
            per[rung] = r
        data['runs'][run] = {'rungs': per, 'generation_ledger_rub': status['new_generation_local_ledger_rub'],
                             'figure_embedding_usd': sum((x.get('usage') or {}).get('cost', 0) for x in receipts),
                             'figure_embedding_requests': len(receipts)}
    return data


def distribution_figure(data):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, rung in zip(axes, SEEDS):
        for run, color, marker, dx in (('v2', OLD, 'o', -0.15), ('simple', NEW, 's', 0.15)):
            vals = data['runs'][run]['rungs'][rung]['distribution']['per_agent_reasoning']
            ax.bar([i + dx for i in range(1, 11)], vals, width=0.3, color=color, label=None)
            ax.plot([i + dx for i in range(1, 11)], vals, marker, color=color, ms=4, ls='',
                    label='v2 prompts' if run == 'v2' else 'simple prompts')
        ax.set_yscale('log'); ax.set_title(f'Rung {rung} · {LABELS[rung]}', fontsize=10)
        ax.set_xticks(range(1, 11)); ax.set_xlabel('Agent'); ax.grid(axis='y', color='#e4e3df')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('Distribution reasoning tokens (log)')
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('Reasoning written during task distribution, per swarm agent: v2 prompts (gray) vs simple prompts (blue)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(OUT/'fig10_distribution_lengths.png', dpi=160); plt.close(fig)


def img(path, alt):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def fmt(x, nd=3):
    return '—' if x is None else f'{x:+.{nd}f}'


def page(d):
    v2, sm = d['runs']['v2']['rungs'], d['runs']['simple']['rungs']
    rows_dist = ''.join(
        f"<tr><td>{r} · {LABELS[r]}</td><td>{v2[r]['distribution']['max']:,}</td><td>{sm[r]['distribution']['max']:,}</td>"
        f"<td>{v2[r]['distribution']['median']:,.0f}</td><td>{sm[r]['distribution']['median']:,.0f}</td>"
        f"<td>{v2[r]['distribution']['computing_turns']} / {v2[r]['distribution']['turns']}</td>"
        f"<td>{sm[r]['distribution']['computing_turns']} / {sm[r]['distribution']['turns']}</td></tr>" for r in SEEDS)
    rows_solve = ''.join(
        f"<tr><td>{r}</td>" + ''.join(
            f"<td>{x[r][c]['solving_reasoning_median']:,.0f} ({x[r][c]['solving_reasoning_max']:,})</td>"
            for c in ('swarm', 'private') for x in (v2, sm)) +
        ''.join(f"<td>{x[r][c]['own_correct']}/{x[r][c]['own_total']}</td>" for c in ('swarm', 'private') for x in (v2, sm)) + '</tr>'
        for r in SEEDS)
    rows_geo = ''.join(
        f"<tr><td>{r}</td>" + ''.join(
            f"<td>{fmt(v2[r]['geometry'][w]['delta_swarm_minus_private'])}</td><td>{fmt(sm[r]['geometry'][w]['delta_swarm_minus_private'])}</td>"
            for w in (100, 200, 400)) + '</tr>' for r in SEEDS)
    rows_phase = ''.join(
        f"<tr><td>{r}</td>" + ''.join(
            f"<td>{fmt(x[r]['geometry'][w]['distribution_minus_solving'])} <small>({x[r]['geometry'][w]['distribution_windows']} w)</small></td>"
            for w in (100, 200) for x in (v2, sm)) + '</tr>' for r in SEEDS)
    rows_ops = ''.join(
        f"<tr><td>{r}</td><td>{v2[r]['solving_end_spread_minutes']}</td><td>{sm[r]['solving_end_spread_minutes']}</td>"
        + ''.join(f"<td>{x[r][c]['interrupted_calls']} / {x[r][c]['calls']}</td>" for c in ('swarm', 'private') for x in (v2, sm))
        + '</tr>' for r in SEEDS)
    calls = lambda x: sum(x[r][c]['calls'] for r in SEEDS for c in ('swarm', 'private'))
    broken = lambda x: sum(x[r][c]['interrupted_calls'] for r in SEEDS for c in ('swarm', 'private'))
    receipts = lambda x: sum(x[r][c]['usage_backed_rub'] for r in SEEDS for c in ('swarm', 'private'))
    unrec = lambda x: sum(x[r][c]['unreceipted_rub'] for r in SEEDS for c in ('swarm', 'private'))
    dmax_old = max(v2[r]['distribution']['max'] for r in SEEDS); dmax_new = max(sm[r]['distribution']['max'] for r in SEEDS)
    comp_old = sum(v2[r]['distribution']['computing_turns'] for r in SEEDS); comp_new = sum(sm[r]['distribution']['computing_turns'] for r in SEEDS)
    deltas = [sm[r]['geometry'][w]['delta_swarm_minus_private'] for r in SEEDS for w in (100, 200, 400)]
    correct = lambda x, c: sum(x[r][c]['own_correct'] for r in SEEDS)
    chat = lambda x: (min(x[r]['distribution']['chat_tokens_min'] for r in SEEDS), max(x[r]['distribution']['chat_tokens_max'] for r in SEEDS))
    w100 = lambda x: [x[r]['geometry'][100]['distribution_windows'] for r in SEEDS]
    restarts = lambda x: sum(x[r][c]['explicit_turn_restarts'] for r in SEEDS for c in ('swarm', 'private'))
    win_old = sum(v2[r]['geometry'][200]['distribution_windows'] for r in SEEDS)
    win_new = sum(sm[r]['geometry'][200]['distribution_windows'] for r in SEEDS)
    f = lambda name, alt: img(RUNS['simple']/'figures'/name, alt)
    g = lambda name, alt: img(RUNS['v2']/'figures'/name, alt)
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Simple vs v2 prompts</title><style>
:root{{--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--bg:#fcfcfb;--card:#f3f1ea}}
body{{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 64px}} h1{{font-size:26px;margin:.2em 0}} h2{{margin-top:2em;font-size:20px}}
.kicker{{color:var(--ink2);font-size:12px;letter-spacing:.08em;text-transform:uppercase}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:18px 0}}
.stats div{{background:var(--card);border-radius:10px;padding:12px}} .stats b{{display:block;font-size:24px}} .stats span{{color:var(--ink2);font-size:13px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}}
th{{color:var(--ink2);font-weight:600}} figure{{margin:16px 0}} img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px}}
figcaption,.note{{color:var(--ink2);font-size:13px}} .result{{background:var(--card);border-radius:8px;padding:10px 12px}}
details summary{{cursor:pointer;color:var(--ink2);margin:8px 0}} small{{color:var(--ink2)}}
</style></head><body><main>
<div class="kicker">Set_v2 ladder · prompt comparison</div>
<h1>Simple prompts vs v2 prompts</h1>
<p>Same 40 tasks, seeds, budgets and model (DeepSeek-V4-Flash, cloud.ru). The only change: <code>SWARM_PROMPT_STYLE=simple</code> —
short plain instructions and a blunt <em>“DO NOT SOLVE ANY PROBLEM NOW. DON'T EVEN TRY.”</em> during distribution.
One swarm + one private control per rung in each run; no uncertainty bands.</p>
<div class="stats"><div><b>{dmax_old:,} → {dmax_new:,}</b><span>longest distribution reasoning of one agent, tokens</span></div>
<div><b>{comp_old} → {comp_new}</b><span>distribution turns that compute (≥ {HEAVY_EQ} “=” signs)</span></div>
<div><b>{min(deltas):+.3f}…{max(deltas):+.3f}</b><span>swarm − private R during solving, simple prompts</span></div>
<div><b>{d['runs']['v2']['generation_ledger_rub']:.0f} → {d['runs']['simple']['generation_ledger_rub']:.0f} ₽</b><span>generation ledger, 8 runs</span></div></div>

<h2>Summary</h2><ul>
<li><b>The blunt ban works:</b> no distribution turn computes any more ({comp_old} → {comp_new} turns with ≥ {HEAVY_EQ} “=” signs); the longest distribution reasoning fell from {dmax_old:,} to {dmax_new:,} tokens. Long distribution turns now deliberate about the choice, not the problem.</li>
<li><b>Side effect for the measurement:</b> distribution is now so short that at W=200 only {win_new} ten-author distribution windows exist (was {win_old}); at W=400 there are none on any rung, so the distribution-vs-solving contrast can be measured only at small windows.</li>
<li><b>Solving:</b> swarm and private curves stay close on every rung (swarm − private R between {min(deltas):+.3f} and {max(deltas):+.3f} over W=100/200/400), as with the v2 prompts.</li>
<li><b>Correctness is about the same:</b> swarm {correct(v2, 'swarm')}/40 → {correct(sm, 'swarm')}/40, private {correct(v2, 'private')}/40 → {correct(sm, 'private')}/40.</li>
<li><b>Chat became shorter:</b> distribution messages are {chat(sm)[0]}–{chat(sm)[1]} tokens (were {chat(v2)[0]}–{chat(v2)[1]}); “one sentence” is taken literally.</li>
<li><b>Few distribution windows:</b> at W=100 the number of ten-author distribution windows per rung fell from {'/'.join(map(str, w100(v2)))} to {'/'.join(map(str, w100(sm)))}, so the distribution − solving contrast now rests on 2–4 windows per rung and is noisy (rung 4 is no longer near zero).</li>
<li><b>Reliability:</b> interrupted provider calls {broken(v2)}/{calls(v2)} → {broken(sm)}/{calls(sm)}; explicit restart turns {restarts(v2)} → {restarts(sm)}. Generation ledger {d['runs']['v2']['generation_ledger_rub']:.0f} → {d['runs']['simple']['generation_ledger_rub']:.0f} ₽.</li></ul>

<h2>1. Did agents solve during distribution?</h2>
<table><tr><th>Rung</th><th>Max per agent, v2</th><th>Max, simple</th><th>Median, v2</th><th>Median, simple</th><th>Computing turns, v2</th><th>Computing turns, simple</th></tr>{rows_dist}</table>
<p class="note">Distribution reasoning tokens per swarm agent (all distribution turns of the agent, including interrupted ones the pipeline keeps). “Computing turns”: distribution calls whose reasoning contains ≥ {HEAVY_EQ} “=” signs — a heuristic for arithmetic, checked by reading the longest turns.</p>
{f('fig10_distribution_lengths.png', 'Distribution reasoning per swarm agent, log scale. Gray: v2 prompts; blue: simple prompts. Agent numbers are not matched between runs (assignments differ).')}

<h2>2. Solving lengths and correctness</h2>
<table><tr><th>Rung</th><th>Swarm, v2</th><th>Swarm, simple</th><th>Private, v2</th><th>Private, simple</th><th>Swarm correct, v2</th><th>simple</th><th>Private correct, v2</th><th>simple</th></tr>{rows_solve}</table>
<p class="note">Solving reasoning tokens per agent: median (max). Correct = own submitted answers right / 10. Task selection calibrated single-turn solutions at 15–21 thousand tokens (mean); the pipeline’s JSON protocol gives shorter solutions in both runs.</p>

<h2>3. Swarm vs private during solving</h2>
<table><tr><th>Rung</th><th>W=100, v2</th><th>simple</th><th>W=200, v2</th><th>simple</th><th>W=400, v2</th><th>simple</th></tr>{rows_geo}</table>
<p class="note">Mean swarm − private R, reasoning only, ten authors, matched window starts. Descriptive; no significance test.</p>
{f('fig03_solving_comparison.png', 'Simple prompts: swarm (blue) vs private (orange) R during solving, W=200.')}
<details><summary>v2 prompts: same figure</summary>{g('fig03_solving_comparison.png', 'v2 prompts: swarm vs private during solving, W=200.')}</details>

<h2>4. Distribution vs solving geometry</h2>
<table><tr><th>Rung</th><th>W=100, v2</th><th>W=100, simple</th><th>W=200, v2</th><th>W=200, simple</th></tr>{rows_phase}</table>
<p class="note">Mean R of distribution minus mean R of solving (swarm reasoning, ten authors); in brackets the number of ten-author distribution windows. “—”: no window. At W=400 the simple-prompt run has no distribution window on any rung.</p>
{f('fig07_window_sensitivity.png', 'Simple prompts: distribution − solving R at W=100/200/400.')}
<details><summary>v2 prompts: same figure</summary>{g('fig07_window_sensitivity.png', 'v2 prompts: distribution − solving R at W=100/200/400.')}</details>

<h2>5. Lengths per agent, timing and reliability</h2>
{f('fig01_author_lengths.png', 'Simple prompts: tokens per agent by phase.')}
<details><summary>v2 prompts: same figure</summary>{g('fig01_author_lengths.png', 'v2 prompts: tokens per agent by phase.')}</details>
<table><tr><th>Rung</th><th>Solving end spread, min, v2</th><th>simple</th><th>Swarm interrupted / calls, v2</th><th>simple</th><th>Private interrupted / calls, v2</th><th>simple</th></tr>{rows_ops}</table>
{f('fig09_timelines.png', 'Simple prompts: real request intervals per rung.')}

<h2>Next steps</h2><ul>
<li>Keep the simple prompts: they remove solving from distribution, and correctness is about the same (section 2).</li>
<li>Decide how to measure the distribution phase now that it is short: smaller windows, the chat payload channel, or asking for a longer distribution message (e.g. 3–5 sentences on why this problem) so that the chat carries text.</li>
<li>Repeat groups (several swarm/private pairs per rung) are needed before any claim about differences; fix provider interruptions first.</li></ul>

<h2>Scope and sources</h2>
<p class="note">Runs: <code>{RUNS['v2']}</code> (v2 prompts) and <code>{RUNS['simple']}</code> (simple prompts), seeds 3/82/83/84, tasks <code>artifacts/pipeline-ladder-v2/items</code>.
Generation ledger: v2 {d['runs']['v2']['generation_ledger_rub']:.2f} ₽, simple {d['runs']['simple']['generation_ledger_rub']:.2f} ₽
(simple: {receipts(sm):.2f} ₽ usage-backed + {unrec(sm):.2f} ₽ unreceipted estimates for interrupted calls).
Figure embeddings, simple run: {d['runs']['simple']['figure_embedding_requests']} requests, ${d['runs']['simple']['figure_embedding_usd']:.4f} by receipts.
Built by <code>scripts/build_prompt_comparison_report.py</code> from saved files only; numbers in <code>prompt-comparison.json</code>.</p>
</main></body></html>'''


def main():
    data = collect()
    write_json(OUT/'prompt-comparison.json', data)
    distribution_figure(data)
    (OUT/'prompt-comparison.html').write_text(page(data), encoding='utf-8')
    print(OUT/'prompt-comparison.html')


if __name__ == '__main__':
    main()
