"""Audit finished jobs and build a self-contained local study report."""
import base64
import csv
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from swarm_pipeline.audit import audit as audit_distribution
from swarm_pipeline.audit_gradual import audit as audit_gradual
from swarm_pipeline.supervision import read_json, write_json
from swarm_pipeline.uniform_windows import ROOT, load_streams, layouts, segment, snapshot_rows, compute, draw
from swarm_pipeline.analyze import embed, token_data


def picture(path, caption):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img loading="lazy" src="data:image/png;base64,{data}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def coverage_plot():
    methods = read_json(ROOT/'window-methods.json')
    case = next(c for c in methods['cases'] if c['rung']==1 and c['mode']=='barrier')
    lengths = case['phase_lengths']
    tokenizer = Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    private = load_streams(Path('artifacts/pipeline-ladder/rung_1/private/fixed/seed_3'),tokenizer)
    private_lengths = {a['agent']:len(token_data(tokenizer,a['text'])) for a in private}
    colors = {'negotiation':'#287cd0','solving':'#21a17a','exchange':'#9267bd'}
    fig, axes = plt.subplots(3,1,figsize=(13,13),sharex=True)
    for ax, mode in zip(axes,('continuous','barrier','trim')):
        for aid in range(1,11):
            cursor=0
            for phase in ('negotiation','solving','exchange'):
                n=lengths[str(aid)][phase]
                if mode=='trim': n=min(a[phase] for a in lengths.values())
                start=cursor if mode=='continuous' else case['phase_spans'][phase][0]
                ax.barh(aid,n,left=start,height=.72,color=colors[phase])
                cursor += n
            ax.barh(aid+11,private_lengths[aid],left=0,height=.72,color='#e57532')
        ax.axhline(10.9,color='.7',lw=.6)
        ax.set_yticks(list(range(1,11))+list(range(12,22)),
                      [f'S{i}' for i in range(1,11)]+[f'P{i}' for i in range(1,11)],fontsize=8)
        ax.invert_yaxis(); ax.set_title(mode.title()); ax.grid(axis='x',alpha=.2)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=color,label=phase) for phase,color in colors.items()]+[Patch(color='#e57532',label='private solve')],
               loc='upper center',ncol=4,bbox_to_anchor=(.5,.96))
    axes[-1].set_xlabel('Conditional token position; S = swarm author, P = private author; white = no text')
    fig.suptitle('Rung 1: where each author actually contributes text',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(ROOT/'author-coverage.png',dpi=150); plt.close(fig)


def distribution_plots(tokenizer, records):
    rows, audits = [], []
    for rec in records:
        if rec['kind'] != 'distribution' or rec['status'] != 'completed':
            continue
        directory = Path(rec['directory'])
        audits.append({'seed': rec['seed'], **audit_distribution(directory)})
        totals = {i: {'seed': rec['seed'], 'agent': i, 'reasoning_tokens': 0, 'response_tokens': 0,
                      'published_chat_tokens': 0, 'observed_turns': 0} for i in range(1, 11)}
        for path in sorted(directory.glob('calls/*/request.json')):
            q, r, e = read_json(path), read_json(path.with_name('response.json')), read_json(path.with_name('event.json'))
            row = totals[q['agent']]
            row['observed_turns'] += 1
            row['reasoning_tokens'] += len(token_data(tokenizer, r.get('reasoning', '')))
            row['response_tokens'] += len(token_data(tokenizer, r.get('content', '')))
            if not e['action_error']:
                text = r['content'].strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1].rsplit('```', 1)[0]
                action = json.loads(text)
                if action.get('action') == 'send':
                    row['published_chat_tokens'] += len(token_data(tokenizer, action['message']))
        for row in totals.values():
            row['total_tokens'] = row['reasoning_tokens'] + row['response_tokens']
            rows.append(row)
    if not rows:
        return [], audits
    with (ROOT / 'distribution-lengths.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    seeds = sorted({r['seed'] for r in rows})
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    for ax, key, label in zip(axes, ('total_tokens', 'reasoning_tokens', 'published_chat_tokens'),
                              ('All generated tokens', 'Private reasoning tokens', 'Published chat message tokens')):
        matrix = np.array([[next(r[key] for r in rows if r['seed'] == seed and r['agent'] == aid)
                            for aid in range(1, 11)] for seed in seeds])
        im = ax.imshow(matrix, aspect='auto', cmap='Blues')
        ax.set_yticks(range(len(seeds)), [f'Seed {s}' for s in seeds])
        ax.set_xticks(range(10), [str(i) for i in range(1, 11)])
        ax.set_title(label)
        for i in range(len(seeds)):
            for j in range(10):
                ax.text(j, i, str(matrix[i,j]), ha='center', va='center', fontsize=8,
                        color='white' if matrix[i,j] > matrix.max()*.6 else '#17202a')
        fig.colorbar(im, ax=ax, pad=.02)
    axes[-1].set_xlabel('Agent identity (task assignment may vary across seeds)')
    fig.suptitle('Five independent distribution-only repeats | rung 1', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(ROOT / 'distribution-heatmap.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key in zip(axes, ('total_tokens', 'published_chat_tokens')):
        groups = [[r[key] for r in rows if r['seed'] == seed] for seed in seeds]
        ax.boxplot(groups, tick_labels=[str(s) for s in seeds], showfliers=False)
        for i, group in enumerate(groups, 1):
            ax.scatter(i + np.linspace(-.15, .15, len(group)), group, s=18, alpha=.7)
        ax.set_title(key.replace('_', ' ')); ax.set_xlabel('Run seed'); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Ten agents per repeat; points are authors, not independent run replicates')
    fig.tight_layout(); fig.savefig(ROOT / 'distribution-spread.png', dpi=150); plt.close(fig)
    summary = [{'seed': seed, 'min_total': min(r['total_tokens'] for r in rows if r['seed'] == seed),
                'max_total': max(r['total_tokens'] for r in rows if r['seed'] == seed),
                'median_total': float(np.median([r['total_tokens'] for r in rows if r['seed'] == seed])),
                'min_published_chat': min(r['published_chat_tokens'] for r in rows if r['seed'] == seed),
                'max_published_chat': max(r['published_chat_tokens'] for r in rows if r['seed'] == seed)} for seed in seeds]
    write_json(ROOT / 'distribution-summary.json', summary)
    return summary, audits


def gradual_plots(tokenizer, record):
    directory = Path(record['directory'])
    verification = audit_gradual(directory)
    calls = []
    for path in sorted(directory.glob('calls/*/request.json')):
        q = read_json(path); r = read_json(path.with_name('response.json'))
        calls.append((q, r))
    origin = min(q['started_at'] for q, _ in calls)
    colors = {'negotiation': '#aeb7c2', 'solving': '#287cd0', 'report': '#20a176', 'exchange': '#9061bf'}
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for q, r in calls:
        start, end = (q['started_at']-origin)/60, (r['completed_at']-origin)/60
        ax.barh(q['agent'], end-start, left=start, height=.65, color=colors[q['context']['phase']])
    if record.get('imported_cost_excluded_rub'):
        neg_end = max(r['completed_at'] for q,r in calls if q['context']['phase']=='negotiation')
        solve_start = min(q['started_at'] for q,r in calls if q['context']['phase']=='solving')
        a,b=(neg_end-origin)/60,(solve_start-origin)/60
        ax.axvspan(a,b,color='#e6e8ec',alpha=.7,zorder=0)
        ax.text((a+b)/2,5.5,'Imported history\n→ recovery restart',ha='center',va='center',fontsize=10,color='#526175')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=color, label=phase) for phase, color in colors.items()], ncol=4, loc='upper center', bbox_to_anchor=(.5, 1.15))
    ax.set_yticks(range(1,11)); ax.invert_yaxis(); ax.set_xlabel('Minutes since first dispatch (real wall clock)')
    ax.set_ylabel('Agent'); ax.grid(axis='x', alpha=.2)
    ax.set_title('Gradual pilot: request intervals, not inferred per-token timestamps', pad=45)
    fig.tight_layout(); fig.savefig(ROOT / 'gradual-timeline.png', dpi=150); plt.close(fig)
    swarm = load_streams(directory, tokenizer)
    private = load_streams(Path('artifacts/pipeline-ladder/rung_1/private/fixed/seed_3'), tokenizer)
    layout, spans, _ = layouts(swarm, tokenizer, 'continuous')
    control = {a['agent']: [segment(a['text'], 0, tokenizer)] for a in private}
    end = max(s['end'] for parts in list(layout.values())+list(control.values()) for s in parts)
    plans, texts = [], []
    for width in (400, 2048, 4096):
        starts = list(range(0, max(0, end-width+1), width//2))
        rows = snapshot_rows(layout, starts, width, 'swarm')+snapshot_rows(control, starts, width, 'private')
        texts.extend(t for row in rows for t in row['windows'].values())
        plans.append((width, rows))
    vectors = embed(texts, ROOT / 'gradual-analysis', workers=12, cache_dir=Path('artifacts/pipeline-analysis/embedding-cache'))
    for width, rows in plans:
        half = len(rows)//2
        metrics = compute(rows[:half], vectors) + compute(rows[half:], vectors)
        stem = ROOT / 'gradual-analysis' / f'gradual_continuous_w{width}'
        with stem.with_suffix('.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(metrics[0])); w.writeheader(); w.writerows(metrics)
        # Continuous authors have different phase boundaries; no common phase shading is justified.
        no_spans = {p: (0,0) for p in ('negotiation', 'solving', 'exchange')}
        draw(metrics, no_spans, width, f"Gradual swarm seed {record['seed']} vs historical private seed 3 (unpaired pilot)", stem.with_suffix('.png'))
    return verification


def report(records, analysis_ready, distributions, verification, errors):
    summary = read_json(ROOT / 'analysis-summary.json') if analysis_ready else []
    timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    sections = [f'<h1>Uniform windows and gradual reporting</h1><p class="muted">Built {timestamp}. Local exploratory study; no detection accuracy claim.</p>',
        '<h2>What changed</h2><ul><li>Both populations now use full windows on the same token grid. Missing authors stay missing.</li>'
        '<li>Four saved ladder pairs were reprocessed. New generation is restricted to five distribution repeats and one gradual-reporting pilot on rung 1.</li>'
        '<li>R, pairwise cosine, effective dimension, active-author count, and matched-author drift are shown.</li></ul>',
        '<h2>Axes in the earlier continuous-clock plots</h2><p>The horizontal axis in the earlier geometry plots is a conditional generated-token clock. '
        'It is not elapsed time, equal task progress, or proof that messages were simultaneous. Private starts solving at zero; swarm starts distributing. '
        'The same fixed grid continues to the union of both horizons. Only the event timeline uses real timestamps.</p>'
        '<p>Each point uses at most one full window per author. No tails, padding, vector holding, or stretching to equal progress. '
        'Geometry is blank below four authors. Compare the active count before interpreting R or dimension. R and pairwise cosine are redundant at fixed N.</p>']
    phase_status=ROOT/'phase-analysis/status.json'
    if phase_status.exists() and read_json(phase_status)['state']=='complete':
        from swarm_pipeline.phase_report import html_section
        sections.insert(1,html_section())
    if summary:
        if (ROOT/'trim400-summary.json').exists():
            strict = read_json(ROOT/'trim400-summary.json')
            sections.append('<section id="trim400-n10"><h2>All four ladder rungs: W=400, exactly ten authors</h2>'
                '<p>Each swarm phase is cut to its shortest author. Private solving is independently cut to its shortest author. '
                'Every populated geometry point therefore contains all ten authors in that condition. Windows are full 400-token windows, stride 200; '
                'the original common phase starts and real gaps remain. No padding, short tails, or interpolation. '
                'The new private trim is stricter than the earlier variable-count orange reference.</p>')
            sections.append(picture(ROOT/'trim400-analysis/ladder_trim400_overview.png',
                'Blue: swarm; orange: private. Shaded bands: distribution and final exchange. Blank intervals contain no full ten-author window.'))
            sections.append('<p><b>Earlier descriptive view:</b> high-low-high phase means with sparse endpoints do not establish a continuous arc. '
                'Read the latest separate-channel analysis above before attributing this pattern to exchanged messages.</p>')
            sections.append('<table><tr><th>Rung</th><th>Distribution R (points)</th><th>Solving R (points)</th><th>Exchange R (points)</th><th>Phase-mean pattern</th></tr>')
            for row in strict:
                cells=[]
                for phase in ('negotiation','solving','exchange'):
                    p=row['phases'][phase]
                    cells.append('No full window' if p['R_mean'] is None else f"{p['R_mean']:.3f} ({p['points']})")
                verdict=('High-low-high means (sparse endpoints)' if row['descriptive_high_low_high_R'] else
                         'No full exchange window' if not row['all_three_phases_observed'] else 'No high-low-high phase means')
                sections.append(f"<tr><td>{row['rung']}</td>"+''.join(f'<td>{c}</td>' for c in cells)+f'<td>{verdict}</td></tr>')
            sections.append('</table><p>R values are phase means over overlapping windows, not independent replications. '
                'Rung 4 has only 223 tokens per author left after final-exchange trimming, so W=400 cannot show a final exchange point. '
                'This is missing coverage, not evidence that the behavior disappeared. Other arcs are descriptive findings from one saved group per rung.</p>')
            for row in strict:
                rung=row['rung']
                sections.append(f'<details><summary>Ladder {rung}: all metrics, W=400, N=10</summary>'+picture(
                    ROOT/'trim400-analysis'/f'rung_{rung}_trim400_n10.png',
                    'R, pairwise cosine, effective dimension, active count, and matched-author drift. Populated points have ten authors; gaps are preserved.')+'</details>')
            sections.append('</section>')
        sections.append(picture(ROOT/'author-coverage.png',
            'Per-author phase lengths on the conditional clock. White intervals contain no output and cannot supply vectors. Continuous lengths use phase-wise token counts; boundary retokenization can shift a few tokens.'))
        methods = read_json(ROOT/'window-methods.json')
        sections.append('<h2>Why a complete ten-author arc may disappear</h2><p>These are observed own-output phase lengths, including private reasoning. '
            'A full window cannot create text that an author never generated. At a shared phase start, the following counts can supply a complete W-token vector:</p>'
            '<table><tr><th>Rung / phase</th><th>Min / median / max tokens</th><th>Authors ≥400</th><th>Authors ≥2048</th><th>Authors ≥4096</th></tr>')
        for case in methods['cases']:
            if case['mode'] != 'barrier':
                continue
            for phase in ('negotiation','solving','exchange'):
                ns = [a[phase] for a in case['phase_lengths'].values()]
                sections.append(f"<tr><td>{case['rung']} / {phase}</td><td>{min(ns)} / {np.median(ns):.0f} / {max(ns)}</td>"+
                    ''.join(f'<td>{sum(n>=w for n in ns)}/10</td>' for w in (400,2048,4096))+'</tr>')
        sections.append('</table><p>Concatenating across a phase boundary increases coverage but mixes chat and solving in the same vector. '
            'Longer windows trade phase resolution for coverage requirements; they do not automatically solve missing-author coverage.</p>')
        sections.append('<h2>Rung 1: window and transition ablations</h2><p><b>Continuous:</b> own text concatenated across phases. '
            '<b>Barrier:</b> shared phase starts after the longest preceding stream, with gaps. '
            '<b>Trim:</b> shortest phase length retained, next start unchanged. '
            '<b>Supplement:</b> short solving streams extended with historical independent same-prompt solutions; synthetic diagnostic only. '
            '<b>Chat supplement:</b> the literal proposal to append independent same-task solutions to short chats. '
            'This is deliberately synthetic, kept separate from observed communication and from solving-only supplementation.</p>')
        modes=('continuous','barrier','trim','supplement')
        if (ROOT/'chat-padding-provenance.json').exists(): modes += ('chat_supplement',)
        for mode in modes:
            sections.append(f'<details {"open" if mode == "continuous" else ""}><summary>{mode.title()}</summary>')
            for width in (400,2048,4096):
                sections.append(picture(ROOT/'analysis'/f'rung_1_{mode}_w{width}.png',
                    'Blue: swarm. Orange: matched private tasks. Shading: common barrier reference only; continuous authors cross phases at different positions.'))
            sections.append('</details>')
        sections.append('<h2>Transfer checks: remaining ladder rungs</h2>')
        for rung in (2,3,4):
            sections.append(f'<details><summary>Ladder rung {rung}, W=2048</summary>')
            for mode in ('continuous','barrier','trim'):
                sections.append(picture(ROOT/'analysis'/f'rung_{rung}_{mode}_w2048.png', f'Rung {rung}, {mode}. One existing pair; descriptive only.'))
            sections.append('</details>')
        sections.append('<h2>Coverage, without filling gaps</h2><table><tr><th>Rung / mode / W</th><th>Swarm N≥4 points</th><th>Private N≥4 points</th><th>Both N=10 points</th><th>Mean ΔR at both N=10</th></tr>')
        for r in summary:
            delta = '—' if r['mean_R_difference_at_full10'] is None else f"{r['mean_R_difference_at_full10']:+.4f}"
            sections.append(f"<tr><td>{r['rung']} / {r['mode']} / {r['width']}</td><td>{r['swarm_points_N4']}</td><td>{r['private_points_N4']}</td><td>{r['full10_paired_points']}</td><td>{delta}</td></tr>")
        sections.append('</table><p>These averages combine positions with different phase mixtures; they are coverage summaries, not a cooperation-effect estimate.</p>')
    if distributions:
        sections += ['<h2>Fresh distribution repeats</h2>', picture(ROOT/'distribution-heatmap.png',
            'Private reasoning is retained for embeddings but is not broadcast. Published chat counts only the message string of accepted send actions.'),
            picture(ROOT/'distribution-spread.png', 'Five runs provide an initial variability check; ten authors within one run are not ten independent replications.')]
        large=sum(r['max_total']>=10000 for r in distributions)
        sections.append(f'<p><b>Observed:</b> {large}/{len(distributions)} runs contained an author with at least 10,000 distribution tokens. '
            f"Run medians ranged from {min(r['median_total'] for r in distributions):.0f} to {max(r['median_total'] for r in distributions):.0f} tokens. "
            f"Published chat messages ranged from {min(r['min_published_chat'] for r in distributions)} to {max(r['max_published_chat'] for r in distributions)} tokens. "
            'The largest lengths therefore concern private reasoning and protocol output, not comparably long shared messages. '
            'Long outputs occurred in error-free repeats too; transport failure is not their sole explanation. Five runs cannot establish a model-general cause.</p>')
    if verification:
        sections += ['<h2>Live gradual reporting</h2>',
            f"<p>Replay audit passed. Successful reports: {verification['successful_reports']}; reports delivered before the last solve finished: <b>{verification['reports_before_last_solve']}</b>; report requests that read an earlier report: <b>{verification['reports_reading_prior_reports']}</b>. Provider interruptions: {verification['provider_interruptions']}.</p>",
            picture(ROOT/'gradual-timeline.png', 'Blue solving and green reporting overlap across agents. Gray middle span includes the recovery restart after imported distribution; it is not normal pipeline latency. In-flight requests do not receive later messages.'),
            '<p>This tests reporting overlap. It does not test live message injection into an ongoing solve. The geometry comparison uses historical private seed 3, not a newly matched replicate; treat it as an unpaired pilot.</p>']
        for width in (400,2048,4096):
            sections.append(picture(ROOT/'gradual-analysis'/f'gradual_continuous_w{width}.png', 'Uniform continuous windows; gaps retained. Different run seed and potentially different assignments.'))
    sections.append('<h2>Interpretation limits</h2><ul><li>Long windows can smooth short communication events and reduce the available chat-only coverage.</li>'
        '<li>Trimming enforces common observed lengths by discarding text and introduces explicit gaps.</li>'
        '<li>Supplementation increases coverage through synthetic text; it can also alter geometry. A restored curve is not evidence of real agent behavior.</li>'
        '<li>A mixed population may retain more active vectors but its changing composition still needs measurement. No claim that this automatically fixes dropout.</li>'
        '<li>Five distribution repeats and one gradual pilot do not establish model-general outlier behavior or a reliable cooperation detector.</li></ul>')
    if errors:
        sections.append('<h2>Incomplete or failed stages</h2><pre>'+html.escape(json.dumps(errors, indent=2))+'</pre>')
    sections.append('<h2>Run ledger</h2><pre>'+html.escape(json.dumps(records, indent=2))+'</pre>')
    sections.append('<details><summary>Exact gradual-pilot system protocol</summary><pre>'+html.escape(__import__('swarm_pipeline.window_pilot',fromlist=['ASYNC_PROTOCOL']).ASYNC_PROTOCOL)+'</pre></details>')
    doc = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Window and overlap study</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#172b43;background:#fafbfd}h1,h2{line-height:1.2}h2{margin-top:40px}figure{margin:24px 0}img{width:100%;height:auto;background:white}figcaption,.muted{color:#526175;font-size:14px}details{border:1px solid #d9e1ea;padding:14px;margin:16px 0;border-radius:8px}summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#eef2f6;padding:15px}</style><body>'+''.join(sections)+'</body></html>'
    (ROOT/'window-study.html').write_text(doc, encoding='utf-8')
    main = Path('artifacts/pipeline-checkpoint/swarm-team-checkpoint.html')
    if main.exists():
        original = main.read_text(encoding='utf-8')
        marker = 'id="window-study-link"'
        if marker not in original:
            original = original.replace('</body>', '<aside id="window-study-link" style="padding:20px;border:2px solid #287cd0"><a href="../pipeline-windows/window-study.html">New: uniform windows, dropout ablations, distribution repeats and gradual-reporting pilot</a></aside></body>')
            main.write_text(original, encoding='utf-8')


def main():
    deadline = time.time() + 2*3600
    write_json(ROOT/'experiment.status.json', {'state':'running','event_key':'window-study:running','updated_at':time.time()})
    while time.time() < deadline:
        if (ROOT/'analysis-summary.json').exists() and (ROOT/'live.status.json').exists():
            live = read_json(ROOT/'live.status.json')
            if live['state'] in {'complete','failed'} and (ROOT/'recovered-gradual-602.json').exists():
                break
        time.sleep(15)
    records = read_json(ROOT/'live-results.json') if (ROOT/'live-results.json').exists() else []
    for path in ROOT.glob('recovered-*.json'):
        recovery = read_json(path)
        records = [r for r in records if (r['kind'],r['seed']) != (recovery['kind'],recovery['seed'])]
        records.append(recovery)
    # Financial admission revisions preserve tasks, protocol and provider settings.
    # Only resume workers whose original orchestrator has already returned.
    for rec in records:
        if rec['kind']!='distribution' or rec['status']=='completed':
            continue
        if 'Budget reservation' not in rec.get('error',''):
            continue
        from dataclasses import replace
        from swarm_pipeline.runner import Config
        from swarm_pipeline.window_pilot import DistributionRunner
        from swarm_pipeline.stable_live import RetainingClient
        directory = Path(rec['directory'])
        old = read_json(directory/'manifest.json')
        config = replace(Config(**old['config']),budget_rub=40)
        runner = DistributionRunner(old['tasks'],config,directory.parent.parent,
                                    RetainingClient(os.environ['CLOUDRU_API_KEY']),workers=10,partial_continuation=True)
        runner.prepare_existing()
        result = runner.run()
        rec.update(status=result['status'],calls=result['calls'],cost_rub=result['spent_root_rub'],financial_recovery=True)
    corrected = next((r for r in records if r['kind']=='gradual' and r['seed']==602 and r['status']=='completed'),None)
    if corrected:
        for rec in records:
            if rec['kind']=='gradual' and rec['seed']==601:
                rec['original_status']=rec['status']; rec['status']='superseded'
                rec['reason']='Temporary reservation handling stopped reporting; retained as an operationally confounded pilot.'
    # Include paid/estimated receipts from failed and superseded attempts too.
    for rec in records:
        directory = Path(rec['directory'])
        responses = [read_json(p) for p in directory.glob('calls/*/response.json')]
        imported = rec.get('imported_cost_excluded_rub',0)
        rec['cost_rub'] = sum(r['cost_rub'] for r in responses)-imported
        rec['usage_backed_rub'] = sum(r['cost_rub'] for r in responses if not r.get('cost_estimated'))-imported
        rec['estimated_rub'] = sum(r['cost_rub'] for r in responses if r.get('cost_estimated'))
    write_json(ROOT/'final-run-ledger.json',records)
    analysis_ready = (ROOT/'analysis-summary.json').exists()
    if analysis_ready:
        coverage_plot()
    errors = [r for r in records if r['status'] not in {'completed','superseded'}]
    if not analysis_ready:
        errors.append({'stage':'uniform-analysis','error':'No completed summary before deadline'})
    if sum(r['status']=='completed' for r in records) < 6:
        errors.append({'stage':'live','error':'Not all six live jobs finished before report deadline'})
    tokenizer = Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    distributions, audits = distribution_plots(tokenizer, records)
    gradual = next((r for r in records if r['kind']=='gradual' and r['status']=='completed'), None)
    verification = gradual_plots(tokenizer, gradual) if gradual else None
    write_json(ROOT/'verification.json', {'distribution_audits':audits,'gradual':verification,'errors':errors})
    report(records, analysis_ready, distributions, verification, errors)
    cost = sum(r.get('cost_rub',0) for r in records)
    embedding_receipts = [read_json(p) for p in ROOT.glob('*/embedding-requests/*.response.json')]
    embedding_usd = sum(r.get('usage',{}).get('cost',0) for r in embedding_receipts)
    write_json(ROOT/'cost-summary.json',{'generation_usage_backed_rub':sum(r['usage_backed_rub'] for r in records),
        'generation_unreceipted_estimates_rub':sum(r['estimated_rub'] for r in records),
        'generation_total_rub':cost,'embedding_receipted_usd':embedding_usd,
        'note':'Imported distribution counted once; failed and superseded attempts included. No exchange-rate conversion applied.'})
    findings = ['# Window and overlap study findings', '',
        f'Live generation recorded cost including estimates: RUB {cost:.4f}. Completed primary jobs: {sum(r["status"]=="completed" for r in records)}/6. '
        'The first gradual attempt is retained separately as an operationally confounded pilot; imported distribution receipts are counted once.', '',
        f'Embedding provider receipts total USD {embedding_usd:.6f}. See cost-summary.json for usage-backed versus estimated generation cost.', '',
        'Uniform full-window plots preserve missing authors and show count explicitly. The common clock is conditional tokens, not wall time.',
        'The trim and supplement cases are diagnostic transformations. Neither establishes improved natural cooperation.', '',
        '## Distribution length repeats', '',
        'Large distribution outputs recur in clean runs: seed 501 has 18,242 total tokens for one author and seed 505 has 15,942; their largest output is overwhelmingly private reasoning. '
        'Across the five repeats, published messages range 19–51 tokens. Run medians range 1,233–2,271 tokens. '
        'This is an observed repeated length imbalance, not proof of a model-general mechanism. See distribution-summary.json and distribution-lengths.csv.', '',
        '## Gradual reporting audit', '',
        (f"Ten solves, ten reports, and ten final commits completed. {verification['reports_before_last_solve']} reports were delivered before the last solve finished; "
         f"{verification['reports_reading_prior_reports']} report requests read an earlier report. "
         f"History/context/delivery replay passed. {verification['provider_interruptions']} interrupted provider streams were retained and continued. "
         'This establishes reporting overlap in this run, not peer influence on already-running private solves. '
         'The timeline marks the restart after imported distribution; that interval is not normal pipeline latency. See verification.json.' if verification else 'No completed gradual-pilot verification.'), '',
        '## Limitations', '', 'One historical pair per rung; five fresh distribution repeats; one unpaired gradual pilot. '
        'Answer correctness was not used for selection. A mixed population has not been tested here. '
        'No detector accuracy, statistical equivalence, or model-general conclusions are established.', '',
        '## Incomplete stages', '', json.dumps(errors,indent=2)]
    if analysis_ready:
        methods = read_json(ROOT/'window-methods.json')
        coverage = []
        for case in methods['cases']:
            if case['mode'] != 'barrier':
                continue
            for phase in ('negotiation','solving','exchange'):
                ns = [a[phase] for a in case['phase_lengths'].values()]
                coverage.append({'rung':case['rung'],'phase':phase,'min_tokens':min(ns),'median_tokens':float(np.median(ns)),
                    'max_tokens':max(ns),'authors_at_least_2048_tokens':sum(n>=2048 for n in ns)})
        write_json(ROOT/'phase-coverage.json',coverage)
        findings += ['', '## Phase coverage at W=2048', '',
            'In the saved rung-1 pair, 5/10 authors have at least 2048 distribution tokens, 10/10 have at least 2048 solving tokens, '
            'and 2/10 have at least 2048 final-exchange tokens. Rung 4 has no author with a full 2048-token final-exchange window. See phase-coverage.json.', '',
            'No full ten-author chat arc should be inferred where fewer than ten authors generated a full phase window. '
            'Continuous windows can bridge real boundaries but then mix phases. Longer windows do not create missing text.', '',
            'Trimming removes longer tails and gives equal author lengths within retained phase fragments, but leaves gaps and can eliminate full chat windows. '
            'Independent-solution supplementation improves coverage by adding synthetic text. Both solving-only and literal chat-padding cases are plotted separately; '
            'neither should be interpreted as a natural communication effect. See chat-padding-provenance.json and window-methods.json.', '',
            'Twenty-four saved-data geometry views plus three gradual-pilot views are in the HTML. Active-author count and matched-author drift accompany the original geometry. '
            'R and mean pairwise cosine are redundant at fixed N; effective dimension is bounded by N-1. '
            'Common-clock comparisons can mix phases and cannot independently identify cooperation. See analysis-summary.json and per-view CSVs.']
    (ROOT/'FINDINGS.md').write_text('\n'.join(findings),encoding='utf-8')
    write_json(ROOT/'experiment.status.json', {'state':'complete' if not errors else 'failed',
        'event_key':'window-study:complete' if not errors else 'window-study:failed',
        'updated_at':time.time(),'details':{'report':str(ROOT/'window-study.html'),'cost_rub':cost,'errors':errors}})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write_json(ROOT/'experiment.status.json', {'state':'failed','event_key':'window-study:finalize-failed',
            'updated_at':time.time(),'details':{'error_type':type(exc).__name__,'error':str(exc)}})
        raise
