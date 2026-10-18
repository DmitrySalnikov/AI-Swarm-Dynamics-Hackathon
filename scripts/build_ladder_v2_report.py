"""Collect every set_v2 ladder figure into one self-contained HTML page.

Reads only the outputs saved by scripts/build_ladder_v2_figures.py; makes no generation or embedding calls.
"""
import base64
import csv
from datetime import datetime, timezone
import glob
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_ladder_v2_figures import LABELS, OUT, ROOT, SEEDS
from swarm_pipeline.supervision import read_json

CHANNEL_NAMES = {'reasoning': 'reasoning only', 'visible_output': 'visible output (JSON protocol)', 'payload': 'accepted payload only'}


def figure(path, caption):
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return (f'<figure><img loading="lazy" alt="{html.escape(caption[:120])}" src="data:image/png;base64,{data}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def facts():
    with (OUT/'author-lengths.csv').open(encoding='utf-8') as f:
        rows = [{k: (int(v) if v.lstrip('-').isdigit() else v) for k, v in r.items()} for r in csv.DictReader(f)]
    swarm = [r for r in rows if r['condition'] == 'swarm']; private = [r for r in rows if r['condition'] == 'private']
    contrasts = read_json(OUT/'phase-analysis/contrasts.json')
    reasoning = [c for c in contrasts if c['channel'] == 'reasoning']
    deltas = [c['mean_swarm_minus_private_R'] for c in reasoning if c['mean_swarm_minus_private_R'] is not None]
    w200 = {c['rung']: c['mean_swarm_minus_private_R'] for c in reasoning if c['width'] == 200}
    commit = {c['rung']: c for c in read_json(OUT/'phase-analysis/coverage.json')
              if c['channel'] == 'reasoning' and c['phase'] == 'exchange'}
    receipts = [read_json(p) for p in glob.glob(str(OUT/'embedding-requests/*.response.json'))]
    cost = sum((r.get('usage') or {}).get('cost', 0) for r in receipts)
    coverage = read_json(OUT/'coverage-facts.json'); timelines = read_json(OUT/'timeline-facts.json')
    peak_private = max(private, key=lambda r: r['solving_reasoning'])
    peak_distribution = max(swarm, key=lambda r: r['negotiation_reasoning'])
    return {
        'rows': rows, 'reasoning': reasoning, 'w200': w200, 'delta_min': min(deltas), 'delta_max': max(deltas),
        'chat_min': min(r['negotiation_payload'] for r in swarm), 'chat_max': max(r['negotiation_payload'] for r in swarm),
        'peak_private': peak_private, 'peak_distribution': peak_distribution,
        'commit_min': {r: commit[r]['min_tokens'] for r in SEEDS},
        'commit_short': {r: sum(v < 100 for v in commit[r]['lengths'].values()) for r in SEEDS},
        'trim_removed': {p: (min(c['trim_removed_share'][p] for c in coverage), max(c['trim_removed_share'][p] for c in coverage))
                         for p in ('negotiation', 'solving', 'exchange')},
        'solve_spread': (min(t['swarm_solving_end_spread_minutes'] for t in timelines), max(t['swarm_solving_end_spread_minutes'] for t in timelines)),
        'timelines': timelines, 'receipts': len(receipts), 'cost': cost, 'trim400': read_json(OUT/'trim400-summary.json'),
    }


def contrast_table(f):
    head = '<tr><th>Rung</th>'+''.join(f'<th>W={w}</th>' for w in (100, 200, 400))+'</tr>'
    body = ''
    for rung in SEEDS:
        cells = ''
        for width in (100, 200, 400):
            c = next(x for x in f['reasoning'] if x['rung'] == rung and x['width'] == width)
            m = c['phase_means_R_N10']
            value = 'no window' if m['negotiation'] is None else f"{m['negotiation']-m['solving']:+.3f}"
            star = '*' if m['negotiation'] is not None and c['phase_points_N10']['negotiation'] < 3 else ''
            cells += f"<td>{value}{star}<br><small>ΔR swarm−private {c['mean_swarm_minus_private_R']:+.4f}</small></td>"
        body += f'<tr><td><b>{rung}</b> · {LABELS[rung]}</td>{cells}</tr>'
    return f'<table><thead>{head}</thead><tbody>{body}</tbody></table>'


def main():
    f = facts()
    date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    pd, pp = f['peak_distribution'], f['peak_private']
    tr = f['trim_removed']
    w200 = ', '.join(f"{f['w200'][r]:+.3f}" for r in SEEDS)
    s = [f'''<header><div class="eyebrow">SET_V2 LADDER · {date}</div>
    <h1>Swarm and private control on the set_v2 ladder</h1>
    <p class="lead">The plots from the post-call report, rebuilt on the new four-rung runs: author lengths, solving geometry, phase windows and real request timing.</p>
    <div class="stats"><div><b>4</b><span>rungs, 10 tasks each</span></div><div><b>8</b><span>runs: one swarm + one private control per rung</span></div><div><b>10</b><span>agents per run</span></div><div><b>${f['cost']:.2f}</b><span>embedding receipts for these plots</span></div></div>
    <nav><a href="#lengths">Lengths</a><a href="#solving">Swarm vs private</a><a href="#phases">Phases & windows</a><a href="#detail">Detailed phase plots</a><a href="#timing">Timing</a><a href="#scope">Scope</a></nav></header>
    <section class="summary"><h2>Summary</h2><ul>
    <li><b>Solving:</b> swarm and private reasoning curves stay close on every rung. Mean swarm−private R at W=200 is {w200} for rungs 1–4. Across W=100/200/400 it ranges from {f['delta_min']:+.3f} to {f['delta_max']:+.3f}.</li>
    <li><b>Phases:</b> distribution reasoning is more aligned than solving reasoning on rungs 1–3, and the gap grows with window width. On rung 4 (same template) the gap almost disappears because solving R is already high.</li>
    <li><b>Lengths:</b> published distribution messages are only {f['chat_min']}–{f['chat_max']} tokens. Private reasoning varies widely; the longest author wrote {pp['solving_reasoning']:,} reasoning tokens (rung {pp['rung']}, private agent {pp['agent']}).</li>
    <li><b>Commit:</b> no ten-author reasoning window exists in the commit phase at any width: on each rung, {min(f['commit_short'].values())}–{max(f['commit_short'].values())} agents write fewer than 100 reasoning tokens there.</li></ul></section>
    <section id="lengths"><h2>1. How much does each agent write?</h2>''']
    s.append(figure(OUT/'fig01_author_lengths.png', 'Bars are individual authors. Reasoning includes every observed turn, including interrupted (transport-error) responses that the pipeline retains. Chat counts accepted send-message text in the distribution phase only.'))
    s.append(f'''<p class="interpret"><b>Result:</b> published chat is {f['chat_min']}–{f['chat_max']} tokens per agent. The tallest blue segment (rung {pd['rung']}, agent {pd['agent']}: {pd['negotiation_reasoning']:,} distribution reasoning tokens) comes mostly from one interrupted provider response whose reasoning is kept; rung 1, agent 6 combines an interrupted response with one long accepted distribution turn. The {pp['solving_reasoning']:,}-token private author on rung {pp['rung']} also spans several requests.</p>
    <details><summary>Length spread per rung</summary>'''+figure(OUT/'fig02_length_spread.png', 'Ten authors per run, one run per condition and rung. Points are authors, not independent run replicates.')+'</details></section>')
    s.append('''<section id="solving"><h2>2. Swarm vs private during solving</h2>
    <p>The swarm distributes tasks, then each agent solves privately. The control gives each agent the same assigned task without any exchange. Blue is swarm; orange is the private control.</p>''')
    s.append(figure(OUT/'fig03_solving_comparison.png', 'Reasoning only, W=200, stride 100. Each curve is trimmed to its shortest author, so every point has ten authors; only window starts present in both conditions are drawn. ΔR is descriptive; no significance test.'))
    t4 = next(t for t in f['trim400'] if t['rung'] == 4)
    s.append(f'''<p class="interpret"><b>Result:</b> mean ΔR at W=200 is {w200} (rungs 1–4); over all three widths it stays within {f['delta_min']:+.3f}…{f['delta_max']:+.3f} and changes sign. No cooperation-specific difference is visible during solving.</p>
    <details><summary>Full-arc view with shortest-phase trimming (W=400, reasoning + visible output)</summary>'''+figure(OUT/'fig04_trim400_overview.png',
        f"Shared phase starts; every populated point has ten authors; shaded bands are distribution and commit. Distribution and commit give only single windows on rungs 1–3; on rung 4 the trimmed commit is shorter than W ({t4['phases']['exchange']['points']} windows).")+'</details></section>')
    s.append(f'''<section id="phases"><h2>3. Phases, windows and author coverage</h2>
    <p>Each phase is measured on its own local token clock, separately for reasoning, visible output and accepted payload. Full windows only, stride W/2.</p>''')
    s.append(figure(OUT/'fig07_window_sensitivity.png', 'Swarm reasoning, ten authors. * the distribution mean rests on one or two windows. n/a: no ten-author distribution window at that width. Overlapping windows are not independent experiments.'))
    s.append('<p>Distribution − solving mean R, and the swarm−private solving difference, from <code>phase-analysis/contrasts.json</code>:</p>'+contrast_table(f))
    s.append(f'''<p class="interpret"><b>Result:</b> the phase contrast depends on window width and shrinks along the ladder. Trimming to the shortest author removes {tr['negotiation'][0]:.0%}–{tr['negotiation'][1]:.0%} of distribution text, {tr['solving'][0]:.0%}–{tr['solving'][1]:.0%} of solving text and {tr['exchange'][0]:.0%}–{tr['exchange'][1]:.0%} of commit text (reasoning + visible output).</p>
    <h3>Where each author actually writes text</h3>''')
    s.append(figure(OUT/'fig06_coverage_rung1.png', 'Rung 1. Continuous own streams, shared phase starts with gaps, and shortest-author trimming. S = swarm, P = private; reasoning + visible output; token positions, not wall time.'))
    s.append('<details><summary>Author coverage for rungs 2–4</summary>'+''.join(
        figure(OUT/f'fig06_coverage_rung{r}.png', f'Rung {r} · {LABELS[r]}.') for r in (2, 3, 4))+'</details></section>')
    s.append('''<section id="detail"><h2>4. Detailed phase plots: geometry with author count</h2>''')
    s.append(figure(OUT/'phase-analysis/rung_2_reasoning_w200.png', 'Example: rung 2, W=200, reasoning only. Solid lines keep all ten authors; dashed lines use however many authors are still writing. Each phase starts at token zero. Full tails are in the CSV files.'))
    for rung in SEEDS:
        items = ''.join(figure(OUT/f'phase-analysis/rung_{rung}_{c}_w{w}.png', f'Rung {rung} · {CHANNEL_NAMES[c]} · W={w}.')
                        for c in ('reasoning', 'visible_output', 'payload') for w in (100, 200, 400))
        s.append(f'<details><summary>Rung {rung} · {LABELS[rung]}: all channels and widths (9 plots)</summary>{items}</details>')
    s.append('</section>')
    lo, hi = f['solve_spread']
    s.append(f'''<section id="timing"><h2>5. Real request timing</h2>''')
    s.append(figure(OUT/'fig09_timelines.png', 'Real request start/finish timestamps. Each rung has its own origin; the private control ran after the swarm. Gaps inside a row are retries or continuations after interrupted responses.'))
    s.append(f'''<p class="interpret"><b>Result:</b> swarm agents finish solving between {lo:.1f} and {hi:.1f} minutes apart depending on the rung, so some agents idle while others are still solving.</p></section>
    <section id="scope"><h2>Scope and sources</h2>
    <p>One swarm run and one matched private control per rung (seeds 3, 82, 83, 84) from <code>artifacts/pipeline-ladder-v2</code>. Model: DeepSeek-V4-Flash on cloud.ru. Embeddings: qwen/qwen3-embedding-4b via OpenRouter, {f['receipts']} requests, ${f['cost']:.4f} by receipts. No answer-correctness filtering.</p>
    <p class="note">Not reproduced: the “three fixes” figure. Its supplementation and early-reporting panels need separate experiments that were not run on set_v2. The coverage figures above show the trimming panel.</p>
    <p class="note">Sources: figures/author-lengths.csv, coverage-facts.json, timeline-facts.json, trim400-summary.json, phase-analysis/*.csv, contrasts.json, coverage.json. Built by scripts/build_ladder_v2_figures.py and scripts/build_ladder_v2_report.py.</p></section>''')
    css = '''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f1f4f8;color:#1c2d42;font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1160px;margin:28px auto;background:white;padding:36px 42px;border-radius:14px}h1{font-size:36px;line-height:1.15;margin:10px 0}h2{font-size:24px;line-height:1.25;margin:0 0 12px}h3{margin:26px 0 8px}p{margin:10px 0}.eyebrow{font-size:12px;letter-spacing:1.4px;color:#65768b}.lead{font-size:19px;color:#51657e}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.stats div{padding:14px;background:#edf3fb;border-radius:9px}.stats b{display:block;font-size:28px;line-height:1.1;color:#2466ae}.stats span{font-size:13px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}nav a{padding:7px 12px;border:1px solid #dce5ef;border-radius:20px;font-size:13px;text-decoration:none}a{color:#205a9d}section{margin-top:32px;scroll-margin-top:15px}.summary{padding:20px;background:#eef4fa;border-left:4px solid #347bc6}.summary li{margin:7px 0}ul,ol{padding-left:22px}.interpret{padding:12px 15px;background:#eef6f2;border-left:3px solid #218268}.note,figcaption{color:#5c6e83;font-size:13px}figure{margin:20px 0}img{display:block;width:100%;height:auto}figcaption{margin-top:7px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:10px;text-align:left;vertical-align:top;border:1px solid #dce4ed}th{background:#eef2f7}small{color:#5c6e83}code{font-size:13px;background:#f1f4f8;padding:1px 4px;border-radius:4px}details{border:1px solid #dce4ed;padding:13px 16px;border-radius:8px;margin-top:18px}summary{font-weight:650;cursor:pointer}@media(max-width:760px){main{margin:0;padding:23px 16px;border-radius:0}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}table{font-size:12px}td,th{padding:6px}}@media print{body{background:white}main{margin:0;max-width:none;padding:10px}nav{display:none}figure,table,.summary{break-inside:avoid}}'''
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>set_v2 ladder figures</title><style>'+css+'</style></head><body><main>'+''.join(s)+'</main></body></html>')
    target = OUT/'ladder-v2-figures.html'; target.write_text(document, encoding='utf-8')
    print(json.dumps({'report': str(target), 'bytes': target.stat().st_size, 'images': document.count('<img ')}))


if __name__ == '__main__':
    main()
