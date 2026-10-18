"""Rebuild the post-call report figures for the set_v2 ladder runs (one swarm/private pair per rung).

Without --embed only token counts, author coverage and wall-clock timelines are drawn (no provider calls).
--dry-run reports how many windows the embedding figures would need and how many are uncached.
--embed requests the missing qwen3-embedding-4b vectors (OpenRouter) and draws the geometry figures.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tokenizers import Tokenizer

from swarm_pipeline.analyze import embed, token_data, vector_key
from swarm_pipeline.phase_windows import CHANNELS, PHASES, draw_case, load_channels, make_rows
from swarm_pipeline.supervision import read_json, write_json
from swarm_pipeline.uniform_windows import compute, layouts, load_streams, segment, snapshot_rows

# LADDER_ROOT selects another run with the same layout (e.g. the simple-prompt rerun); default unchanged
ROOT = Path(os.environ.get('LADDER_ROOT', 'artifacts/pipeline-ladder-v2'))
OUT = ROOT/'figures'
SEEDS = {1: 3, 2: 82, 3: 83, 4: 84}
LABELS = {1: 'Unrelated domains', 2: 'Physics subfields', 3: 'Linear algebra', 4: 'Same determinant template'}
WIDTHS = (100, 200, 400)
COLORS = {'negotiation': '#347bc6', 'solving': '#24a078', 'exchange': '#9467bd'}
PHASE_NAMES = {'negotiation': 'Distribution', 'solving': 'Solving', 'exchange': 'Commit'}
SWARM, PRIVATE = '#2476d2', '#e57532'
CACHE = ROOT/'embedding-cache'


def run_dir(rung, condition):
    folder = 'swarm/cooperative' if condition == 'swarm' else 'private/fixed'
    return ROOT/f'rung_{rung}'/folder/f'seed_{SEEDS[rung]}'


def lengths_table(tokenizer):
    """Per author: reasoning, visible output and published payload tokens in every phase."""
    rows, channels = [], {}
    for rung in SEEDS:
        for condition in ('swarm', 'private'):
            streams, _ = load_channels(run_dir(rung, condition))
            channels[(rung, condition)] = streams
            for aid in range(1, 11):
                row = {'rung': rung, 'condition': condition, 'agent': aid}
                for phase in PHASES:
                    for channel in CHANNELS:
                        row[f'{phase}_{channel}'] = len(token_data(tokenizer, streams[(phase, channel)][aid]))
                row['reasoning_total'] = sum(row[f'{p}_reasoning'] for p in PHASES)
                row['generated_total'] = row['reasoning_total']+sum(row[f'{p}_visible_output'] for p in PHASES)
                rows.append(row)
    with (OUT/'author-lengths.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows, channels


def lengths_bars(rows):
    """Figure 1 analogue: per-author reasoning by phase, private reasoning, published chat."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.4), sharey='row', sharex=True)
    top = max(r['reasoning_total'] for r in rows)
    for col, rung in enumerate(SEEDS):
        swarm = sorted([r for r in rows if r['rung'] == rung and r['condition'] == 'swarm'], key=lambda r: r['agent'])
        private = sorted([r for r in rows if r['rung'] == rung and r['condition'] == 'private'], key=lambda r: r['agent'])
        x = np.arange(1, 11)
        ax = axes[0, col]; bottom = np.zeros(10)
        for phase in PHASES:
            values = np.array([r[f'{phase}_reasoning'] for r in swarm])
            ax.bar(x, values, bottom=bottom, color=COLORS[phase], width=.8, label=PHASE_NAMES[phase]); bottom += values
        ax.set_title(f'Rung {rung} · {LABELS[rung]}', fontsize=10)
        ax = axes[1, col]; values = [r['solving_reasoning'] for r in private]
        ax.bar(x, values, color=PRIVATE, width=.8)
        ax = axes[2, col]; values = [r['negotiation_payload'] for r in swarm]
        ax.bar(x, values, color='#e38a3b', width=.8)
        for row, series in ((0, [r['reasoning_total'] for r in swarm]), (1, [r['solving_reasoning'] for r in private])):
            peak = max(series); idx = series.index(peak)+1
            axes[row, col].text(idx, peak+top*.02, f'{peak:,}', ha='center', fontsize=8)
        for row in range(3):
            axes[row, col].grid(axis='y', alpha=.2); axes[row, col].set_axisbelow(True)
        axes[2, col].set_xticks(x, [str(i) for i in x], fontsize=7); axes[2, col].set_xlabel('Agent', fontsize=9)
    axes[0, 0].set_ylim(0, top*1.12); axes[1, 0].set_ylim(0, top*1.12)
    axes[0, 0].set_ylabel('Swarm reasoning tokens\n(stacked by phase)')
    axes[1, 0].set_ylabel('Private-control\nreasoning tokens')
    axes[2, 0].set_ylabel('Published distribution\nchat tokens')
    axes[0, 0].legend(fontsize=8, loc='upper left')
    fig.suptitle('set_v2: how much each agent writes · one swarm and one private control per rung, ten agents each', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(OUT/'fig01_author_lengths.png', dpi=155); plt.close(fig)


def lengths_spread(rows):
    """Figure 2 analogue: box plus points per rung."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    rng = np.random.default_rng(0)
    for ax, title in zip(axes, ('All generated tokens (reasoning + visible output)', 'Published distribution chat tokens')):
        for i, rung in enumerate(SEEDS):
            for j, (condition, color) in enumerate((('swarm', SWARM), ('private', PRIVATE))):
                if title.startswith('Published') and condition == 'private':
                    continue
                key = 'generated_total' if title.startswith('All') else 'negotiation_payload'
                values = [r[key] for r in rows if r['rung'] == rung and r['condition'] == condition]
                pos = i+1+(j-.5)*.36 if title.startswith('All') else i+1
                ax.boxplot([values], positions=[pos], widths=.3, showfliers=False,
                           medianprops={'color': color}, boxprops={'color': '#333'})
                ax.scatter(pos+rng.uniform(-.1, .1, len(values)), values, s=16, alpha=.75, color=color,
                           label=condition if i == 0 else None)
        ax.set_xticks(range(1, 5), [f'Rung {r}' for r in SEEDS]); ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    axes[0].legend(fontsize=9)
    fig.suptitle('Ten agents per run; points are authors, not independent run replicates')
    fig.tight_layout(); fig.savefig(OUT/'fig02_length_spread.png', dpi=155); plt.close(fig)


def coverage(tokenizer):
    """Figure 6 analogue for every rung: continuous / shared phase starts / trimmed."""
    facts = []
    for rung in SEEDS:
        swarm = sorted(load_streams(run_dir(rung, 'swarm'), tokenizer), key=lambda a: a['agent'])
        private = load_streams(run_dir(rung, 'private'), tokenizer)
        lengths = {a['agent']: {p: len(token_data(tokenizer, a['phases'][p])) for p in PHASES} for a in swarm}
        private_lengths = {a['agent']: len(token_data(tokenizer, a['text'])) for a in private}
        _, spans, _ = layouts(swarm, tokenizer, 'barrier')
        minima = {p: min(l[p] for l in lengths.values()) for p in PHASES}
        fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)
        for ax, mode in zip(axes, ('continuous', 'barrier', 'trim')):
            for aid in range(1, 11):
                cursor = 0
                for p in PHASES:
                    n = minima[p] if mode == 'trim' else lengths[aid][p]
                    start = cursor if mode == 'continuous' else spans[p][0]
                    ax.barh(aid, n, left=start, height=.72, color=COLORS[p]); cursor += n
                ax.barh(aid+11, private_lengths[aid], left=0, height=.72, color=PRIVATE)
            ax.axhline(10.9, color='.7', lw=.6)
            ax.set_yticks(list(range(1, 11))+list(range(12, 22)),
                          [f'S{i}' for i in range(1, 11)]+[f'P{i}' for i in range(1, 11)], fontsize=8)
            ax.invert_yaxis(); ax.set_title({'continuous': 'Continuous', 'barrier': 'Shared phase starts',
                                            'trim': 'Trimmed to the shortest author'}[mode]); ax.grid(axis='x', alpha=.2)
        fig.legend(handles=[Patch(color=COLORS[p], label=PHASE_NAMES[p].lower()) for p in PHASES]+
                   [Patch(color=PRIVATE, label='private solve')], loc='upper center', ncol=4, bbox_to_anchor=(.5, .96))
        axes[-1].set_xlabel('Conditional token position (reasoning + visible output); S = swarm author, P = private author; white = no text')
        fig.suptitle(f'Rung {rung} ({LABELS[rung]}): where each author actually contributes text', fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(OUT/f'fig06_coverage_rung{rung}.png', dpi=150); plt.close(fig)
        total = {p: sum(l[p] for l in lengths.values()) for p in PHASES}
        facts.append({'rung': rung, 'phase_min': minima, 'phase_max': {p: max(l[p] for l in lengths.values()) for p in PHASES},
                      'trim_removed_share': {p: 1-10*minima[p]/total[p] for p in PHASES},
                      'private_min': min(private_lengths.values()), 'private_max': max(private_lengths.values())})
    write_json(OUT/'coverage-facts.json', facts)
    return facts


def timelines():
    """Figure 9 analogue: real request intervals of every call, swarm and private, per rung."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 17))
    facts = []
    for ax, rung in zip(axes, SEEDS):
        origin = None; calls = []
        for condition, offset in (('swarm', 0), ('private', 11)):
            for path in sorted(run_dir(rung, condition).glob('calls/*/request.json')):
                q = read_json(path); r = read_json(path.with_name('response.json'))
                calls.append((condition, offset+q['agent'], q['context']['phase'], q['started_at'], r.get('completed_at') or q['started_at']))
        origin = min(c[3] for c in calls)
        for condition, y, phase, start, end in calls:
            color = PRIVATE if condition == 'private' else COLORS[phase]
            ax.barh(y, (end-start)/60, left=(start-origin)/60, height=.65, color=color, edgecolor='white', linewidth=.3)
        ax.axhline(10.9, color='.7', lw=.6)
        ax.set_yticks(list(range(1, 11))+list(range(12, 22)), [f'S{i}' for i in range(1, 11)]+[f'P{i}' for i in range(1, 11)], fontsize=8)
        ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
        ax.set_title(f'Rung {rung} · {LABELS[rung]}', fontsize=11)
        swarm_end = max(c[4] for c in calls if c[0] == 'swarm'); private_end = max(c[4] for c in calls if c[0] == 'private')
        solve_end = {c[1]: c[4] for c in calls if c[0] == 'swarm' and c[2] == 'solving'}
        facts.append({'rung': rung, 'swarm_minutes': (swarm_end-min(c[3] for c in calls if c[0] == 'swarm'))/60,
                      'private_minutes': (private_end-min(c[3] for c in calls if c[0] == 'private'))/60,
                      'swarm_solving_end_spread_minutes': (max(solve_end.values())-min(solve_end.values()))/60,
                      'swarm_calls': sum(c[0] == 'swarm' for c in calls), 'private_calls': sum(c[0] == 'private' for c in calls)})
    axes[-1].set_xlabel('Minutes since the first request of the rung (real wall clock)')
    fig.legend(handles=[Patch(color=COLORS[p], label=PHASE_NAMES[p].lower()) for p in PHASES]+[Patch(color=PRIVATE, label='private solve')],
               loc='upper center', ncol=4, bbox_to_anchor=(.5, .982))
    fig.suptitle('Request intervals, not inferred per-token timestamps', fontsize=13, y=.995)
    fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUT/'fig09_timelines.png', dpi=140); plt.close(fig)
    write_json(OUT/'timeline-facts.json', facts)
    return facts


def plan_windows(tokenizer, channels):
    """Same windows as swarm_pipeline.phase_windows and scripts/build_trim400_ladder, on the v2 runs."""
    phase_plans, trim_plans, texts, cov = [], [], [], []
    for rung in SEEDS:
        sources = {c: channels[(rung, c)] for c in ('swarm', 'private')}
        for channel in CHANNELS:
            for width in WIDTHS:
                groups = []
                for phase in PHASES:
                    for condition in ('swarm', 'private'):
                        if condition == 'private' and phase != 'solving':
                            continue
                        available, trim, info = make_rows(sources[condition][(phase, channel)], tokenizer, width, condition)
                        if width == WIDTHS[0]:
                            cov.append({'rung': rung, 'phase': phase, 'channel': channel, 'condition': condition, **info})
                        for mode, rows in (('available', available), ('trim', trim)):
                            groups.append((phase, condition, mode, rows))
                            if mode == 'available':
                                texts.extend(t for r in rows for t in r['windows'].values())
                phase_plans.append((rung, channel, width, groups))
        swarm = load_streams(run_dir(rung, 'swarm'), tokenizer)
        private = load_streams(run_dir(rung, 'private'), tokenizer)
        layout, spans, _ = layouts(swarm, tokenizer, 'trim')
        control = {a['agent']: [segment(a['text'], 0, tokenizer)] for a in private}
        private_min = min(s['end'] for parts in control.values() for s in parts)
        for parts in control.values():
            parts[0]['end'] = private_min; parts[0]['offsets'] = parts[0]['offsets'][:private_min]
        end = max([s['end'] for parts in layout.values() for s in parts]+[private_min])
        starts = list(range(0, max(0, end-400+1), 200))
        rows = snapshot_rows(layout, starts, 400, 'swarm')+snapshot_rows(control, starts, 400, 'private')
        texts.extend(t for row in rows for t in row['windows'].values())
        trim_plans.append((rung, spans, layout, rows))
    return phase_plans, trim_plans, texts, cov


def seed_cache(texts):
    """Reuse vectors already embedded for the per-rung analyses (same model and key)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = {vector_key(t) for t in texts if t.strip()}
    sources = [p for r in SEEDS for p in [ROOT/f'rung_{r}/analysis/embedding-cache']]
    linked = 0
    for key in keys:
        target = CACHE/f'{key}.npy'
        if target.exists():
            continue
        for source in sources:
            if (source/f'{key}.npy').exists():
                target.write_bytes((source/f'{key}.npy').read_bytes()); linked += 1; break
    todo = [t for k, t in {vector_key(t): t for t in texts if t.strip()}.items() if not (CACHE/f'{k}.npy').exists()]
    return {'unique_windows': len(keys), 'copied_from_rung_caches': linked, 'uncached': len(todo),
            'uncached_chars': sum(len(t) for t in todo),
            'estimated_usd_at_0.02_per_M_tokens': sum(len(t)/2*.02/1e6 for t in todo)}


def phase_figures(phase_plans, cov, vectors):
    """Figures 3, 7, 8 analogues: phase-local windows by channel and width."""
    target = OUT/'phase-analysis'; target.mkdir(parents=True, exist_ok=True)
    write_json(target/'coverage.json', cov)
    summaries, records_all = [], []
    for rung, channel, width, groups in phase_plans:
        case = []
        for phase, condition, mode, rows in groups:
            records = compute(rows, vectors)
            for r in records:
                r.update(rung=rung, channel=channel, width=width, phase=phase, mode=mode)
            case.extend(records)
            full = [r for r in records if r['n_active'] == 10]
            summaries.append({'rung': rung, 'channel': channel, 'width': width, 'phase': phase, 'condition': condition, 'mode': mode,
                              'points_N10': len(full), 'mean_R_N10': float(np.mean([r['R'] for r in full])) if full else None})
        if case:
            with (target/f'rung_{rung}_{channel}_w{width}.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(case[0])); writer.writeheader(); writer.writerows(case)
        draw_case(rung, channel, width, case, [c for c in cov if c['rung'] == rung and c['channel'] == channel],
                  target/f'rung_{rung}_{channel}_w{width}.png')
        records_all.extend(case)
    contrasts = []
    for rung in SEEDS:
        for channel in CHANNELS:
            for width in WIDTHS:
                means = {p: next(s for s in summaries if s['rung'] == rung and s['channel'] == channel and s['width'] == width
                                 and s['phase'] == p and s['condition'] == 'swarm' and s['mode'] == 'trim') for p in PHASES}
                solve = [r for r in records_all if r['rung'] == rung and r['channel'] == channel and r['width'] == width
                         and r['phase'] == 'solving' and r['mode'] == 'trim' and r['n_active'] == 10]
                sw = {r['start']: r for r in solve if r['condition'] == 'swarm'}
                pr = {r['start']: r for r in solve if r['condition'] == 'private'}
                common = sorted(set(sw) & set(pr))
                contrasts.append({'rung': rung, 'channel': channel, 'width': width,
                                  'phase_means_R_N10': {p: means[p]['mean_R_N10'] for p in PHASES},
                                  'phase_points_N10': {p: means[p]['points_N10'] for p in PHASES},
                                  'matched_solving_points_N10': len(common),
                                  'mean_swarm_minus_private_R': float(np.mean([sw[x]['R']-pr[x]['R'] for x in common])) if common else None})
    write_json(target/'summary.json', summaries); write_json(target/'contrasts.json', contrasts)

    # Figure 3: solving, reasoning only, W=200, trimmed, matched starts.
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharey=True)
    for rung, ax in zip(SEEDS, axes.flat):
        rows = [r for r in records_all if r['rung'] == rung and r['channel'] == 'reasoning' and r['width'] == 200
                and r['phase'] == 'solving' and r['mode'] == 'trim' and r['n_active'] == 10]
        streams = {c: {r['start']: r for r in rows if r['condition'] == c} for c in ('swarm', 'private')}
        positions = sorted(set(streams['swarm']) & set(streams['private']))
        for condition, color, label in (('swarm', SWARM, 'Swarm'), ('private', PRIVATE, 'Private control')):
            chosen = [streams[condition][x] for x in positions]
            ax.plot([r['x']/1000 for r in chosen], [r['R'] for r in chosen], '.-', color=color, lw=1.3, ms=3, label=label)
        delta = float(np.mean([streams['swarm'][x]['R']-streams['private'][x]['R'] for x in positions]))
        ax.set_title(f'Rung {rung} · mean ΔR = {delta:+.3f}', fontsize=11)
        ax.set_ylim(0, 1); ax.grid(alpha=.2); ax.set_xlabel('Local generated tokens, thousands'); ax.set_ylabel('Mean vector length R')
    axes[0, 0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Solving phase: swarm vs private control\nReasoning only · W=200 · same local window starts · ten authors in both populations', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .91)); fig.savefig(OUT/'fig03_solving_comparison.png', dpi=155); plt.close(fig)

    # Figure 7: distribution minus solving mean R by width.
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(4); bar = .23; lowest = 0
    for i, (width, color) in enumerate(((100, '#b4c9e9'), (200, '#6b98d1'), (400, '#2a66b6'))):
        rows = [next(c for c in contrasts if c['rung'] == r and c['channel'] == 'reasoning' and c['width'] == width) for r in SEEDS]
        for j, c in enumerate(rows):
            m = c['phase_means_R_N10']
            if m['negotiation'] is None or m['solving'] is None:
                ax.text(x[j]+(i-1)*bar, .004, 'n/a', ha='center', fontsize=8); continue
            d = m['negotiation']-m['solving']; lowest = min(lowest, d)
            ax.bar(x[j]+(i-1)*bar, d, width=bar, color=color, label=f'W={width}' if j == 0 else None)
            star = '*' if c['phase_points_N10']['negotiation'] < 3 else ''
            ax.text(x[j]+(i-1)*bar, d+(.006 if d >= 0 else -.02), f'{d:.3f}{star}', ha='center', fontsize=8)
    ax.axhline(0, color='.4', lw=.7)
    ax.set_xticks(x, [f'Rung {r}' for r in SEEDS]); ax.set_ylabel('Distribution mean R − solving mean R')
    ax.set_ylim(min(0, lowest*1.3), ax.get_ylim()[1]*1.12)
    ax.set_title('The phase contrast changes with window width (swarm, reasoning, N=10)', pad=12)
    ax.legend(ncol=3, loc='upper right'); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT/'fig07_window_sensitivity.png', dpi=155); plt.close(fig)
    return contrasts


def trim400_figure(trim_plans, vectors):
    """Figure 4 analogue: mixed channel, shared phase starts, shortest-author trimming, W=400."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8)); summary = []
    for (rung, spans, layout, rows), ax in zip(trim_plans, axes.flat):
        half = len(rows)//2
        records = compute(rows[:half], vectors)+compute(rows[half:], vectors)
        phases = {}
        for phase in PHASES:
            start = spans[phase][0]
            end = min(s['end'] for parts in layout.values() for s in parts if s['phase'] == phase)
            selected = [r for r in records[:half] if r['n_active'] == 10 and r['start'] >= start and r['start']+400 <= end]
            phases[phase] = {'points': len(selected), 'R_mean': float(np.mean([r['R'] for r in selected])) if selected else None}
        summary.append({'rung': rung, 'phases': phases})
        for condition, color in (('swarm', SWARM), ('private', PRIVATE)):
            series = [r for r in records if r['condition'] == condition]
            ax.plot([r['x'] for r in series], [r['R'] if r['n_active'] == 10 else np.nan for r in series], '.-', ms=3, color=color, label=condition)
        for phase in ('negotiation', 'exchange'):
            ax.axvspan(*spans[phase], color='#689bce', alpha=.1)
        missing = [PHASE_NAMES[p].lower() for p in PHASES if phases[p]['points'] == 0]
        ax.set_ylim(0, 1); ax.set_title(f'Rung {rung}'+(f' — no full window: {", ".join(missing)}' if missing else ''))
        ax.set_xlabel('Conditional token position'); ax.set_ylabel('R (ten authors)'); ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle('Uniform W=400, stride=200; shortest-phase trimming; N=10 at every populated point\nReasoning + visible output. Gaps stay blank. Private trimmed to its shortest author; no private chat phases.', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(OUT/'fig04_trim400_overview.png', dpi=160); plt.close(fig)
    write_json(OUT/'trim400-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--embed', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    rows, channels = lengths_table(tokenizer)
    if not args.dry_run:
        lengths_bars(rows); lengths_spread(rows)
        facts = {'coverage': coverage(tokenizer), 'timelines': timelines()}
        print(json.dumps(facts, indent=1))
    if args.dry_run or args.embed:
        phase_plans, trim_plans, texts, cov = plan_windows(tokenizer, channels)
        estimate = seed_cache(texts); print(json.dumps(estimate))
        if args.embed:
            vectors = embed(texts, OUT, workers=12, cache_dir=CACHE)
            contrasts = phase_figures(phase_plans, cov, vectors)
            print(json.dumps([c for c in contrasts if c['channel'] == 'reasoning'], indent=1))
            print(json.dumps(trim400_figure(trim_plans, vectors), indent=1))


if __name__ == '__main__':
    main()
