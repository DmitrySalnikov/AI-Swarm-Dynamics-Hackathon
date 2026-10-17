"""Interpret and render the saved-data phase/channel sensitivity study."""
import base64
import csv
from html import escape
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import time

from swarm_pipeline.supervision import read_json, write_json

ROOT=Path('artifacts/pipeline-windows')
OUT=ROOT/'phase-analysis'
PHASES=('negotiation','solving','exchange')
CHANNELS=('reasoning','visible_output','payload')


def findings():
    contrasts=read_json(OUT/'contrasts.json')
    coverage=read_json(OUT/'coverage.json')
    bullets=[]
    for width in (100,200,400):
        reason=[r for r in contrasts if r['channel']=='reasoning' and r['width']==width]
        complete=sum(r['three_phases_observed'] for r in reason)
        shape=sum(r['high_low_high_phase_means'] is True for r in reason)
        bullets.append(f'At W={width}, private reasoning has ten-author coverage in all three phases on {complete}/4 ladder rungs; '
                      f'{shape}/4 have higher distribution and commit phase-mean R than solving. These are phase summaries, not continuous arcs or independent replications.')
    messages=[c for c in coverage if c['channel']=='payload' and c['phase']=='negotiation' and c['condition']=='swarm']
    low=min(c['min_tokens'] for c in messages);high=max(c['max_tokens'] for c in messages)
    bullets.append(f'Actual accepted distribution-message payloads range {low}–{high} tokens per author across the four saved groups. '
                   'The distribution phase has no ten-author payload window at any tested width.' if all(c['min_tokens']<100 for c in messages) else
                   f'Accepted distribution-message payloads range {low}–{high} tokens per author; inspect coverage before comparing widths.')
    bullets.append('The commit payload is the same canonical answer map for every author by protocol. Its R near one is expected copying of a shared result, not independent evidence of an emergent cooperation signal.')
    deltas=[r['mean_swarm_minus_private_R_same_local_token'] for r in contrasts
            if r['channel']=='reasoning' and r['mean_swarm_minus_private_R_same_local_token'] is not None]
    bullets.append(f'Within private solving, the mean swarm-minus-private R differences at matching local token starts and N=10 range from {min(deltas):+.3f} to {max(deltas):+.3f} across levels and widths. '
                   'The sign varies by level. This is not a statistical significance assessment.')
    level4=[r for r in contrasts if r['rung']==4 and r['channel']=='reasoning']
    changes=[(r['width'],r['phase_means_R_N10']['negotiation']-r['phase_means_R_N10']['solving']) for r in level4]
    bullets.append('On level 4, the private-reasoning distribution-minus-solving R contrast depends on width: '+
                   ', '.join(f'W={w}: {d:+.3f}' for w,d in changes)+'. A strong width-independent phase contrast is not established there.')
    for phase,title in (('negotiation','distribution'),('exchange','commit')):
        rows=[c for c in coverage if c['channel']=='reasoning' and c['condition']=='swarm' and c['phase']==phase]
        losses=[100*c['discarded_by_trim_tokens']/max(c['total_tokens'],1) for c in rows]
        bullets.append(f'Shortest-author trimming discards {min(losses):.1f}–{max(losses):.1f}% of observed private-reasoning tokens in {title}. '
                       'It enforces a constant author count by removing text; it does not solve unequal output lengths without loss.')
    bullets.append('The four saved groups use different tasks and one group per rung. Window overlap, trimming, and changing author count prevent treating plotted points as independent experimental repetitions.')
    return bullets


def picture(path,caption):
    data=base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img loading="lazy" src="data:image/png;base64,{data}"><figcaption>{escape(caption)}</figcaption></figure>'


def html_section():
    contrasts=read_json(OUT/'contrasts.json')
    coverage=read_json(OUT/'coverage.json')
    sections=['<section id="phase-channels"><h2>Latest: separate phases, channels and window widths</h2>',
        '<p><b>Saved data only; no new generation.</b> Each phase starts at local token zero. '
        'Columns are independent phases; they are not connected into a continuous time trajectory. '
        'Full windows: W=100, 200, 400; stride W/2. Solid curves: shortest-author trim, exactly ten authors. '
        'Dashed curves: all available authors, geometry only for N≥4; read the author-count row. '
        'The two curves coincide within their common full-ten-author prefix. Ladder level means one of the four task-similarity levels.</p>',
        '<p><b>Channels:</b> private reasoning = observed provider reasoning, never broadcast; visible output = the own model response including JSON protocol; '
        'payload = accepted send message during distribution, private answer + notes during solving, canonical common answer map during commit. '
        'Claims have no message payload. Visible output is not synonymous with a peer message. '
        'Interrupted/rejected own output remains in the first two channels; payload uses accepted actions only.</p>',
        '<h3>What the reanalysis says</h3><ul>']
    sections += ['<li>'+escape(text)+'</li>' for text in findings()]
    sections += ['</ul><h3>Private reasoning: phase means with exactly ten authors</h3>',
        '<table><tr><th>Level</th><th>W</th><th>Distribution R (windows)</th><th>Solving R (windows)</th><th>Commit R (windows)</th><th>Solving ΔR vs private</th></tr>']
    for r in contrasts:
        if r['channel']!='reasoning':continue
        cells=[]
        for phase in PHASES:
            mean=r['phase_means_R_N10'][phase];n=r['phase_points_N10'][phase]
            cells.append('No full window' if mean is None else f'{mean:.3f} ({n})')
        delta=r['mean_swarm_minus_private_R_same_local_token']
        cells.append('—' if delta is None else f"{delta:+.3f} ({r['matched_solving_points_N10']} matched positions)")
        sections.append(f"<tr><td>{r['rung']}</td><td>{r['width']}</td>"+''.join(f'<td>{c}</td>' for c in cells)+'</tr>')
    sections += ['</table><p>ΔR uses only the solving phase, equal local token-window starts and N=10 in both conditions. '
        'It is a descriptive difference, not a causal estimate or a statistical test. Means with one or two windows are especially sparse; overlapping windows are not independent trials.</p>',
        '<h3>Read the plots</h3><p>The three rows show R, effective dimension, and active authors. '
        'Private controls appear only in solving. No private chat/commit phases are invented. '
        'Empty panels mean insufficient full windows, not zero similarity. All figures use the same Y limits. '
        'For readability, each X axis ends after the final window with N≥4; complete low-count tails and lengths remain in the CSV and coverage table.</p>']
    for rung in range(1,5):
        sections.append(f'<details {"open" if rung==1 else ""}><summary>Ladder rung {rung}: phase/channel plots</summary>')
        for channel,label in (('reasoning','Private reasoning only'),('visible_output','Visible model output, including protocol'),('payload','Accepted message / result payload')):
            sections.append(f'<details {"open" if channel=="reasoning" else ""}><summary>{label}</summary>')
            for width in (100,200,400):
                sections.append(f'<details {"open" if width==200 else ""}><summary>Window {width}, stride {width//2}</summary>'+picture(
                    OUT/f'rung_{rung}_{channel}_w{width}.png',
                    'Solid: exactly ten authors after trim. Dashed: available authors, changing N. Each column starts at its own token zero; no synthetic text or cross-phase window.')+'</details>')
            sections.append('</details>')
        sections.append('</details>')
    sections += ['<details><summary>Length coverage and text discarded by trimming</summary><table><tr><th>Rung</th><th>Condition</th><th>Phase</th><th>Channel</th><th>Min / median / max tokens</th><th>Discarded by trim</th></tr>']
    for c in coverage:
        lost=c['discarded_by_trim_tokens']/max(c['total_tokens'],1)*100
        sections.append(f"<tr><td>{c['rung']}</td><td>{c['condition']}</td><td>{c['phase']}</td><td>{c['channel']}</td><td>{c['min_tokens']} / {c['median_tokens']:.0f} / {c['max_tokens']}</td><td>{lost:.1f}%</td></tr>")
    sections += ['</table></details><p><b>Conclusion:</b> evaluate the channels separately. A phase contrast in private reasoning is not automatically a contrast in exchanged messages. '
        'Smaller windows can recover some short-phase coverage, but do not create longer conversations or provide independent repetitions. '
        'The common-packet commit is an explicit protocol intervention. These results do not isolate a general cooperation signature.</p></section>']
    return ''.join(sections)


class ImageCheck(HTMLParser):
    def __init__(self):super().__init__();self.count=0
    def handle_starttag(self,tag,attrs):
        if tag=='img':
            src=dict(attrs)['src'];assert src.startswith('data:image/png;base64,')
            assert base64.b64decode(src.split(',',1)[1]).startswith(bytes([137,80,78,71,13,10,26,10]))
            self.count+=1


def main():
    deadline=time.time()+3600
    while time.time()<deadline:
        if (OUT/'status.json').exists() and read_json(OUT/'status.json')['state']=='complete':break
        error_path=ROOT/'phase.stderr.log'
        if error_path.exists() and 'Traceback' in error_path.read_text(encoding='utf-8'):
            raise RuntimeError('Phase analysis failed; inspect phase.stderr.log')
        time.sleep(10)
    else:raise TimeoutError('Phase analysis did not finish within the report wait')
    summaries=read_json(OUT/'summary.json')
    fieldnames=['condition','start','x','n_active','agents','R','eff_dim','pair_cosine','matched_drift','n_matched','rung','channel','width','phase','mode']
    for rung in range(1,5):
        for channel in CHANNELS:
            for width in (100,200,400):
                path=OUT/f'rung_{rung}_{channel}_w{width}.csv'
                if not path.exists():
                    assert all(s['max_active']==0 for s in summaries if s['rung']==rung and s['channel']==channel and s['width']==width)
                    with path.open('w',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fieldnames).writeheader()
    checks=[]
    for path in sorted(OUT.glob('rung_*_w*.csv')):
        with path.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        assert all(int(r['n_active']) in (0,10) for r in rows if r['mode']=='trim')
        assert all(math.isfinite(float(r['R']))==(int(r['n_active'])>=4) for r in rows)
        assert all(float(r['eff_dim'])<=int(r['n_active'])-1+1e-4 for r in rows if int(r['n_active'])>=4)
        checks.append({'csv':path.name,'rows':len(rows),'count_integrity':True})
    assert len(checks)==36
    # Full-ten-author metrics must coincide between methods on the retained common prefix.
    for r in summaries:
        if r['mode']=='trim':
            other=next(s for s in summaries if s['mode']=='available' and all(s[k]==r[k] for k in ('rung','channel','width','phase','condition')))
            assert r['points_N10']==other['points_N10']
            assert r['mean_R_N10']==other['mean_R_N10']
    from scripts.finalize_window_study import report
    report(read_json(ROOT/'final-run-ledger.json'),True,read_json(ROOT/'distribution-summary.json'),read_json(ROOT/'verification.json')['gradual'],[])
    document=(ROOT/'window-study.html').read_text(encoding='utf-8')
    parser=ImageCheck();parser.feed(document);assert parser.count==72
    assert 'id="phase-channels"' in document
    cost=read_json(ROOT/'cost-summary.json')
    receipts=[read_json(p) for p in ROOT.glob('*/embedding-requests/*.response.json')]
    cost['embedding_receipted_usd']=sum(r.get('usage',{}).get('cost',0) for r in receipts)
    phase_cost=sum(read_json(p).get('usage',{}).get('cost',0) for p in (OUT/'embedding-requests').glob('*.response.json'))
    cost['phase_channel_followup_embedding_usd']=phase_cost
    write_json(ROOT/'cost-summary.json',cost)
    lines=['# Separate-phase, separate-channel reanalysis','',
        'Saved four ladder pairs only. No new generation. Widths 100/200/400, stride W/2; each phase has an independent local token origin. '
        'All available authors versus shortest-author trimming. No padding, interpolation, held vectors, or cross-phase windows.','',
        '## Findings','']+['- '+s for s in findings()]+['',
        '## Scope of interpretation','',
        'Inspect the local solving comparison separately from phase contrasts. Only the solving phase has a corresponding private control. '
        'Raw reasoning and visible output retain interrupted/rejected generated fragments. Semantic payload includes accepted actions only. '
        'R and pairwise cosine are algebraically linked at fixed N. Effective dimension depends on author count. '
        'No statistical independence, model-general cooperation effect, or classifier performance is established.','',
        f'Embedding receipts for this follow-up: USD {phase_cost:.6f}. New generation cost: RUB 0.','',
        'See contrasts.json for all widths and channels; coverage.json for lengths and discarded text; manifest.json for exact channel definitions, source hashes and operational errors.']
    (OUT/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    main_findings=ROOT/'FINDINGS.md'
    old=main_findings.read_text(encoding='utf-8').split('## Separate-phase, separate-channel follow-up')[0].rstrip()
    main_findings.write_text(old+'\n\n## Separate-phase, separate-channel follow-up\n\n'+'\n'.join('- '+s for s in findings())+
        '\n\nSee phase-analysis/FINDINGS.md and the latest phase-channels section in the HTML. No new generation.\n',encoding='utf-8')
    write_json(OUT/'report-qa.json',{'csv_checks':checks,'embedded_pngs':parser.count,'html_bytes':len(document.encode()),
        'trim_matches_full_author_prefix':True,'embedding_cost_usd':phase_cost,'new_generation_rub':0})
    write_json(ROOT/'report-qa.json',{'embedded_pngs':parser.count,'html_bytes':len(document.encode()),
        'phase_channel_csv_checks':36,'phase_detail':'phase-analysis/report-qa.json','new_generation_rub':0})
    write_json(OUT/'report.status.json',{'state':'complete','updated_at':time.time(),'figures':36,'embedding_cost_usd':phase_cost})
    print(json.dumps({'state':'complete','embedded_pngs':parser.count,'embedding_cost_usd':phase_cost}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write_json(OUT/'report.status.json',{'state':'failed','updated_at':time.time(),'error_type':type(exc).__name__,'error':str(exc)})
        raise
