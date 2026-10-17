"""Summarise the work since the team call, using saved data and plots only."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from tokenizers import Tokenizer

from scripts.build_windows_brief import Check, figure, read
from swarm_pipeline.analyze import token_data

ROOT=Path('artifacts/pipeline-windows')
OUT=ROOT/'post-call-assets'


def stream_comparison():
    """Show actual coverage and recorded synthetic spans, not inferred token times."""
    tokenizer=Tokenizer.from_file('artifacts/pipeline-analysis/deepseek-tokenizer.json')
    methods=read(ROOT/'window-methods.json')
    case=next(c for c in methods['cases'] if c['rung']==1 and c['mode']=='barrier')
    phases=('negotiation','solving','exchange')
    lengths=case['phase_lengths'];spans=case['phase_spans']
    minima={p:min(a[p] for a in lengths.values()) for p in phases}
    private={}
    for aid in range(1,11):
        path=Path(f'artifacts/pipeline-ladder/rung_1/private/fixed/seed_3/generated/agent_{aid}.generated.txt')
        private[aid]=len(token_data(tokenizer,path.read_text(encoding='utf-8')))
    gradual={}
    for aid in range(1,11):
        path=ROOT/f'live/gradual-602/cooperative/seed_602/generated/agent_{aid}.generated.txt'
        text=path.read_text(encoding='utf-8');index=read(path.with_suffix('.index.json'))
        gradual[aid]={p:len(token_data(tokenizer,'\n\n'.join(text[r['start']:r['end']] for r in index if r['phase']==p)))
                      for p in ('negotiation','solving','report','exchange')}
    supplements=read(ROOT/'chat-padding-provenance.json')['supplements']
    colors={'negotiation':'#347bc6','solving':'#24a078','report':'#23afbf','exchange':'#9467bd'}
    fig,axes=plt.subplots(1,3,figsize=(14,7),sharey=True)
    for panel,ax in enumerate(axes):
        for aid in range(1,11):
            if panel<2:
                for p in phases:
                    n=minima[p] if panel==0 else lengths[str(aid)][p]
                    ax.barh(aid,n,left=spans[p][0],height=.73,color=colors[p])
                if panel==1:
                    for supplement in supplements:
                        if supplement['agent']!=aid:continue
                        p=supplement['phase']
                        assert spans[p][0]+lengths[str(aid)][p]<=supplement['start']<supplement['end']<=spans[p][1]
                        ax.barh(aid,supplement['end']-supplement['start'],left=supplement['start'],height=.73,
                                color='#e3e6eb',edgecolor='#727e8c',hatch='///',linewidth=.4)
            else:
                cursor=0
                for p in ('negotiation','solving','report','exchange'):
                    n=gradual[aid][p]
                    ax.barh(aid,n,left=cursor,height=.73,color=colors[p]);cursor+=n
            ax.barh(aid+11,private[aid],left=0,height=.73,color='#e57532')
        ax.axhline(10.9,color='#9eaaba',lw=.7)
        ax.set_yticks(list(range(1,11))+list(range(12,22)),
                      [f'S{i}' for i in range(1,11)]+[f'P{i}' for i in range(1,11)],fontsize=8)
        ax.set_ylim(21.8,.2);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        maximum=max(max(private.values()),max(sum(a.values()) for a in gradual.values()),spans['exchange'][1])
        ax.set_xlim(0,maximum*1.03);ax.tick_params(axis='x',labelsize=8)
        ax.set_xlabel('Shared phase slots (tokens)' if panel<2 else 'Own output order (tokens)',fontsize=9)
    axes[0].set_title('1. Trim each phase\nto the shortest author',fontsize=11)
    axes[1].set_title('2. Supplement short chats\nwith other same-task solutions',fontsize=11)
    axes[2].set_title('3. Report as soon as finished\nseparate live pilot',fontsize=11)
    fig.suptitle('How the three fixes change per-author text coverage\nS1–S10: swarm · P1–P10: original private reference · reasoning + visible output · not wall time',fontsize=12)
    handles=[Patch(color=colors[p],label=label) for p,label in (
        ('negotiation','Distribution'),('solving','Solving'),('report','Early report'),('exchange','Commit'))]
    handles += [Patch(color='#e57532',label='Private solving'),Patch(facecolor='#e3e6eb',edgecolor='#727e8c',hatch='///',label='Synthetic supplement')]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,0))
    fig.tight_layout(rect=(0,.08,1,.89));fig.savefig(OUT/'three-fixes-streams.png',dpi=155);plt.close(fig)
    return {'swarm_authors_per_panel':10,'private_authors_per_panel':10,'source_level':1,
            'trim_and_supplement_seed':3,'gradual_seed':602,'private_reference_seed':3,
            'synthetic_spans':len(supplements),'clock':'Token coverage only, not elapsed time; gradual is a separate unpaired pilot.'}


def distributions():
    with (ROOT/'distribution-lengths.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    seeds=sorted({int(r['seed']) for r in rows})
    assert len(seeds)==5
    fig,axes=plt.subplots(2,5,figsize=(13,5.6),sharey='row',sharex=True)
    for col,seed in enumerate(seeds):
        group=sorted([r for r in rows if int(r['seed'])==seed],key=lambda r:int(r['agent']))
        assert len(group)==10
        for row,key,color in ((0,'reasoning_tokens','#347bc6'),(1,'published_chat_tokens','#e38a3b')):
            values=[int(r[key]) for r in group]
            ax=axes[row,col];ax.bar(range(1,11),values,color=color,width=.8)
            ax.set_xticks(range(1,11),[str(i) for i in range(1,11)],fontsize=7)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            if row==0:
                ax.set_title(f'Group {col+1} · seed {seed}',fontsize=10)
                ax.set_ylim(0,20000)
                peak=max(values);idx=values.index(peak)+1
                ax.text(idx,peak+450,f'{peak:,}',ha='center',fontsize=8)
            else:ax.set_ylim(0,70);ax.set_xlabel('Agent',fontsize=9)
    axes[0,0].set_ylabel('Private reasoning tokens')
    axes[1,0].set_ylabel('Published chat tokens')
    fig.suptitle('Five fresh distribution-only repeats: ten agents in each group',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'distribution-bars.png',dpi=155);plt.close(fig)
    return len(seeds)


def solving_comparison():
    fig,axes=plt.subplots(2,2,figsize=(11,6.2),sharey=True)
    facts=[]
    for level,ax in zip(range(1,5),axes.flat):
        with (ROOT/'phase-analysis'/f'rung_{level}_reasoning_w200.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        streams={}
        for condition in ('swarm','private'):
            streams[condition]={int(r['start']):r for r in rows if r['phase']=='solving' and r['mode']=='trim'
                                and r['condition']==condition and int(r['n_active'])==10}
        positions=sorted(set(streams['swarm'])&set(streams['private']))
        assert positions
        for condition,color,label in (('swarm','#2476d2','Swarm'),('private','#e57532','Private control')):
            chosen=[streams[condition][x] for x in positions]
            ax.plot([float(r['x'])/1000 for r in chosen],[float(r['R']) for r in chosen],'.-',color=color,
                    lw=1.3,ms=3,label=label)
        delta=float(np.mean([float(streams['swarm'][x]['R'])-float(streams['private'][x]['R']) for x in positions]))
        facts.append({'level':level,'width':200,'matched_positions':len(positions),'delta_R':delta})
        ax.set_title(f'Level {level} · mean ΔR = {delta:+.3f}',fontsize=11)
        ax.set_ylim(0,1);ax.grid(alpha=.2);ax.set_xlabel('Local generated tokens, thousands')
        ax.set_ylabel('Mean vector length R')
    axes[0,0].legend(loc='lower right',fontsize=9)
    fig.suptitle('During solving, swarm and private trajectories remain close\nReasoning only · W=200 · same local window starts · ten authors in both populations',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(OUT/'solving-comparison.png',dpi=155);plt.close(fig)
    return facts


def window_sensitivity():
    contrasts=read(ROOT/'phase-analysis/contrasts.json')
    fig,ax=plt.subplots(figsize=(11,3.8))
    x=np.arange(4);width=.23
    for i,(window,color) in enumerate(((100,'#b4c9e9'),(200,'#6b98d1'),(400,'#2a66b6'))):
        rows=[next(r for r in contrasts if r['rung']==level and r['channel']=='reasoning' and r['width']==window) for level in range(1,5)]
        delta=[r['phase_means_R_N10']['negotiation']-r['phase_means_R_N10']['solving'] for r in rows]
        ax.bar(x+(i-1)*width,delta,width=width,color=color,label=f'W={window}')
        for j,(r,d) in enumerate(zip(rows,delta)):
            star='*' if r['phase_points_N10']['negotiation']<3 else ''
            ax.text(x[j]+(i-1)*width,d+.008,f'{d:.3f}{star}',ha='center',fontsize=8)
    ax.set_xticks(x,[f'Level {i}' for i in range(1,5)])
    ax.set_ylim(0,.335);ax.set_ylabel('Distribution mean R − solving mean R')
    ax.set_title('The phase contrast changes with window width',pad=12)
    ax.legend(ncol=3,loc='upper right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(OUT/'window-sensitivity.png',dpi=155);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    groups=distributions();facts=solving_comparison();window_sensitivity();stream_facts=stream_comparison()
    verification=read(ROOT/'verification.json')
    v=verification['gradual'];cost=read(ROOT/'cost-summary.json')
    assert len(verification['distribution_audits'])==5
    assert v['context_history_delivery_replay'] and v['successful_reports']==10
    date=datetime.now(timezone.utc).strftime('%Y-%m-%d')
    sections=[f'''<header><div class="eyebrow">TEAM CHECKPOINT · {date}</div>
    <h1>Work since the team call</h1>
    <p class="lead">Agent-length variability, cooperation geometry, and three ways to handle agents finishing at different positions.</p>
    <div class="stats"><div><b>{groups}</b><span>fresh distribution repeats</span></div><div><b>4</b><span>saved task levels reanalysed</span></div><div><b>3</b><span>length-handling ideas tested</span></div><div><b>1</b><span>completed gradual-report pilot</span></div></div>
    <nav><a href="#lengths">Chat lengths</a><a href="#cooperation">Cooperation results</a><a href="#windows">Windows & fixes</a><a href="#pilot">Live overlap</a><a href="#conclusions">Conclusions</a></nav></header>
    <section class="summary"><h2>Current status</h2><ul>
    <li><b>Pipeline:</b> task distribution, private solving and result sharing are implemented. Early reporting was tested live and audited.</li>
    <li><b>Data:</b> large length imbalances recur, mainly in private reasoning. Actual shared messages are short.</li>
    <li><b>Research result:</b> phase contrasts exist in some measurements, but a robust cooperation-specific geometric signature is not established.</li></ul></section>
    <section><h2>What was done</h2><p>Reused the four completed swarm/private task pairs as the baseline; ran five new ten-agent distribution repeats; tested trimming and independent-solution supplementation; implemented and tested completion-driven reporting. Recomputed uniform windows, then split phases and channels. Retained interrupted outputs, verified histories/deliveries, and corrected temporary-budget-reservation handling.</p>
    <p class="note">The four-level task data were already available. The new generation here was distribution repeats and the timing pilot, not a new four-level generation sweep.</p></section>
    <section id="lengths"><h2>1. How much does each agent actually write?</h2>''']
    sections.append(figure(OUT/'distribution-bars.png','Bars are individual authors. Reasoning includes all observed turns and recovery fragments; chat counts accepted send-message text only. Agent task assignments can differ between repeats.'))
    sections.append('''<p class="interpret"><b>Result:</b> private reasoning alone reached 18,179 and 15,869 tokens in two error-free repeats. Published messages across the five repeats were only 19–51 tokens. The imbalance repeats; it is not solely a transport-failure artifact.</p>
    <details><summary>Distribution spread across the five groups</summary>'''+figure(ROOT/'distribution-spread.png','Ten authors per group; five independent generation repeats. Authors and overlapping windows are not independent run replications.')+'''</details></section>
    <section id="cooperation"><h2>2. What happened to the cooperation comparison?</h2>
    <p>Swarm distributes tasks, then agents solve privately. The control gives each agent its assigned task without peer exchange. During solving, both populations perform similar work. Blue is swarm; orange is the private control.</p>''')
    sections.append(figure(OUT/'solving-comparison.png','Matched window starts and ten authors in both conditions. These are reasoning-only solving trajectories, not complete interaction arcs. ΔR is descriptive; no significance or equivalence test was performed.'))
    sections.append('''<p class="interpret"><b>Result:</b> solving curves remain close. Across W=100/200/400, mean swarm-minus-private R differences range from −0.012 to +0.023 and change sign by task level. The overlap is an observed result, not a plotting error; it does not establish absence of interaction elsewhere.</p>
    <p>Distribution and final commit are separate phases. The private control has no equivalent chat/commit phases. In v2 the final answers are copied from an identical shared packet, so high final payload similarity is expected from the protocol.</p>
    <details><summary>Earlier mixed-channel full-arc view, with its limitations</summary>'''+figure(ROOT/'trim400-analysis/ladder_trim400_overview.png','W=400; every populated point has ten authors. These mix reasoning and visible output. Large gaps came from trimming to the shortest stream while retaining phase starts after the longest. Sparse endpoints and missing level-4 commit coverage prevent claiming a convincing continuous arc.')+'''</details></section>
    <section id="windows"><h2>3. Windows, active authors and the three fixes</h2>
    <p>Tested W=400/2048/4096, then W=100/200/400. The latest plots use independent local token origins for each phase and separate reasoning, visible JSON responses and accepted message/result payload. Full windows only; stride W/2; active-author count is explicit.</p>
    <table><thead><tr><th>Idea tested</th><th>Implementation</th><th>Observed result / limit</th></tr></thead><tbody>
    <tr><td><b>1. Trim</b></td><td>Cut each phase to its shortest author; keep ten vectors where a full window exists.</td><td>Removed 70–84% of distribution reasoning and 52–93% of commit reasoning. Fixed count costs substantial text; it does not restore missing output.</td></tr>
    <tr><td><b>2. Supplement</b></td><td>Append independently generated same-task solution text from historical controls. Tested solving-only and literal short-chat variants.</td><td>Can supply more full windows. It is synthetic content, not newly observed communication; geometric changes cannot establish natural cooperation.</td></tr>
    <tr><td><b>3. Gradual reporting</b></td><td>Dispatch a report as soon as its author finishes; deliver earlier messages in later reporting requests.</td><td>Actual overlap confirmed. It changes scheduling, but does not guarantee equal lengths or inject messages into an ongoing solve.</td></tr>
    </tbody></table>''')
    sections.append('<h3>What the author streams look like against private</h3>')
    sections.append(figure(OUT/'three-fixes-streams.png',
        'Level 1. Each row is one author; white means no text. Panels 1–2 use the same saved pair: trimming leaves gaps, synthetic additions are hatched. Panel 3 uses the separate gradual-report pilot and concatenates each author’s own output. Orange is the original historical private reference, not a newly matched timing run. These token diagrams do not compare runtimes or establish per-token simultaneity.'))
    sections.append('<p class="interpret"><b>Read it this way:</b> trimming removes long tails; supplementation fills space with independent solution text; gradual reporting changes when a message is delivered. The third diagram shows output order only — the real overlap is measured in the timestamp timeline below.</p>')
    sections.append('<details><summary>Original stream drawing: continuous / shared phase starts / trimmed</summary>'+figure(
        ROOT/'author-coverage.png','Earlier level-1 coverage diagram: continuous own streams, common phase-start offsets with gaps, and shortest-author trimming. These are three representations, not the three fixes above. S = swarm, P = private; phase-wise token counts include reasoning and visible output.')+'</details>')
    sections.append(figure(OUT/'window-sensitivity.png','This compares distribution versus solving inside swarm, not swarm versus private. Private reasoning, N=10. * Distribution mean has only one or two windows. Changing width also changes which fragments enter each mean; overlapping windows are not independent experiments.'))
    sections.append('''<p class="interpret"><b>Result:</b> the apparent phase contrast depends on window width. On level 4 it is about 0.005 / 0.037 / 0.080 at W=100/200/400. Actual distribution-message payloads in the four saved task groups are only 25–62 tokens, so none of these windows fits that message channel.</p>
    <details><summary>Detailed phase plots: geometry together with author count</summary>'''+figure(ROOT/'phase-analysis/rung_2_reasoning_w200.png','Example: level 2, W=200, reasoning only. Solid lines retain all ten authors; dashed lines use a changing number. Each phase starts at token zero. Full low-count tails remain in the underlying CSV.')+'''</details></section>
    <section id="pilot"><h2>4. Did gradual reporting actually work?</h2>''')
    sections.append(figure(ROOT/'gradual-timeline.png','Real request timestamps, separate level-1 pilot. Gray middle interval includes recovery/restart after imported distribution; it is not normal pipeline latency. Blue = solving; green = reporting.'))
    sections.append(f'''<p class="interpret"><b>Verified:</b> {v['reports_before_last_solve']}/10 reports were delivered before the last solve finished; {v['reports_reading_prior_reports']}/10 reporting requests included an earlier report. Ten agents solved, reported and committed. History/context/delivery replay passed.</p>
    <p class="note">Four provider interruptions were retained and explicitly continued. The initial budget-stopped timing attempt is archived separately; the corrected pilot establishes reporting overlap, not peer influence on already-running private solves.</p></section>
    <section id="conclusions" class="summary"><h2>What we can take to the team</h2><ol>
    <li><b>Keep:</b> the working pipeline, completion-driven reporting, retained logs, uniform-window and author-count diagnostics.</li>
    <li><b>Length issue:</b> variability is real and recurrent. Trimming and supplementation are sensitivity checks rather than complete solutions.</li>
    <li><b>Effect interpretation:</b> earlier mixed-channel curves combined private planning, solving and a copied final packet. They do not independently establish a general cooperation signature.</li>
    <li><b>Next design decision:</b> choose the observation channel and a suitable unit for short messages; use matched communication-access controls and repeat groups before making a stronger effect claim.</li>
    </ol></section>
    <details><summary>Scope, costs and source artifacts</summary>
    <p>One saved swarm/private pair per task level; five fresh distribution repeats on level 1; one completed gradual-report pilot after a stopped preliminary attempt. No answer-correctness filtering.</p>
    <p>Post-call generation: RUB {cost['generation_usage_backed_rub']:.2f} usage-backed + RUB {cost['generation_unreceipted_estimates_rub']:.2f} unreceipted estimates = RUB {cost['generation_total_rub']:.2f}. Both timing attempts are included; imported distribution is counted once. Embedding receipts: USD {cost['embedding_receipted_usd']:.4f}. Building this report made no generation or embedding calls.</p>
    <p>Sources: distribution-lengths.csv; phase-analysis/contrasts.json and coverage.json; verification.json; cost-summary.json. R measures vector alignment, not communication access; effective dimension depends on author count.</p></details>
    <footer><a href="window-study.html#phase-channels">Full study: plots, channel definitions, counts and findings</a> · All images embedded for sharing.</footer>''')
    css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f1f4f8;color:#1c2d42;font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1160px;margin:28px auto;background:white;padding:36px 42px;border-radius:14px}h1{font-size:36px;line-height:1.15;margin:10px 0}h2{font-size:24px;line-height:1.25;margin:0 0 12px}p{margin:10px 0}.eyebrow{font-size:12px;letter-spacing:1.4px;color:#65768b}.lead{font-size:19px;color:#51657e}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.stats div{padding:14px;background:#edf3fb;border-radius:9px}.stats b{display:block;font-size:28px;line-height:1.1;color:#2466ae}.stats span{font-size:13px}nav{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}nav a{padding:7px 12px;border:1px solid #dce5ef;border-radius:20px;font-size:13px;text-decoration:none}a{color:#205a9d}section{margin-top:32px;scroll-margin-top:15px}.summary{padding:20px;background:#eef4fa;border-left:4px solid #347bc6}.summary li{margin:7px 0}ul,ol{padding-left:22px}.interpret{padding:12px 15px;background:#eef6f2;border-left:3px solid #218268}.note,figcaption{color:#5c6e83;font-size:13px}figure{margin:20px 0}img{display:block;width:100%;height:auto}figcaption{margin-top:7px}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:12px;text-align:left;vertical-align:top;border:1px solid #dce4ed}th{background:#eef2f7}td:first-child{width:16%}details{border:1px solid #dce4ed;padding:13px 16px;border-radius:8px;margin-top:18px}summary{font-weight:650;cursor:pointer}footer{font-size:13px;border-top:1px solid #dce4ed;padding-top:15px;margin-top:25px}@media(max-width:760px){main{margin:0;padding:23px 16px;border-radius:0}.stats{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}table{font-size:12px}td,th{padding:8px}}@media print{body{background:white}main{margin:0;max-width:none;padding:10px}nav{display:none}figure,table,.summary{break-inside:avoid}details{display:none}}'''
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Work since the team call — results</title><style>'+css+'</style></head><body><main>'+''.join(sections)+'</main></body></html>'
    target=ROOT/'work-since-team-call.html';target.write_text(document,encoding='utf-8')
    # Update the already-open short-report URL to the corrected scope too.
    (ROOT/'windows-and-agent-lengths-brief.html').write_text(document,encoding='utf-8')
    parser=Check();parser.feed(document);assert parser.images==9
    qa={'report':str(target),'also_updated':'windows-and-agent-lengths-brief.html','embedded_pngs':parser.images,
        'bytes':target.stat().st_size,'matched_solving_comparisons':facts,'stream_diagrams':stream_facts,'new_generation':False,'new_embeddings':False}
    (OUT/'report-qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa))


if __name__=='__main__':main()
