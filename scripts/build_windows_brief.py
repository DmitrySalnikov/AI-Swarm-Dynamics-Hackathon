"""Build a concise, self-contained briefing from existing window-study artifacts."""
import base64
from datetime import datetime, timezone
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('artifacts/pipeline-windows')
OUT=ROOT/'brief-assets'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def figure(path,caption):
    data=base64.b64encode(path.read_bytes()).decode()
    return '<figure><img src="data:image/png;base64,'+data+'" alt="'+escape(caption,quote=True)+'"><figcaption>'+escape(caption)+'</figcaption></figure>'


def length_chart(coverage):
    fig,ax=plt.subplots(figsize=(11,3.8))
    colors={'reasoning':'#246ac2','payload':'#17896b'}
    for channel,shift,label in (('reasoning',-.13,'Private reasoning'),('payload',.13,'Accepted chat-message text')):
        for level in range(1,5):
            row=next(c for c in coverage if c['rung']==level and c['condition']=='swarm' and c['phase']=='negotiation' and c['channel']==channel)
            y=level+shift
            ax.plot([row['min_tokens'],row['max_tokens']],[y,y],color=colors[channel],lw=5,solid_capstyle='round',alpha=.7)
            ax.plot(row['median_tokens'],y,'o',color=colors[channel],ms=7,label=label if level==1 else None)
    for w in (100,200,400):
        ax.axvline(w,color='#a26722',ls=':',lw=1)
        ax.text(w,.48,str(w),ha='center',fontsize=9,color='#81521e')
    ax.set_xscale('log');ax.set_xlim(15,30000);ax.set_ylim(4.5,.35)
    ax.set_yticks(range(1,5),[f'Level {i}' for i in range(1,5)])
    ax.set_xlabel('Distribution tokens per author — logarithmic scale; bars = min–max, dots = median')
    ax.set_title('The chat is short; most distribution text is private reasoning',pad=22)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,0),ncol=2,fontsize=9)
    ax.grid(axis='x',alpha=.15)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(OUT/'channel-lengths.png',dpi=155);plt.close(fig)


def interpretation_chart(contrasts):
    fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    example=next(c for c in contrasts if c['rung']==2 and c['channel']=='reasoning' and c['width']==200)
    phases=('negotiation','solving','exchange')
    values=[example['phase_means_R_N10'][p] for p in phases]
    axes[0].scatter([0,1,2],values,color='#246ac2',s=60,label='Swarm: phase means')
    private=values[1]-example['mean_swarm_minus_private_R_same_local_token']
    axes[0].scatter(1,private,color='#e57531',s=60,label='Private: matched solving')
    for x,p,v in zip(range(3),phases,values):
        axes[0].annotate(f"{v:.3f}; n={example['phase_points_N10'][p]}",(x,v),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    axes[0].set_xticks(range(3),['Distribution','Solving','Common-map\ncommit'])
    axes[0].set_ylim(0,1.05);axes[0].set_xlim(-.4,2.4);axes[0].set_ylabel('R, exactly ten authors')
    axes[0].set_title('Level 2, W=200: sparse phase summaries')
    axes[0].legend(loc='lower left',fontsize=8);axes[0].grid(axis='y',alpha=.2)
    widths=(100,200,400)
    differences=[]
    for w in widths:
        row=next(c for c in contrasts if c['rung']==4 and c['channel']=='reasoning' and c['width']==w)
        differences.append(row['phase_means_R_N10']['negotiation']-row['phase_means_R_N10']['solving'])
    axes[1].bar(range(3),differences,width=.55,color='#7497d0')
    for x,v in enumerate(differences):axes[1].text(x,v+.004,f'{v:+.3f}',ha='center',fontsize=10)
    axes[1].set_xticks(range(3),[f'W={w}' for w in widths]);axes[1].set_ylim(0,.105)
    axes[1].set_ylabel('Distribution mean R − solving mean R')
    axes[1].set_title('Level 4: the contrast depends on window size')
    axes[1].grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'interpretation.png',dpi=155);plt.close(fig)


class Check(HTMLParser):
    def __init__(self):super().__init__();self.images=0
    def handle_starttag(self,tag,attrs):
        if tag=='img':
            source=dict(attrs)['src'];assert source.startswith('data:image/png;base64,')
            assert base64.b64decode(source.split(',',1)[1]).startswith(bytes([137,80,78,71,13,10,26,10]))
            self.images+=1


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    coverage=read(ROOT/'phase-analysis/coverage.json')
    contrasts=read(ROOT/'phase-analysis/contrasts.json')
    verification=read(ROOT/'verification.json')['gradual']
    assert verification['context_history_delivery_replay']
    assert verification['successful_solves']==verification['successful_reports']==10
    length_chart(coverage);interpretation_chart(contrasts)
    today=datetime.now(timezone.utc).strftime('%Y-%m-%d')
    body=f'''<header><div class="eyebrow">TEAM BRIEF · {today} · SAVED RESULTS</div>
    <h1>Windows & unequal agent lengths</h1>
    <p class="lead">What we tried when agents finish at different token positions — and what the geometry can currently tell us.</p>
    <div class="tags"><span>10 authors per group</span><span>4 task-similarity levels</span><span>3 proposed fixes</span></div></header>
    <section><h2>The measurement problem</h2>
    <p>One vector represents one author's text window. When that author runs out of text, its vector disappears. Both the number and identity of remaining authors can change R and effective dimension. A changing curve can therefore reflect changing coverage.</p>
    <div class="callout"><b>Our window rule:</b> full windows only, one vector per available author, stride W/2. Report author count; keep missing text explicit. No repeated final vectors, short-tail padding or interpolation.</div>
    <p>We tried W=400/2048/4096, then W=100/200/400 separately within each phase. The latest view starts each phase at its own token zero; it is a token-position comparison, not elapsed time. Reasoning, visible JSON output and accepted message/result payload are analysed separately.</p>
    </section>
    <section><h2>Three proposed fixes</h2>
    <div class="cards">
      <article><div class="number">1</div><h3>Trim to the shortest author</h3>
      <p><b>Rule:</b> keep the common prefix of every author in a phase.</p>
      <p><b>Tested:</b> fixed-ten-author geometry. Keeping the next phase after the original longest span created large empty gaps; later plots separate the phase clocks.</p>
      <p class="take"><b>Finding:</b> loses 70–84% of distribution reasoning and 52–93% of commit reasoning. Useful as a sensitivity check; expensive in lost data.</p></article>
      <article><div class="number">2</div><h3>Supplement short streams</h3>
      <p><b>Rule:</b> append another independently generated solution of the same task.</p>
      <p><b>Tested:</b> solving-only supplementation and the literal short-chat supplementation proposal, using historical outputs with source provenance.</p>
      <p class="take"><b>Finding:</b> can supply more full windows, but changes the semantic content. A diagnostic synthetic construction; it adds no observed communication.</p></article>
      <article><div class="number">3</div><h3>Report as soon as finished</h3>
      <p><b>Rule:</b> first finisher reports immediately. Later reporting requests receive messages already published.</p>
      <p><b>Live pilot:</b> {verification['reports_before_last_solve']}/10 reports arrived before the last solve finished; {verification['reports_reading_prior_reports']}/10 reporting requests read earlier reports.</p>
      <p class="take"><b>Finding:</b> overlap works. In-flight solves remain frozen; this does not guarantee equal text lengths or peer influence on ongoing reasoning.</p></article>
    </div></section>
    <section><h2>What the checks revealed</h2>
    <ul><li><b>Real distribution messages contain only 25–62 tokens per author.</b> W=100/200/400 cannot form a full window in that message channel. Earlier long “chat” streams were mostly private reasoning.</li>
    <li><b>The final phase commits an identical shared answer map.</b> High similarity of that payload is expected from the protocol. It does not independently establish a cooperation-specific geometric signal.</li>
    <li><b>During private solving, swarm and private-control means remain close.</b> At matched local token positions with ten authors, mean ΔR ranges from −0.012 to +0.023 across widths and task levels. Significance was not assessed.</li></ul>
    '''
    body+=figure(OUT/'channel-lengths.png','Four saved groups. Vertical reference lines mark W=100/200/400. Message text is shorter than every tested window; min–max ranges are not confidence intervals.')
    body+=figure(OUT/'interpretation.png','Reasoning only. Left: n counts overlapping windows, not independent runs; the private point uses the same local solving positions. Right: changing W also changes which full windows enter the phase mean. No continuous arc is inferred between phase summaries.')
    body+='''</section><section class="conclusion"><h2>Current interpretation</h2>
    <p><b>The pipeline implements task distribution and result sharing; the gradual-report pilot implements genuine reporting overlap.</b> A robust cooperation-specific signature in embedding geometry is not established by these checks.</p>
    <p>Keep the window/count diagnostics. Treat trimming and supplementation as sensitivity checks. For messages this short, reconsider the observation unit before requesting longer outputs solely to obtain more windows.</p></section>
    <details><summary>Live timing and scope</summary>'''
    body+=figure(ROOT/'gradual-timeline.png','Separate rung-1 timing pilot. Blue = solving, green = reporting. The gray interval is a recovery restart after imported distribution, not normal pipeline latency. Four interrupted streams were retained and continued; the earlier budget-stopped pilot is archived separately.')
    body+='''<p>Saved-data checks: one swarm/private pair per task level; five additional distribution-only repeats on level 1. The timing pilot is a separate run, not a matched four-level replication. These figures do not establish statistical independence or a universal detection threshold. This briefing generated no new model outputs or embeddings.</p></details>
    <footer><a href="window-study.html#phase-channels">Full plots, coverage tables and interpretations</a> ·
    Sources: phase-analysis/coverage.json, phase-analysis/contrasts.json, verification.json.<br>
    R measures alignment of unit vectors; it does not encode communication access. Effective dimension depends on author count. Higher alignment can occur without interaction on similar tasks.</footer>'''
    css='''*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#1c2c40;font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}main{max-width:1120px;margin:32px auto;background:white;padding:38px 44px;border-radius:16px}h1{font-size:36px;line-height:1.12;margin:10px 0}h2{font-size:24px;margin:0 0 12px}h3{font-size:18px;line-height:1.25;margin:12px 0}p{margin:10px 0}.eyebrow{font-size:12px;letter-spacing:1.3px;color:#63738b}.lead{font-size:19px;max-width:820px;color:#506079}.tags{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.tags span{background:#eaf0f8;padding:5px 10px;border-radius:20px;font-size:13px}section{margin-top:30px}.callout{padding:13px 16px;border-left:4px solid #246ac2;background:#eef4fc}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.cards article{padding:18px;background:#f5f7fa;border:1px solid #dfe6ef;border-radius:10px;font-size:14px}.number{width:27px;height:27px;background:#246ac2;border-radius:50%;text-align:center;color:white;font-weight:700;line-height:27px}.take{border-top:1px solid #dfe6ef;padding-top:12px}ul{padding-left:22px}li{margin:9px 0}figure{margin:22px 0}img{display:block;width:100%;height:auto}figcaption{font-size:12px;color:#5e6d82;margin-top:8px}.conclusion{padding:20px;background:#edf5f3;border-left:4px solid #17896b}details{margin-top:24px;padding:14px 16px;border:1px solid #dfe6ef;border-radius:8px}summary{cursor:pointer;font-weight:650}footer{margin-top:28px;border-top:1px solid #dfe6ef;padding-top:14px;color:#63738b;font-size:12px}a{color:#205daa}@media(max-width:780px){main{margin:0;padding:24px 18px;border-radius:0}.cards{grid-template-columns:1fr}h1{font-size:30px}}@media print{body{background:white}main{margin:0;padding:12px;max-width:none}article,figure{break-inside:avoid}details{display:none}footer{font-size:10px}}'''
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Windows & unequal agent lengths — brief</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>'
    path=ROOT/'windows-and-agent-lengths-brief.html'
    path.write_text(document,encoding='utf-8')
    parser=Check();parser.feed(document);assert parser.images==3
    qa={'file':str(path),'embedded_pngs':parser.images,'bytes':path.stat().st_size,
        'self_contained_images':True,'new_generation':False,'new_embeddings':False,
        'source_verification':'verification.json','source_statistics':'phase-analysis/contrasts.json'}
    (OUT/'brief-qa.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    print(json.dumps(qa))


if __name__=='__main__':main()
