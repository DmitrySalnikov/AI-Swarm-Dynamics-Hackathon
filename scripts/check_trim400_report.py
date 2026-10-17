"""Validate fixed-author plots and persist the follow-up findings and cost ledger."""
import base64
import csv
from html.parser import HTMLParser
import json
import math
from pathlib import Path

ROOT = Path('artifacts/pipeline-windows')


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.count = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            src = dict(attrs)['src']
            assert src.startswith('data:image/png;base64,')
            assert base64.b64decode(src.split(',',1)[1]).startswith(bytes([137,80,78,71,13,10,26,10]))
            self.count += 1


def main():
    checks = []
    for rung in range(1,5):
        with (ROOT/'trim400-analysis'/f'rung_{rung}_trim400_n10.csv').open(encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        assert all(int(r['n_active']) in (0,10) for r in rows)
        assert all(math.isfinite(float(r['R'])) == (int(r['n_active'])==10) for r in rows)
        assert all(r['agents']=='1,2,3,4,5,6,7,8,9,10' for r in rows if int(r['n_active'])==10)
        assert all(float(r['eff_dim'])<=9.00001 for r in rows if int(r['n_active'])==10)
        checks.append({'rung':rung,'populated_points':sum(int(r['n_active'])==10 for r in rows),'exactly_ten_authors':True})
    path=ROOT/'window-study.html'
    document=path.read_text(encoding='utf-8')
    parser=Images();parser.feed(document)
    assert parser.count==36
    assert 'id="trim400-n10"' in document
    report={'strict_ten_author_checks':checks,'embedded_pngs':parser.count,'html_bytes':path.stat().st_size}
    (ROOT/'trim400-qa.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    summary=json.loads((ROOT/'trim400-summary.json').read_text(encoding='utf-8'))
    lines=['## Fixed-count 400-token follow-up','',
           'All four saved ladder pairs now have W=400, stride=200 figures. Both swarm phases and private solving are trimmed to their own shortest author. '
           'Every populated geometry point has exactly ten authors; original phase-start offsets and gaps remain.', '',
           '| Rung | Distribution mean R | Solving mean R | Exchange mean R | Phase-window counts |',
           '| --- | --- | --- | --- | --- |']
    for row in summary:
        phases=[row['phases'][p] for p in ('negotiation','solving','exchange')]
        means=['Missing' if p['R_mean'] is None else f"{p['R_mean']:.3f}" for p in phases]
        lines.append('| '+str(row['rung'])+' | '+' | '.join(means)+' | '+' / '.join(str(p['points']) for p in phases)+' |')
    lines += ['', 'Rungs 1–3 show a descriptive high-low-high pattern in R. Rung 3 has only one distribution point. '
              'Rung 4 has 223 final-exchange tokens after trimming, so no full W=400 exchange window exists. '
              'Missing coverage does not establish absence of final convergence. Phase means use overlapping windows from one saved group per rung; they are not independent replications.', '',
              'No new generation was performed. Additional embedding receipts and fixed-count validation are recorded in cost-summary.json and trim400-qa.json. '
              'The HTML includes an overview, phase means/counts, and individual five-metric figures.']
    findings=ROOT/'FINDINGS.md'
    old=findings.read_text(encoding='utf-8').split('## Fixed-count 400-token follow-up')[0].rstrip()
    findings.write_text(old+'\n\n'+'\n'.join(lines)+'\n',encoding='utf-8')
    cost_path=ROOT/'cost-summary.json'
    cost=json.loads(cost_path.read_text(encoding='utf-8'))
    receipts=[json.loads(p.read_text(encoding='utf-8')) for p in ROOT.glob('*/embedding-requests/*.response.json')]
    cost['embedding_receipted_usd']=sum(r.get('usage',{}).get('cost',0) for r in receipts)
    cost['trim400_followup_embedding_usd']=sum(json.loads(p.read_text(encoding='utf-8')).get('usage',{}).get('cost',0)
        for p in (ROOT/'trim400-analysis/embedding-requests').glob('*.response.json'))
    cost_path.write_text(json.dumps(cost,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
