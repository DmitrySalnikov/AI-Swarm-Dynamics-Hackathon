"""Baseline simulation without cooperation: an opening task from set A, then the ten tasks of set B.

Every agent solves 1 + |B| tasks one after another in one continuous conversation, with no team and no chat. Its own
reasoning is fed back: each assistant turn in the history is the provider reasoning wrapped in <think>...</think>
followed by the visible JSON reply. Regimes (the order of the opening task and of set B; all drawn from the run seed):
  same     one opening task and one order of B, shared by every agent;
  random   each agent draws its opening task at random (repeats allowed) and its own random order of B;
  cyclic   different opening tasks (a permutation of A); B is a cyclic shift of a random base order, agent i starting
           at position i, so at every position the agents hold different tasks (a cyclic Latin square);
  latin    different opening tasks; B is a random Latin square built row by row from random perfect matchings, so the
           step from one task to the next differs between agents (no two agents share a successor for long).
Set A is the new ladder set (bench/ladder/set_v3), set B the set used in the swarm runs (bench/ladder/set_v2), so the
main phase can be compared with them. Provider: Together AI, DeepSeek-V4.1-Flash; the budget and the recorded costs
are in USD (the inherited fields keep their `_rub` names). --tasks-per-agent K stops every agent after its first K
tasks (the opening task counts); the run directory is then <regime>_k<K>. --order-seed draws the task orders from
another seed than the run seed (which also seeds the requests), so a run can repeat the orders of an earlier one with
new sampling; its directory is seed_<seed>_order_<order seed>.

  python -m swarm_pipeline.baseline_sim --set-a bench/ladder/set_v3/rung_1.jsonl --set-b bench/ladder/set_v2/rung_1.jsonl \\
      --regime latin --seed 1 --out <root> --offline
  python -m swarm_pipeline.baseline_sim ... --budget-usd 5        # live; key: TOGETHER_API_KEY or Keychain together-api
"""
import argparse,hashlib,json,os,random,subprocess,sys,urllib.request
from pathlib import Path
from swarm_pipeline import runner as base
from swarm_pipeline.stable_live import RetainingClient
from swarm_pipeline.sequential_control import SequentialRunner,SequentialOfflineClient

REGIMES=('same','random','cyclic','latin')
TOGETHER_URL='https://api.together.xyz/v1'
TOGETHER_MODEL='deepseek-ai/DeepSeek-V4.1-Flash'
PRICE_IN,PRICE_OUT=0.30,1.20  # USD per million tokens (Together, 2026-10-26); cached input is not discounted here


def random_latin_rows(symbols,rng):
    """n×n Latin square over `symbols` (rows = agents, columns = positions), each row a random perfect matching of
    positions to the symbols still free in that column (a Latin rectangle always extends, so this never fails)."""
    n=len(symbols);rows=[]
    for _ in range(n):
        free=[[s for s in symbols if all(r[c]!=s for r in rows)] for c in range(n)]
        match={}  # symbol -> column
        def augment(c,seen):
            for s in rng.sample(free[c],len(free[c])):
                if s in seen:continue
                seen.add(s)
                if s not in match or augment(match[s],seen):
                    match[s]=c;return True
            return False
        for c in rng.sample(range(n),n):
            assert augment(c,set())
        row=[None]*n
        for s,c in match.items():row[c]=s
        rows.append(row)
    return rows


def orders(regime,a_ids,b_ids,agents,seed):
    """Per-agent task order: [opening task from A] + all tasks of B."""
    rng=random.Random(f'{seed}:{regime}:orders')
    if regime=='same':
        start=rng.choice(a_ids);b=rng.sample(b_ids,len(b_ids))
        return [[start]+b for _ in range(agents)]
    if regime=='random':
        return [[rng.choice(a_ids)]+rng.sample(b_ids,len(b_ids)) for _ in range(agents)]
    if agents>len(a_ids) or agents!=len(b_ids):
        raise ValueError('cyclic and latin need one agent per task of B and at most |A| agents')
    starts=rng.sample(a_ids,agents)
    if regime=='cyclic':
        base_order=rng.sample(b_ids,len(b_ids))
        rows=[base_order[i:]+base_order[:i] for i in range(agents)]
    else:
        rows=random_latin_rows(b_ids,rng)
    return [[s]+r for s,r in zip(starts,rows)]


class BaselineRunner(SequentialRunner):
    def __init__(self,*args,regime,a_ids,b_ids,cut=None,order_seed=None,**kwargs):
        super().__init__(*args,**kwargs)
        if regime not in REGIMES:raise ValueError('Unknown regime')
        if cut is not None and not 1<=cut<=len(b_ids)+1:raise ValueError('--tasks-per-agent must be between 1 and |B| + 1')
        self.regime,self.a_ids,self.b_ids,self.cut=regime,a_ids,b_ids,cut
        self.order_seed=self.config.seed if order_seed is None else order_seed
        self.manifest.update(protocol_version='baseline-sim-v1',regime=regime,set_a=a_ids,set_b=b_ids,currency='USD',
                             memory='own reasoning in <think> tags + visible reply',provider=base.API_URL,
                             tasks_per_agent=cut or len(b_ids)+1,order_seed=self.order_seed,
                             baseline_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        suffix='' if self.order_seed==self.config.seed else f'_order_{self.order_seed}'
        self.directory=self.root/(regime+(f'_k{cut}' if cut else ''))/f'seed_{self.config.seed}{suffix}'

    def initial_state(self):
        state=super().initial_state()
        for i,o in enumerate(orders(self.regime,self.a_ids,self.b_ids,self.config.agents,self.order_seed),1):
            state['agents'][str(i)]['order']=o[:self.cut] if self.cut else o
        return state

    def apply(self,q,r):
        event=super().apply(q,r)
        history=self.state['agents'][str(q['agent'])]['history']
        k=max(j for j,m in enumerate(history) if m['role']=='assistant')
        thought=(r.get('reasoning') or '').strip()
        if thought:history[k]={'role':'assistant','content':f'<think>\n{thought}\n</think>\n\n{r["content"]}'}
        return event


def together_key():
    key=os.environ.get('TOGETHER_API_KEY')
    if not key and sys.platform=='darwin':
        p=subprocess.run(['security','find-generic-password','-s','together-api','-a',os.environ.get('USER',''),'-w'],capture_output=True,text=True)
        key=p.stdout.strip() if p.returncode==0 else None
    return key


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set-a',type=Path,required=True);parser.add_argument('--set-b',type=Path,required=True)
    parser.add_argument('--regime',choices=REGIMES,required=True)
    parser.add_argument('--agents',type=int,default=10);parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--offline',action='store_true')
    parser.add_argument('--budget-usd',type=float,default=0);parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--max-tokens',type=int,default=40000);parser.add_argument('--temperature',type=float,default=1.0)
    parser.add_argument('--max-context-bytes',type=int,default=3_000_000)
    parser.add_argument('--tasks-per-agent',type=int,help='stop each agent after its first K tasks (opening task included)')
    parser.add_argument('--order-seed',type=int,help='seed of the task orders (default: --seed)')
    args=parser.parse_args()
    a,b=base.load_tasks(args.set_a),base.load_tasks(args.set_b)
    tasks=a+b
    key=None if args.offline else together_key()
    if not args.offline and (not key or args.budget_usd<=0):
        parser.error('Live runs need a Together key (TOGETHER_API_KEY or Keychain together-api) and a positive --budget-usd')
    root=args.out.resolve();repo=Path(__file__).resolve().parents[1]
    if root==repo or any(root==p or p in root.parents for p in (repo/n for n in ('bench','results','embedding_eval','.git'))):
        parser.error('Use a separate artifact directory outside source and baseline data')
    base.API_URL=TOGETHER_URL  # read by the streaming client
    # Together's front end rejects Python's default urllib User-Agent (Cloudflare error 1010)
    opener=urllib.request.build_opener();opener.addheaders=[('User-Agent','swarm-pipeline-baseline/1.0')]
    urllib.request.install_opener(opener)
    config=base.Config(condition='fixed',agents=args.agents,seed=args.seed,model=TOGETHER_MODEL,temperature=args.temperature,
                       max_tokens=args.max_tokens,max_calls=args.agents*(len(b)+1)*3,max_context_bytes=args.max_context_bytes,
                       budget_rub=args.budget_usd,price_in=PRICE_IN/1e6,price_out=PRICE_OUT/1e6,offline=args.offline)
    client=SequentialOfflineClient() if args.offline else RetainingClient(key)
    runner=BaselineRunner(tasks,config,root,client,workers=args.workers,partial_continuation=True,regime=args.regime,
                          a_ids=[t['id'] for t in a],b_ids=[t['id'] for t in b],cut=args.tasks_per_agent,order_seed=args.order_seed)
    result=runner.run()
    print(json.dumps({k:result.get(k) for k in ('status','offline','calls','spent_root_rub')}))
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
