"""Independent agents that solve every task one after another; no negotiation or peer exchange.

The order comes from the base runner: `fixed` gives every agent the supplied order (task 1..N), `shuffled` an
independent deterministic shuffle per agent, `disjoint` cyclic offsets. With --unique-start (shuffled only) the
first tasks are a random permutation, so no two agents start on the same task, and each agent then takes its
remaining tasks in its own random order (`shuffled_unique_start/`). With --shared-random-order (fixed only) every
agent gets the same order, drawn at random per run seed (`fixed_random_order/`). --tasks-per-agent K keeps the
first K tasks of every order (directory suffix `_kK`). Each task is solved
in one turn with one submit action, as in the swarm solving phase and the one-task private control, so only the
number and order of tasks differ. The prompt texts match private_control (plain wording, no style selection).

  python -m swarm_pipeline.sequential_control --items <tasks.jsonl> --out <root> --condition fixed --seed 3 \\
      --workers 10 --budget-rub 300
  python -m swarm_pipeline.sequential_control ... --condition shuffled --unique-start
  python -m swarm_pipeline.sequential_control ... --condition fixed --shared-random-order --seed 1003
"""
import argparse,hashlib,json,random,sys
from pathlib import Path
from swarm_pipeline.parallel import ParallelRunner
from swarm_pipeline.stable_live import RetainingClient
from swarm_pipeline import runner as base
from swarm_pipeline.answer_format import canonical_answer

PROTOCOL='''You get one problem per turn. Solve it and reply with exactly one JSON object, nothing else:
{"action":"submit","task_id":"<problem id>","answer":"<final answer>","notes":"<short description of your solution, 2-3 sentences>"}
Then you get your next problem.'''
INSTRUCTION='Solve your current problem. Return one submit action with the answer and a short description of your solution (2-3 sentences).'


def condition_dir(condition,unique_start=False,shared_random_order=False,tasks_per_agent=None):
    """Run directory name: fixed, fixed_random_order, shuffled, shuffled_unique_start (+ _k<N> with --tasks-per-agent)."""
    name='shuffled_unique_start' if unique_start else 'fixed_random_order' if shared_random_order else condition
    return name+(f'_k{tasks_per_agent}' if tasks_per_agent is not None else '')


class SequentialRunner(ParallelRunner):
    def __init__(self,*args,unique_start=False,shared_random_order=False,tasks_per_agent=None,**kwargs):
        super().__init__(*args,**kwargs)
        if self.team:raise ValueError('Sequential control requires an independent condition')
        if unique_start and self.config.condition!='shuffled':raise ValueError('--unique-start applies to the shuffled condition only')
        if shared_random_order and self.config.condition!='fixed':raise ValueError('--shared-random-order applies to the fixed condition only')
        if tasks_per_agent is not None and not 1<=tasks_per_agent<=len(self.tasks):raise ValueError('--tasks-per-agent must be between 1 and the number of tasks')
        self.unique_start,self.shared_random_order,self.tasks_per_agent=unique_start,shared_random_order,tasks_per_agent
        self.manifest.update(protocol_version='sequential-control-v1',protocol_hash=base.digest([PROTOCOL,INSTRUCTION]),
            protocol_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if unique_start:self.manifest.update(order_mode='unique-start-then-own-shuffle-v1')
        if shared_random_order:self.manifest.update(order_mode='shared-random-order-v1')
        if tasks_per_agent is not None:self.manifest.update(tasks_per_agent=tasks_per_agent)
        self.directory=self.root/condition_dir(self.config.condition,unique_start,shared_random_order,tasks_per_agent)/f'seed_{self.config.seed}'

    def initial_state(self):
        state=super().initial_state()  # per-condition task order (fixed / shuffled / disjoint)
        if self.shared_random_order:  # one random order per run, identical for every agent
            order=list(self.tasks);random.Random(f'{self.config.seed}:shared').shuffle(order)
            for a in state['agents'].values():a['order']=list(order)
        if self.unique_start:
            starts=list(self.tasks);random.Random(f'{self.config.seed}:starts').shuffle(starts)
            for i,a in state['agents'].items():
                rest=[t for t in self.tasks if t!=starts[int(i)-1]]
                random.Random(f'{self.config.seed}:{i}').shuffle(rest)
                a['order']=[starts[int(i)-1]]+rest
        if self.tasks_per_agent is not None:
            for a in state['agents'].values():a['order']=a['order'][:self.tasks_per_agent]
        for i,a in state['agents'].items():
            a['history'][0]['content']=f'You are Agent {i}. Solve your problems one by one, on your own. There is no team and no chat.\n'+PROTOCOL
        return state

    def context(self,i):
        c=super().context(i)
        c['instruction']=INSTRUCTION
        return c

    def apply(self,q,r):
        a=self.state['agents'][str(q['agent'])]
        a['history'] += [{'role':'user','content':json.dumps(q['context'],ensure_ascii=False)},{'role':'assistant','content':r['content']}]
        a['seen_inbox']=len(q['context']['inbox']);error=None
        try:
            if r['finish_reason']!='stop':raise ValueError('Incomplete provider response: '+str(r['finish_reason']))
            text=r['content'].strip()
            if text.startswith('```') and text.endswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
            action=json.loads(text);task=q['context']['current_task']
            action['answer']=canonical_answer(action.get('answer'))
            if action.get('action')!='submit' or action.get('task_id')!=task or not isinstance(action.get('answer'),str) or not isinstance(action.get('notes'),str) or not action['notes'].strip() or len(action['notes'])>2000:
                raise ValueError('Solving requires one submit for the current task with brief notes under 2000 characters')
            a['answers'][task]=action['answer']
            if all(len(x['answers'])==len(x['order']) for x in self.state['agents'].values()):self.state['status']='completed'
        except (ValueError,KeyError,TypeError,AttributeError) as e:
            error=str(e);a['history'].append({'role':'user','content':'Action rejected: '+error+'. Follow the JSON action protocol.'})
        if r['finish_reason']!='stop':self.state['status']='provider_incomplete'
        self.state['calls']+=1;self.state['next_agent']=q['agent']%self.config.agents+1;self.state['pending']=None
        return {'call':q['call'],'phase':q['context']['phase'],'agent':q['agent'],'deliveries':[],'action_error':error}


class SequentialOfflineClient(base.OfflineClient):
    def complete(self,q):
        return {'content':json.dumps({'action':'submit','task_id':q['context']['current_task'],'answer':'offline-placeholder','notes':'Private explanation.'}),'reasoning':'Private offline reasoning.','finish_reason':'stop','usage':{'prompt_tokens':1,'completion_tokens':1}}


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--workers',type=int,default=10)
    parser.add_argument('--unique-start',action='store_true')
    parser.add_argument('--shared-random-order',action='store_true')
    parser.add_argument('--tasks-per-agent',type=int)
    args,rest=parser.parse_known_args();sys.argv[1:]=rest
    base.Runner=lambda *a,**kw:SequentialRunner(*a,workers=args.workers,partial_continuation=True,unique_start=args.unique_start,
                                                shared_random_order=args.shared_random_order,tasks_per_agent=args.tasks_per_agent,**kw)
    base.CloudClient=RetainingClient;base.OfflineClient=SequentialOfflineClient;base.main()

if __name__=='__main__':main()
