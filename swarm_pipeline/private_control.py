"""One matched task per independent agent; no negotiation or peer exchange."""
import argparse,hashlib,json,sys
from pathlib import Path
from swarm_pipeline.parallel import ParallelRunner
from swarm_pipeline.stable_live import RetainingClient
from swarm_pipeline import runner as base
from swarm_pipeline.answer_format import canonical_answer

PROTOCOL='''Return exactly one JSON object without markdown fences.
Solve your assigned task privately in one turn and return
{"action":"submit","task_id":"your task","answer":"answer","notes":"brief explanation"}.
Keep notes under 2000 characters. No work or send actions. There is no peer
communication or shared result packet. Submit only your assigned task.'''

class PrivateRunner(ParallelRunner):
    def __init__(self,*args,assignments,**kwargs):
        super().__init__(*args,**kwargs)
        if self.team:raise ValueError('Private control requires an independent condition')
        self.assignments=assignments
        assert set(assignments)=={str(i) for i in range(1,self.config.agents+1)}
        assert sorted(assignments.values())==sorted(self.tasks)
        self.manifest.update(protocol_version='private-single-task-v1',assignments=assignments,
            protocol_hash=base.digest(PROTOCOL),protocol_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def initial_state(self):
        state=super().initial_state()
        for i,a in state['agents'].items():
            a['order']=[self.assignments[i]]
            a['history'][0]['content']=f'You are Agent {i}. Solve independently.\n'+PROTOCOL
        return state

    def context(self,i):
        c=super().context(i)
        c['instruction']='Solve your assigned task privately. Return one submit action with an answer and a brief explanation under 2000 characters. Do not message peers or return work actions.'
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
                raise ValueError('Solving requires one submit for the assigned task with brief notes under 2000 characters')
            a['answers'][task]=action['answer']
            if all(len(x['answers'])==1 for x in self.state['agents'].values()):self.state['status']='completed'
        except (ValueError,KeyError,TypeError,AttributeError) as e:
            error=str(e);a['history'].append({'role':'user','content':'Action rejected: '+error+'. Follow the JSON action protocol.'})
        if r['finish_reason']!='stop':self.state['status']='provider_incomplete'
        self.state['calls']+=1;self.state['next_agent']=q['agent']%self.config.agents+1;self.state['pending']=None
        return {'call':q['call'],'phase':q['context']['phase'],'agent':q['agent'],'deliveries':[],'action_error':error}

class PrivateOfflineClient(base.OfflineClient):
    def complete(self,q):
        return {'content':json.dumps({'action':'submit','task_id':q['context']['current_task'],'answer':'offline-placeholder','notes':'Private explanation.'}),'reasoning':'Private offline reasoning.','finish_reason':'stop','usage':{'prompt_tokens':1,'completion_tokens':1}}

def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--assignments',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=10)
    args,rest=parser.parse_known_args();sys.argv[1:]=rest
    mapping=json.loads(args.assignments.read_text())
    base.Runner=lambda *a,**kw:PrivateRunner(*a,assignments=mapping,workers=args.workers,partial_continuation=True,**kw)
    base.CloudClient=RetainingClient;base.OfflineClient=PrivateOfflineClient;base.main()

if __name__=='__main__':main()
