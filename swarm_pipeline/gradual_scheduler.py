"""Completion-driven scheduling with reservation backpressure and sticky stops."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import time

from swarm_pipeline import runner as base
from swarm_pipeline.window_pilot import GradualRunner
from swarm_pipeline.supervision import read_json, write_json


class BudgetAwareGradualRunner(GradualRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest['execution']['scheduler'] = 'completion-driven-backpressure-v2'
        self.manifest['scheduler_source_hash'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def run(self):
        with base.output_lock(self.root), ThreadPoolExecutor(max_workers=self.workers) as pool:
            manifest_path = self.directory / 'manifest.json'
            state_path = self.directory / 'state.json'
            if manifest_path.exists() and read_json(manifest_path) != self.manifest:
                raise ValueError('Manifest differs; use a new output directory')
            write_json(manifest_path, self.manifest)
            self.state = read_json(state_path) if state_path.exists() else self.initial_state()
            known = set(self.state['applied_order'])
            for path in sorted((self.directory / 'calls').glob('*/request.json')):
                q = read_json(path)
                if q['call'] in known:
                    continue
                if not path.with_name('response.json').exists():
                    raise RuntimeError('Unknown pending request outcome: ' + str(path))
                if q['call'] not in self.state['pending_calls']:
                    self.state['pending_calls'].append(q['call'])
                self.consume(q, read_json(path.with_name('response.json')))
            futures = {}
            while self.state['status'] == 'running' or futures:
                busy = {q['agent'] for q in futures.values()}
                if self.state['status'] == 'running':
                    for i in self.eligible():
                        if i in busy or len(futures) >= self.workers:
                            continue
                        if self.state['next_call'] >= self.config.max_calls:
                            self.state['status'] = 'call_limit'; break
                        context = self.context(i)
                        messages = self.state['agents'][str(i)]['history'] + [{'role':'user','content':json.dumps(context,ensure_ascii=False)}]
                        size = len(json.dumps(messages,ensure_ascii=False).encode())
                        reserved = size*self.config.price_in + self.config.max_tokens*self.config.price_out
                        if size > self.config.max_context_bytes:
                            self.state['status'] = 'context_limit'; break
                        if not self.config.offline and self.spent()+reserved > self.config.budget_rub:
                            # Pending calls hold conservative maximum reservations.
                            # Wait for their actual receipts before declaring a permanent stop.
                            if not futures:
                                self.state['status'] = 'budget_limit'
                            break
                        call = self.state['next_call']
                        q = {'call':call,'agent':i,'context':context,'messages':messages,
                             'model':self.config.model,'temperature':self.config.temperature,'max_tokens':self.config.max_tokens,
                             'seed':int(base.digest([self.config.seed,i,call])[:8],16)%(2**31-1),
                             'reserved_rub':reserved,'started_at':time.time()}
                        write_json(self.directory/'calls'/f'{call:06d}'/'request.json',q)
                        self.state['next_call'] += 1; self.state['pending_calls'].append(call)
                        write_json(state_path,self.state)
                        futures[pool.submit(self.complete,q)] = q
                if not futures:
                    break
                done, _ = wait(futures,return_when=FIRST_COMPLETED)
                for future in sorted(done,key=lambda f:f.result()['completed_at']):
                    q = futures.pop(future)
                    prior_status = self.state['status']
                    self.consume(q,future.result())
                    if prior_status not in {'running','completed'}:
                        self.state['status'] = prior_status
                        write_json(state_path,self.state)
            self.state['pending'] = None
            write_json(state_path,self.state)
            result = self.results(); write_json(self.directory/'result.json',result)
            self.export()
            return result
