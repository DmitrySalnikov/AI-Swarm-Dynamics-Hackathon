"""Distribution repeats and a causal, completion-driven exchange pilot."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
from pathlib import Path
import time

from swarm_pipeline import runner as base
from swarm_pipeline.protocol_v2 import V2Runner, V2OfflineClient, PROTOCOL
from swarm_pipeline.supervision import read_json, write_json


class DistributionRunner(V2Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest['protocol_version'] = 'distribution-only-v2'
        self.manifest['pilot_source_hash'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def apply(self, request, response):
        event = super().apply(request, response)
        if self.state['phase'] == 'solving' and self.state['status'] == 'running':
            self.state['status'] = 'completed'
        return event


ASYNC_PROTOCOL = PROTOCOL.replace(
    'Your answer and the short description are sent to the team chat.',
    'Immediately after submitting, you will be asked to report to the shared chat while peers may still be solving. '
    'Use a send action to all with a brief result and explanation. Read already published messages in that request. '
    'The coordinator does not inject messages into requests already in flight. '
    'After everyone reports, an identical shared packet is delivered for final commits.')


class GradualRunner(V2Runner):
    """Apply each receipt immediately; one outstanding request per agent."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.manifest.update(protocol_version='gradual-exchange-v1', protocol_hash=base.digest(ASYNC_PROTOCOL),
                             pilot_source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.manifest['execution']['scheduler'] = 'completion-driven-v1'

    def initial_state(self):
        state = super().initial_state()
        state.update(next_call=0, pending_calls=[], applied_order=[], stage_attempts={})
        for i, agent in state['agents'].items():
            agent['history'][0]['content'] = f'You are Agent {i}. Distribute tasks, solve privately, report immediately, and commit common results.\n' + ASYNC_PROTOCOL
        return state

    def context(self, i):
        context = super().context(i)
        agent = self.state['agents'][str(i)]
        if self.state['phase'] != 'negotiation' and agent['answers'] and not agent['reported']:
            context['phase'] = 'report'
            context.pop('shared_packet', None)
            context.pop('shared_packet_hash', None)
            context['own_notes'] = agent['solution_notes']
            context['instruction'] = ('You have finished your assigned task. Report now while peers may still be solving. '
                'Read the available shared chat and send your result and brief explanation: '
                '{"action":"send","to":"all","message":"your result and explanation"}. '
                'Do not solve again or change submitted answers.')
        return json.loads(json.dumps(context))

    def eligible(self):
        if self.state['phase'] == 'negotiation':
            return [self.next_agent()]
        all_reported = all(a['reported'] for a in self.state['agents'].values())
        return [int(i) for i, a in self.state['agents'].items()
                if not a['finished'] and (not a['reported'] or all_reported)]

    def apply(self, request, response):
        if request['context']['phase'] != 'report':
            return super().apply(request, response)
        agent = self.state['agents'][str(request['agent'])]
        agent['history'] += [{'role': 'user', 'content': json.dumps(request['context'], ensure_ascii=False)},
                             {'role': 'assistant', 'content': response['content']}]
        agent['seen_inbox'] = len(request['context']['inbox'])
        deliveries, error = [], None
        try:
            if response['finish_reason'] != 'stop':
                raise ValueError('Incomplete provider response: ' + str(response['finish_reason']))
            text = response['content'].strip()
            if text.startswith('```') and text.endswith('```'):
                text = text.split('\n', 1)[1].rsplit('```', 1)[0]
            action = json.loads(text)
            if (action.get('action') != 'send' or action.get('to') != 'all'
                    or not isinstance(action.get('message'), str) or not action['message'].strip()):
                raise ValueError('Report requires one nonempty send action to all')
            for recipient, record in self.state['agents'].items():
                if int(recipient) == request['agent']:
                    continue
                message = {'from': request['agent'], 'message': action['message'], 'call': request['call']}
                record['inbox'].append(message)
                deliveries.append({'to': int(recipient), **message})
            agent['reported'] = True
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            error = str(exc)
            agent['history'].append({'role': 'user', 'content': 'Action rejected: ' + error + '. Follow the JSON action protocol.'})
        if response['finish_reason'] != 'stop':
            self.state['status'] = 'provider_incomplete'
        self.state['calls'] += 1
        self.state['pending'] = None
        return {'call': request['call'], 'phase': 'report', 'agent': request['agent'],
                'deliveries': deliveries, 'action_error': error}

    def consume(self, request, response):
        call = request['call']
        folder = self.directory / 'calls' / f'{call:06d}'
        write_json(folder / 'response.json', response)
        event = self.apply(request, response)
        event['applied_at'] = time.time()
        self.state['applied_order'].append(call)
        self.state['pending_calls'].remove(call)
        if response['finish_reason'] != 'stop' or event['action_error']:
            key = f"{request['agent']}:{request['context']['phase']}"
            attempts = self.state['stage_attempts'].get(key, 0) + 1
            self.state['stage_attempts'][key] = attempts
            # Preserve partial output as an explicitly observed fragment, never as a successful answer.
            if response['finish_reason'] != 'stop':
                fragment = '\n'.join(response.get(k, '') for k in ('reasoning', 'content'))
                message = {'role': 'user', 'content': 'The previous request was interrupted. Its billing may be estimated. '
                           'Continue the same stage from this observed fragment and return a valid action:\n' + fragment}
                self.state['agents'][str(request['agent'])]['history'].append(message)
                write_json(self.directory / 'operational_events' / f'continuation_{call:06d}.json',
                           {'source_call': call, 'message': message, 'attempt': attempts, 'at': time.time()})
            self.state['status'] = 'running' if attempts <= 3 else 'retry_limit'
        write_json(folder / 'event.json', event)
        write_json(self.directory / 'state.json', self.state)
        print(f"call {call}: agent {request['agent']} {event['phase']}; {self.state['status']}; error={event['action_error']}", flush=True)

    def run(self):
        with base.output_lock(self.root), ThreadPoolExecutor(max_workers=self.workers) as pool:
            manifest_path = self.directory / 'manifest.json'
            state_path = self.directory / 'state.json'
            if manifest_path.exists() and read_json(manifest_path) != self.manifest:
                raise ValueError('Manifest differs; use a new output directory')
            write_json(manifest_path, self.manifest)
            self.state = read_json(state_path) if state_path.exists() else self.initial_state()
            # Never replay a call whose provider outcome is unknown.
            known = set(self.state['applied_order'])
            journals = sorted((self.directory / 'calls').glob('*/request.json'))
            for path in journals:
                request = read_json(path)
                if request['call'] in known:
                    continue
                if not path.with_name('response.json').exists():
                    raise RuntimeError('Unknown pending request outcome: ' + str(path))
                if request['call'] not in self.state['pending_calls']:
                    self.state['pending_calls'].append(request['call'])
                self.consume(request, read_json(path.with_name('response.json')))
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
                        messages = self.state['agents'][str(i)]['history'] + [{'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}]
                        size = len(json.dumps(messages, ensure_ascii=False).encode())
                        reserved = size * self.config.price_in + self.config.max_tokens * self.config.price_out
                        if size > self.config.max_context_bytes:
                            self.state['status'] = 'context_limit'; break
                        if not self.config.offline and self.spent() + reserved > self.config.budget_rub:
                            self.state['status'] = 'budget_limit'; break
                        call = self.state['next_call']
                        request = {'call': call, 'agent': i, 'context': context, 'messages': messages,
                            'model': self.config.model, 'temperature': self.config.temperature, 'max_tokens': self.config.max_tokens,
                            'seed': int(base.digest([self.config.seed, i, call])[:8], 16) % (2**31-1),
                            'reserved_rub': reserved, 'started_at': time.time()}
                        write_json(self.directory / 'calls' / f'{call:06d}' / 'request.json', request)
                        self.state['next_call'] += 1
                        self.state['pending_calls'].append(call)
                        write_json(state_path, self.state)
                        futures[pool.submit(self.complete, request)] = request
                if not futures:
                    break
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda f: f.result()['completed_at']):
                    request = futures.pop(future)
                    self.consume(request, future.result())
            self.state['pending'] = None
            write_json(state_path, self.state)
            result = self.results()
            write_json(self.directory / 'result.json', result)
            self.export()
            return result


class GradualOfflineClient(V2OfflineClient):
    def complete(self, request):
        if request['context']['phase'] == 'report':
            return {'content': json.dumps({'action': 'send', 'to': 'all', 'message': str(request['context']['own_answers'])}),
                    'reasoning': 'Scripted report reasoning.', 'finish_reason': 'stop',
                    'usage': {'prompt_tokens': 1, 'completion_tokens': 1}}
        return super().complete(request)
