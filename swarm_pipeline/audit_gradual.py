"""Replay asynchronous dispatch/application order and verify frozen contexts."""
from dataclasses import fields
from pathlib import Path
import tempfile

from swarm_pipeline.runner import Config
from swarm_pipeline.window_pilot import GradualRunner, GradualOfflineClient
from swarm_pipeline.supervision import read_json, write_json


def audit(directory):
    directory = Path(directory)
    manifest, state = read_json(directory / 'manifest.json'), read_json(directory / 'state.json')
    assert state['status'] == 'completed' and not state['pending_calls']
    requests, responses, events = {}, {}, {}
    for path in sorted(directory.glob('calls/*/request.json')):
        q = read_json(path); call = q['call']
        requests[call] = q
        responses[call] = read_json(path.with_name('response.json'))
        events[call] = read_json(path.with_name('event.json'))
    assert sorted(requests) == list(range(len(requests)))
    assert sorted(state['applied_order']) == sorted(requests)
    repairs = {r['source_call']: r for r in
               (read_json(p) for p in directory.glob('operational_events/continuation_*.json'))}
    timeline = sorted([(q['started_at'], 0, call) for call, q in requests.items()] +
                      [(e['applied_at'], 1, call) for call, e in events.items()])
    pending = {}
    with tempfile.TemporaryDirectory() as temp:
        config = Config(**{f.name: manifest['config'][f.name] for f in fields(Config)})
        replay = GradualRunner(manifest['tasks'], config, Path(temp), GradualOfflineClient(), workers=10)
        replay.state = replay.initial_state()
        for _, kind, call in timeline:
            q = requests[call]; aid = str(q['agent'])
            agent = replay.state['agents'][aid]
            if kind == 0:
                assert aid not in pending, 'Two outstanding requests for one agent'
                assert q['context'] == replay.context(q['agent']), f'Context differs at dispatch {call}'
                assert q['messages'][:-1] == agent['history'], f'History differs at dispatch {call}'
                assert q['context']['inbox'] == agent['inbox'], f'Inbox differs at dispatch {call}'
                pending[aid] = call
            else:
                assert pending.pop(aid) == call
                e = replay.apply(q, responses[call])
                for key in ('call', 'agent', 'phase', 'deliveries', 'action_error'):
                    assert e[key] == events[call][key], f'Event replay differs: {call} {key}'
                if call in repairs:
                    agent['history'].append(repairs[call]['message'])
                    replay.state['status'] = 'running'
        assert not pending
        assert replay.state['agents'] == state['agents']
        assert replay.state['shared_packet'] == state['shared_packet']
        assert replay.state['status'] == 'completed'
    for aid in state['agents']:
        text = ''.join(r.get(field, '') + '\n\n'
                       for call, r in sorted(responses.items()) if str(requests[call]['agent']) == aid
                       for field in ('reasoning', 'content') if r.get(field))
        assert text == (directory / 'generated' / f'agent_{aid}.generated.txt').read_text(encoding='utf-8')
    solves = [e for e in events.values() if e['phase'] == 'solving' and not e['action_error']]
    reports = [e for e in events.values() if e['phase'] == 'report' and not e['action_error']]
    last_solve = max(e['applied_at'] for e in solves)
    report_calls = {e['call'] for e in reports}
    with_prior_reports = sum(any(m.get('call') in report_calls for m in requests[e['call']]['context']['inbox']) for e in reports)
    record = {'completed': True, 'calls': len(requests), 'context_history_delivery_replay': True,
              'exports_match_observed_output': True, 'successful_solves': len(solves),
              'successful_reports': len(reports), 'reports_before_last_solve': sum(e['applied_at'] < last_solve for e in reports),
              'reports_reading_prior_reports': with_prior_reports,
              'action_errors': [{'call': e['call'], 'error': e['action_error']} for e in events.values() if e['action_error']],
              'provider_interruptions': sum(r['finish_reason'] != 'stop' for r in responses.values()),
              'cost_rub': sum(r['cost_rub'] for r in responses.values()),
              'estimated_cost_calls': [call for call, r in responses.items() if r.get('cost_estimated')]}
    write_json(directory / 'audit-gradual.json', record)
    return record
