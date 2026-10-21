"""Continue a run halted with status provider_incomplete, without changing its manifest.

For every interrupted solving call that has no recovery event yet, an explicit recovery turn is appended to the
agent's history (the same text as swarm_pipeline/resume_parallel.py, followed by the observed fragment, if any) and
recorded in operational_events/. The state is set back to running; then rerun the same generation command with the
same arguments (same manifest), which resumes from the saved state. Works for the private and sequential controls,
whose manifests resume_parallel cannot reproduce.

  python scripts/restart_halted_run.py <run directory>   # e.g. <root>/rung_3/fixed/seed_83
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swarm_pipeline.runner import digest, output_lock
from swarm_pipeline.supervision import read_json, write_json

NOTE = ("The previous attempt was interrupted and may have been billed. Its outcome is unknown. "
        "This is an explicit recovery turn. Continue solving your current task from your retained "
        "completed notes and answers. No missing output has been fabricated. Return one JSON action.")


def main():
    d = Path(sys.argv[1])
    with output_lock(d.parent.parent):
        state = read_json(d/'state.json')
        assert state['pending'] is None and not state.get('pending_batch'), 'outstanding requests'
        assert state['status'] == 'provider_incomplete', state['status']
        max_bytes = read_json(d/'manifest.json')['config']['max_context_bytes']
        events = d/'operational_events'
        covered = {read_json(p)['source_call'] for p in events.glob('continuation_*.json')}
        repairs = []
        for path in sorted((d/'calls').glob('*/response.json')):
            response, call = read_json(path), int(path.parent.name)
            if response['finish_reason'] == 'stop' or call in covered:
                continue
            request = read_json(path.with_name('request.json'))
            assert request['context']['phase'] == 'solving' and request['context'].get('current_task')
            agent = str(request['agent'])
            if request['context']['current_task'] in state['agents'][agent]['answers']:
                continue
            note = NOTE
            if response.get('reasoning') or response.get('content'):
                note += '\nYour observed incomplete private fragment follows:\n' + (response.get('reasoning') or '') + '\n' + (response.get('content') or '')
            message = {'role': 'user', 'content': note}
            history = state['agents'][agent]['history'] + [message]
            assert len(json.dumps(history, ensure_ascii=False).encode()) <= max_bytes, 'context limit'
            state['agents'][agent]['history'] = history
            repairs.append({'kind': 'explicit_interrupted_turn_restart', 'before_call': state['calls'], 'source_call': call,
                            'agent': request['agent'], 'message': message, 'source_response_hash': digest(response),
                            'at': time.time(), 'original_request_replayed': False,
                            'reason': 'Provider stream interruptions halted the run; user-authorized completion'})
        assert repairs and len(repairs) <= 10, len(repairs)
        write_json(events/f"before_restart_{state['calls']:06d}.state.json", read_json(d/'state.json'))
        for r in repairs:
            write_json(events/f"continuation_{r['source_call']:06d}.json", r)
        state['status'] = 'running'
        write_json(d/'state.json', state)
    print(json.dumps({'dir': str(d), 'resumed_at_call': state['calls'], 'restarted': [(r['source_call'], r['agent']) for r in repairs]}))


if __name__ == '__main__':
    main()
