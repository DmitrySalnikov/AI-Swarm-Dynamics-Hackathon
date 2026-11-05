"""Streaming transport that journals every received text delta with its reception time.

`JournalingClient.complete` is the stream loop of `stable_live.RetainingClient` (kept unchanged, since earlier runners
hash that module) plus one line per non-empty delta in `request['delta_journal']`: delta number, wall time (UTC
epoch seconds), monotonic time, channel (`reasoning` or `content`) and text. Lines are flushed as they arrive, so an
interrupted call keeps the deltas observed before the interruption. These are client reception times, not model
token times. `read_deltas` and `check_deltas` reconstruct and verify a call's text from its journal. Setting
`client.cancel` (a threading.Event) ends every open stream at its next chunk with finish_reason `cancelled`.
"""
from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import swarm_pipeline.runner as base

JOURNAL_SCHEMA = 'delta-journal-v1'


class DeltaJournal:
    def __init__(self, path):
        self.path, self.n, self.stream = (Path(path) if path else None), 0, None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.path.open('a', encoding='utf-8')

    def add(self, channel, text):
        if not text or self.stream is None:
            return
        self.stream.write(json.dumps({'n': self.n, 'at': time.time(), 'mono': time.monotonic(), 'channel': channel,
                                      'text': text}, ensure_ascii=False) + '\n')
        self.stream.flush()
        self.n += 1

    def close(self):
        if self.stream is not None:
            self.stream.close()


def read_deltas(path):
    path = Path(path)
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def check_deltas(deltas, response):
    """True if the journal reproduces the response text of both channels exactly (no gaps, no repeats)."""
    joined = {'reasoning': '', 'content': ''}
    for d in deltas:
        joined[d['channel']] += d['text']
    return joined['reasoning'] == (response.get('reasoning') or '') and joined['content'] == (response.get('content') or '')


def segments(deltas):
    """Consecutive deltas of one channel, in reception order: [{'channel', 'text', 'marks': [[end offset, at], ...]}]."""
    out = []
    for d in deltas:
        if not out or out[-1]['channel'] != d['channel']:
            out.append({'channel': d['channel'], 'text': '', 'marks': []})
        s = out[-1]
        s['text'] += d['text']
        s['marks'].append([len(s['text']), d['at']])
    return out


class JournalingClient(base.CloudClient):
    def __init__(self, key, timeout=900):
        super().__init__(key, timeout=timeout)
        self.cancel = threading.Event()

    def complete(self, request):
        payload = {k: request[k] for k in ('model', 'messages', 'temperature', 'max_tokens', 'seed')}
        payload.update(stream=True, stream_options={'include_usage': True}, reasoning_effort='high')
        req = urllib.request.Request(base.API_URL + '/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        journal = DeltaJournal(request.get('delta_journal'))
        parts, thoughts = [], []
        usage, finish, response_id, model, fingerprint, field = None, None, None, None, None, None
        chunks, done, output_bytes, error = 0, False, 0, None
        status_code, retry_after = None, None
        first_chunk, last_chunk, max_gap, previous = None, None, 0, None
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as stream:
                for raw in stream:
                    line = raw.decode('utf-8').strip()
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        done = True
                        break
                    chunk = json.loads(data)
                    now = time.time()
                    first_chunk = first_chunk or now
                    if previous is not None:
                        max_gap = max(max_gap, now - previous)
                    previous = last_chunk = now
                    chunks += 1
                    usage = chunk.get('usage') or usage
                    response_id = chunk.get('id') or response_id
                    model = chunk.get('model') or model
                    fingerprint = chunk.get('system_fingerprint') or fingerprint
                    for choice in chunk.get('choices', []):
                        delta = choice.get('delta', {})
                        content = delta.get('content') or ''
                        thought = delta.get('reasoning_content') or delta.get('reasoning') or ''
                        if thought:
                            field = 'reasoning_content' if delta.get('reasoning_content') else 'reasoning'
                        journal.add('reasoning', thought)
                        journal.add('content', content)
                        parts.append(content)
                        thoughts.append(thought)
                        output_bytes += len((content + thought).encode())
                        finish = choice.get('finish_reason') or finish
                    if output_bytes > request['max_tokens'] * 4:
                        finish = 'client_output_limit'
                        break
                    if self.cancel.is_set():
                        finish = 'cancelled'
                        break
        except urllib.error.HTTPError as exc:
            error = type(exc).__name__
            status_code = exc.code
            retry_after = exc.headers.get('Retry-After')
            finish = 'transport_error'
        except Exception as exc:
            error = type(exc).__name__
            finish = 'transport_error'
        finally:
            journal.close()
        if finish is None:
            finish = 'stream_incomplete'
        return {'content': ''.join(parts), 'reasoning': ''.join(thoughts), 'usage': usage,
                'finish_reason': finish, 'response_id': response_id, 'actual_model': model,
                'system_fingerprint': fingerprint, 'reasoning_field': field, 'error_type': error,
                'http_status': status_code, 'retry_after': retry_after, 'delta_journal': JOURNAL_SCHEMA,
                'stream_diagnostics': {'chunks': chunks, 'done_seen': done, 'first_chunk_at': first_chunk,
                                       'last_chunk_at': last_chunk, 'max_gap_seconds': max_gap,
                                       'received_output_bytes': output_bytes}}
