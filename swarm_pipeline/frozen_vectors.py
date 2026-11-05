"""Frozen agent vectors on a swarm-token axis.

Every agent's stream is all text it generated, call by call; inside a call its channel segments (reasoning, visible
content) follow in reception order. The primary view embeds both channels, the visible view (`channels=('content',)`)
only the visible replies; neither contains prompts, received messages or replayed history. Each character gets the
reception time of the streamed delta that carried it (calls/<n>/deltas.jsonl, `timing: journal`); a call without a
journal is spread evenly between its first and last streamed chunk (`timing: interpolated`, older runs). A token is
final when the character after it has arrived (the separator after a segment counts as arriving with the segment's
last delta), so a window is available only once its last token boundary has been observed. An interrupted call and
its continuation are two calls: the continuation's tokens follow the fragment, each with its own journal times.

An agent's vector appears when the agent has generated `first` tokens and is replaced every further `step` tokens by
the embedding of its last `width` tokens; between updates it stays frozen, also while the agent waits and generates
nothing. The axis is swarm tokens: the number of tokens all agents of the run have generated up to that moment
(swarm_clock), so waiting does not move it and only generation does. At every point, R = |mean of the current unit
vectors| over the agents that already have a vector (N of them), with the age of those vectors (axis units since
their last update) and how many updated since the previous point.

Times are seconds from the run start. A baseline run of the covert-channel experiment imports the prelude of its
swarm twin (twin clock); its own calls and events are shifted by state['clock_shift'] so that its start falls on the
twin's barrier.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from swarm_pipeline.analyze import token_data
from swarm_pipeline.supervision import read_json

FIRST, STEP, WIDTH = 1000, 500, 1000


def run_clock(directory):
    """(zero, shift): times in the generated index and events minus zero are seconds from the run start; own (not
    imported) times are first moved by shift (non-zero only for a baseline run, onto its swarm twin's clock)."""
    state = read_json(Path(directory)/'state.json')
    if state.get('twin_started_at') is not None:
        return state['twin_started_at'], state.get('clock_shift', state['twin_started_at'] - state['started_at'])
    return state['started_at'], 0.0


def _char_times(index, n, zero, shift):
    """Reception time (s from the run start) of every character of an agent's generated text."""
    t = np.zeros(n)
    calls = {}
    for p in index:
        calls.setdefault((p['imported'], p['call']), []).append(p)
    for parts in calls.values():
        move = 0.0 if parts[0]['imported'] else shift
        if all(p.get('timing') == 'journal' and p.get('marks') for p in parts):
            for p in parts:
                previous = 0
                for end, at in p['marks']:
                    t[p['start'] + previous:p['start'] + end] = at + move - zero
                    previous = end
                t[p['end']:p['end'] + 2] = p['marks'][-1][1] + move - zero
        else:
            p = parts[0]
            a = (p.get('first_chunk_at') or p['started_at']) + move - zero
            b = (p.get('last_chunk_at') or p['completed_at']) + move - zero
            lo, hi = min(q['start'] for q in parts), max(q['end'] for q in parts) + 2
            t[lo:hi] = a + (b - a)*np.arange(hi - lo)/max(hi - lo - 1, 1)
    return np.maximum.accumulate(t) if n else t


def agent_streams(directory, tokenizer, channels=None):
    """Per agent: text, token offsets, token times (s from run start), phase, task and channel of every token, call
    spans. channels=None keeps every segment; ('content',) gives the visible view."""
    directory = Path(directory)
    zero, shift = run_clock(directory)
    state = read_json(directory/'state.json')
    out = []
    for aid in sorted(state['agents'], key=int):
        path = directory/'generated'/f'agent_{aid}.generated.txt'
        full, index = path.read_text(encoding='utf-8'), read_json(path.with_suffix('.index.json'))
        times_full = _char_times(index, len(full), zero, shift)
        keep = [p for p in index if channels is None or p['part'] in channels]
        text, times, owners, segs, calls = '', [], [], [], {}
        for p in keep:  # rebuild the (possibly filtered) text with its separators and character times
            seg = dict(p, start=len(text), end=len(text) + p['end'] - p['start'])
            text += full[p['start']:p['end'] + 2]; times.append(times_full[p['start']:p['end'] + 2])
            owners.append(np.full(p['end'] - p['start'] + 2, calls.setdefault((p['imported'], p['call']), len(calls))))
            segs.append(seg)
        char_t = np.concatenate(times) if times else np.zeros(0)
        owner = np.concatenate(owners) if owners else np.zeros(0, dtype=int)
        offsets = token_data(tokenizer, text)
        if len(offsets):
            # final when the next character of the same call has arrived; the last token of a call at the call's end
            last, nxt = offsets[:, 1] - 1, np.minimum(offsets[:, 1], len(text) - 1)
            same = owner[nxt] == owner[last]
            token_t = np.maximum.accumulate(np.where(same, char_t[nxt], char_t[last]))
            starts = np.array([s['start'] for s in segs])
            k = np.clip(np.searchsorted(starts, offsets[:, 0], side='right') - 1, 0, len(segs) - 1)
        else:
            token_t, k = np.zeros(0), np.zeros(0, dtype=int)
        spans = {}
        for s in segs:
            key = (s['imported'], s['call'])
            lo, hi = float(char_t[s['start']]) if s['end'] > s['start'] else None, float(char_t[max(s['end'] - 1, s['start'])])
            span = spans.setdefault(key, {'call': s['call'], 'imported': s['imported'], 'phase': s['phase'], 'task': s.get('task_id'),
                                          't0': lo, 't1': hi, 'timing': s.get('timing', 'interpolated')})
            span['t0'] = lo if span['t0'] is None else min(span['t0'], lo if lo is not None else span['t0'])
            span['t1'] = max(span['t1'], hi)
        out.append({'agent': int(aid), 'text': text, 'offsets': offsets, 'times': token_t,
                    'phase': [segs[j]['phase'] for j in k], 'task': [segs[j].get('task_id') for j in k],
                    'channel': [segs[j]['part'] for j in k],
                    'spans': sorted(spans.values(), key=lambda s: (s['t0'] if s['t0'] is not None else s['t1'])),
                    'timing': sorted({s.get('timing', 'interpolated') for s in segs})})
    return out


def swarm_clock(streams):
    """Sorted times of every token of every agent; swarm_tokens(clock, t) = tokens the run has generated by t."""
    return np.sort(np.concatenate([s['times'] for s in streams])) if streams else np.zeros(0)


def swarm_tokens(clock, t):
    return int(np.searchsorted(clock, t, side='right'))


def updates(stream, first=FIRST, step=STEP, width=WIDTH):
    """Vector updates of one agent: availability time, token range and text of the window that replaces its vector."""
    n, out = len(stream['offsets']), []
    for end in range(first, n + 1, step):
        start = max(0, end - width)
        o = stream['offsets']
        out.append({'agent': stream['agent'], 't': float(stream['times'][end - 1]), 'start': start, 'end': end,
                    'chars': [int(o[start, 0]), int(o[end - 1, 1])],
                    'text': stream['text'][int(o[start, 0]):int(o[end - 1, 1])],
                    'phase': stream['phase'][end - 1], 'task': stream['task'][end - 1],
                    'phases': sorted(set(stream['phase'][start:end])), 'tasks': sorted({str(x) for x in stream['task'][start:end]}),
                    'channels': sorted(set(stream['channel'][start:end]))})
    return out


def window_records(per_agent_updates, key='g'):
    """windows.jsonl rows: agent, token and character ranges, content hash, channels, availability time and axis value."""
    return [{'agent': u['agent'], 'start': u['start'], 'end': u['end'], 'chars': u['chars'],
             'sha256': hashlib.sha256(u['text'].encode()).hexdigest(), 'channels': u['channels'], 'phases': u['phases'], 'phase': u['phase'],
             'tasks': u.get('tasks'),
             'task': u['task'], 'available_s': u['t'], key: u.get(key)} for ups in per_agent_updates for u in ups]


def canonical_points(per_agent_updates, key='g'):
    """The union of all vector-update positions: the stepwise trajectory changes only there."""
    return sorted({u[key] for ups in per_agent_updates for u in ups})


def geometry(V):
    """Metrics of unit vectors V (N×d), as embedding_eval.dynamics.metrics_at: R = |m|; the total spread
    mean|v - m|^2 = 1 - R^2 splits into var_par (variance of the projections on the mean direction) and var_perp (the
    rest, across the mean direction); eff_dim = participation ratio of the covariance spectrum, via the N×N Gram matrix
    of the centred vectors (1 when the agents differ along one line, at most N - 1)."""
    m = V.mean(0); R = float(np.linalg.norm(m))
    var_par = float((V @ (m/max(R, 1e-12))).var())
    Xc = V - m
    lam = np.clip(np.linalg.eigvalsh(Xc @ Xc.T/len(V)), 0, None)
    eff = float(lam.sum()**2/(lam**2).sum()) if (lam**2).sum() > 1e-24 else None
    return {'R': R, 'var_par': var_par, 'var_perp': 1 - R*R - var_par, 'eff_dim': eff}


def timeline(per_agent_updates, vectors, grid, key='t'):
    """R, var_perp, eff_dim (geometry), N, vector age and the phase mix of the current (frozen) vectors at every grid
    point; key is the update field on the grid axis ('t' seconds or 'g' swarm tokens) and names the axis field of the
    rows."""
    from swarm_pipeline.analyze import vector_key
    rows = []
    cursors = [0]*len(per_agent_updates)
    for t in grid:
        current, updated = [], 0
        for j, ups in enumerate(per_agent_updates):
            before = cursors[j]
            while cursors[j] < len(ups) and ups[cursors[j]][key] <= t:
                cursors[j] += 1
            updated += cursors[j] > before
            if cursors[j]:
                current.append(ups[cursors[j] - 1])
        if not current:
            rows.append({key: float(t), 'R': None, 'var_perp': None, 'var_par': None, 'eff_dim': None, 'N': 0, 'phases': {},
                         'updated': 0})
            continue
        V = np.stack([vectors[vector_key(u['text'])] for u in current])
        V = V/np.linalg.norm(V, axis=1, keepdims=True)
        phases = {}
        for u in current:
            phases[u['phase']] = phases.get(u['phase'], 0) + 1
        tasks = [u['task'] for u in current]
        ages = [t - u[key] for u in current]
        rows.append({key: float(t), **geometry(V), 'N': len(current), 'phases': phases,
                     'same_task_share': max(tasks.count(x) for x in set(tasks))/len(tasks), 'updated': updated,
                     'age_mean': float(np.mean(ages)), 'age_max': float(np.max(ages))})
    return rows


def stage_changes(directory):
    """Per agent: [(seconds from the run start, stage)] from events.jsonl, starting with the prelude at 0 (for a
    baseline the imported prelude ends are on the twin clock)."""
    directory = Path(directory)
    zero, shift = run_clock(directory)
    state = read_json(directory/'state.json')
    out = {int(a): [(0.0, 'prelude')] for a in state['agents']}
    path = directory/'events.jsonl'
    events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    for e in events:
        if e['kind'] == 'stage':
            t = e['twin_at'] - zero if e.get('imported') else e['at'] + shift - zero
            out[e['agent']].append((t, e['after']))
    return {a: sorted(v, key=lambda x: x[0]) for a, v in out.items()}
