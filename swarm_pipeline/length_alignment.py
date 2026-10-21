"""Length alignment of solving streams for runs with one task per agent (swarm solving phase, one-task control).

A stream is one agent's solving output (reasoning + visible response of every solving call, in call order, joined
by blank lines), tokenized with the analysis tokenizer. When agents produce different amounts of text, fewer of them
remain at later token positions and R over fewer vectors is biased upwards. Two alignments:

  (a) trim        cut every stream of a run to the shortest one; only ten-author positions remain. Nothing is
                  deleted from the logs; the cut text is only excluded from the windows. trim_facts() reports how
                  much text was cut per agent.
  (b) supplement  extend every shorter stream to the longest one of its run with independent solutions of the same
                  task (same task id and prompt) by other agents in other runs (the sequential controls). Donor texts
                  are whole per-task outputs of one agent; each donor is used at most once over all recipients; the
                  last donor is cut at the target length. Provenance (run, agent, task, sha256, tokens) is recorded.
                  This is SYNTHETIC text: it was never produced by the recipient and must be labeled as such.
"""
import hashlib
from pathlib import Path

import numpy as np

from swarm_pipeline.analyze import token_data
from swarm_pipeline.supervision import read_json

SEP = '\n\n'


def _pieces(directory, aid):
    path = Path(directory)/'generated'/f'agent_{aid}.generated.txt'
    text, index = path.read_text(encoding='utf-8'), read_json(path.with_suffix('.index.json'))
    pieces = sorted((p for p in index if p['phase'] == 'solving'), key=lambda p: (p['call'], p['start']))
    return [(text[p['start']:p['end']], p.get('task_id')) for p in pieces]


def solving_streams(directory, tokenizer):
    """Per agent: solving text in call order, token offsets, the task id of every token, original token count."""
    state = read_json(Path(directory)/'state.json')
    assert state['status'] == 'completed', directory
    out = []
    for aid in sorted(state['agents'], key=int):
        parts, ends = [], []
        for chunk, task in _pieces(directory, aid):
            parts.append(chunk)
            ends.append((len(SEP.join(parts)), task))
        joined = SEP.join(parts)
        offsets = token_data(tokenizer, joined)
        bounds = np.array([e for e, _ in ends]); ids = [t for _, t in ends]
        piece = np.minimum(np.searchsorted(bounds, offsets[:, 0], side='right'), len(ids) - 1) if len(offsets) else []
        token_task = [ids[i] for i in piece]
        out.append({'agent': int(aid), 'text': joined, 'offsets': offsets, 'token_task': token_task,
                    'original_tokens': len(offsets), 'synthetic_from': len(offsets)})
    return out


def trim(streams):
    """(a) Every stream cut to the shortest stream of the run (text kept, offsets cut)."""
    n = min(len(s['offsets']) for s in streams)
    return [{**s, 'offsets': s['offsets'][:n], 'token_task': s['token_task'][:n]} for s in streams]


def trim_facts(streams):
    n = min(len(s['offsets']) for s in streams)
    cut = [len(s['offsets']) - n for s in streams]
    total = sum(len(s['offsets']) for s in streams)
    return {'kept_tokens_per_agent': n, 'cut_tokens_per_agent': cut, 'cut_tokens_total': sum(cut),
            'original_tokens_total': total, 'cut_share': sum(cut)/total if total else 0.0}


def donor_pool(directories, tokenizer):
    """Task id -> donors: the whole output of one agent on one task in a completed run (prompt hash kept for checks)."""
    pool = {}
    for directory in directories:
        directory = Path(directory)
        state, manifest = read_json(directory/'state.json'), read_json(directory/'manifest.json')
        assert state['status'] == 'completed', directory
        prompts = {t['id']: t['prompt'] for t in manifest['tasks']}
        for aid in sorted(state['agents'], key=int):
            grouped = {}
            for chunk, task in _pieces(directory, aid):
                grouped.setdefault(task, []).append(chunk)
            for task, chunks in grouped.items():
                text = SEP.join(chunks)
                pool.setdefault(task, []).append({
                    'text': text, 'run': str(directory), 'agent': int(aid), 'task': task,
                    'prompt_sha256': hashlib.sha256(prompts[task].encode()).hexdigest(),
                    'sha256': hashlib.sha256(text.encode()).hexdigest(), 'tokens': len(token_data(tokenizer, text))})
    for task in pool:  # a fixed pseudo-random order per task, independent of directory order
        pool[task].sort(key=lambda d: hashlib.sha256(f"{task}:{d['sha256']}".encode()).hexdigest())
    return pool


def supplement(streams, tasks, prompts, pool, used, tokenizer, label):
    """(b) Extend each stream to the longest one of the run with unused same-task donors.

    tasks: agent -> task id; prompts: task id -> prompt text of the recipient run; used: set of donor sha256 shared
    over all recipients (mutated). Returns (streams, provenance)."""
    target = max(len(s['offsets']) for s in streams)
    out, provenance = [], []
    for s in streams:
        need = target - len(s['offsets'])
        if need <= 0:
            out.append(s)
            continue
        task, prompt_sha = tasks[s['agent']], hashlib.sha256(prompts[tasks[s['agent']]].encode()).hexdigest()
        text, chosen = s['text'], []
        while len(token_data(tokenizer, text)) < target:
            donor = next((d for d in pool.get(task, []) if d['sha256'] not in used and d['prompt_sha256'] == prompt_sha), None)
            if donor is None:
                raise RuntimeError(f'{label}: no unused donor left for {task}')
            used.add(donor['sha256']); chosen.append(donor)
            text = text + SEP + donor['text']
        offsets = token_data(tokenizer, text)[:target]
        boundary = len(s['text'])
        synthetic_from = int(np.searchsorted(offsets[:, 0], boundary, side='left'))
        out.append({**s, 'text': text, 'offsets': offsets, 'token_task': [task]*target, 'synthetic_from': synthetic_from})
        provenance.append({'recipient': label, 'agent': s['agent'], 'task': task, 'original_tokens': s['original_tokens'],
                           'target_tokens': target, 'synthetic_tokens': target - synthetic_from,
                           'donors': [{'run': d['run'], 'agent': d['agent'], 'sha256': d['sha256'], 'tokens': d['tokens']} for d in chosen]})
    return out, provenance
