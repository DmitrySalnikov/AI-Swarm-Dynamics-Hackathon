"""Apply the upstream embedding geometry to persisted stateful agent outputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tokenizers import Tokenizer

from embedding_eval.dynamics import metrics_at, unit
from swarm_pipeline.supervision import write_json, read_json

MODEL = "qwen/qwen3-embedding-4b"
URL = "https://openrouter.ai/api/v1/embeddings"
PHASES = ("negotiation", "solving", "exchange")
STYLES = {
    "cooperative": ("#2a78d6", "-", "o", "swarm"),
    "fixed": ("#eb6834", "--", "s", "independent, fixed order"),
    "shuffled": ("#1baf7a", ":", "^", "independent, shuffled order"),
    "disjoint": ("#9a7634", "--", "v", "independent, cyclic offsets"),
    "mixed": ("#6750a4", "-.", "D", "5 swarm + 5 shuffled"),
}
PANELS = [("R", "R — length of the mean vector"),
          ("var_perp_std", "Spread across the mean direction"), ("eff_dim", "Effective dimension"),
          ("n_active", "Vectors included (swarm windows retained by progress alignment)"),
          ("pair_cosine", "Mean pairwise cosine similarity")]


def token_data(tokenizer, text):
    return np.asarray(tokenizer.encode(text, add_special_tokens=False).offsets, dtype=np.int64).reshape(-1, 2)


def windows(text, offsets, width, limit=None):
    """Upstream rule: stride W/2; retain a tail with at least W/2 tokens."""
    n = len(offsets) if limit is None else min(len(offsets), limit)
    if not n:
        return []
    starts = list(range(0, n, width // 2))
    starts = [s for s in starts if n - s >= width // 2] or [0]
    return [{"text": text[int(offsets[s, 0]):int(offsets[min(s + width, n) - 1, 1])],
             "start": s, "end": min(s + width, n), "x": (s + min(s + width, n)) / 2}
            for s in starts]


def load_agents(root, condition, seed, tokenizer):
    directory = root / condition / f"seed_{seed}"
    state, manifest = read_json(directory / "state.json"), read_json(directory / "manifest.json")
    assert state["status"] == "completed" and state["pending"] is None
    agents = []
    for agent in range(1, manifest["config"]["agents"] + 1):
        path = directory / "generated" / f"agent_{agent}.generated.txt"
        text = path.read_text(encoding="utf-8")
        index = read_json(path.with_suffix(".index.json"))
        if condition != "cooperative":
            # Only the first 32k tokens enter these figures. Tokenize enough
            # complete text to cover that prefix, rather than every full solve.
            chars = len(text) if manifest.get('protocol_version')=='private-single-task-v1' else min(len(text), 160000)
            while True:
                offsets = token_data(tokenizer, text[:chars])
                if len(offsets) >= 32000 or chars == len(text):
                    break
                chars = min(len(text), chars * 2)
            agents.append({"agent": agent, "text": text, "offsets": offsets, "phases": {}})
            continue
        record = {"agent": agent, "text": text, "offsets": token_data(tokenizer, text), "phases": {},"protocol":manifest.get("protocol_version","v1")}
        prefix = 0
        for phase in PHASES:
            pieces = [p for p in index if p["phase"] == phase]
            if not pieces:
                continue
            begin, end = min(p["start"] for p in pieces), max(p["end"] + 2 for p in pieces)
            phase_text = text[begin:end]
            offsets = token_data(tokenizer, phase_text)
            calls = sorted(set(p["call"] for p in pieces))
            turns = []
            for call in calls:
                parts = [p for p in pieces if p["call"] == call]
                a, b = min(p["start"] for p in parts) - begin, max(p["end"] + 2 for p in parts) - begin
                start = int(np.searchsorted(offsets[:, 0], a))
                stop = int(np.searchsorted(offsets[:, 0], b))
                turns.append({"text": phase_text[a:b], "x": prefix + (start + stop) / 2, "call": call})
            record["phases"][phase] = {"text": phase_text, "offsets": offsets, "prefix": prefix, "turns": turns}
            prefix += len(offsets)
        agents.append(record)
    return agents, manifest


def vector_key(text):
    return hashlib.sha256((MODEL + "|no-instruction|" + text).encode()).hexdigest()


def embed(texts, out, workers=4, cache_dir=None):
    cache = Path(cache_dir) if cache_dir else out / "embedding-cache"
    cache.mkdir(parents=True, exist_ok=True)
    unique = {vector_key(t): t for t in texts if t.strip()}
    todo = [(key, text) for key, text in unique.items() if not (cache / f"{key}.npy").exists()]
    if not todo:
        print(f"Embeddings: all {len(unique)} windows cached; no provider requests", flush=True)
        return {tag: np.load(cache / f"{tag}.npy", allow_pickle=False) for tag in unique}
    batches, batch, chars = [], [], 0
    for item in todo:
        if batch and (len(batch) >= 32 or chars + len(item[1]) > 180000):
            batches.append(batch); batch, chars = [], 0
        batch.append(item); chars += len(item[1])
    if batch:
        batches.append(batch)
    # The existing Windows proxy is needed for OpenRouter; Cloud.ru stays direct
    # in the generation launcher. This process changes no global system setting.
    os.environ.pop("NO_PROXY", None); os.environ.pop("no_proxy", None)
    proxies = urllib.request.getproxies()
    key = os.environ["OPENROUTER_API_KEY"]
    estimate_usd = sum(len(t) / 2 * .02 / 1e6 for _, t in todo)
    if estimate_usd > 1:
        raise RuntimeError("Embedding volume exceeds the $1 diagnostic envelope")
    print(f"Embeddings: {len(unique)} unique windows, {len(todo)} uncached, {len(batches)} batches; estimated ${estimate_usd:.4f}", flush=True)

    def request_batch(batch):
        identity = hashlib.sha256("".join(k for k, _ in batch).encode()).hexdigest()
        request_path = out / "embedding-requests" / f"{identity}.json"
        old = read_json(request_path) if request_path.exists() else {}
        attempts = old.get("attempts", 0)
        for retry in range(4):
            attempts += 1
            write_json(request_path, {"model": MODEL, "hashes": [k for k, _ in batch], "attempts": attempts,
                                      "estimated_usd_per_attempt": sum(len(t) / 2 * .02 / 1e6 for _, t in batch),
                                      "started_at": time.time(), "via": "system_proxy"})
            payload = {"model": MODEL, "input": [t for _, t in batch], "encoding_format": "float"}
            req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
                with opener.open(req, timeout=120) as stream:
                    response = json.load(stream)
                vectors = sorted(response["data"], key=lambda d: d["index"])
                assert len(vectors) == len(batch) and [v["index"] for v in vectors] == list(range(len(batch)))
                matrix = np.asarray([v["embedding"] for v in vectors], dtype=np.float32)
                assert matrix.shape[1] == 2560 and np.isfinite(matrix).all()
                matrix = unit(matrix)
                receipt = {"model": response.get("model"), "usage": response.get("usage"),
                           "attempts": attempts, "vectors": len(batch), "completed_at": time.time()}
                write_json(request_path.with_suffix(".response.json"), receipt)
                for (tag, _), vector in zip(batch, matrix):
                    np.save(cache / f"{tag}.npy", vector)
                return len(batch)
            except urllib.error.HTTPError as exc:
                if exc.code in (400, 401, 402, 403) or retry == 3:
                    raise RuntimeError(f"Embedding HTTP {exc.code}") from None
            except (OSError, ValueError, KeyError):
                if retry == 3:
                    raise
            time.sleep(2 ** retry)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(request_batch, batch) for batch in batches]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 10 == 0 or i == len(futures):
                print(f"Embedding batches {i}/{len(futures)}", flush=True)
    return {tag: np.load(cache / f"{tag}.npy", allow_pickle=False) for tag in unique}


def attach(rows, vectors):
    for row in rows:
        row["vector"] = vectors[vector_key(row["text"])]

def chat_endpoints(agent, phase):
    turns=agent['phases'][phase]['turns']
    return [turns[-1]] if agent.get('protocol')=='v2' and phase=='exchange' else [turns[0],turns[-1]]


def phase_rows(cooperative, width, vectors):
    per = []
    for a in cooperative:
        phases = a["phases"]
        solve = windows(phases["solving"]["text"], phases["solving"]["offsets"], width)
        for row in solve:
            row["x"] += phases["solving"]["prefix"]
        attach(solve, vectors)
        neg, exchange = chat_endpoints(a,'negotiation'),chat_endpoints(a,'exchange')
        ends = neg + exchange
        attach(ends, vectors)
        per.append({"solve": solve, "ends": ends})
    count = max(len(a["solve"]) for a in per)
    snapshots = []
    for position in range(count + len(per[0]['ends'])):
        if position < 2:
            chosen = [a["ends"][position] for a in per]; phase = "negotiation"
        elif position >= count + 2:
            chosen = [a["ends"][position - count] for a in per]; phase = "exchange"
        else:
            chosen = [a["solve"][round((position - 2) * (len(a["solve"]) - 1) / max(count - 1, 1))] for a in per]
            phase = "solving"
        snapshots.append({"x": float(np.mean([r["x"] for r in chosen])), "phase": phase,
                          "X": np.stack([r["vector"] for r in chosen]), "position": position})
    spans = {phase: (float(np.mean([a["phases"][phase]["prefix"] for a in cooperative])),
                     float(np.mean([a["phases"][phase]["prefix"] + len(a["phases"][phase]["offsets"]) for a in cooperative]))) for phase in PHASES}
    return snapshots, spans


def control_rows(wins):
    count = min(len(w) for w in wins)
    return [{"x": float(np.mean([w[i]["x"] for w in wins])), "phase": "independent",
             "X": np.stack([w[i]["vector"] for w in wins]), "position": i} for i in range(count)]


def metrics(rows, mean=None):
    result=[]
    for r in rows:
        X=unit(r['X']-mean) if mean is not None else r['X']
        m=metrics_at(X);n=len(X)
        m['pair_cosine']=(n*m['R']**2-1)/(n-1) if n>1 else float('nan')
        result.append({'x':r['x'],'phase':r['phase'],**m})
    return result


def plot(series, spans, width, path, limit=None, centred=False, protocol_label="stateful"):
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(11, 14), sharex=True)
    end = limit or spans["exchange"][1]
    for ax, (metric, title) in zip(axes, PANELS):
        for phase in ("negotiation", "exchange"):
            lo, hi = spans[phase]
            if lo < end:
                ax.axvspan(lo, min(hi, end), color="#eeede9", zorder=0)
        for condition, rows in series.items():
            color, line, marker, label = STYLES[condition]
            selected = [r for r in rows if r["x"] <= end]
            ax.plot([r["x"] for r in selected], [r[metric] for r in selected],
                    color=color, linestyle=line, marker=marker, markersize=3, linewidth=1.8, label=label)
        ax.set_title(title, loc="left", fontsize=11, pad=30 if metric == "R" else 8)
        ax.grid(alpha=.25); ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 1)
    axes[3].set_ylim(0,11)
    axes[4].set_ylim(-.2,1)
    for phase, label in (("negotiation", "task distribution"), ("solving", "parallel solving"), ("exchange", "results exchange")):
        lo, hi = spans[phase]
        if lo < end:
            axes[0].text((lo + min(hi, end)) / 2, 1.025, label, transform=axes[0].get_xaxis_transform(), ha="center", fontsize=8)
    axes[-1].set_xlim(0, end)
    axes[-1].set_xlabel("generated tokens (mean phase-aligned swarm position; control window centres)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .945), ncol=2, frameon=False, fontsize=9)
    variant = "centred" if centred else "raw, L2-normalised"
    fig.suptitle(f"Rung 1: {protocol_label} swarm vs independent agents | Qwen3-Embedding-4B ({variant})\n"
                 f"Solving/control windows: {width} tokens, stride {width // 2}; one group of 10 per condition, no replicate bands\n"
                 "Swarm solve windows aligned by relative progress; shorter solves hold windows. Token alignment is approximate.",
                 fontsize=10, y=.99)
    fig.subplots_adjust(left=.09, right=.985, bottom=.06, top=.87, hspace=.4)
    fig.savefig(path, dpi=160, facecolor="white"); fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/pipeline-live"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/pipeline-analysis"))
    parser.add_argument("--widths", type=int, nargs="+", default=[2048, 400, 4096])
    parser.add_argument("--swarm-root", type=Path)
    parser.add_argument("--swarm-seed", type=int, default=1)
    parser.add_argument("--control-seed", type=int, default=3)
    parser.add_argument("--sources", type=Path, help="JSON mapping conditions to isolated output roots")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--prefix", default="rung1")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    token_path = args.out / "deepseek-tokenizer.json"
    if token_path.exists():
        tokenizer = Tokenizer.from_file(str(token_path))
    else:
        tokenizer = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
        tokenizer.save(str(token_path))
    agents, manifests = {}, {}
    sources=read_json(args.sources) if args.sources else {}
    for condition in ("cooperative", "fixed", "shuffled", "disjoint"):
        source = Path(sources[condition]) if condition in sources else ((args.swarm_root or args.root) if condition == "cooperative" else args.root)
        agents[condition], manifests[condition] = load_agents(source, condition, args.swarm_seed if condition == "cooperative" else args.control_seed, tokenizer)
    assert len({m["task_hash"] for m in manifests.values()}) == 1, "Conditions must use the same frozen task set"
    all_texts, prepared = [], {}
    for width in args.widths:
        control = {}
        for condition in ("fixed", "shuffled", "disjoint"):
            control[condition] = [windows(a["text"], a["offsets"], width, limit=32000) for a in agents[condition]]
            all_texts.extend(r["text"] for rows in control[condition] for r in rows)
        for a in agents["cooperative"]:
            solve = a["phases"]["solving"]
            all_texts.extend(r["text"] for r in windows(solve["text"], solve["offsets"], width))
            for phase in ("negotiation", "exchange"):
                all_texts.extend(r['text'] for r in chat_endpoints(a,phase))
        prepared[width] = control
    vectors = embed(all_texts, args.out, cache_dir=args.cache_dir)
    reports = []
    for width, control in prepared.items():
        for per in control.values():
            for rows in per:
                attach(rows, vectors)
        swarm, spans = phase_rows(agents["cooperative"], width, vectors)
        snapshots = {"cooperative": swarm, **{c: control_rows(w) for c, w in control.items()}}
        # Separate natural token clock: completed agents disappear instead of
        # repeating windows. Do not confuse this with the progress-aligned arc.
        solve_windows=[]
        for a in agents['cooperative']:
            p=a['phases']['solving'];ws=windows(p['text'],p['offsets'],width)
            attach(ws,vectors);solve_windows.append(ws)
        active=[]
        for i in range(max(map(len,solve_windows))):
            chosen=[ws[i] for ws in solve_windows if i<len(ws)]
            if len(chosen)>=4:
                active.append({'x':float(np.mean([w['x'] for w in chosen])), 'phase':'solving', 'X':np.stack([w['vector'] for w in chosen])})
        active_metrics=metrics(active)
        active_path=args.out/f'{args.prefix}_w{width}_active_only.csv'
        with active_path.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(active_metrics[0]));writer.writeheader();writer.writerows(active_metrics)
        mixed = []
        shuffled = snapshots["shuffled"]
        for row in swarm:
            nearest = min(shuffled, key=lambda r: abs(r["x"] - row["x"]))
            mixed.append({**row, "X": np.concatenate([row["X"][:5], nearest["X"][5:]])})
        snapshots["mixed"] = mixed
        unique = {vector_key(t): vectors[vector_key(t)] for t in all_texts if t.strip()}
        # Centre once over this width's experiment, not separately per condition.
        experiment = [r["vector"] for per in control.values() for rows in per for r in rows]
        for a in agents["cooperative"]:
            p = a["phases"]["solving"]
            experiment.extend(vectors[vector_key(r["text"])] for r in windows(p["text"], p["offsets"], width))
            for phase in ("negotiation", "exchange"):
                experiment.extend(vectors[vector_key(r["text"])] for r in chat_endpoints(a,phase))
        mean = np.stack(experiment).mean(0)
        for centred in (False, True):
            series = {c: metrics(rows, mean if centred else None) for c, rows in snapshots.items()}
            name = f"{args.prefix}_w{width}_{'centred' if centred else 'raw'}"
            label=manifests['cooperative'].get('protocol_version','stateful')
            plot(series, spans, width, args.out / f"{name}.png", centred=centred,protocol_label=label)
            if not centred:
                plot(series, spans, width, args.out / f"{name}_first10k.png", limit=10000,protocol_label=label)
            rows = [{"condition": c, **r} for c, data in series.items() for r in data]
            with (args.out / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            reports.append({"file": name, "phase_spans": spans, "points": {c:len(r) for c,r in series.items()}})
    receipts = [read_json(p) for p in (args.out / "embedding-requests").glob("*.response.json")]
    manifest = {"model": MODEL, "provider": "OpenRouter via system proxy", "dimensions":2560,
        "tokenizer":"deepseek-ai/DeepSeek-V3", "instruction":None, "normalisation":"L2", "widths":args.widths,
        "source":"own generated reasoning and visible JSON only; all answers regardless of correctness; interrupted observed fragments retained",
        "replicates":1, "bands":None, "phase_alignment":"two distribution endpoints; one commit for v2, two exchange endpoints for v1; solving resampled to maximum window count, retaining 10 agents",
        "controls":"natural generated-token windows; 32000-token prefix", "mixed":"five swarm agents + five shuffled at nearest token centre; not a new simulation",
        "task_hash":manifests['cooperative']['task_hash'], "runs":{c:m['config']['seed'] for c,m in manifests.items()},
        "protocol_version":manifests['cooperative'].get('protocol_version','v1'),
        "baseline_root":str(args.root.resolve()),"swarm_root":str((args.swarm_root or args.root).resolve()),
        "condition_roots":sources,
        "plots":reports,"embedding_input_tokens":sum((r.get('usage')or{}).get('prompt_tokens',0) for r in receipts),
        "usage_backed_usd":sum((r.get('usage')or{}).get('cost',0) for r in receipts),
        "limitations":["Different seeds and recovery policies", "One group per condition; no uncertainty estimate", "Phase-aligned token coordinates approximate", "Window holding can affect temporal dynamics", "Correctness not used for selection"]}
    write_json(args.out / "analysis-manifest.json", manifest)
    print(json.dumps({k:manifest[k] for k in ['dimensions','embedding_input_tokens','usage_backed_usd','widths']}),flush=True)


if __name__ == "__main__":
    main()
