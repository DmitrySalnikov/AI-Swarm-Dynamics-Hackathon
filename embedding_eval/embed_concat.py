# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "tokenizers", "huggingface_hub"]
# ///
"""Sliding-window embeddings for every concat/<set>/**/*.txt (built by build_concat.py).

    uv run embedding_eval/embed_concat.py [--budget-usd 1] [--dry-run]

Windows: DeepSeek-V3 tokens (the generator's tokenizer), length W=2048, start every STEP=1024 tokens
(tokens 1-2048, 1025-3072, ...). A window is kept while at least MIN_LEFT=1024 tokens remain from its start,
so the last kept window may be shorter than 2048; shorter remainders are dropped.
Each window's raw text goes to Qwen3-Embedding-4B via OpenRouter WITHOUT an instruction prefix; vectors are
L2-normalised, not centred.

Output next to each <name>.txt:
  <name>.vectors.npy    float32 (n_windows, 2560)
  <name>.windows.json   per window: idx, tok_start, tok_end (0-based, end exclusive), char_start, char_end,
                        pieces = indices into <name>.index.json that the window overlaps
concat/embeddings_manifest.json: settings, totals, spend. Files already embedded are skipped (resumable).
Key: OPENROUTER_API_KEY or macOS Keychain service 'openrouter-api'. The key is never printed.
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONCAT = HERE / "concat"
W, STEP, MIN_LEFT = 2048, 1024, 1024
URL, MODEL = "https://openrouter.ai/api/v1/embeddings", "qwen/qwen3-embedding-4b"
PRICE = 0.02 / 1e6                      # $ per input token
BATCH = 24                              # windows per request

_tok, lock = None, threading.Lock()
spent = {"usd": 0.0, "tokens": 0, "requests": 0, "providers": set()}


def tokenizer():
    global _tok
    if _tok is None:
        from tokenizers import Tokenizer
        _tok = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    return _tok


def windows(text):
    off = tokenizer().encode(text, add_special_tokens=False).offsets
    n, out = len(off), []
    for s in range(0, n, STEP):
        if n - s < MIN_LEFT:
            break
        e = min(s + W, n)
        cs, ce = off[s][0], off[e - 1][1]
        out.append({"idx": len(out), "tok_start": s, "tok_end": e, "char_start": cs, "char_end": ce})
    return out, n


def key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k and sys.platform == "darwin":
        p = subprocess.run(["security", "find-generic-password", "-s", "openrouter-api", "-a", os.environ.get("USER", ""),
                            "-w"], capture_output=True, text=True)
        k = p.stdout.strip() if p.returncode == 0 else None
    if not k:
        sys.exit("no OpenRouter key (OPENROUTER_API_KEY or Keychain 'openrouter-api')")
    return k


def embed(k, texts, budget):
    with lock:
        if spent["usd"] + sum(map(len, texts)) / 2 * PRICE > budget:      # ~2 chars/token: pessimistic
            raise RuntimeError(f"budget ${budget} reached (spent ${spent['usd']:.4f})")
    data, err = json.dumps({"model": MODEL, "input": texts, "encoding_format": "float"}).encode(), None
    for attempt in range(6):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {k}",
                                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
            if "data" in out and len(out["data"]) == len(texts):
                break
            err = str(out.get("error", out))[:300]
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code in (400, 401, 402, 403):
                raise RuntimeError(err) from None
        except Exception as e:  # noqa: BLE001 - network: retry
            err = repr(e)
        time.sleep(min(60, 2 ** attempt))
    else:
        raise RuntimeError(f"OpenRouter failed 6 times: {err}")
    X = np.zeros((len(texts), len(out["data"][0]["embedding"])), np.float32)
    for d in out["data"]:
        X[d["index"]] = d["embedding"]
    u = out.get("usage") or {}
    with lock:
        spent["usd"] += float(u.get("cost") or u.get("prompt_tokens", 0) * PRICE)
        spent["tokens"] += u.get("prompt_tokens", 0)
        spent["requests"] += 1
        spent["providers"].add(out.get("provider"))
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def do_file(path, k, budget):
    vec_path, win_path = path.with_suffix(".vectors.npy"), path.with_suffix(".windows.json")
    if vec_path.exists() and win_path.exists():
        return path, "skipped", 0
    text = path.read_text()
    pieces = json.loads(path.with_suffix(".index.json").read_text())
    ws, n_tok = windows(text)
    for w in ws:
        w["pieces"] = [j for j, p in enumerate(pieces) if p["start"] < w["char_end"] and p["end"] > w["char_start"]]
    texts = [text[w["char_start"]:w["char_end"]] for w in ws]
    X = np.concatenate([embed(k, texts[i:i + BATCH], budget) for i in range(0, len(texts), BATCH)]) if ws \
        else np.zeros((0, 2560), np.float32)
    np.save(vec_path, X)
    win_path.write_text(json.dumps({"n_tokens": n_tok, "windows": ws}, indent=1))
    return path, "done", len(ws)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-usd", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = sorted(p for p in CONCAT.rglob("*.txt"))
    if a.dry_run:
        tot = 0
        for p in files:
            ws, _ = windows(p.read_text()); tot += len(ws)
        print(f"files {len(files)}, windows {tot}, est. cost ${tot * W * PRICE:.3f}")
        return
    k = key()
    t0, done, n_win = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for path, status, nw in ex.map(lambda p: do_file(p, k, a.budget_usd), files):
            done += 1; n_win += nw
            if status == "done":
                print(f"[{done}/{len(files)}] {path.relative_to(CONCAT)}: {nw} windows  spent ${spent['usd']:.4f}", flush=True)
    manifest = {"model": MODEL, "via": "openrouter", "providers": sorted(filter(None, spent["providers"])),
                "instruction": None, "normalised": "L2", "centred": False, "tokenizer": "deepseek-ai/DeepSeek-V3",
                "window": W, "step": STEP, "min_tokens_left": MIN_LEFT, "files": len(files),
                "windows_this_run": n_win, "tokens_this_run": spent["tokens"], "usd_this_run": round(spent["usd"], 6),
                "seconds_this_run": round(time.time() - t0)}
    (CONCAT / "embeddings_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
