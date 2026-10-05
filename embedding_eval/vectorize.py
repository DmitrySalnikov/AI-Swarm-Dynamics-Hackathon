# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn", "tokenizers", "huggingface_hub"]
# ///
"""Vectorization of agent reasoning for δ(t): windows + Qwen3-Embedding-4B (OpenRouter) + TF-IDF char (hybrid).

Self-contained module: can be imported into a pipeline or run as a CLI.
The scheme has been validated (embedding_eval/REPORT.md; choice of window and centering — results/improve_eval.md):
  • windows of W = 2048 tokens, stride 1024 (DeepSeek-V3 tokenizer); the last window is aligned to the end of the text;
  • Qwen3-Embedding-4B on the ORIGINAL window text with the INSTR instruction (as in validation), L2 norm;
  • TF-IDF char 3–5 (sublinear) on text with masked numbers/times/URLs; the vocabulary and IDF are fitted once
    on all windows of the experiment;
  • Qwen vectors are centered: the mean vector of ALL windows of the experiment is subtracted from each, then L2 norm;
  • hybrid: h = [√α·q, √(1−α)·t], α = 0.5 → h_i·h_j = 0.5·cos_q + 0.5·cos_tfidf, ‖h‖ = 1.

Usage in code:
    import vectorize as V
    wins = V.windows(text)                               # windows of one log
    Q, meta = V.embed_logs({"agent3/step2": text, ...})  # dense Qwen vectors of all windows, cached
    T, vec = V.tfidf_fit_transform([m["text"] for m in meta])   # TF-IDF over all windows of the experiment
    S = V.hybrid_sim(Q, T)                               # hybrid similarity matrix (centering inside)
CLI: solo solutions (gen_runs.py: **/<name>.reasoning.txt) and/or chats (chat_runs.py: agent_*.jsonl, solve/):
    uv run vectorize.py --logs-dir runs/deepseek-v4-flash --chat-dir runs/chat --out vectors/ --budget-usd 1
    uv run vectorize.py --logs-dir ... --chat-dir ... --dry-run   # windows, tokens and cost only
  Both directories in one run → TF-IDF and centering are fitted on all windows of the experiment at once.
OpenRouter key: OPENROUTER_API_KEY environment variable or the macOS keychain
    security add-generic-password -U -s openrouter-api -a "$USER" -w
The key is never printed or stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- scheme parameters (do not change w/o re-validation)
W, STEP = 2048, 1024
OR_URL = "https://openrouter.ai/api/v1/embeddings"
OR_MODEL = "qwen/qwen3-embedding-4b"
PRICE_PER_TOKEN = 0.02 / 1e6          # $ per input token (openrouter.ai, 2026-10-04)
INSTR = ("Instruct: Given a fragment of a step-by-step solution, retrieve fragments that solve the same problem"
         "\nQuery:")
ALPHA = 0.5
CACHE_DIR = Path(os.environ.get("VECTORIZE_CACHE", Path(__file__).resolve().parent / "cache"))

# ---------------------------------------------------------------- windows
_TOK = None


def _tokenizer():
    global _TOK
    if _TOK is None:
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
    return _TOK


def windows(text: str, w: int = W, step: int = STEP) -> list[dict]:
    """Windows over DeepSeek-V3 tokens. Returns dicts: idx, text, tok_start, tok_end, n_tok, rel_center, is_first, is_last."""
    off = np.array(_tokenizer().encode(text, add_special_tokens=False).offsets, dtype=np.int64)
    n = len(off)
    if n == 0:
        return []
    starts = list(range(0, max(n - w, 0) + 1, step)) or [0]
    if starts[-1] + w < n:
        starts.append(n - w)               # last window aligned to the end, the tail is not lost
    out = []
    for k, s in enumerate(starts):
        e = min(s + w, n)
        out.append({"idx": k, "text": text[int(off[s, 0]):int(off[e - 1, 1])], "tok_start": s, "tok_end": e,
                    "n_tok": n, "rel_center": round((s + e) / (2 * n), 5), "is_first": k == 0,
                    "is_last": k == len(starts) - 1})
    return out


# ---------------------------------------------------------------- masking (for TF-IDF)
_URL = re.compile(r"https?://\S+|www\.\S+")
_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AaPp][Mm])?\b")
_NUM = re.compile(r"(?<![A-Za-z_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|(?<![\w.])\.\d+")


def mask_text(t: str) -> str:
    return _NUM.sub("<NUM>", _TIME.sub("<TIME>", _URL.sub("<URL>", t)))


# ---------------------------------------------------------------- OpenRouter
def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k and sys.platform == "darwin":
        p = subprocess.run(["security", "find-generic-password", "-s", "openrouter-api", "-a", os.environ.get("USER", ""),
                            "-w"], capture_output=True, text=True)
        k = p.stdout.strip() if p.returncode == 0 else None
    if not k:
        raise RuntimeError("no OpenRouter key (OPENROUTER_API_KEY or Keychain 'openrouter-api')")
    return k


class Budget:
    """Hard spending limit; log in CACHE_DIR/spend.jsonl."""

    def __init__(self, limit_usd: float):
        self.limit, self.path = limit_usd, CACHE_DIR / "spend.jsonl"
        self.spent = sum(json.loads(l)["usd"] for l in self.path.read_text().splitlines() if l.strip()) \
            if self.path.exists() else 0.0

    def check(self, est_tokens):
        if self.spent + est_tokens * PRICE_PER_TOKEN > self.limit:
            raise RuntimeError(f"limit ${self.limit}: spent ${self.spent:.4f}")

    def add(self, tokens, usd):
        self.spent += usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"tokens": tokens, "usd": usd, "t": time.time()}) + "\n")


def _post(key, payload, retries=6):
    data, err = json.dumps(payload).encode(), None
    for i in range(retries):
        req = urllib.request.Request(OR_URL, data=data, headers={"Authorization": f"Bearer {key}",
                                                                  "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read())
            if "data" in out:
                return out
            err = out.get("error")
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}: {e.read()[:300]!r}"
            if e.code in (400, 401, 402, 403):
                raise RuntimeError(err) from None
        except Exception as e:  # noqa: BLE001 — network: retry
            err = repr(e)
        time.sleep(min(60, 2 ** i))
    raise RuntimeError(f"OpenRouter: {retries} failed attempts: {err}")


def encode(texts: list[str], budget: Budget | None = None, batch_max=64, batch_chars=200_000, log=print) -> np.ndarray:
    """Qwen3-Embedding-4B via OpenRouter, with the INSTR instruction. Returns L2-normalized (n, 2560)."""
    key, inputs = _key(), [INSTR + t for t in texts]
    out, i = [None] * len(inputs), 0
    while i < len(inputs):
        j, chars = i, 0
        while j < len(inputs) and j - i < batch_max and (chars + len(inputs[j]) <= batch_chars or j == i):
            chars += len(inputs[j]); j += 1
        if budget:
            budget.check(chars / 3.5)
        r = _post(key, {"model": OR_MODEL, "input": inputs[i:j], "encoding_format": "float"})
        for d in r["data"]:
            out[i + d["index"]] = d["embedding"]
        u = r.get("usage") or {}
        tok = u.get("prompt_tokens") or u.get("total_tokens") or int(chars / 3.5)
        usd = float(u["cost"]) if u.get("cost") is not None else tok * PRICE_PER_TOKEN
        if budget:
            budget.add(tok, usd)
        log(f"[vectorize] {j}/{len(inputs)} windows, +{tok} tok., ${usd:.5f}")
        i = j
    X = np.asarray(out, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- cache by (log, window index)
def _cache_path(log_id: str) -> Path:
    tag = hashlib.sha1(f"{OR_MODEL}|{INSTR}|{W}|{STEP}".encode()).hexdigest()[:8]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", log_id)
    return CACHE_DIR / tag / f"{safe}.npz"


def embed_logs(logs: dict[str, str], budget_usd: float = 1.0, log=print):
    """logs: {log_id: reasoning text}. Returns (Q (n_windows, 2560), meta: list of dicts with log_id and window fields).
    A window is recomputed only if its text has changed (cache by log_id + window index + sha1 of the text)."""
    budget = Budget(budget_usd)
    metas, todo, cached = [], [], {}
    for lid, text in logs.items():
        ws = windows(text)
        p = _cache_path(lid)
        old = dict(np.load(p, allow_pickle=False)) if p.exists() else {}
        hashes = [hashlib.sha1(w["text"].encode()).hexdigest() for w in ws]
        for w, h in zip(ws, hashes):
            m = dict(w, log_id=lid, sha1=h)
            metas.append(m)
            ok = "sha1" in old and w["idx"] < len(old["sha1"]) and old["sha1"][w["idx"]] == h
            if ok:
                cached[(lid, w["idx"])] = old["X"][w["idx"]]
            else:
                todo.append(m)
    by_log = {}
    for m in metas:
        by_log.setdefault(m["log_id"], []).append(m)

    def save(lids):
        for lid in lids:
            ms = by_log[lid]
            if all((lid, m["idx"]) in cached for m in ms):
                p = _cache_path(lid)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(p, X=np.stack([cached[(lid, m["idx"])] for m in ms]), sha1=np.array([m["sha1"] for m in ms]))

    # in chunks by log: if the process is interrupted, the paid-for vectors are already on disk
    todo_by_log = {}
    for m in todo:
        todo_by_log.setdefault(m["log_id"], []).append(m)
    chunk, chunk_logs = [], []
    for lid, ms in list(todo_by_log.items()) + [(None, [])]:
        if lid is not None:
            chunk += ms; chunk_logs.append(lid)
        if chunk and (lid is None or len(chunk) >= 500):
            Xn = encode([m["text"] for m in chunk], budget, log=log)
            for m, x in zip(chunk, Xn):
                cached[(m["log_id"], m["idx"])] = x
            save(chunk_logs)
            chunk, chunk_logs = [], []
    save([lid for lid in by_log if lid not in todo_by_log])
    Q = np.stack([cached[(m["log_id"], m["idx"])] for m in metas]) if metas else np.zeros((0, 2560), np.float32)
    log(f"[vectorize] windows {len(metas)}, from cache {len(metas) - len(todo)}, new {len(todo)}; "
        f"total spent ${budget.spent:.4f}")
    return Q, metas


# ---------------------------------------------------------------- TF-IDF and hybrid
def tfidf_fit_transform(window_texts: list[str]):
    """TF-IDF char_wb 3–5, sublinear, on masked text. Fit ONCE on all windows of the experiment."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, dtype=np.float32)
    T = vec.fit_transform([mask_text(t) for t in window_texts])  # rows are L2-normalized
    return T, vec


def center(Q: np.ndarray, mu: np.ndarray | None = None) -> np.ndarray:
    """Subtract the mean vector (default: mean of all passed windows, i.e. the whole experiment) and L2-normalize.
    Pass ALL windows of the experiment at once or explicitly the same mu: otherwise the vectors are not comparable."""
    mu = Q.mean(0) if mu is None else mu
    Z = Q - mu
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)


def smooth(Q: np.ndarray, meta: list[dict], k: int = 1) -> np.ndarray:
    """Optional: mean of windows idx−k…idx+k of the same log, then L2 norm (not part of the default scheme)."""
    pos = {(m["log_id"], m["idx"]): i for i, m in enumerate(meta)}
    out = np.empty_like(Q)
    for i, m in enumerate(meta):
        js = [pos[(m["log_id"], j)] for j in range(m["idx"] - k, m["idx"] + k + 1) if (m["log_id"], j) in pos]
        v = Q[js].mean(0)
        out[i] = v / max(np.linalg.norm(v), 1e-12)
    return out


def hybrid_sim(Q: np.ndarray, T, alpha: float = ALPHA, centered: bool = True) -> np.ndarray:
    """Hybrid similarity: α·cos(Qwen, centered) + (1−α)·cos(TF-IDF). Q is all windows of the experiment."""
    Qc = center(Q) if centered else Q
    return alpha * (Qc @ Qc.T) + (1 - alpha) * (T @ T.T).toarray()


def hybrid_vectors(Q: np.ndarray, T, alpha: float = ALPHA, centered: bool = True):
    """One vector per window: [√α·q, √(1−α)·t] (sparse, ‖h‖ = 1); q centered over all experiment windows."""
    from scipy import sparse
    Qc = center(Q) if centered else Q
    return sparse.hstack([sparse.csr_matrix(np.sqrt(alpha) * Qc), np.sqrt(1 - alpha) * T]).tocsr()


# ---------------------------------------------------------------- chats (chat_runs.py format)
def load_chat_logs(chat_dir, fields=("reasoning", "message"), include_solve=True, include_invalid=False):
    """Chats from chat_runs.py → ({log_id: text}, {log_id: metadata}).

    One fragment = one turn of one agent: ITS OWN `fields` from agent_<i>.jsonl
    (default reasoning + message). The shared chat text and other agents' messages are not used, otherwise all
    agents would have identical vectors by construction. log_id = <run>/agent_<i>/turn_<NNN>. A turn longer than
    a window is split into windows.
    solve/agent_<i>.json (solution of the agent's own task in merge) → log_id = <run>/agent_<i>/solve; reasoning
    is used, or response if reasoning is empty (this is recorded in the solve_text metadata).
    *_invalid_* runs are skipped by default.
    """
    chat_dir = Path(chat_dir)
    logs, extra = {}, {}

    def run_info(run_dir):
        rel = run_dir.relative_to(chat_dir)
        parts = rel.parts
        kind = next((x for x in parts if x in ("negotiation", "merge")), None)
        bench = next((x for x in parts if x.startswith("set_")), None)
        return str(rel), kind, bench

    for f in sorted(chat_dir.rglob("agent_*.jsonl")):
        if "_invalid_" in f.parent.name and not include_invalid:
            continue
        rel, kind, bench = run_info(f.parent)
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            text = "\n\n".join(str(r.get(k) or "").strip() for k in fields if str(r.get(k) or "").strip())
            if not text:
                continue
            lid = f"{rel}/agent_{r['agent']}/turn_{int(r['turn']):03d}"
            logs[lid] = text
            extra[lid] = {"source": "chat", "run": rel, "kind": kind, "bench": bench, "agent": r["agent"],
                          "turn": r["turn"], "round": r.get("round")}
    if include_solve:
        for f in sorted(chat_dir.rglob("solve/agent_*.json")):
            run_dir = f.parent.parent
            if "_invalid_" in run_dir.name and not include_invalid:
                continue
            rel, kind, bench = run_info(run_dir)
            r = json.loads(f.read_text())
            agent = int(re.search(r"agent_(\d+)", f.name).group(1))
            reasoning, response = (r.get("reasoning") or "").strip(), (r.get("response") or "").strip()
            text = reasoning or response
            if not text:
                continue
            lid = f"{rel}/agent_{agent}/solve"
            logs[lid] = text
            extra[lid] = {"source": "solve", "run": rel, "kind": kind, "bench": bench, "agent": agent,
                          "task_id": r.get("task_id"), "solve_text": "reasoning" if reasoning else "response"}
    return logs, extra


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", help="directory with *.reasoning.txt (solo solutions, recursive)")
    ap.add_argument("--chat-dir", help="chat_runs.py chat directory (agent_*.jsonl, solve/agent_*.json)")
    ap.add_argument("--chat-fields", default="reasoning,message", help="agent turn fields: reasoning,message")
    ap.add_argument("--out", default="vectors", help="where to save qwen.npy, tfidf.npz, index.jsonl")
    ap.add_argument("--budget-usd", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true", help="windows, tokens and cost estimate only")
    a = ap.parse_args()
    if not a.logs_dir and not a.chat_dir:
        ap.error("--logs-dir and/or --chat-dir is required")
    logs, extra = {}, {}
    if a.logs_dir:
        files = sorted(p for p in Path(a.logs_dir).rglob("*.reasoning.txt") if ".failed." not in p.name)
        for p in files:
            lid = "solo/" + str(p.relative_to(a.logs_dir)).removesuffix(".reasoning.txt")
            logs[lid] = p.read_text()
            extra[lid] = {"source": "solo"}
    if a.chat_dir:
        cl, ce = load_chat_logs(a.chat_dir, tuple(a.chat_fields.split(",")))
        logs.update({"chat/" + k: v for k, v in cl.items()})
        extra.update({"chat/" + k: v for k, v in ce.items()})
        n_resp = sum(1 for v in ce.values() if v.get("solve_text") == "response")
        if n_resp:
            print(f"[vectorize] warning: {n_resp} solve files have empty reasoning — response used instead")
    if a.dry_run:
        n_win = n_tok = 0
        for t in logs.values():
            ws = windows(t)
            n_win += len(ws)
            n_tok += sum(w["tok_end"] - w["tok_start"] for w in ws)
        print(f"logs {len(logs)}, windows {n_win}, DeepSeek tokens ≈ {n_tok}; "
              f"estimated cost ≈ ${n_tok * 1.1 * PRICE_PER_TOKEN:.4f} (×1.1 for tokenizer difference)")
        return
    Q, meta = embed_logs(logs, a.budget_usd)
    T, _ = tfidf_fit_transform([m["text"] for m in meta])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from scipy import sparse
    np.save(out / "qwen.npy", Q)
    sparse.save_npz(out / "tfidf.npz", T)
    with open(out / "index.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps({**{k: v for k, v in m.items() if k != "text"}, **extra.get(m["log_id"], {})},
                               ensure_ascii=False) + "\n")
    print(f"saved: {out}/qwen.npy {Q.shape}, tfidf.npz {T.shape}, index.jsonl ({len(meta)} windows)")


if __name__ == "__main__":
    main()
