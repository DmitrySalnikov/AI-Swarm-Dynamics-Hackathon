# vectorize.py — embeddings of reasoning and chats for δ(t)

A single file; dependencies are declared in the header (`uv run` installs them itself). The scheme was chosen and
validated in a separate evaluation (repository `MadExplorer/AI-Village`, `embedding_eval/REPORT.md`,
`results/improve_eval.md`); do not change the parameters without re-validation.

## Scheme

1. The text is split into windows of **2048** DeepSeek-V3 tokens with a stride of **1024**; the last window is aligned to the end.
2. Window → **Qwen3-Embedding-4B** via OpenRouter (2560 numbers), with a fixed instruction.
3. Qwen vectors are **centered** over all windows of the experiment (subtract the mean, length 1).
4. **TF-IDF char 3–5** on text with masked numbers, fitted on all windows of the experiment.
5. Hybrid: `sim = 0.5·cos(Qwen) + 0.5·cos(TF-IDF)`.

On `set_1`/`set_2` (the test half of the tasks), this scheme vs. a 1024 window without centering:
d′ 3.28 → 4.03, AUC 0.972 → 0.979, same-type tasks ("twins") 0.615 → 0.742.

## What gets embedded

| Source | Fragment | `log_id` in the index |
|---|---|---|
| solo solutions `**/<task>.reasoning.txt` (`gen_runs.py`) | reasoning windows | `solo/<run_k>/<set>/<domain>` |
| agent turn in a chat `agent_<i>.jsonl` (`chat_runs.py`) | this agent's **own** `reasoning` + `message` in this turn | `chat/<set>/<kind>/run_<k>/agent_<i>/turn_<NNN>` |
| solution in merge `solve/agent_<i>.json` | `reasoning`, or `response` if it is empty | `chat/<set>/merge/run_<k>/agent_<i>/solve` |

Other agents' messages and the shared chat text are not used: otherwise all agents would have identical vectors by
construction. `*_invalid_*` runs are skipped.

## Running

OpenRouter key — once in the terminal (input is hidden, the key goes into the macOS keychain):
```
security add-generic-password -U -s openrouter-api -a "$USER" -w
```
or the `OPENROUTER_API_KEY` variable.

```
# estimate volume and cost
uv run embedding_eval/vectorize.py --logs-dir embedding_eval/runs/deepseek-v4-flash --chat-dir embedding_eval/runs/chat --dry-run
# everything at once: solo solutions + chats → one shared TF-IDF and one centering
uv run embedding_eval/vectorize.py --logs-dir embedding_eval/runs/deepseek-v4-flash \
    --chat-dir embedding_eval/runs/chat --out embedding_eval/vectors --budget-usd 1
```
Output: `vectors/qwen.npy` (windows × 2560, not centered), `vectors/tfidf.npz`, `vectors/index.jsonl`.
For each window the index has: `log_id`, `source` (solo / chat / solve), `idx`, `tok_start`, `tok_end`, `n_tok`,
`rel_center`; for chats also `run`, `kind`, `bench`, `agent`, `turn`, `round`; for solve — `task_id`, `solve_text`.
For 80 solutions and 40 chat runs: ≈ 4.5M tokens, ≈ $0.10.

From code:
```python
import vectorize as V
logs, extra = V.load_chat_logs("embedding_eval/runs/chat")      # or your own {log_id: text}
Q, meta = V.embed_logs(logs, budget_usd=1.0)                    # cache by (log, window index)
T, _ = V.tfidf_fit_transform([m["text"] for m in meta])
S = V.hybrid_sim(Q, T)          # Qwen centering inside; pass ALL windows of the experiment at once
delta = 1 - S[np.ix_(ids, ids)].mean()   # δ for the set of windows ids (unit-length vectors)
```

## Important

- **TF-IDF and centering — once, on all windows of the experiment** (solo solutions + chats together).
- **One generator model for the whole experiment.** Solo solutions and chats must come from the same model,
  otherwise δ(t) jumps at the seams because of the model switch.
- **Same-type tasks** (one task with different data) are distinguished worse than different-type ones (AUC ≈ 0.74 vs 0.98).
  Do not put biology or music from different sets into one run, nor computer_science from `set_1` and `set_3`.
- The vector cache is `embedding_eval/cache/` (path is changed via `VECTORIZE_CACHE`); do not commit it or
  `embedding_eval/vectors/` to git. The spending limit is hard; the log is `cache/spend.jsonl`.
