# Development setup

Run commands from the repository root in PowerShell.

The new stateful runner and its offline checks are documented in [PIPELINE.md](PIPELINE.md).

```powershell
uv sync --locked
uv run python -X utf8 scripts/check_setup.py
uv run python -X utf8 embedding_eval/gen_runs.py --help
uv run python -X utf8 embedding_eval/chat_runs.py --help
```

The local interpreter is `.venv\Scripts\python.exe`; select it in PyCharm.
Python 3.13 is the local default; upstream scripts require Python 3.10 or newer.
`uv.lock` pins the shared development environment. Use `uv run python <script>`
to use that environment: `uv run <script>` may instead use the script's inline
dependency metadata and create a separate environment.

On Windows, use `python -X utf8` for these scripts. The local default encoding
is CP1251, which cannot print some symbols in CLI help and can misread UTF-8
artifacts. In PyCharm, add `PYTHONUTF8=1` to the run configuration environment.

## Provider credentials

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
# Edit .env locally, then:
uv run --env-file .env python -X utf8 embedding_eval/gen_runs.py --help
```

Generation uses `TOGETHER_API_KEY` or `CLOUDRU_API_KEY`; embeddings use
`OPENROUTER_API_KEY`. Existing scripts also have macOS Keychain fallbacks,
which are not available on Windows. Environment variables work on Windows.
The `.env` file is loaded explicitly by uv, not automatically by the scripts.
Setup verification does not require credentials or call model APIs.

## Existing workflow

See the upstream README for the generation, concatenation, embedding, and
dynamics scripts. `bench/` holds the two public ten-question sets;
`embedding_eval/runs/`, `concat/`, and `dynamics/` contain existing artifacts.
Several scripts write directly into those directories. Use a separate output
copy or review their paths before reproducing plots to avoid replacing baseline
artifacts. Data licensing is described in `LICENSE-DATA`.

The setup checker validates dependencies, Python source syntax, benchmark
structure, and simple known-vector cases for the existing metrics. It does not
download a tokenizer, regenerate embeddings, or validate the research claims.
