# AI-Swarm-Dynamics-Hackathon

https://swarmchasing.com/

A benchmark for experiments with agent swarms: four sets of 10 questions each.
`set_1`–`set_3` cover 10 different domains: `set_1` and `set_2` are hard (long reasoning,
10–20k tokens on average), `set_3` is easy (2–5k tokens). `set_4` is entirely mathematical:
10 different branches, 1–5k tokens. All questions are solvable by a mid-tier model; the answer is short and checked automatically.

> **Do not publish.** Some of the questions are taken from GPQA, HLE and LAB-Bench, whose authors ask
> that they not be posted in plain text or used for training. These questions keep their
> `canary` field. The repository must remain private.

## Structure

```
bench/set_{1,2,3,4}.jsonl                questions
results/<model>/<set>/<domain>.json     model answer, tokens, verdict
results/<model>/<set>/<domain>.reasoning.txt   model reasoning
results/<model>/summary.md              summary table
scripts/run.py                           run a model on the sets
```

## Sets

`set_1`–`set_3` have one question per domain: `computer_science`, `mathematics`, `finance`,
`linguistics`, `physics`, `biology`, `chemistry`, `classics`, `economics`, `music`.

| Domain | set_1 | set_2 | set_3 (easy) |
|---|---|---|---|
| computer_science | breaking Diffie–Hellman, p=1009 (HLE) | output of a Scheme program with `call/cc` (HLE) | Diffie–Hellman, p=227 (original) |
| mathematics | probability, 12 letters into pairs (AIME 2025 #7) | quadrilateral formed by circumcenters (HMMT 2025 #26) | numbers using digits 1–8 divisible by 22 (AIME 2025 #5) |
| finance | retirement annuity (TheoremQA) | PE ratio (TheoremQA) | project NPV (original) |
| linguistics | Guazacapán Xinka verb forms (Linguini) | matching Zuni words (Linguini) | numerals of an invented language (original) |
| physics | mass of an incandescent lamp filament (OlympiadBench) | olympiad problem #900 (OlympiadBench) | parachutist with quadratic drag (TheoremQA) |
| biology | GC content of a sequence (LAB-Bench) | GC content of a sequence (LAB-Bench) | GC content of 150 nucleotides (original) |
| chemistry | number of stereoisomers (GPQA Diamond) | 4-step synthesis, H types (GPQA Diamond) | NaHCO₃ and MgCO₃ mixture from mass loss (original) |
| classics | Latin hexameter scansion (HLE) | scansion of a line of Plautus (HLE) | 6 Roman calendar dates (original) |
| economics | welfare at equilibrium (HLE) | job search and unemployment benefit (HLE) | certainty equivalent (TheoremQA) |
| music | just intonation, "Hänschen klein" (original) | just intonation, second melody (original) | just intonation, 13 notes (original) |

GPQA questions are posed open-ended, without answer options: the answer is a number and cannot be guessed.
The music questions were written by us (the melody and the interval table are given in the problem statement); the reference answer
was computed by code. For the first melody it matched the reference answer of the original HLE question 66f57e3ddc7259d8b5bb0b46.
In `set_3`, 7 of the 10 questions are our own, modeled on problems from `set_1`/`set_2` with smaller parameters:
the model could not solve the hard questions from GPQA and Linguini within 5000 tokens.

### set_4: mathematics

| Branch | Problem | Source |
|---|---|---|
| geometry | distance between the circumcenter and the orthocenter | original |
| linear_algebra | determinant of a 5×5 integer matrix | original |
| calculus | improper integral ∫₀^∞ x²e⁻ˣ sin x dx | original |
| differential_equations | y″ + 2y′ + 5y = 10 cos x, find y(π) | original |
| series | double series | TheoremQA |
| number_theory | divisors of 9! ending in 1 | HMMT Feb 2025 #1 |
| combinatorics | coloring the segments of a 2×2 grid | AIME 2025 #18 |
| probability | random subset of the divisors of 2025 | AIME 2025 #22 |
| algebra | product of logarithms | AIME 2025 #19 |
| complex_numbers | system with moduli of complex numbers | AIME 2025 #8 |

The reference answers of our own problems were verified independently: by substitution into the equation, by numerical
integration, by a second formula.

### Sources

| Question | Benchmark | ID in source |
|---|---|---|
| `set_1/computer_science` | [HLE](https://huggingface.co/datasets/cais/hle) | `67192b9472c6fd14e759e369` |
| `set_1/mathematics` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | problem 7 |
| `set_1/finance` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | question "Aisha graduates college…" |
| `set_1/linguistics` | [Linguini](https://huggingface.co/datasets/facebook/linguini) | `012023010100` |
| `set_1/physics` | [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | `OE_TO_physics_en_COMP`, id 908 |
| `set_1/biology` | [LAB-Bench SeqQA](https://huggingface.co/datasets/futurehouse/lab-bench) | `97e6e1a0-f207-46d0-9201-f0081f40e19a` |
| `set_1/chemistry` | [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | `rectXfsCM1dj4Kv2c` |
| `set_1/classics` | [HLE](https://huggingface.co/datasets/cais/hle) | `66fc698fd90ebe461bfd0cc4` |
| `set_1/economics` | [HLE](https://huggingface.co/datasets/cais/hle) | `66fc23cfa7be4edbe85cf177` |
| `set_1/music` | original, based on HLE `66f57e3ddc7259d8b5bb0b46` | — |
| `set_2/computer_science` | [HLE](https://huggingface.co/datasets/cais/hle) | `66f4aa5df382ae9214c8dc9b` |
| `set_2/mathematics` | [HMMT February 2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) | problem 26 |
| `set_2/finance` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | question "Estimate the PE ratio…" |
| `set_2/linguistics` | [Linguini](https://huggingface.co/datasets/facebook/linguini) | `012021020100` |
| `set_2/physics` | [OlympiadBench](https://huggingface.co/datasets/Hothan/OlympiadBench) | `OE_TO_physics_en_COMP`, id 900 |
| `set_2/biology` | [LAB-Bench SeqQA](https://huggingface.co/datasets/futurehouse/lab-bench) | `64c5e31b-5b98-494a-92a7-1019723204d6` |
| `set_2/chemistry` | [GPQA Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | `recJZ3QEfRKjYw9a7` |
| `set_2/classics` | [HLE](https://huggingface.co/datasets/cais/hle) | `67015a7f6a2b21f149f3aaba` |
| `set_2/economics` | [HLE](https://huggingface.co/datasets/cais/hle) | `6711e5e05e64a53ed09449fd` |
| `set_2/music` | original | — |
| `set_3/computer_science` | original | — |
| `set_3/mathematics` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | problem 5 |
| `set_3/finance` | original | — |
| `set_3/linguistics` | original | — |
| `set_3/physics` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | question "A parachutist with mass m=80 kg…" |
| `set_3/biology` | original | — |
| `set_3/chemistry` | original | — |
| `set_3/classics` | original | — |
| `set_3/economics` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | question "An investor has utility function…" |
| `set_3/music` | original | — |
| `set_4/geometry` | original | — |
| `set_4/linear_algebra` | original | — |
| `set_4/calculus` | original | — |
| `set_4/differential_equations` | original | — |
| `set_4/series` | [TheoremQA](https://huggingface.co/datasets/TIGER-Lab/TheoremQA) | question "Sum the series …" |
| `set_4/number_theory` | [HMMT February 2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025) | problem 1 |
| `set_4/combinatorics` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | problem 18 |
| `set_4/probability` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | problem 22 |
| `set_4/algebra` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | problem 19 |
| `set_4/complex_numbers` | [AIME 2025](https://huggingface.co/datasets/MathArena/aime_2025) | problem 8 |

Linguini is built from problems of the International Linguistics Olympiad (IOL), LAB-Bench SeqQA from
questions on working with DNA sequences, and OlympiadBench from olympiad physics problems.

### Question format

```json
{
  "id": "set_1/computer_science",
  "set": "set_1",
  "domain": "computer_science",
  "title": "Breaking Diffie–Hellman, p=1009",
  "source": {"dataset": "HLE", "id": "67192b9472c6fd14e759e369", "url": "https://huggingface.co/datasets/cais/hle"},
  "prompt": "… finish with a final line of the form `ANSWER: <your answer>`.",
  "answer": "760",
  "check": {"type": "exact"},
  "canary": "…"
}
```

`prompt` is the ready-made text for the model; the answer is expected as the last line `ANSWER: …`.
Check type `check.type`:
- `exact` — string match ignoring case and whitespace;
- `list` — match of a comma-separated list, in order;
- `num` — a number with relative tolerance `rel_tol`.

### How the questions were selected

The benchmark target is the "Diffie–Hellman level": the solution method is clear, but it requires long, careful
work (calculations, enumeration, applying rules), and the answer is unambiguous. Candidates were run on the
target model. For `set_1`/`set_2` we took questions on which it spends roughly 5–30k reasoning tokens;
ones that were too heavy (the model went to 50–100k tokens or did not answer) were replaced with
simpler ones. For `set_3` we aimed at 2–3k, for `set_4` at 1–5k, with a hard limit of 5000:
a candidate that did not fit within the limit was discarded, and one that was too easy was replaced with a harder one. Model errors are acceptable: the sets contain unsolved questions.

## Model

**DeepSeek V4 Flash** (`deepseek-ai/DeepSeek-V4-Flash-0731`) via Together AI, `reasoning_effort=high`,
one run per question; `max_tokens=100000` for `set_1`/`set_2` and 5000 for `set_3`/`set_4`. The model returns its reasoning in a separate field; its token
count is `reasoning_tokens` from `usage`.

Result: **set_1 — 8/10, set_2 — 7/10, set_3 — 10/10, set_4 — 10/10**, details in
[`results/deepseek-v4-flash-0731/summary.md`](results/deepseek-v4-flash-0731/summary.md).

This is a single sample per question. Reliably estimating solvability requires 3–5 runs.

## Running

You need [uv](https://docs.astral.sh/uv/) and a Together AI key:

```bash
export TOGETHER_API_KEY=...
uv run scripts/run.py --set set_1                       # DeepSeek V4 Flash, reasoning high
uv run scripts/run.py --set set_2 --model Qwen/Qwen3.7-Plus
uv run scripts/run.py --set set_3 --set set_4 --max-tokens 5000   # easy sets, as during selection
uv run scripts/run.py --base-url https://other-provider/v1 --model ...
```

Any OpenAI-compatible API works; the key is taken from `TOGETHER_API_KEY`. Results are written
to `results/<model>/`; questions already computed are skipped. To recompute a question, delete
its `.json`. Runs go 5 in parallel. A single question takes from a minute to ~20 minutes.

## Agent team chats

`embedding_eval/chat_runs.py` simulates a team chat of 10 agents on 10 benchmark problems (`set_3` or
`set_4`). Each turn is a separate DeepSeek-V4-Flash call; the agent's reasoning and message are recorded
separately in order to measure the spread of their embeddings over time.

- **negotiation** — 2 rounds: the agents distribute the problems (the order of problems and agents is shuffled), the outcome is `ASSIGNMENT`;
- **merge** — the agents solve the problems according to the assignment from `negotiation/run_<k>`, discuss the answers
  for 2 rounds, then Agent 1 writes `FINAL`.

```bash
uv run embedding_eval/chat_runs.py --bench set_3 --kind negotiation --runs 10
uv run embedding_eval/chat_runs.py --bench set_3 --kind merge --runs 10
```

Completed runs (10 of each kind for `set_3` and `set_4`) are in
`embedding_eval/runs/chat/<bench>/<kind>/run_<k>/`: `agent_<i>.jsonl` (turn: `turn`, `round`, `agent`,
`reasoning`, `message`, `reasoning_tokens`, `seed`), `transcript.json`, `result.json`, and for merge also
`solve/agent_<i>.json` (the agent's solution to its problem). Invalid attempts are saved as `run_<k>_invalid_<n>`.

The provider sometimes cuts off a long stream. In that case the `.json` will have `finish_reason: null` and an empty
answer: delete the file and run again.
