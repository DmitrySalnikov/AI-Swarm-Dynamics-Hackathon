# Why the legacy separation differs from the stateful pilot

Verified on 13 October 2026 using local source, saved vectors and transcript
metadata. No paid generation or embedding requests. Reference images were
visually re-inspected. `python -m swarm_pipeline.legacy_diagnosis` reproduces the
allocation diagnostic and numerical evidence under ignored artifacts.

## Verified construction

The supplied set_4 figure corresponds to the legacy extended phase pipeline:
`embedding_eval/dynamics_mixed.py` loads `load_set(..., extended=True)` and seven
2048-token solve windows. `embed_solve_ext.py` appends independently generated
solutions of the same assigned question until the stream reaches 8192 tokens.
Each appended solution is used once. All 100 extended agent streams in set_4
use extra solutions, with 2 to 12 source solutions per stream including its own.
This is repeated same-task reasoning, not one continuous long solve.

`chat_runs.py::solve` sends ONLY the benchmark question as one user message.
Negotiation history is absent. Thus the old solving-phase separation cannot be
attributed to retained interaction context during those solves. Negotiation and
merge do contain exchanged messages, but the middle texts are private solves.

`build_concat.py` constructs independent fixed/shuffled streams by concatenating
ten separately generated solutions. Shuffling changes their order; it does not
generate a retained-context sequential trajectory. `dynamics_mixed.py` forms
ten seeded subsets of ten from twenty saved streams. Those control groups
overlap; they are not ten independent generated populations.

The swarm assigns ten distinct questions per group, then its extended solve
stays on the assigned question. Legacy shuffled windows contain approximately
4.6 to 6.1 distinct DOMINANT questions across ten agents, averaging over the
original subsets and window centres through 9216 tokens. Dominant task is
computed by character overlap against recorded piece spans; mixed windows can
contain additional tasks. This is a diagnostic label, not exact token weighting.

The original `bench/set_4.jsonl` is NOT ladder rung_4. It contains different
mathematics questions, such as geometry, determinant, series and calculus.
The current rung_1 contains different domains and deliberately longer tasks.
The older generator is Together DeepSeek-V4-Flash-0731; current generation uses
Cloud.ru DeepSeek-V4-Flash. Context, solution length, task set and provider/model
identifier changed together. Embedding model and geometry formulas were retained.

## Controlled offline allocation test

Reuse the same 100 legacy extended-solve vector sequences: ten runs per question,
ten questions, seven windows per sequence. At each synthetic group, choose a
different source run for each agent, then vary task allocation only:

1. One distinct question per agent.
2. Independently sampled questions, allowing collisions.
3. The same question for every agent, using distinct source runs.

There are 200 deterministic reassemblies per condition. These are resampled
layouts of existing vectors, not 200 new independent experiments. No new peer
information is provided to the solve texts. Metrics use the same upstream
implementation and normalized vectors. Each snapshot contains ten vectors.

| Allocation | R | Perpendicular spread | Effective dimension |
|---|---:|---:|---:|
| One question each | 0.5971 | 0.6414 | 8.1674 |
| Random questions with collisions | 0.6446 | 0.5753 | 5.0037 |
| Same question | 0.9705 | 0.0566 | 5.8436 |

One-question-each reconstruction nearly reproduces the original swarm solving
plateau (R 0.5967, dimension 8.1574 in the inspected 1500–8000 token interval).
Task allocation alone therefore produces a substantial dimension and dispersion
gap without communication during solving. The random-allocation ablation does
NOT fully reproduce original shuffled R (0.7154), because it uses the extended
solve pool rather than the original concatenated solo streams. Additional
effects from corpus composition, concatenation, lengths and window mixtures
remain unresolved. Do not claim allocation explains the entire R gap.

## What changed and what should be corrected

The earlier comparisons changed multiple experimental factors at once rather
than isolating the transition from assembled to stateful histories. The new
cyclic-offset control deliberately matches diverse tasks without interaction,
which was absent from the original supplied plot. It is expected to be harder
than a shuffled population with task collisions. A smaller gap against this
control is not an embedding implementation failure.

The geometry arc persists. Effective-dimension separation against shuffled also
persists; the large legacy R gap is weaker. Current results do not establish
which individual change reduced it. The offline test shows a concrete competing
explanation for the old middle-phase signal: distinct task allocation and
constructed long same-task streams. Rewriting prompts to maximize this gap
could increase task/format separation without improving interaction detection.

The appropriate next design is a bridge on the exact original set_4 and the
current frozen generator/embedding settings. Compare assembled versus stateful
histories while retaining the same task-allocation design, and add a one-task-
per-agent control with no negotiation or peer results. Analyze distribution,
private solving and result transfer separately. Keep all observed responses;
do not regenerate until a desired curve appears. If a real interaction effect
is sought during solving, tasks must require peer information or the protocol
must include measured transfer during solving; that is a different architecture
from the user's current private-solving single arc and needs an explicit design.

No prompt changes or new live generation were made in this diagnosis. Existing
research results remain preserved. All new repository contributions are English.

## Evidence

- [Task-allocation figure](../artifacts/pipeline-legacy-diagnosis/allocation_without_communication.png)
- [Numerical diagnosis](../artifacts/pipeline-legacy-diagnosis/diagnosis.json)
- Original construction: `embedding_eval/chat_runs.py`, `build_concat.py`,
  `embed_solve_ext.py`, `dynamics_mixed.py`.
- Current three-group findings: [DIAGNOSTICS_AND_NEXT_EXPERIMENTS.md](DIAGNOSTICS_AND_NEXT_EXPERIMENTS.md).

## Original-set matched pair completed

On 13 October 2026 the authorized bridge ran the original set_4 through current
stateful v2 and a one-task-per-agent independent control with identical realized
agent-task assignments. First seed73 exposed a parser bug: numeric JSON answers
were rejected as non-string answers, causing 24 rejected actions and repeated
reasoning. That pair remains archived. A shared finite-scalar canonicalization
fix was tested, then seed74 completed both conditions without any action errors,
partial continuations or missing usage. Swarm made40calls, private made10.
Private agents had no negotiation, peer input, shared packet or final team map.

| Window | Swarm R | Private R | Swarm effective dimension | Private effective dimension |
|---|---:|---:|---:|---:|
| 400 | 0.5704 | 0.5743 | 8.0544 | 8.0835 |
| 2048 | 0.6286 | 0.6109 | 8.0086 | 8.0141 |
| 4096 | 0.6405 | 0.6300 | 8.0140 | 8.0364 |

The large legacy separation is absent during solving in this matched pair. The
swarm arc remains at distribution and common final commit. This is consistent
with task-allocation and assembly confounding in the legacy middle phase, but
one pair does not prove statistical equivalence or absence of interaction signals.
Negotiation context, lengths, generation randomness and final output differ.
Most solves are short; one private agent generated33115tokens in one successful
call. Relative-progress curves repeat windows, so their dense points are not
independent observations. True token-clock plots display shrinking N separately.

The corrected pair cost RUB17.5766594502 generation with complete receipts, plus
USD0.00978486 embedding receipts. The earlier malformed-format pair is separate.
Both audits rerun, assignments verified, primary figures visually inspected.

- [Full swarm arc and private solving](../artifacts/pipeline-set4-clean/analysis/full_arc_w2048.png)
- [400-token paired progress and active-count views](../artifacts/pipeline-set4-clean/analysis/paired_w400.png)
- [Verified findings](../artifacts/pipeline-set4-clean/FINDINGS.md)
