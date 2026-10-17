# Stateful swarm pilot findings

Recorded on 12 October 2026. This note compares the supplied legacy plots with
the completed rung 1 stateful pilot and explains why separation during solving
is less clear. The research target is interaction versus independent work.
Answer correctness is not used to select texts, windows, agents, or conditions.

## Main finding

The stateful swarm reproduces a convergence-divergence-convergence arc:
agents start with task distribution, solve different assigned tasks in parallel,
and converge during result exchange and individual final commits.

During solving, its geometry substantially overlaps the independent cyclic-order
control. Instantaneous dispersion therefore does not uniquely identify
interaction. The complete sequence of transitions is the more promising signal
to investigate; this pilot does not validate a detector or establish its accuracy.

## Observed geometry

These descriptive means use raw, L2-normalized Qwen3-Embedding-4B vectors and
2048-token windows with a 1024-token stride. Swarm solving windows are aligned
by relative progress; control means use natural window centres in the same
approximate token interval. They are not confidence estimates.

| Condition | Mean vector length R | Perpendicular spread | Effective dimension |
|---|---:|---:|---:|
| Cooperative swarm | 0.631 | 0.600 | 8.504 |
| Independent fixed order | 0.843 | 0.281 | 3.349 |
| Independent shuffled order | 0.657 | 0.565 | 5.775 |
| Independent cyclic offsets | 0.642 | 0.585 | 7.901 |
| Derived five swarm plus five shuffled | 0.662 | 0.559 | 6.030 |

Swarm R is approximately 0.946 at the first distribution endpoint and 0.957
at the last exchange endpoint. The middle-phase R gap is approximately 0.026
against shuffled and 0.011 against cyclic offsets. Effective dimension retains
some separation, but overlapping trajectories and one pilot per condition do
not establish a reliable decision rule.

## Why overlap during solving is expected

The cooperative protocol requires task distribution and result exchange, while
communication during solving is optional. The completed cooperative pilot
recorded zero peer deliveries in the solving phase. Each agent solved its
assigned task using private retained context.

The independent cyclic-order control also places agents on different tasks
without communication. Both groups consequently generate texts about different
problems. Similar solving-phase dispersion is consistent with this shared task
diversity. The embedding of an agent's own text does not directly observe whether
a chat channel is available.

Shuffled agents likewise often work on different tasks at a given token position.
Fixed-order agents more often discuss the same task, which is consistent with
their higher R and lower effective dimension. This is an interpretation of the
observations, not an isolated causal experiment.

## Differences from the supplied legacy plots

| Aspect | Legacy examples | Stateful pilot |
|---|---|---|
| Task set | Supplied set_4 examples | Ladder rung_1 |
| Trajectory construction | Separately generated chats and solutions assembled into a trajectory; extra solutions extend the solving phase | Continuous private histories across distribution, solving, exchange, and final commit |
| Groups | Ten groups with means and SD bands | One ten-agent group per condition, no replicate bands |
| Independent controls | Fixed and shuffled order | Fixed, shuffled, and cyclic offsets |
| Display horizon | First 10000 generated tokens | Full arc to approximately 19000, plus first-10000 views |
| Embedding model and metrics | Qwen3-Embedding-4B and the upstream geometry metrics | Same embedding model and metric implementation |

Several factors changed together: the task set, trajectory generation and memory,
the construction of the solving phase, and the number of groups. The contribution
of each factor to the smaller swarm-versus-shuffled gap has not been established.
An explanation attributing the change to a single factor would be premature.

Final convergence is stronger in the new plot: R ends near 0.96, compared with
approximately 0.84 in the supplied legacy example. The new final commit contains
a complete answer map for each agent; the common output format should be
considered when interpreting this convergence.

## Interpretation constraints

- A channel being available is different from agents using it during solving.
- Task diversity can produce dispersion without interaction.
- The five-plus-five line is derived from existing vectors, not a separately
  generated population or a linear interpolation of metric values.
- Different seeds, worker configurations, and recovery turns are present across
  the pilot conditions. Recovery can change later generated text.
- Shorter swarm solves hold windows during relative-progress alignment, following
  the legacy plotting approach. Coordinates are approximate generated-token
  positions, not wall-clock time.
- Variable solution lengths and changes in the number of active vectors remain
  a separate research question. This alignment is not a general solution to it.

## Follow-up comparison

To identify why the legacy separation changed, keep tasks, generation model,
embedding model, windowing, and alignment fixed while comparing assembled and
stateful trajectories. Add repeated groups before estimating uncertainty or
detector performance. Evaluate the full distribution-solving-exchange sequence
alongside instantaneous metrics, including the cyclic-order negative control.

These are follow-up directions, not new experiments started by this note.

## Local evidence

- Method and execution details: [PIPELINE.md](PIPELINE.md).
- Main figure: [2048-token raw geometry](../artifacts/pipeline-analysis/rung1_w2048_raw.png).
- Window variants: [400 tokens](../artifacts/pipeline-analysis/rung1_w400_raw.png)
  and [4096 tokens](../artifacts/pipeline-analysis/rung1_w4096_raw.png).
- Descriptive values: [summary2048.json](../artifacts/pipeline-analysis/summary2048.json).
- Parameters and provenance: [analysis-manifest.json](../artifacts/pipeline-analysis/analysis-manifest.json).
- Reproduced legacy example: [set_4 mixed plot](../artifacts/pipeline-analysis/legacy-reference/set_4_mixed.png).

The linked experiment artifacts are local and ignored by Git. This note and all
new repository contributions use English; no remote publication was performed.


## Protocol v2 completed pilot

Verified on 12 October 2026. The full generation, audit, embedding analysis and
comparison figure completed. Ten agents distributed ten rung 1 tasks, solved
privately in parallel, received one immutable results packet, and each committed
its complete answer map once. The audit verified packet identity, exactly one
successful submit and commit per agent, and no solving or commit discussion.

Generation took 10.36 minutes: 20 distribution calls, 10 parallel solving calls,
and 10 parallel commit calls. All 40 calls have usage receipts; no action errors,
HTTP retries, partial continuations or explicit restarts occurred. The local
usage-backed generation cost was RUB 44.6823834046, excluding the earlier tiny
smoke. New embeddings cost USD 0.01721614 for 860807 input tokens in 25 batches.
Of 6930 unique analysis windows, 750 required new embeddings; 6180 reused cache.
Existing independent controls were reused without regeneration. These costs are
local usage accounting, not a provider billing invoice.

The v1 and v2 solving trajectories are almost coincident. Simplifying the protocol
preserved the convergence-divergence-convergence arc, but did not establish clearer
instantaneous separation from independent cyclic offsets. Final convergence is
stronger in v2. A shared packet and the required common answer-map format are
part of this effect; it cannot be attributed exclusively to interaction.

The v2 exchange phase has one measured commit point. Lines connecting it to
solving are visual connections, not observations of gradual exchange dynamics.
The comparison uses within-phase relative progress, not elapsed time. There is
one pilot per version, with different seeds and historical recovery policies;
this is a descriptive comparison, not a causal estimate or validated detector.
Answer correctness never filters the analysis. Variable solution lengths and
holding windows remain a future methodological issue.

| Metric | v1 solving mean | v2 solving mean | v1 final | v2 final |
|---|---:|---:|---:|---:|
| Mean vector length R | 0.6309 | 0.6341 | 0.9566 | 0.9752 |
| Perpendicular spread | 0.6000 | 0.5958 | 0.0845 | 0.0489 |
| Effective dimension | 8.5037 | 8.5200 | 5.5778 | 5.0183 |

Primary figures at 400, 2048 and 4096 tokens and the v1/v2 comparison were visually
checked for readable labels and legends. Raw, centred, first-10000-token views,
PDFs and CSVs are saved locally. No new generation is running for this pilot.
No Git commit or push was performed.

- [V1 versus v2 figure](../artifacts/pipeline-analysis-v2/protocol_comparison.png)
- [V2 versus independent controls](../artifacts/pipeline-analysis-v2/rung1_v2_w2048_raw.png)
- [V2 execution audit](../artifacts/pipeline-v2/cooperative/seed_3/audit.json)
- [Analysis manifest](../artifacts/pipeline-analysis-v2/analysis-manifest.json)
- [Comparison values](../artifacts/pipeline-analysis-v2/comparison-summary.json)
