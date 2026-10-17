# Stateful agent pipeline

The runner is an initial implementation for the reviewed research plan. It reads
the provisional `bench/ladder/rung_*.jsonl` sets without generating replacement
tasks or modifying baseline artifacts.

## Run locally

```powershell
uv run --locked python -X utf8 -m unittest discover -s tests -v
uv run --locked python -X utf8 -m swarm_pipeline.runner --items bench/ladder/rung_1.jsonl --condition cooperative --offline --out artifacts/pipeline-offline
```

Offline mode uses scripted agents with placeholder answers. It tests execution,
not research performance, and needs no credentials or downloads. Run the other
conditions with `--condition fixed`, `shuffled`, or `disjoint`.

The local `.env` holds the Cloud.ru key and is ignored by Git. To check credentials
through the model catalogue without requesting generation:

```powershell
uv run --locked --env-file .env python -X utf8 -m swarm_pipeline.runner --check-api
```

A live run requires an explicit positive `--budget-rub` and the key. Output goes
to `<out>/<condition>/seed_<seed>/`; the configured RUB limit applies cumulatively
to all conditions and seeds under the same output root. Keep live and offline
experiments in separate roots. Use the reviewed spending cap, not a placeholder.

## Execution contract

- `fixed`: each agent solves all tasks in the supplied order, one at a time.
- `shuffled`: each agent receives a deterministic, independently shuffled order.
- `disjoint`: cyclic offsets produce distinct current tasks when all agents advance
  at the same rate. Different solution durations can create overlap; this is an
  order control, not enforced synchronization on elapsed time or token count.
- `cooperative`: one task per agent; each agent broadcasts a task preference
  before claiming ownership, with claim receipts delivered to peers. Agents solve with
  retained histories, send messages during work, then report and assemble results.
  Agents that finish early can respond when new messages arrive. The MVP requires
  equal numbers of agents and tasks for this condition.

Scheduling is deterministic round-robin between calls. Communication reaches
the recipient's next call, not the middle of an already streaming call. Agents
can voluntarily return intermediate `work` notes or `send` actions instead of
finishing their entire solution in one call. This provides communication at
call boundaries; the pilot must verify that the model actually uses it.

An agent's private history contains its prior visible JSON responses and working
notes. Peers receive only explicitly sent messages. Raw provider reasoning is
stored for observation but is not replayed automatically; this memory policy
is explicit and shared across conditions. Context growth stops at a configured
byte limit rather than silently dropping earlier turns. The initial task
presentation is full statements for negotiation and one task at a time for
independent agents; ID-only negotiation and all-tasks-upfront controls remain
future variants.

The experiment compares different amounts of task work per agent: independent
agents solve the whole set, while cooperative agents split it. Report task
coverage, generated tokens, calls, elapsed time, and correctness alongside metrics.

## Audit and recovery

Each run has a manifest with the full private benchmark, task hash, configuration,
protocol/code hashes, and pricing assumptions. Reference answers are excluded from model requests.
Each call has `request.json`, `response.json`, and `event.json`: actual context,
output/reasoning, provider usage, delivered messages, and protocol errors.
Credentials are never serialized. `state.json` checkpoints histories, assignments,
inboxes, and results after every response. An output-root lock prevents concurrent
writers, including different conditions sharing a budget.

Re-running an identical command resumes from the checkpoint. Completed runs do
not repeat requests. A durable response is applied after a crash without another
provider call. If a request has no durable response, the outcome/billing is
unknown: automatic replay is refused and its reservation counts against the
shared budget. Provider errors and incomplete outputs halt the run; they are not
automatically retried. Different settings or task content require a new run path.

The `generated/` export contains each agent's own reasoning and response text,
with character spans and phase/call metadata. It excludes received text and
replayed history. Provider token totals are attached to call segments; exact
window/token alignment still requires the downstream tokenizer. Existing
embedding and plot scripts have not yet been adapted to these artifacts.

## Cost controls

Pricing defaults reproduce the ladder's snapshot: 43.2978 RUB per million input
tokens and 86.5834 per million output tokens. Override both rates if the tariff
changes. Requests reserve estimated input cost plus the configured output limit;
provider usage settles actual cost, and missing usage conservatively charges the
reservation. Global call and context limits are also enforced.

The ladder report records Cloud.ru exceeding `max_tokens`. The client closes an
oversized stream and stops incomplete runs, but server-side generation or billing
may continue. A local spending cap is therefore an admission limit, not a promise
of a hard account-side cap. Do not launch the full statistics grid before reviewing
usage and cancellation behavior in the pilot.

The public [Cloud.ru model catalogue](https://cloud.ru/products/evolution-ai-factory/catalog-foundation-models)
was checked on 12 October 2026 and lists the same V4 Flash rates.
Using the stored selected-task runs, rung 1's sum of mean solution lengths is
178,843 output tokens. One run of each condition with ten agents entails
100 + 100 + 100 + 10 = 310 task solutions, projecting approximately 480 RUB in
solution output and 491 RUB including the original isolated-call inputs.
These are reference projections, not live-run forecasts: persistent history,
additional working turns, negotiation, and exchange add cost, and model behavior
can change in context. A proposed first-stage allocation is 1,000 RUB total,
with at most 100 RUB for a small protocol probe before the full one-rung pilot.
The user approved up to 3,000 RUB on 12 October 2026, preferring about 1,000 RUB
if sufficient. Start with a 1,000 RUB admission cap; raise it only if verified
pilot costs require more, never above the approved 3,000 RUB across the whole
pilot including protocol checks. The first two-agent live API/protocol check
completed correctly in eight calls for 0.3586 RUB. It predates the mandatory
preference broadcast and remains a historical probe rather than a pipeline result.
The subsequent mandatory-preference probe also completed correctly (ten calls);
the combined recorded cost of both probes is 0.8877 RUB.
The first ten-agent cooperative pilot completed in 64 calls and 51.28 minutes,
costing 78.9926 RUB (79.8803 RUB including probes). Its audit verified complete
history carryover, message delivery, phase progression, usage, and generated-only
exports. All ten task submissions were correct. During result exchange, a late
incorrect economics correction propagated to three final answer maps; seven
agents ended with 10/10 and three with 9/10. The protocol currently lacks a final
consensus barrier. Four malformed action attempts were rejected and recovered.
These observations remain part of the pilot rather than being edited out.

## Long pilot supervision

`python -m swarm_pipeline.pilot` supervises one condition, freezes its task input,
and publishes a small status JSON plus stdout/stderr paths and process IDs.
It inherits credentials from its launching environment without serializing them.
The local event-monitor skill can watch that status and queue a completion or
failure event to the owning chat. Initial running and unchanged progress are
silent; queue acceptance must not be confused with automatic consumption.
Actual acceptance and owning-chat consumption were observed for the completed
cooperative pilot. `control_batch` now runs the three independent controls using
the same frozen tasks and scientific configuration, with a shared spending cap.
The initial fixed-order control was interrupted technically: its supervisor
encountered a Windows status-write lock, and its still-running worker later had
a connection reset without a usage receipt. The partial seed-1 trial is preserved.
Supervision now retries transient write locks and observes immutable call journals.
A separately labelled seed-2 replacement uses the unchanged scientific runner
and seed-1 cooperative reference, with a 900 RUB cumulative admission cap and
100 RUB set aside for uncertain billing. This changes the sampling realization,
not the task set or agent protocol. It is not an exact replay of the unresolved
request. The uncertainty reserve is not reported as an actual charge.
The worker and watcher need an awake computer. Stopping the watcher does not
stop the pilot. No recurring automation is installed by this workflow.

## Next validation milestone

The seed-2 control also stopped with an incomplete provider stream and no usage
receipt. Both failed attempts are preserved. Full live benchmarks are stopped;
the current priority is fast pipeline validation and concurrent execution.

`pilot` and `control_batch` now default to `--workers 10`. For direct execution,
use `python -m swarm_pipeline.parallel --workers 10` with the usual runner flags.
Independent agents solve concurrently while retaining their own sequential task
histories. Cooperative solving uses bounded rounds: each agent has at most one
outstanding turn, all inputs are frozen at round start, and messages enter the
next round. Negotiation and exchange remain serial. A single coordinator journals
requests and reserves the shared budget before dispatch, persists responses as
they arrive, and applies them in deterministic order. Unknown interrupted calls
are never replayed. The manifest records scheduler, worker count and source hash;
existing serial artifacts cannot be resumed with this scheduler. Cooperative
parallel rounds change message timing and require a new labelled experiment.
The three control conditions run in sequence, with up to ten agents within each.

Use `--workers 1` for the original serial pilot. Optional `--partial-continuation`
also supports parallel rounds: identifiable, nonempty interrupted solving fragments
enter only their own agent's private history on a new turn. Every continuation is
journaled and audited; it changes memory on interrupted calls and must be labelled
experimental recovery. Empty or unidentified outcomes remain halted. Admission
accounting reserves 100 RUB exposure per unreceipted incomplete call, separately
from recorded cost estimates. No original request is silently replayed.

On 12 October the user explicitly requested completing the paid pilot. Seed-3
parallel controls were launched with ten workers and fragment continuation, using
the frozen rung-1 tasks and historical serial cooperative seed-1 reference.
The cumulative admission cap is 1,000 RUB; prior failed attempts are included.
Each completed control is audited automatically. Cooperative reference timing
and recovered memory differ, so this is a pipeline validation comparison rather
than a fully matched clean research result. The human ceiling remains 3,000 RUB.

Seed 3's fixed trial later halted at 70 calls: 59/100 submitted solutions, 55
correct, followed by five HTTP errors without captured status codes or output.
An explicit recovery preserves all completed work and original journals, adds
logged private restart turns for those five unresolved attempts, reduces workers
to four, and raises the cumulative admission cap to 2,000 RUB within the approved
ceiling. These new turns are retries from retained context, with unresolved billing
reserved; they are not reconstructions of missing responses. Execution revisions
are archived and audited against historical worker bounds. HTTP status and
Retry-After are now captured. Verified usage-backed cost before recovery was
153.8663 RUB; estimates and admission exposure are reported separately. Recovered
trajectories must be labelled experimental rather than clean research controls.

The user requested `NO_PROXY=*` on 12 October. It is set in ignored `.env` and
the pilot launcher for newly spawned workers, including future conditions from
an already-running batch. Python had discovered Windows system proxies despite
no proxy environment variables. With bypass enabled, a catalogue request completed
in 0.254 seconds and a tiny generation completed normally with a usage receipt in
4.487 seconds (0.0024244 RUB). Existing worker connections were not interrupted;
this probe alone does not establish the cause of earlier stream failures.

The direct fixed run subsequently reached 78/100 submissions and halted at call
94 on HTTP 500. It was resumed from retained state. Optional parallel recovery now
also creates explicit logged new turns after empty HTTP 429/500/502/503/504 errors,
with 10/20/40-second backoff and at most three automatic retries per agent/task.
Retry-After can extend the wait up to 60 seconds. All recovery records share the
20-record condition limit, and unknown billing still reserves 100 RUB per failed
call. Automatic HTTP retries are counted separately by the auditor. Other unknown
outcomes and request journals without durable responses remain halted. The
admission cap remains 2,000 RUB and direct transport remains enabled.

At call 110 the admission exposure reached 2,002.6432 RUB and blocked the next
request. This was an admission stop, not a new provider failure: 88/100 fixed
solutions were retained, with no outstanding requests. Usage-backed spend was
202.6432 RUB and unreceipted estimates 169.3584 RUB; conservative exposure is not
known billing. A budget-only manifest revision raised the cap to the previously
approved 3,000 RUB ceiling, preserving all scientific and transport settings.
The worker resumed at call 110. This ceiling must not be exceeded. Pilot status
publication now excludes stale result files from earlier worker attempts.

Latest user steering keeps the existing architecture and requests completion of
the queued pilots, with approximately 200 RUB per run and 1,000 RUB total as
expectations. New parallel workers no longer add the arbitrary 100 RUB buffer per
unreceipted failure; they use the original request cost estimate when usage is
missing and keep billing uncertainty visible. The previous approved 3,000 RUB
ceiling remains. The 20-record recovery cutoff was raised to 300; per-agent/task
HTTP retry limits remain bounded. The active fixed worker was left uninterrupted
at 97/100 solutions; subsequent controls load the revised accounting. The monitor
was replaced independently to incorporate the latest instructions.

Fixed seed 3 completed: 127 calls, 100 submitted answers, 89 correct, integrity
audit passed. Its condition ledger is 235.8939 RUB including estimated costs.
Shuffled reached 51 correct submissions out of 51 saved and halted at call 72 on
an empty TimeoutError. It was resumed without regenerating fixed or earlier
answers. The transport timeout is now 900 seconds; known empty TimeoutError and
ConnectionResetError have logged, bounded recovery turns. The batch reaudits and
reuses completed matching conditions instead of launching their workers again.

The rung-1 generation pilot is now complete. All four integrity audits were rerun
and passed. Fixed seed 3: 89/100 correct, 127 calls; shuffled seed 3: 96/100,
130 calls; disjoint seed 3: 94/100, 113 calls, ten workers. The cooperative seed-1
reference had 10/10 initial submissions; three final maps lost one correct answer
during exchange. Independent runs had zero peer deliveries. Comparison metadata
is saved in ignored `artifacts/pipeline-live/comparison-rung1.json`.
Entire-root accounting, including probes and prior failed trials: 553.1371 RUB
usage-backed plus 357.7590 RUB unreceipted estimates, 910.8961 RUB combined ledger.
This is not a verified invoice. No pipeline workers remain active. Embeddings,
window metrics and detection plots were subsequently computed as described below;
these recovered single-trial trajectories remain pilot data, not a clean
statistical comparison.

## Embedding geometry of the stateful pilot

`swarm_pipeline/analyze.py` applies the original `embedding_eval/dynamics.py`
metric implementation to each agent's generated exports. Correctness is not used
to select agents, solutions, or windows. Inputs contain only own generated
reasoning and visible output, including observed interrupted fragments, without
inboxes, prompt replay, or private reference answers.

The original plots use Qwen3-Embedding-4B through OpenRouter, without an instruction
prefix, with L2 normalization and the DeepSeek-V3 tokenizer. The new plots use
the same model and metrics, windows of 400/2048/4096 tokens and half-window stride.
Negotiation and exchange each use whole first/last own turns. Solving is aligned
by relative progress to the longest solve, holding windows for shorter agents,
as in the original phase plot code. Coordinates are mean generated-token
positions, not wall-clock time. This is explicit alignment, not a solved general
correction for variable agent lengths. Independent streams use their own natural
token windows; mixed curves combine five swarm and five shuffled agents and are
derived views, not newly generated cooperative populations.

Artifacts: ignored `artifacts/pipeline-analysis/`, with nine PNG/PDF figures,
six metric CSVs, `analysis-manifest.json`, `summary2048.json`, and content-hashed
vector caches. Both raw and globally centered figures are saved, along with raw
first-10k views. One pilot group exists per condition, so no replicate SD bands
are drawn. The same-task hash, source export hashes, analysis source snapshot and
renderer hash are recorded. All 7,053 vectors were checked for finite values,
dimension 2,560 and unit norm. Embedding usage: 8,099,502 tokens, $0.16199004,
excluding tiny smoke checks. OpenRouter works through the existing Windows system
proxy; generation on Cloud.ru retains NO_PROXY=*. Only the analysis process
changes its proxy environment; the system configuration is untouched.

The original set_3/set_4 mixed plots were reproduced from committed vectors into
`legacy-reference/` without new requests or modifying baseline files. Fresh Git
fetch found no newer commit than the already-reviewed ladder commit17de56b.

At width 2048 the mean raw R in the swarm solving phase is0.631; independent
fixed0.843, shuffled0.657, disjoint0.642 (control windows in the corresponding
token interval). Swarm endpoints rise to approximately0.946 and0.957. The arc is
visible, but solving-phase swarm geometry substantially overlaps disjoint: these
plots demonstrate a trajectory pattern, not a validated cooperation classifier.

Run with the local OpenRouter credential:

```powershell
uv run --locked --env-file .env python -X utf8 -m swarm_pipeline.analyze
```

Cached vectors are reused without provider requests; no generation is repeated.

## Protocol v2 one pass experiment

The user authorized the v2 implementation and full pilot with ten parallel model
requests. V1 data and independent controls are preserved. `protocol_v2.py` changes
only cooperation: serial task distribution, a barrier before private parallel
solving, a coordinator-built common packet after every agent submits, then
parallel final commits. Solving accepts only submit actions with brief published
notes under 2000 characters. No peer sends or work actions are accepted in that
phase. Each final commit preserves all published answers exactly; no discussion,
recomputation or corrections occur. This deliberately shared format must be
considered when interpreting final embedding convergence.

The packet contains task ID, author, submitted answer and brief explanation,
ordered by the frozen task list. It contains no private provider reasoning or
reference answers. Agents retain their separate histories. Packet identity and
the two barriers are audited, including exactly one successful submit and commit
per agent. Technical retry attempts remain separately journaled and unreceipted
billing stays estimated. Empty distribution and commit failures have bounded
recovery with the original shared state, not invented outputs.

Offline suite passed31 tests; a two-agent live smoke completed10 calls for
0.5405 RUB, with packet and single-commit audit passed. Full rung1 v2 uses seed3,
ten workers, the same generation model/temperature and a local500 RUB admission
envelope; expected additional spend is around200 RUB or less. The prior human
overall ceiling3000 RUB remains. No arbitrary100 RUB buffers are added.

`v2_experiment.py` supervises generation, audit, analysis and v1/v2 comparison:

```powershell
uv run --locked --env-file .env python -X utf8 -m swarm_pipeline.v2_experiment --workers 10 --seed 3 --budget-rub 500
```

Generation/status output is in ignored `artifacts/pipeline-v2/`; figures and CSVs
are in `artifacts/pipeline-analysis-v2/`. Independent control and v1 vector caches
are reused; only new v2 windows require embedding requests. V2 exchange plots
contain one final-commit point rather than a fabricated second chat round.
`protocol_comparison.png` compares v1 and v2 at relative progress within each
phase, separately from the natural token-axis control figures. Correctness is
not an analysis selection criterion. The event monitor watches the entire
experiment and reports completion only after audit and plot generation.

1. Confirm the live RUB cap and rates.
2. Run a small provider/protocol probe, checking action formatting, reasoning,
   usage fields, memory behavior, and output-limit handling.
3. Run one ten-agent pilot per condition on one provisional rung and manually
   read the transcripts, including evidence that received messages affect work.
4. Adapt window embeddings and phase plots to the event logs, then expand across
   rungs, repeats, and a second model after reviewing the pilot.
