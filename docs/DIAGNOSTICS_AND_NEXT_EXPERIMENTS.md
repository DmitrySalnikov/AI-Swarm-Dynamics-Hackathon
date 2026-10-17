# V2 diagnostics and next experiments

Recorded on 12 October 2026. Re-read the public text export of Andrey's research
plan, saved locally under ignored artifacts/pipeline-diagnostics. Its instructions
are research proposals, not independent authorization. The user's current scope
prioritizes interaction detection and does not filter on answer correctness.

## New analysis completed

Ran `python -m swarm_pipeline.diagnostics` using completed v2 seed 3, existing
independent controls, and cached Qwen3-Embedding-4B vectors. No paid requests.
Five figures are saved as PNG/PDF, with numerical JSON and CSV evidence:

- [Separate control comparisons](../artifacts/pipeline-diagnostics/controls_side_by_side.png)
- [Solving-phase zoom](../artifacts/pipeline-diagnostics/solving_zoom.png)
- [Alignment sensitivity](../artifacts/pipeline-diagnostics/alignment_sensitivity.png)
- [Vector-count experiment](../artifacts/pipeline-diagnostics/vector_count_effect.png)
- [Per-agent solution lengths](../artifacts/pipeline-diagnostics/solution_lengths.png)

The control comparisons use common vertical scales and omit the mixed population
to make each negative control visible. Shaded boundaries are approximate: first
and last solving-window centres, not exact wall-clock phase boundaries. V2 final
exchange has one measured commit point; connecting lines are not intermediate
observations. There is no replicate uncertainty estimate.

## Unequal lengths and disappearing vectors

Own solving lengths range from 7535 to 15872 tokens: a 2.106 ratio. Comparing the
same window index across agents eventually drops shorter solutions. These are
token-index snapshots, not simultaneous wall-clock observations.

Three clocks/policies were evaluated with the same cached 2048-token windows:

1. Active only: remove agents without another window at that index.
2. Hold: retain the final available window of completed agents.
3. Relative progress: map each agent's whole solve to a common progress grid,
   as in the current main figures. Windows repeat when solves are shorter.

Holding retains stale text. Relative progress compares different absolute token
positions and can smooth or shift events. Neither is a universal repair. Active
only changes both sample size and task composition. Active metrics are omitted
below four vectors; counts remain visible for the full horizon.

An exhaustive subset experiment removes vectors from ONE fixed mid-solve
snapshot, without changing texts. Across all subsets, mean R changes from 0.6405
at N=10 to 0.7129 at N=4 and 0.8197 at N=2. Mean effective dimension changes from
8.5212 to 2.9373 to 1.0000. Thus sample count alone can mimic convergence.
The centred covariance has rank at most N-1: a large dimension drop after agent
completion can occur even with no new shared information.

For unit vectors, mean off-diagonal pairwise cosine is exactly
`C = (N * R**2 - 1) / (N - 1)`. This removes the self-similarity/count term in R.
Mean C across all equally weighted subsets remains 0.3448 at every subset size;
the numerical invariant and dimension bound were checked. Subset percentile
bands measure sensitivity to composition, not uncertainty between independent
runs. Actual dropout is not random: hard tasks may survive longer, so C can still
change. Effective dimension divided by N-1 remains sample-dependent and is only
a diagnostic, not a proven unbiased replacement.

## Research-plan status

The public plan proposes a four-rung task ladder, retained-context cooperation
and sequential controls, prompt/transcript review, a second generator, then
two generators/two embeddings/four sets/four conditions/ten repeats. The stated
total 400 does not equal this product (640 analysis combinations).

The ladder exists upstream; retained-context runs and all four pilot conditions
exist locally; one generator and one embedding model have been analyzed. V2
implements the user's simpler distribution/parallel-solve/shared-result arc.
Independent repeated groups, a second generator, a second embedding model,
the other three rungs in this pipeline, and held-out detection remain unfinished.
More agents are a separate factor from more independent groups.

## Recommended experiment order

1. Extend outputs with N, mean pairwise cosine, finish masks, and explicit clock
   labels. Use the common all-active token interval as one conservative view,
   with full relative-progress trajectories as a separate phase-based view.
2. Generate three matched independent groups per condition on rung 1 with the
   current generator and ten agents. Freeze prompts, task assignments, recovery
   policy, token budget and windowing; preserve all observed outputs. Treat
   paired seeds as a design aid, not proof of deterministic provider sampling.
   Plot individual groups and between-group dispersion before selecting features.
3. Add a negative control with the same task allocation, common input and output
   schema, and no peer information. Match workload and final output size. Existing
   controls solve ten tasks each while swarm agents solve one each, so they do
   not isolate communication alone. A common packet of identical answers can
   cause final convergence irrespective of how those answers were acquired.
4. Ablate final representation using existing outputs: prose/reasoning versus
   JSON, common packet text versus individual explanation. New text partitions
   may require new embeddings, but do not require regenerating solutions.
5. Repeat the frozen design with a second generator, then test rung 4 (same task,
   different parameters) against rung 1. This addresses model and task-diversity
   dependence before changing population size or embedding model.
6. Test agent counts only with a defined workload design: fixed total tasks and
   fixed tasks per agent answer different questions. Compare metrics at matched
   N, including exhaustive/random subsets, rather than raw dimension alone.
7. Estimate discrimination on held-out GROUPS, not randomly split overlapping
   windows. Compare a temporal arc feature to instantaneous geometry. Group-level
   splits prevent transcript leakage; no current detector-accuracy claim.

Do not describe v1/v2 differences as within noise without independent repeats.
Current evidence supports near-coincident solving trajectories, not a noise test.
Do not assume centered embeddings remove task or format confounds. Overlapping
windows are dependent; short phase turns and long solution windows differ in
length; protocol JSON, shared prompts, interrupted reasoning and held windows
can affect geometry. Token chronology and actual wall-clock concurrency are
different observables. Agent finish timestamps should be retained if wall-clock
analysis is later added.

These recommendations are a staged follow-up design. No additional generation
grid was started by this diagnostic run. Code, documents and new figure labels
are English; original reference text is local and ignored. Nothing was pushed.

## Three-group series completed on 13 October 2026

The user subsequently authorized three groups per each of four conditions and
completion through figures and conclusions. Seeds 3, 41 and 42 are now complete:
four historical groups reused, eight new runs generated, all twelve audits passed.
Generation journals were not replayed during recovery. The final embedding stage
failed after repeated connection resets; cache preserved all but nineteen vectors.
Recovery embedded those nineteen and completed the figures. Local generation
accounting for eight new runs totals RUB 1158.1484311916, including estimates for
unreceipted calls. Embedding receipts total USD 0.35169922; possible charges for
connection failures without a response are not confirmed by these receipts.

The aggregate covers 26 of 33 progress-grid points, up to progress 0.78125, where
every condition and all three groups have observations. The longest swarm's
late trajectory exceeds the 32000-token control prefix. No extrapolation was
used. A preliminary aggregate had missing group-three tail values; verification
restricted curves, bands, summaries and paired differences to this common domain.
No windows were selected by correctness or desired separation.

| Condition | Pairwise cosine, group mean +/- sample SD | Effective dimension, group mean +/- sample SD |
|---|---:|---:|
| Cooperative | 0.3321 +/- 0.0058 | 8.5120 +/- 0.0274 |
| Fixed | 0.6635 +/- 0.0456 | 3.7766 +/- 0.5787 |
| Shuffled | 0.3790 +/- 0.0135 | 5.8486 +/- 0.2140 |
| Cyclic offsets | 0.3519 +/- 0.0103 | 7.5825 +/- 0.8714 |

Swarm mean pairwise cosine is lower than each control in all three paired groups.
The gap against cyclic offsets is small and trajectories overlap, so this does
not establish reliable classification. Fixed-order separation remains dominated
by same-task versus different-task text. Effective dimension separates shuffled
more strongly but lies near its rank ceiling for swarm: approximately 8.51 out
of a maximum 9. Its low observed group dispersion need not imply a robust
interaction detector; task diversity and metric saturation can contribute.

Natural token-clock plots confirm that dimension declines as agents disappear.
Pairwise cosine does not show the same mechanically required dimension decline,
but still varies with surviving task composition. Main progress-aligned figures
retain N=10 and should not be interpreted as actual ongoing agent counts.

Execution differences remain material. Historical controls contain repairs and
different worker settings. New swarm seed 41 had two action errors; seed 42 had
thirty, including repeated invalid submissions and four partial continuations.
All reached one successful submit and commit per agent. Failed actions and
observed reasoning remain in the embedding input. There was no protocol change
to force cleaner curves and no regeneration to remove an inconvenient group.

The next experiment should match task allocation/workload and final output
format while withholding peer information. That control is more informative
than increasing population size before count and format effects are isolated.
Three groups support descriptive replication, not a held-out detector score.

- [Three-group solving geometry](../artifacts/pipeline-replicates/solving_three_groups.png)
- [Natural-clock vector dropout](../artifacts/pipeline-replicates/active_vectors_three_groups.png)
- [Numerical findings](../artifacts/pipeline-replicates/FINDINGS.md)
- [Audits, accounting and verification](../artifacts/pipeline-replicates/verification.json)
