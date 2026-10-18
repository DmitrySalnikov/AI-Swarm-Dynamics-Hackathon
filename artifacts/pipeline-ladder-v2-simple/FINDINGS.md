# Four-rung full-arc comparison

One swarm and one matched one-task-per-agent private control per rung. All runs are new. Tasks: artifacts/pipeline-ladder-v2/items. Model/settings and finite-scalar normalization retained. Correctness never filters trajectories.

| Rung | Dataset | Swarm R | Private R | Swarm dimension | Private dimension |
|---|---|---:|---:|---:|---:|
| 1 | Unrelated domains | 0.6090 | 0.6062 | 8.6396 | 8.6389 |
| 2 | Physics subfields | 0.6630 | 0.6777 | 8.4619 | 8.2523 |
| 3 | Linear algebra | 0.8105 | 0.8163 | 6.3282 | 6.0906 |
| 4 | Same determinant template | 0.9038 | 0.9202 | 5.6992 | 6.7907 |

Full-arc blue curves retain real distribution/solving/commit observations. Orange has solving only, mapped by relative progress: no independent chat endpoints are invented. Paired and active-count figures at400/2048/4096 are saved per rung. One pair per rung; no uncertainty bands or causal claim from cross-rung differences. Lengths, held windows, short windows, final output format and recovery remain interpretation constraints. Inspect all errors/lengths; no desired-effect selection.
