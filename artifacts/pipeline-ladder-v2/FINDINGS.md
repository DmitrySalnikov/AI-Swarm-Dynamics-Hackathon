# Four-rung full-arc comparison

One swarm and one matched one-task-per-agent private control per rung. All runs are new. Tasks: artifacts/pipeline-ladder-v2/items. Model/settings and finite-scalar normalization retained. Correctness never filters trajectories.

| Rung | Dataset | Swarm R | Private R | Swarm dimension | Private dimension |
|---|---|---:|---:|---:|---:|
| 1 | Unrelated domains | 0.6087 | 0.6175 | 8.6377 | 8.6421 |
| 2 | Physics subfields | 0.6714 | 0.6564 | 8.2841 | 8.3675 |
| 3 | Linear algebra | 0.8128 | 0.8078 | 6.5255 | 6.3671 |
| 4 | Same determinant template | 0.9045 | 0.9005 | 6.2425 | 6.1353 |

Full-arc blue curves retain real distribution/solving/commit observations. Orange has solving only, mapped by relative progress: no independent chat endpoints are invented. Paired and active-count figures at400/2048/4096 are saved per rung. One pair per rung; no uncertainty bands or causal claim from cross-rung differences. Lengths, held windows, short windows, final output format and recovery remain interpretation constraints. Inspect all errors/lengths; no desired-effect selection.
