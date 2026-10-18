# Ladder rung 4: Same determinant template paired pilot

Same generator/settings, tasks and agent-task mapping. Private control gets only its own task and own history: no distribution chat, peer results, or common final commit. One group per condition; this isolates task allocation more closely but does not match prior context or final output.

| Window | Condition | R | Pairwise cosine | Effective dimension |
|---|---|---:|---:|---:|
| 400 | Swarm v2 | 0.8393 | 0.6725 | 5.8668 |
| 400 | Private: same assigned tasks | 0.8541 | 0.7002 | 6.1423 |
| 2048 | Swarm v2 | 0.9038 | 0.7972 | 5.6992 |
| 2048 | Private: same assigned tasks | 0.9202 | 0.8302 | 6.7907 |
| 4096 | Swarm v2 | 0.9375 | 0.8657 | 5.6441 |
| 4096 | Private: same assigned tasks | 0.9489 | 0.8895 | 6.9901 |

Means use private-solving relative progress, ten retained vectors. See active-token panels for true dropout. No fake private distribution or exchange is plotted. Swarm full-arc CSVs retain real phase observations. Compare actual solution lengths in analysis/manifest.json before interpreting small-window/large-window differences. Short text windows may contain whole solutions. Original legacy set_4 used independently generated same-task extensions; this run does not. No desired-effect selection, confidence intervals or detector accuracy claim.
