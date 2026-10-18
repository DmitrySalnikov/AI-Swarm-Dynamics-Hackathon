# Ladder rung 4: Same determinant template paired pilot

Same generator/settings, tasks and agent-task mapping. Private control gets only its own task and own history: no distribution chat, peer results, or common final commit. One group per condition; this isolates task allocation more closely but does not match prior context or final output.

| Window | Condition | R | Pairwise cosine | Effective dimension |
|---|---|---:|---:|---:|
| 400 | Swarm v2 | 0.8509 | 0.6938 | 6.5107 |
| 400 | Private: same assigned tasks | 0.8420 | 0.6779 | 6.1555 |
| 2048 | Swarm v2 | 0.9045 | 0.7982 | 6.2425 |
| 2048 | Private: same assigned tasks | 0.9005 | 0.7908 | 6.1353 |
| 4096 | Swarm v2 | 0.9348 | 0.8601 | 6.0315 |
| 4096 | Private: same assigned tasks | 0.9310 | 0.8525 | 6.0696 |

Means use private-solving relative progress, ten retained vectors. See active-token panels for true dropout. No fake private distribution or exchange is plotted. Swarm full-arc CSVs retain real phase observations. Compare actual solution lengths in analysis/manifest.json before interpreting small-window/large-window differences. Short text windows may contain whole solutions. Original legacy set_4 used independently generated same-task extensions; this run does not. No desired-effect selection, confidence intervals or detector accuracy claim.
