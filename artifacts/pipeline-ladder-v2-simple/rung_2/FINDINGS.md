# Ladder rung 2: Physics subfields paired pilot

Same generator/settings, tasks and agent-task mapping. Private control gets only its own task and own history: no distribution chat, peer results, or common final commit. One group per condition; this isolates task allocation more closely but does not match prior context or final output.

| Window | Condition | R | Pairwise cosine | Effective dimension |
|---|---|---:|---:|---:|
| 400 | Swarm v2 | 0.6406 | 0.3454 | 8.4155 |
| 400 | Private: same assigned tasks | 0.6514 | 0.3614 | 8.2107 |
| 2048 | Swarm v2 | 0.6630 | 0.3775 | 8.4619 |
| 2048 | Private: same assigned tasks | 0.6777 | 0.3997 | 8.2523 |
| 4096 | Swarm v2 | 0.6971 | 0.4291 | 8.3432 |
| 4096 | Private: same assigned tasks | 0.7123 | 0.4530 | 8.1638 |

Means use private-solving relative progress, ten retained vectors. See active-token panels for true dropout. No fake private distribution or exchange is plotted. Swarm full-arc CSVs retain real phase observations. Compare actual solution lengths in analysis/manifest.json before interpreting small-window/large-window differences. Short text windows may contain whole solutions. Original legacy set_4 used independently generated same-task extensions; this run does not. No desired-effect selection, confidence intervals or detector accuracy claim.
