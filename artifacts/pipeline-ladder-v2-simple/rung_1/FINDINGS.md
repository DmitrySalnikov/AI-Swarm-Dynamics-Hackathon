# Ladder rung 1: Unrelated domains paired pilot

Same generator/settings, tasks and agent-task mapping. Private control gets only its own task and own history: no distribution chat, peer results, or common final commit. One group per condition; this isolates task allocation more closely but does not match prior context or final output.

| Window | Condition | R | Pairwise cosine | Effective dimension |
|---|---|---:|---:|---:|
| 400 | Swarm v2 | 0.5984 | 0.2870 | 8.5274 |
| 400 | Private: same assigned tasks | 0.5865 | 0.2714 | 8.5216 |
| 2048 | Swarm v2 | 0.6090 | 0.3011 | 8.6396 |
| 2048 | Private: same assigned tasks | 0.6062 | 0.2973 | 8.6389 |
| 4096 | Swarm v2 | 0.6462 | 0.3529 | 8.6476 |
| 4096 | Private: same assigned tasks | 0.6494 | 0.3576 | 8.6646 |

Means use private-solving relative progress, ten retained vectors. See active-token panels for true dropout. No fake private distribution or exchange is plotted. Swarm full-arc CSVs retain real phase observations. Compare actual solution lengths in analysis/manifest.json before interpreting small-window/large-window differences. Short text windows may contain whole solutions. Original legacy set_4 used independently generated same-task extensions; this run does not. No desired-effect selection, confidence intervals or detector accuracy claim.
