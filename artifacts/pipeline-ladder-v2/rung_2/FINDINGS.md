# Ladder rung 2: Physics subfields paired pilot

Same generator/settings, tasks and agent-task mapping. Private control gets only its own task and own history: no distribution chat, peer results, or common final commit. One group per condition; this isolates task allocation more closely but does not match prior context or final output.

| Window | Condition | R | Pairwise cosine | Effective dimension |
|---|---|---:|---:|---:|
| 400 | Swarm v2 | 0.6410 | 0.3461 | 8.3090 |
| 400 | Private: same assigned tasks | 0.6314 | 0.3327 | 8.3687 |
| 2048 | Swarm v2 | 0.6714 | 0.3901 | 8.2841 |
| 2048 | Private: same assigned tasks | 0.6564 | 0.3681 | 8.3675 |
| 4096 | Swarm v2 | 0.7069 | 0.4443 | 8.2118 |
| 4096 | Private: same assigned tasks | 0.6946 | 0.4253 | 8.2355 |

Means use private-solving relative progress, ten retained vectors. See active-token panels for true dropout. No fake private distribution or exchange is plotted. Swarm full-arc CSVs retain real phase observations. Compare actual solution lengths in analysis/manifest.json before interpreting small-window/large-window differences. Short text windows may contain whole solutions. Original legacy set_4 used independently generated same-task extensions; this run does not. No desired-effect selection, confidence intervals or detector accuracy claim.
