# Ladder rung 3: Linear algebra paired pilot

Same generator/settings, tasks and agent-task mapping. Private control gets only its own task and own history: no distribution chat, peer results, or common final commit. One group per condition; this isolates task allocation more closely but does not match prior context or final output.

| Window | Condition | R | Pairwise cosine | Effective dimension |
|---|---|---:|---:|---:|
| 400 | Swarm v2 | 0.7824 | 0.5696 | 6.6694 |
| 400 | Private: same assigned tasks | 0.7722 | 0.5519 | 6.5021 |
| 2048 | Swarm v2 | 0.8128 | 0.6233 | 6.5255 |
| 2048 | Private: same assigned tasks | 0.8078 | 0.6143 | 6.3671 |
| 4096 | Swarm v2 | 0.8337 | 0.6613 | 6.5688 |
| 4096 | Private: same assigned tasks | 0.8299 | 0.6546 | 6.4936 |

Means use private-solving relative progress, ten retained vectors. See active-token panels for true dropout. No fake private distribution or exchange is plotted. Swarm full-arc CSVs retain real phase observations. Compare actual solution lengths in analysis/manifest.json before interpreting small-window/large-window differences. Short text windows may contain whole solutions. Original legacy set_4 used independently generated same-task extensions; this run does not. No desired-effect selection, confidence intervals or detector accuracy claim.
