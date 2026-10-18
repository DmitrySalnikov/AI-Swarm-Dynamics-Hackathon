# Ladder rung 1: Unrelated domains paired pilot

Same generator/settings, tasks and agent-task mapping. Private control gets only its own task and own history: no distribution chat, peer results, or common final commit. One group per condition; this isolates task allocation more closely but does not match prior context or final output.

| Window | Condition | R | Pairwise cosine | Effective dimension |
|---|---|---:|---:|---:|
| 400 | Swarm v2 | 0.5867 | 0.2716 | 8.5901 |
| 400 | Private: same assigned tasks | 0.5910 | 0.2773 | 8.6098 |
| 2048 | Swarm v2 | 0.6087 | 0.3008 | 8.6377 |
| 2048 | Private: same assigned tasks | 0.6175 | 0.3128 | 8.6421 |
| 4096 | Swarm v2 | 0.6472 | 0.3545 | 8.6637 |
| 4096 | Private: same assigned tasks | 0.6518 | 0.3611 | 8.6586 |

Means use private-solving relative progress, ten retained vectors. See active-token panels for true dropout. No fake private distribution or exchange is plotted. Swarm full-arc CSVs retain real phase observations. Compare actual solution lengths in analysis/manifest.json before interpreting small-window/large-window differences. Short text windows may contain whole solutions. Original legacy set_4 used independently generated same-task extensions; this run does not. No desired-effect selection, confidence intervals or detector accuracy claim.
