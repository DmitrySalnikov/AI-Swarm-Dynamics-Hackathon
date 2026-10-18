# Ladder rung 3: Linear algebra paired pilot

Same generator/settings, tasks and agent-task mapping. Private control gets only its own task and own history: no distribution chat, peer results, or common final commit. One group per condition; this isolates task allocation more closely but does not match prior context or final output.

| Window | Condition | R | Pairwise cosine | Effective dimension |
|---|---|---:|---:|---:|
| 400 | Swarm v2 | 0.7720 | 0.5520 | 6.4449 |
| 400 | Private: same assigned tasks | 0.7734 | 0.5541 | 6.2800 |
| 2048 | Swarm v2 | 0.8105 | 0.6192 | 6.3282 |
| 2048 | Private: same assigned tasks | 0.8163 | 0.6296 | 6.0906 |
| 4096 | Swarm v2 | 0.8304 | 0.6553 | 6.4892 |
| 4096 | Private: same assigned tasks | 0.8399 | 0.6730 | 6.3890 |

Means use private-solving relative progress, ten retained vectors. See active-token panels for true dropout. No fake private distribution or exchange is plotted. Swarm full-arc CSVs retain real phase observations. Compare actual solution lengths in analysis/manifest.json before interpreting small-window/large-window differences. Short text windows may contain whole solutions. Original legacy set_4 used independently generated same-task extensions; this run does not. No desired-effect selection, confidence intervals or detector accuracy claim.
