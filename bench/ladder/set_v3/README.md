# Ladder set v3: ten new topics per rung

`rung_{1,2,3,4}.jsonl`, 10 tasks each, in the `set_v2` format. Together with `../set_v2/` they give 20 tasks per
rung on 20 different topics (rung 4: 20 different determinant matrices). No v3 task repeats a task of `set_v1`
(`../rung_*.jsonl`) or `set_v2`, and no v3 topic occurs in `set_v2`.

| Rung | Topics |
|---|---|
| 1, different fields | accounting, astronomy, demography, epidemiology, geometry, logic, number theory, pharmacology, probability, statistics |
| 2, physics sections | AC circuits, capacitors, electromagnetic induction, gravitation, hydrostatics, kinetic theory, oscillations, photoelectric effect, polarisation, projectile motion |
| 3, linear algebra types | adjugate, Cholesky, Gram–Schmidt, inverse, leading principal minors, matrix norms, minimal polynomial, LU with partial pivoting, Smith normal form, vector algebra |
| 4, one template | 7×7 integer determinant (`la_det_e4`, n = 7, seeds 6000–6016; set_v2 uses n = 6) |

## Selection

Model: `deepseek-ai/DeepSeek-V4.1-Flash` via Together AI, temperature 1.0, `reasoning_effort` high,
`max_tokens` 40000, no system prompt. Every task was solved 5 times; it is kept if at least 4 runs are correct,
the mean length (`usage.completion_tokens`) is in 12000–26000 tokens and the coefficient of variation is at most
0.25. Two exceptions, named in the export command: `r2_kinetic_gas` (mean 26156 tokens) and `r3_smith` (CV 0.251).
The statistics of each task are in its `selection` field.

## Files

- Generators: `generators/r1_fields.py`, `r1_fields_b.py`, `r2_physics_c.py`, `r3_linalg_c.py` (new), plus the
  team's generators for geometry, number theory, logic and five linear-algebra types. Each generator recomputes
  its answer by an independent method; `generators/selftest.py` checks this.
- Task files sent to the model: `candidates/v41_*.jsonl` (built by `v41_items.py`).
- Runs: `embedding_eval/runs/ladder/deepseek-v4.1-flash/` (`calib/` probes, `main/` the final tasks,
  `spend.jsonl` the cost log; total USD 7.49).
- Rebuild the set from the runs:

```
uv run --with sympy --with numpy --with scipy --with mpmath --with reasoning-gym==0.1.19 \
    python bench/ladder/export_v3.py --allow r2_kinetic_gas --allow-cv r3_smith
```

Generators that were written or probed but not selected (too long, too short or unstable on this model):
maximum flow, great-circle navigation, humidity, Huffman codes, M/M/c queues, cubic crystals (rung 1);
3-D rotations, polynomial operator, permanent, affine maps, Sylvester equation (rung 3).
