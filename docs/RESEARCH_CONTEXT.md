# Research context

Recorded on 12 October 2026 from the user's supplied conversation, three plots,
and the linked research plan. These sources are reference material. The user
subsequently approved planning and implementation of a stateful agent pipeline.

## Sources

- Repository: https://github.com/DmitrySalnikov/AI-Swarm-Dynamics-Hackathon
- Slides: https://docs.google.com/presentation/d/1k9MR5GTWHC71ojWzF12GmQOjPWZ87HOo1_QvFY9fjaE/edit
- Research plan: https://docs.google.com/document/d/1QnsFs9T5jZip1CD4DlfS0O9ihRH9y92-BaYHS3t5qQg/edit
- Events: https://aibelab.ru/category/7/
- Task-collection suggestion linked by the plan: https://claude.ai/share/56bfba46-a372-4e6b-8d2a-ac8b68131053

The research plan was read through its public text export. The slide deck and
the linked task-collection suggestion have not been reviewed during setup.
The plan is mutable; the notes below are a paraphrase of the retrieved version.

## Objective and supplied observations

Detect cooperation among agents when their communication channel is unobserved,
using temporal changes in the geometry of embeddings of generated text.
The supplied plots compare ten-agent swarms against independent agents with
fixed and shuffled task orders on set_4, over the first 10,000 generated tokens.
They show mean-vector length, spread perpendicular to its direction, and
effective dimension, averaged over ten groups with one-standard-deviation bands.

The supplied 50/50 mixed-population curve lies between the swarm and shuffled
independent curves. The 400-token windows with stride 200 show less separation;
4096-token windows with stride 2048 show more separation in the solving phase.
Larger windows can blur short communication episodes. These are preliminary
observations from supplied figures, not a replicated detection result. Swarm
turns and independent-agent windows have approximate token-axis alignment.
The first figure does not state its window size in the title.

Copies of the supplied figures are preserved in `docs/reference/`:
`set4-mixed-population.png`, `set4-window400-stride200.png`, and
`set4-window4096-stride2048.png`.

## Proposed next work from the plan

1. Collect four sets of ten long tasks: unrelated domains, different subfields
   of a broad domain, one field, and the same task with different parameters.
   Target roughly 20,000 solution tokens where practical. Assess each candidate
   with five runs for length variability; reject tasks with more than one failure.
   A numerical threshold for acceptable length variability remains unspecified.
2. Preserve context across task assignment, negotiation, solving, and result
   exchange. Compare presenting full task statements versus task identifiers at
   the beginning; consider repeating statements at solve time.
3. Build context-preserving sequential controls: tasks issued one at a time,
   or all tasks issued together with solutions requested individually.
4. Have researchers review prompts and read complete pilot transcripts, then
   repeat the pilot with a second model before scaling.
5. Compare fixed order, random order, non-overlapping task orders without
   discussion, and cooperation. Begin with one generation model and one
   embedding model before scaling across two of each, four sets, and ten repeats.
6. Plot the results and prepare a publication after validation.

The plan's total of 400 conflicts with its listed factor product:
2 generation models × 2 embedding models × 4 sets × 4 conditions × 10 repeats
= 640 analysis combinations. With shared generation across embedding models,
that would be 320 generation combinations. The unit of a combination and actual
API call count need clarification before estimating cost.

No new experiments, paid generation, or embedding requests were performed as
part of setup. The next implementation task and provider/model choices remain
to be assigned by the user.

## Existing follow-up work found in Git

The fetched `origin/ladder` branch (commit `17de56b`) already adds four ten-task
sets, task generators, selection/calibration code, and recorded runs. Its
`bench/ladder/REPORT_ladder.md` reports task lengths, variation, correctness,
and the generation conditions. This is existing work to review before collecting
another ladder. The report's results have not been independently verified here.
It uses Cloud.ru DeepSeek V4 Flash rather than the main branch's Together 0731
baseline; comparisons should preserve that distinction.

The working branch `research/setup` was initially based on `main` at `1638e8d`
and has now advanced to `ladder` at `17de56b` for pipeline development. Inspect
the task-selection report:

```powershell
git show origin/ladder:bench/ladder/REPORT_ladder.md
```

The user approved the stateful-runner plan and supplied Cloud.ru credentials.
The credentials are stored only in ignored local configuration. The runner's
contract and remaining live-validation milestones are in `docs/PIPELINE.md`.
