# Explicit distribution actions

The corrected step-specific prompts are used automatically. There is no runtime
prompt selection. Historical prompt versions are preserved in Git; use their
original checkout to replay or resume old runs. Existing run manifests prevent
silently resuming a historical run with changed prompts.

During distribution, every request asks for exactly one action:

- Before an accepted broadcast: return one `send` preference message.
- After an accepted broadcast: return one `claim` for an available task.

The instruction is selected from the runner's accepted `negotiated` state.
Rejected actions do not advance this state. The system overview says the
coordinator requests these actions in separate turns. Full task statements,
the emphatic prohibition on solving during distribution, task assignment rules,
solving instructions, the shared-result barrier and final commits are retained.
The private control uses the same prompts as `simple`.

This addresses observed deliberation about combining two actions into one JSON
object. It does not impose a reasoning cap, change model settings, hide task
content or enable communication during an in-flight solve.

## Distribution-only pilot

Run `scripts/run_step_prompt_pilot.py` with local credentials loaded by the
environment. It uses the tasks, seeds and generation settings from the uploaded
`pipeline-ladder-v2-simple` manifests. Four independent levels run concurrently;
each level retains serial negotiation and stops before solving. The defaults
allow at most 25 RUB per level and 40 calls. Source runs are never overwritten.

Outputs are saved to `artifacts/pipeline-step-prompts`. Inspect actual reasoning,
accepted actions, errors and per-agent counts; do not infer success solely from
shorter output. One group per level is a pilot, not a reproducibility study.

Offline checks cover accepted-action transitions, rejection recovery, full task
visibility, complete run barriers and identical private-control instructions.

## Pilot outcome

All four ten-agent groups completed distribution. Raw requests and events confirm
ten accepted preference broadcasts and ten accepted claims per group, unique task
assignments and no solution submissions. Full task text was retained. Token counts
include observed interrupted fragments and use the same saved tokenizer as the
uploaded comparison.

| Level | Previous simple maximum | Explicit-step maximum | New median |
| --- | ---: | ---: | ---: |
| 1 | 1,639 | 523 | 364.5 |
| 2 | 4,552 | 431 | 344 |
| 3 | 4,225 | 729 | 376.5 |
| 4 | 2,898 | 492 | 256.5 |

Total distribution reasoning decreased from 50,701 to 13,536 tokens (73.3%).
Within-level maximum/minimum ratios are 2.48–2.81, previously 7.77–17.15.
The longest new author deliberated about task availability and other agents'
preferences rather than calculating a solution. Message length also changed:
published preferences now contain 10–26 tokens, previously 7–16.

There were no action-format rejections. Two provider transport interruptions
were retained and retried, resulting in 82 calls for 80 accepted actions.
Level 2 stopped temporarily at a conservative budget admission reservation;
unused allowance from completed groups was reallocated and recorded without
changing scientific settings. Total ledger spending is 53.32 RUB, comprising
43.16 RUB in provider-backed usage and 10.16 RUB in unreceipted estimates.
The total pilot ceiling remained 100 RUB.

This is one group per level and style; it supports the operational correction,
not a claim that future length outliers are impossible. No solving, embedding,
geometry update, commit or push was performed for this pilot.

## Three additional repeats per level

The same distribution-only protocol was repeated with seed offsets 1000, 2000
and 3000, retaining model settings and full tasks. Twelve groups ran concurrently;
all completed. Together with the first pilot, this provides four groups and 40
authors per level, 160 authors overall. Each group has ten accepted broadcasts
and ten accepted unique claims, with no solving calls.

Across all four groups, per-agent reasoning spans are 151–631, 137–677, 221–729
and 138–492 tokens for levels 1–4. Group medians range from 231 to 425.5 tokens.
The previous 4–5k reasoning outliers did not recur in this sample. This does not
guarantee the absence of rare future outliers.

Published preference messages span 9–26 tokens. Their lengths are compact within
groups but vary between groups: agents often use similar wording to earlier
messages, so a longer shared phrasing can shift a whole group's token count.
These are message lengths, not evidence of more interaction.

The twelve additional groups incurred ten transport interruptions and one
empty-content response marked `stop` with the JSON in its reasoning field.
All recovery output remains included. There were no observed multi-action
wrappers among these failures. One group required a recorded financial-admission
budget reallocation; the total ceiling remained 420 RUB. New repeat spending was
187.54 RUB: 132.50 provider-backed plus 55.04 in unreceipted estimates.

Saved artifacts: `artifacts/pipeline-step-repeats/author-lengths.csv`,
`summary.json` and `distribution-repeats.png`. The plot shows individual authors
and group medians without uncertainty bands or independence claims about authors
that share an inbox. The existing HTML report was not modified.
