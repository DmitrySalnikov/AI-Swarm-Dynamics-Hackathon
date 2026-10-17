# Uniform windows and gradual reporting study

Approved scope: reprocess existing ladder runs, five fresh rung-1 distribution-only repeats, and one rung-1 gradual-reporting pilot. Work allocation: two to three hours. No new full factorial study or classifier training.

## Questions

1. Does a visible arc survive when both conditions use the same full-window rule, instead of whole-turn chat embeddings versus fixed-size solving windows?
2. How much coverage is lost because agents produce different amounts of text?
3. Do trimming and independent same-task supplementation change coverage and geometry? These are diagnostic transformations, not additional evidence of natural cooperation.
4. Can a completion-driven scheduler publish early results while other agents are still solving?
5. Do extreme distribution lengths repeat across independent generations?

## Measurement contract

- Keep every author's generated reasoning and visible response in that author's stream. Do not embed received prompts, inboxes, or replayed history as new output.
- Use windows of 400, 2048, and 4096 logical tokenizer tokens, stride W/2. Retain only full windows. No held vectors, repeated tails, progress interpolation, or synthetic private-chat endpoints.
- Anchor the grid at private token zero. Extend the same grid to the union of observed horizons, allowing either condition to be absent. This is a conditional token clock, not wall time or evidence of simultaneous production.
- Report active author count at every grid point. Show geometry only for at least four authors. Show adjacent-window drift only over authors present in both windows, with the matched count in CSV.
- R and mean pairwise cosine are algebraically dependent at fixed N. They are not two independent confirmations.
- Effective dimension depends on N and is bounded by N-1. Interpret changing-N regions as coverage diagnostics. Full-ten-author paired summaries are recorded separately.
- Preserve unsuccessful provider fragments and explicitly log retries. Never filter on answer correctness.

## Offline comparisons

1. **Continuous:** each author's own output is concatenated across phases. Windows can cross real phase boundaries. Barrier shading is only a reference and does not describe every author's actual boundary.
2. **Barrier:** phases begin at common offsets; each phase occupies the longest observed author length. Shorter streams have real gaps before the next phase.
3. **Trim:** retain only the shortest author length in each phase. The next phase still begins after the original longest span. Discarded text is excluded from embeddings, not deleted from logs.
4. **Supplement:** on rung 1, extend solving streams toward the longest original solve using independent solutions of exactly the same task prompt from historical controls. Each donor is used at most once per recipient. Donor provenance and hashes are recorded. This is a synthetic ablation and can change the semantic distribution.
5. **Chat supplement:** separately test the literal proposal to append independent same-task solutions to short negotiation and exchange streams. This deliberately mixes solution text into a chat representation and is labeled synthetic throughout. It is never presented as an observed message or real communication. Different donors are used for the same author's two phases. Preserve the natural-output baseline for comparison.

All four ladder rungs receive continuous, barrier, and trim checks at W=2048. Rung 1 receives all widths and both supplementation diagnostics. One existing pair per rung supports descriptive findings, not statistical detection claims.

## Live comparisons

- Distribution repeats: seeds 501–505, same rung-1 tasks, unchanged v2 distribution prompts. Stop after ten unique claims. Each run has a RUB25 admission cap.
- Gradual pilot: seed 601, ten concurrent solves after assignment. Apply each completed receipt immediately and dispatch its author's report without waiting for every solve. Later report requests include chat published before dispatch. Requests already in flight remain frozen. After all ten reports, all agents receive the identical result packet and commit it. RUB75 admission cap.
- Initial configured generation caps summed to RUB200. Distribution seed502 needed a logged admission revision from RUB25 to RUB35 after a missing usage receipt. The first gradual attempt (601) stopped because temporary outstanding reservations were incorrectly treated as a permanent budget stop; its observed output and cost remain in the ledger. A corrected scheduler waits for outstanding receipts before declaring a budget stop. Recovery pilot602 imports only the completed distribution and starts a separately seeded solving replicate with a RUB125 admission cap. Imported receipts are counted once in total spending. Actual usage and estimated costs are reported separately; admission caps are not spending totals.
- The gradual pilot changes reporting timing. It does not inject peer messages into ongoing private reasoning or establish peer-assisted problem solving.

## Verification and outputs

Test full-window extraction, real-boundary crossing, missing-gap exclusion, distribution termination, early-report overlap, and immutable in-flight contexts. Replay live request histories and delivery/application order. Verify exports against observed provider output.

Local outputs live under `artifacts/pipeline-windows/`: raw journals, CSVs, method/provenance JSON, repeated-length plots, event timeline, and a standalone English HTML report. Preserve prior results. Do not commit or publish without authorization.
