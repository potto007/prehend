---
status: "accepted"
date: "2026-06-24"
deciders: "potto"
consulted: "debugging session (multihop chaining root-cause + live A/B)"
---

# Query-independent extraction MAP for multihop chaining

## Context and Problem Statement

The map-reduce seam ([ADR-0010](0010-auto-chunk-enforcement-for-oversized-subcalls.md), driven through
the served solver [ADR-0016](0016-sglang-as-served-solver.md)) splits an oversized
`context=` blob into chunks, runs a per-chunk MAP, and tree-REDUCEs the partials.
The MAP instruction was the USER QUERY (plus the no-info sentinel directive,
[`_MAP_SENTINEL_DIRECTIVE`](../../prehend/utils/mapreduce.py)): each chunk was
asked the user's question and returned its answer or a droppable `NO_RELEVANT_INFO`
sentinel.

On a MULTIHOP task this cannot chain hops. Example (multihop subset, "What does
Carol own?" -> Carol lives in Denver -> the person in Denver owns a golden key):

- The chunk holding the INTERMEDIATE hop ("Carol moved to Denver") names Carol,
  so the per-query MAP keeps it.
- The chunk holding the TERMINAL hop ("the person who lives in Denver owns a
  golden key") does NOT name Carol, so the per-query MAP judges it irrelevant to
  "what does Carol own?" and drops it as a sentinel.

Result: only the intermediate hop survives the MAP (`reduce_levels=0`, `dropped=15`
of 16 chunks), and the seam returns "Carol moved to Denver" - the wrong answer.
Re-tuning the sentinel directive to preserve background facts was tried twice and
did not fix it: the problem is not the directive wording, it is that the query is
used as the per-chunk filter at all, which structurally discards any hop that does
not mention the queried entity by name.

Measured live (2026-06-24, int4-ct v13 on `:8080`, direct `map_reduce` probe over
the 5-task multihop subset): legacy per-query MAP scores **1/5** (only the task
whose terminal chunk happened to also name the entity passed - by luck, per the
prior handoff).

## Decision

**In the map-reduce seam, run the MAP step QUERY-INDEPENDENTLY: extract every fact
about every named entity in the chunk, not the user query.** The user query then
drives only the REDUCE, where the now-complete set of facts can be CHAINED into an
answer.

- New `_EXTRACTION_MAP_INSTRUCTION` in `prehend/utils/mapreduce.py` ("list every
  fact about any named person/place/org/thing ... include background relationships
  ... reply `NO_RELEVANT_INFO` only if the text states no fact about any named
  entity"). It carries the sentinel clause itself, so entity-free filler chunks
  still drop out and cannot dilute the reduce.
- `map_reduce(..., extraction_map: bool = False)` selects it; the model's `prompt`
  becomes the REDUCE instruction (the existing `reduce_prompt` default).
- The seam (`local_repl.py::_dispatch_with_context`) sets `extraction_map=True`
  via the module constant `_SEAM_EXTRACTION_MAP` (one-line revert). Chunk sizing
  now subtracts the ACTUAL MAP instruction envelope (the fixed extraction
  instruction), and the map-reduce skip-guard also checks the REDUCE prompt room,
  since the user prompt now drives the reduce rather than the map.

Live A/B with the fix, same probe, same model: extraction MAP scores **5/5**
(legacy 1/5). The chaining is genuine, not substring luck: e.g. multihop_056
returns an explicit derivation ("1. Carol moved to Denver. 2. The person who lives
in Denver owns a jade ring. 3. Therefore, Carol owns a jade ring."), and a
Denver/silver distractor injected next to the Chicago/golden chain is correctly
rejected by the reduce.

## Consequences

- **Good:** multihop subset 1/5 -> 5/5 with no latency cost (same chunk count,
  same number of sub-calls, ~26s either way) and no regression on the single task
  legacy already passed. The fix is structural, not a prompt nudge, so it does not
  depend on directive wording the model can ignore.
- **Trade-off:** the MAP now extracts ALL entity facts, not just query-relevant
  ones, so on DENSE single-hop contexts partials are larger (more reduce tokens,
  possibly more reduce levels). The seam only fires for OVERSIZED `context=`
  sub-calls (deliberate large-blob retrieval), where extracting all facts is the
  right behavior; sparse multihop contexts (the failure case) stay compact because
  most chunks are filler -> sentinel. Default-on is justified by the evidence; a
  dense-context regression would show up as reduce bloat and is one-line revertible
  via `_SEAM_EXTRACTION_MAP = False`.
- **Validation scope:** validated by direct `map_reduce` probe (isolates the seam
  from the orchestrator loop). Full end-to-end GATE #2 through the orchestrator is
  still pending and is entangled with the separately-open orchestrator re-scan and
  harness timeout-leak issues; this ADR is accepted on the isolated seam evidence.
- Supersedes the prior attempts to fix chaining by tuning `_MAP_SENTINEL_DIRECTIVE`
  (kept only for the legacy `extraction_map=False` path).
