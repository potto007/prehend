# Spec: Auto-chunk enforcement for oversized sub-calls (ADR-0010)

Status: APPROVED (design), pre-implementation
Date: 2026-06-22
Extends: ADR-0009 (sub-call input-size context guard). Companion ADRs: 0002, 0003, 0008.
Source of truth for TDD. Where this doc and code disagree, this doc wins until amended.

## Problem

ADR-0009 eliminated the context-overflow bug with a deterministic reject-with-hint
guard: an oversized sub-call prompt is not sent; instead the harness returns an
actionable string telling the model to chunk and map-reduce. That fixed correctness
(plain-multihop cold 40% -> warm 53.3%, +13.3pp, 0 overflow) but left a LATENCY tail:
the 12B orchestrator (`gemma-4-12b-it-sft-kb-v13-sft`) does NOT reliably honor the
advised chunk size on hard tasks (e.g. `multihop_002` made 2 giant chunks and timed
out at 660s). The fix is to move decomposition from ADVICE to MECHANISM: when a
sub-call carries oversized data, the harness itself splits -> maps -> reduces, so
latency/correctness no longer depend on the model chunking perfectly.

This is the option explicitly DEFERRED in ADR-0009 ("auto-chunk was rejected for v1 as
too invasive"). The evidence in this session (model ignores chunk advice on hard tasks
-> timeout tail, chunk-size tune b150371 inconclusive) justifies escalating now.

## Goals

1. When a sub-call carries oversized DATA, the harness transparently map-reduces it
   across the server's parallel slots, so a single hard task no longer serializes into
   1-2 giant prefills.
2. Zero regression: the change is purely ADDITIVE. A bare oversized `prompt` (no data
   channel) still hits ADR-0009 reject-with-hint and can never overflow.
3. The map-reduce engine is pure and unit-testable with no sockets.

## Non-goals

- Heuristically splitting an opaque `prompt` string (rejected: the harness should not
  guess what is instruction vs data). The model declares the data via `context=`.
- Chunk overlap (deferred to a future lever; see Open risks).
- Changing the depth/recursion model of RLM itself (ADR-0002) or the per-ask sub-call
  circuit breaker (ADR-0003). Both are inherited unchanged.

## Decisions (settled in brainstorming)

- **API**: add keyword-only `context=None` and `reduce=None` to all four sub-call
  primitives. `prompt` is the instruction (map step); `context` is the large data;
  `reduce` is the optional combine instruction (defaults to `prompt`).
- **Reduce strategy**: hierarchical tree-reduce, depth-bounded (`max_reduce_depth=3`);
  truncate + warn at the bound.
- **Budget**: inherited by delegating every batch to the EXISTING batched primitives,
  which already enforce ADR-0003. No new budget logic.
- **No chunk overlap** in v1.
- **Adoption is empirical**: if v13 never emits `context=`, reject-with-hint still
  prevents overflow. Whether v13 adopts `context=` is a validation question, not a
  correctness gate.

## API surface

All four primitives gain two keyword-only args. The existing `priority=None` positional
param on the `llm_*` primitives is PRESERVED (review finding R2-2: dropping it breaks
callers and the priority thread into `LMRequest`/`send_lm_request_batched`). `context`
and `reduce` are keyword-only, so all existing positional callers are unaffected.

```python
llm_query(prompt, model=None, priority=None, *, context=None, reduce=None) -> str
llm_query_batched(prompts, model=None, priority=None, *, context=None, reduce=None) -> list[str]
rlm_query(prompt, model=None, *, context=None, reduce=None) -> str
rlm_query_batched(prompts, model=None, *, context=None, reduce=None) -> list[str]
```

(`rlm_*` have no `priority` param today; not adding one.)

Batched semantics: `context`/`reduce` are scalars applied to EVERY prompt in the
batch (the common case is "run this same instruction over each of these prompts, each
with its own big context"). Per-prompt context is out of scope for v1; a caller needing
distinct contexts per prompt calls `llm_query`/`rlm_query` in a loop (or the model
composes its own). If `context` is a non-`str`, it is coerced via `str(context)` (mirrors
the existing `_subcall` coercion of `prompt`).

## Dispatch logic (per seam)

Let `fits(text)` mean `oversize_rejection(text, limit, model) is None` (ADR-0009
arithmetic; the hard ceiling, ~250674 chars @98304). Let `compose(instr, data,
label="Text")` = `f"{instr}\n\n{label}:\n{data}"`. Let `R = recommended_chunk_chars(limit,
model)` (the advisory size, ~88473 chars @98304).

`_fill_placeholders` runs on BOTH `prompt` and `context` (and on each batched prompt +
the scalar context) BEFORE compose/chunk (review finding R2-4: a `{var}` placeholder in
`context` must be filled before slicing, else chunks split an unfilled string).

For each primitive, on a single (prompt, context) pair:

1. `context is None` -> UNCHANGED ADR-0009 behavior. `prompt` is sent as-is; if `prompt`
   itself is oversized it gets reject-with-hint. (This path is the ONLY guard against a
   bare oversized prompt; it must remain.)
2. `context` given and `len(compose(prompt, context)) <= R` -> send the single composed
   prompt normally (one call). `reduce` is ignored (no reduction needed).
3. `context` given and `len(compose(prompt, context)) > R` -> route to the map-reduce
   engine (below). Return the engine's combined answer string.

The threshold in steps 2/3 is the RECOMMENDED size `R`, NOT the `fits` ceiling (review
finding R1-6, the most consequential latency fix): a context of ~200K chars *fits* the
98304-token window (`<= 250674`) but prefills slowly as one giant call -- this is exactly
`multihop_002`'s 660s timeout. Using `R` as the dispatch threshold sends that band
through map-reduce (several fast parallel chunks) instead of one slow prefill, which is
the whole point of this work. Inlining is reserved for genuinely small composed prompts
(`<= R`).

`subcall_context_limit is None` (guard disabled) -> `context` is still inlined via
`compose` (step 2 framing) but never triggers map-reduce; there is no limit, hence no
`R`, so behavior collapses to "inline and send" and an oversized `context` here CAN
overflow -- this is the caller's responsibility, consistent with ADR-0009's "no limit ->
no guard" (review finding R2-7). No current caller passes `context=`, so this is not a
regression.

## Map-reduce engine

New module `prehend/utils/mapreduce.py`. Pure: all LM I/O is injected via `run_batch`;
no token-accounting or socket dependency of its own.

```python
def map_reduce(
    prompt: str,
    context: str,
    *,
    run_batch: Callable[[list[str]], list[str]],
    fits: Callable[[str], bool],
    chunk_chars: int,            # data budget per chunk (envelope already subtracted)
    reduce_prompt: str | None = None,
    max_reduce_depth: int = 3,
    is_control: Callable[[str], bool] = _is_control,
    compose: Callable[[str, str, str], str] = _compose,
) -> MapReduceResult: ...
```

- `run_batch(prompts) -> responses`: injected, CONTEXT-FREE. The seam passes a PRIVATE
  send helper (`_send_batched` / `_rlm_send_batched`) that contains the existing
  send + ADR-0009 per-prompt guard + ADR-0003 circuit-breaker logic and NEVER inspects
  `context`. It is NOT the public `*_batched` primitive (review finding R1/R2-1,
  CRITICAL: if the engine called the public batched primitive, which now interprets
  `context=`, a mis-sized group could re-enter map-reduce -> infinite recursion). The
  engine always hands `run_batch` already-composed, no-`context` prompts. Responses are
  positional and 1:1 with inputs.
- `fits(text) -> bool`: injected ADR-0009 ceiling check (testable with a char-length
  stub). Used ONLY for greedy reduce-group packing.
- `chunk_chars`: the DATA budget per chunk = `R - len(compose_overhead)` computed by the
  seam so that `compose(prompt, chunk)` is guaranteed `<= R` (review findings R1-3/R1-5:
  the cut bound must leave room for the instruction + labels + newlines, not be the raw
  advisory size). `compose_overhead` for label `"Text"` = `len(prompt) + len("Text") + 4`.
  If `chunk_chars <= 0` (the instruction alone exceeds `R`), the seam does NOT map-reduce:
  it falls through to a bare reject-with-hint on `compose(prompt, context)` (degraded,
  documented; the model is told to shrink its instruction). The engine reuses this single
  `chunk_chars` data budget for map chunking, partial hard-cut, and the truncate cut;
  reduce-group packing is governed by `fits` (the ceiling), not `chunk_chars`.
- `reduce_prompt`: defaults to `prompt` when None. Note it is part of the sized envelope
  (review finding R2-8): a huge `reduce_prompt` shrinks the per-group data budget; the
  seam computes the reduce data budget from `reduce_prompt`, and the `chunk_chars << R`
  invariant below assumes `reduce_prompt << limit`.
- `is_control(text) -> bool`: True for a non-answer control string -- the ADR-0009 guard
  prefix (`"Sub-call input guard rejected this call:"`), the per-ask budget message, and
  any `"Error: "`-prefixed string. Default implementation checks these known prefixes;
  injectable for tests (review findings R1-8/R2-3, CRITICAL).
- `MapReduceResult`: `{answer: str, n_chunks: int, reduce_levels: int, truncated: bool,
  dropped: int, budget_exhausted: bool}`. The seam returns `.answer`; the rest is for
  logging/metadata and test assertions. `dropped` = count of control-string partials
  excluded; `budget_exhausted` = a map/reduce batch returned the budget message.

### Algorithm

1. **Split**: cut `context` into consecutive slices each `<= chunk_chars` characters (no
   overlap), `chunks = [context[i:i+chunk_chars] for i in range(0, max(len(context),1),
   chunk_chars)]`. Exactly `max(1, ceil(len(context)/chunk_chars))` chunks; an empty
   context yields one empty chunk. (Review finding R1-7: pin the slice generator; test
   `len == chunk_chars`, `chunk_chars+1`, `0`, `1`.)
2. **Map**: `partials = run_batch([compose(prompt, c, "Text") for c in chunks])`.
3. **Filter** (every level, after every `run_batch`): drop partials where
   `is_control(p)` is True; increment `dropped`. If a batch contained the budget message,
   set `budget_exhausted=True` and STOP issuing further map/reduce batches -- reduce only
   the real partials gathered so far (review findings R1-8/R2-3). If NO real partials
   remain after filtering, return the first control string verbatim as `.answer` (so the
   model sees the hint/error) with `reduce_levels=0`.
4. **Reduce loop** (`level = 0`):
   - If `len(partials) == 1`: return `partials[0]`, `reduce_levels = level`.
   - If `level >= max_reduce_depth`: TRUNCATE -- `joined = "\n\n".join(partials)`;
     `cut = joined[:chunk_chars - len(NOTE)]`; `final = compose(reduce_prompt, cut + NOTE,
     "Partial results")` where `NOTE = "\n\n[note: reduce truncated at max depth; some
     partial results omitted]"`; assert `fits(final)` (cutting to `chunk_chars` leaves the
     compose envelope headroom under the ceiling for any sane `reduce_prompt`; if it still
     fails, cut harder rather than emit an oversized final reduce -- review finding R1-4);
     run ONE final reduce via `run_batch([final])`; return its (filtered) answer with
     `truncated=True`, `reduce_levels = level + 1`.
   - Otherwise GROUP: first hard-cut each partial to `chunk_chars` (so no partial exceeds
     the data budget -- review finding R1-2: partials are model answers with no inherent
     size bound, and an unbounded reduce-answer partial is what made the old proof false).
     Then greedily pack consecutive partials into groups with a test-add-then-check
     invariant (review finding R1-10): add `p` to the current group iff
     `fits(compose(reduce_prompt, "\n\n".join(group + [p]), "Partial results"))`; else
     close the group and start a new one with `p`.
   - `partials = run_batch([compose(reduce_prompt, "\n\n".join(g), "Partial results")
     for g in groups])`; filter (step 3); `level += 1`; loop.

Termination (review finding R1-1, the original proof was unsound):

- **Unconditional:** `max_reduce_depth` is the hard backstop. The loop runs at most
  `max_reduce_depth` grouping passes, then exits via the single-shot truncated reduce.
  Termination does NOT rely on per-level progress.
- **Progress (efficiency) invariant:** because every partial is hard-cut to `chunk_chars`
  and `chunk_chars <= R`, while groups are packed against the `fits` ceiling (`~250674` =
  `~2.83 * R`), any two adjacent capped partials satisfy `2 * chunk_chars + envelope <
  ceiling`, so each group holds `>= 2` partials whenever `>= 2` remain. Thus
  `len(groups) <= ceil(len(partials)/2)` and the count strictly halves per level -- the
  bound is reached only for `> ~2.83^max_reduce_depth` chunks (`~22` chunks `~= 2M` chars
  at depth 3; the eval's ~317K contexts -> 4 chunks -> 2 levels, well clear). This
  invariant holds as long as `RECOMMENDED_CHUNK_FRAC` keeps `2*R + envelope <= ceiling`
  (true at 0.30; breaks only above ~0.42). If the frac were raised past that, progress
  degrades to the depth bound but termination still holds.

### Why delegate to the private send helpers

- Parallelism: `_rlm_send_batched` uses a thread pool bounded by
  `max_concurrent_subcalls`; `_send_batched` uses `send_lm_request_batched`. The map
  fan-out gets server-slot parallelism for free.
- Budget (ADR-0003): the send helpers increment `_subcall_count` and short-circuit on
  the per-ask cap, returning the budget message string for the overflow. The engine
  DETECTS that string (via `is_control`), sets `budget_exhausted`, and stops -- so the
  budget message never silently poisons a reduce (review findings R1-8/R2-3). A runaway
  map-reduce cannot blow the budget. No new accounting.
- Guard (ADR-0009): composed chunk/group prompts are sized to fit `R` (the seam subtracts
  the compose envelope), so the per-prompt guard inside the send helper never fires on
  them. Belt-and-suspenders: if a group were somehow mis-sized, the inner guard returns a
  hint string, which `is_control` catches and filters (degraded, not overflow, not
  poisoned).

## Seam wiring

`prehend/environments/local_repl.py`:
- Extract PRIVATE context-free send helpers (review finding R1/R2-1, CRITICAL re-entrancy
  fix):
  - `_send_batched(prompts, model=None, priority=None) -> list[str]`: the CURRENT body of
    `_llm_query_batched` (placeholder fill, per-prompt ADR-0009 guard, circuit-breaker,
    `send_lm_request_batched`, append to `_pending_llm_calls`). Never inspects `context`.
  - `_rlm_send_batched(prompts, model=None) -> list[str]`: the CURRENT body of
    `_rlm_query_batched` (thread pool over `subcall_fn`, append completions). Never
    inspects `context`.
- `_llm_query` / `_rlm_query` gain `context`/`reduce` and implement the 3-way dispatch.
  When `context` triggers map-reduce, inject `run_batch=self._send_batched` (resp.
  `self._rlm_send_batched`) -- NOT the public batched primitive -- and
  `fits=lambda t: oversize_rejection(t, limit=..., model=...) is None`. Compute
  `chunk_chars` = `R - compose_overhead(prompt, "Text")`; if `<= 0`, skip map-reduce and
  return reject-with-hint on the composed prompt. Return `result.answer`. On the
  map-reduce branch, do NOT append a synthetic completion to `_pending_llm_calls` -- the
  send helpers already appended each underlying call (review finding R2-5).
- `_llm_query_batched` / `_rlm_query_batched` gain `context`/`reduce`. `context is None`
  -> delegate straight to `_send_batched`/`_rlm_send_batched` (today's behavior). Else
  each prompt is independently run through the same single-pair dispatch helper as
  `_llm_query`/`_rlm_query` (inline-if-`<=R` / map-reduce-if-not), preserving order. The
  public batched primitive is therefore NEVER the engine's `run_batch`, which makes
  re-entrancy structurally impossible.
- Factor the single-pair dispatch into one shared helper (e.g. `_dispatch_with_context(
  prompt, context, reduce, *, run_batch, send_one)`) used by both the single and batched
  primitives so the logic exists in exactly one place.

`prehend/core/rlm.py`:
- `_subcall` is UNCHANGED (review finding R2-6: adding `context=`/`reduce=` there would be
  dead code -- the REPL composes per-chunk prompts and calls `subcall_fn(prompt, model)`
  positionally, so `_subcall` never receives a `context` kwarg). `_subcall` keeps its
  ADR-0009 reject-with-hint: if the REPL ever hands it an oversized composed prompt (e.g.
  the degraded `chunk_chars <= 0` path), it is still rejected rather than overflowing.
- The REPL's `_rlm_query`/`_rlm_send_batched` own ALL compose/chunk/reduce logic and only
  ever pass `_subcall` a single pre-composed prompt that fits.

Propagation: no new constructor params. `subcall_context_limit` already threads
Harness -> SRLM -> RLM -> LocalREPL (ADR-0009). `R = recommended_chunk_chars` is computed
at the seam from the already-threaded `subcall_context_limit` + `model_name`.

## Prompt teaching (`prehend/utils/prompts.py`)

Update the function-list entries for the four primitives to document `context=`/`reduce=`
as the PREFERRED way to query large text:

- Add to the `llm_query`/`rlm_query` descriptions: "To query LARGE text, pass it as
  `context=` and the harness will automatically chunk it, run your `prompt` over each
  chunk in parallel, and combine the results (pass `reduce=` to give a distinct combine
  instruction; it defaults to `prompt`). You no longer need to slice `context` by hand
  for the common case." Example:
  `llm_query("Which items does Dave own?", context=context)` and
  `llm_query("Extract every date Dave is mentioned", context=context, reduce="What is the earliest date?")`.
- KEEP the existing "sub-calls do NOT see your `context`" block and the manual
  chunk-and-map-reduce examples as the fallback story (still correct when the model
  passes a bare slice). Add one line: "Passing `context=` is the easy path; manual
  slicing still works."
- The `{subcall_char_budget}` field and `recommended_chunk_chars` wiring are unchanged.

## ADR-0010

`docs/decisions/0010-auto-chunk-enforcement-for-oversized-subcalls.md`: records the shift
from reject-only (ADR-0009) to auto-chunk for the `context=` path; the `context=`/`reduce=`
API; tree-reduce + depth bound; budget-by-delegation; no-overlap-v1; the additive/no-
regression property; references 0009/0003/0002. Update `docs/decisions/README.md` index.
ADR-0009 stays immutable; 0010 supersedes its "auto-chunk deferred" stance for this path
only.

## Test plan (TDD; author tests before code)

Engine (pure, fake `run_batch` recording prompts; `fits` = `len <= N` stub):
- `test_context_fits_single_chunk_one_map_no_reduce`: small context -> 1 chunk, 1 map
  call, 0 reduce levels, answer is the single partial.
- `test_oversized_context_splits_into_expected_chunk_count`: chunk count = ceil(len/chunk_chars).
- `test_map_runs_prompt_over_each_chunk_in_one_batch`: `run_batch` called once for map
  with N composed prompts, each containing `prompt` and a chunk.
- `test_reduce_combines_partials`: N>1 partials -> reduce level(s) produced; final answer
  comes from a reduce call whose input contains the partials.
- `test_reduce_uses_reduce_prompt_when_given_else_prompt`: composed reduce prompt carries
  `reduce` text when provided, else `prompt`.
- `test_tree_reduce_multiple_levels`: many partials that cannot fit one reduce group ->
  `reduce_levels >= 2`, each level's `run_batch` has >1 prompt then fewer.
- `test_max_reduce_depth_truncates_and_warns`: force depth bound -> `truncated is True`,
  final reduce input contains the truncation note, exactly one final reduce call at the
  bound.
- `test_single_partial_too_large_is_hardcut`: a partial longer than chunk_chars is cut to
  chunk_chars before composing into a reduce group.
- `test_empty_context_yields_one_empty_chunk`: no crash; one map call.
- `test_run_batch_order_preserved`: partials map positionally to chunks.
- `test_reduce_levels_value_per_exit_path`: single-chunk -> 0; one reduce -> 1; truncated
  exit -> `max_reduce_depth + 1` (review finding R1-9).
- `test_boundary_chunk_counts`: `len == chunk_chars` -> 1 chunk; `chunk_chars + 1` -> 2
  chunks (1-char tail); `len == 0` -> 1 empty chunk; `len == 1` -> 1 chunk (R1-7).
- `test_greedy_group_packs_two_capped_partials`: two `chunk_chars`-capped partials land
  in ONE group (the progress invariant), not two (R1-1/R1-2).
- `test_control_string_partial_is_filtered`: a map partial for which `is_control` is True
  is excluded from reduce; `dropped == 1`; the answer contains only real partials (R1-8).
- `test_all_partials_control_returns_first_verbatim`: if every partial is a control
  string, `.answer` is the first one and no reduce runs.
- `test_budget_message_stops_and_flags`: a batch containing the budget message sets
  `budget_exhausted`, stops further batches, reduces only real partials (R2-3).

Seam dispatch (LocalREPL with a fake LM handler / fake subcall_fn; assert no socket):
- `test_llm_query_context_small_inlined_single_call`: composed `<= R` -> one composed
  send, no map-reduce.
- `test_llm_query_midband_context_triggers_mapreduce`: composed in the `R..ceiling` band
  (e.g. ~200K chars) -> map-reduce, NOT a single inline send (review finding R1-6, the
  latency band).
- `test_llm_query_no_context_unchanged_reject_with_hint`: bare oversized prompt still
  returns the ADR-0009 hint (regression guard).
- `test_rlm_query_oversized_context_fans_out_to_children`: large context -> child RLM
  calls per chunk via subcall_fn.
- `test_batched_context_applies_to_every_prompt_order_preserved`.
- `test_batched_no_context_delegates_to_send_helper_unchanged`: `context is None` batched
  path is byte-for-byte today's behavior.
- `test_engine_run_batch_is_context_free_no_reentrancy`: the `run_batch` passed to the
  engine is `_send_batched`/`_rlm_send_batched` and never re-enters dispatch (R1/R2-1).
- `test_context_coerced_to_str`.
- `test_placeholder_filled_in_context_before_chunk`: a `{var}` in `context` is filled
  before slicing (review finding R2-4).
- `test_priority_param_preserved_and_forwarded`: `llm_query(..., priority=...)` still
  works and priority reaches the send (review finding R2-2).
- `test_no_synthetic_pending_call_on_mapreduce_branch`: `_pending_llm_calls` length equals
  the number of underlying sends, with no extra synthetic entry (R2-5).
- `test_guard_disabled_inlines_context_no_mapreduce`: `subcall_context_limit is None` ->
  composed and sent, never map-reduced.
- `test_huge_instruction_skips_mapreduce_rejects`: `chunk_chars <= 0` (instruction alone
  > R) -> no map-reduce; composed prompt returns reject-with-hint (R1-5).
- `test_subcall_unchanged_rejects_oversized_composed_prompt`: `_subcall` given an
  oversized pre-composed prompt still returns reject-with-hint (R2-6).

Prompt:
- `test_prompt_documents_context_arg_for_large_text`: prompt text mentions `context=`
  and `reduce=` as the large-text path.
- `test_prompt_keeps_manual_slicing_fallback`: the "sub-calls do NOT see your context"
  block is still present.

Regression: full suite (`uv run pytest`) -> 706+ green, 9 skipped unchanged.

## Validation (live, after green suite)

- Server-free smoke: construct LocalREPL with a fake handler; assert oversized `context=`
  map-reduces (combined answer) without hitting the socket.
- Timeout-tail subset (v13 :8080 ctx 98304, bge :8084, e4b :8083): run `multihop`
  `--max-tasks 3 --timeout 600 --subcall-context-limit 98304`. PASS = the heavy tasks
  (002/003/010) now COMPLETE under 600s; `grep -c 'exceeds the available context size'`
  stays flat (no new overflow).
- Fresh cold/warm on plain-multihop vs baseline (cold 40% / warm 53.3%). Watch that
  auto-chunk does NOT regress accuracy on tasks the model solved single-shot (partition
  risk).

## Open risks

- **Adoption**: v13 was trained on `llm_query(selected_docs)`; it may never emit
  `context=`. Mitigated: additive, no regression. A couple of trajectory probes during
  validation will show whether teaching `context=` helps or confuses it.
- **Partition validity**: splitting can sever a multi-hop link across a chunk boundary
  (MIT paper warns partition isn't always valid). v1 has no overlap; validation must
  watch accuracy, not just latency. Overlap is the documented future lever if accuracy
  regresses.
- **Reduce information loss**: tree-reduce over partial answers can drop a detail a
  single-shot read would keep. The depth bound + truncation note make loss visible in
  logs.

## Adversarial review findings incorporated (2 reviewers, 2026-06-22)

- **R1/R2-1 (CRITICAL, re-entrancy):** engine `run_batch` is a private context-free send
  helper (`_send_batched`/`_rlm_send_batched`), never the public `*_batched` primitive;
  re-entrancy is structurally impossible.
- **R1-1/R1-2 (CRITICAL/HIGH, termination):** original progress proof was false. Now:
  depth bound is the unconditional backstop; partials are hard-cut to `chunk_chars` so
  the `2*R+envelope < ceiling` invariant guarantees groups hold >= 2 (genuine progress).
- **R1-8/R2-3 (CRITICAL, control-string poisoning):** `is_control` filters guard/error/
  budget strings out of the reduce; budget message stops further batches and flags
  `budget_exhausted`; all-control map returns the first control string verbatim.
- **R2-2 (HIGH, backward-compat):** `priority=None` preserved on the `llm_*` primitives.
- **R1-6 (HIGH, latency):** inline-vs-mapreduce threshold is `R` (recommended), not the
  fits ceiling -- the ~88K..250K char band (the slow-prefill timeout case) now
  map-reduces instead of sending one giant call.
- **R1-3/R1-4/R1-5 (HIGH/MED, envelope):** cut bounds derive from the data budget
  (`R`/ceiling minus the compose envelope), not the raw advisory size; an instruction
  larger than `R` skips map-reduce and reject-with-hints.
- **R2-4 (MED):** `_fill_placeholders` runs on `context` too, before chunking.
- **R2-5 (MED):** map-reduce branch does not append a synthetic `_pending_llm_calls`
  entry (send helpers already appended each call).
- **R2-6 (MED):** `_subcall` is UNCHANGED -- the dead `context=`/`reduce=` params are
  dropped; the REPL composes before calling `subcall_fn` positionally.
- **R1-7/R1-9/R1-10 (MED/LOW):** slice generator pinned with boundary tests; `reduce_levels`
  pinned per exit path; greedy grouping uses test-add-then-check.
- **R2-7 (LOW):** guard-disabled + oversized `context` can overflow (caller's
  responsibility, no current caller passes `context=`).

## References

- ADR-0009: `docs/decisions/0009-subcall-input-context-guard.md`
- Plan/history: `docs/superpowers/plans/2026-06-22-subcall-context-limit-coherence.md`
- Guard helpers: `prehend/utils/subcall_guard.py` (`oversize_rejection`,
  `recommended_chunk_chars`, `safe_chunk_chars`)
- Seams: `prehend/environments/local_repl.py` (`_llm_query`, `_llm_query_batched`,
  `_rlm_query`, `_rlm_query_batched`); `prehend/core/rlm.py` (`_subcall`)
- Prompt: `prehend/utils/prompts.py`
- Papers: rlm-trainer `docs/recursive_language_models_mit_paper.md` (Appendix A
  summary-agent chunking; D.1), `docs/srlm_apple_paper.md` (s2.1, s3.5)
- Commits: prehend `ab59ed3`, `b150371`; rlm-trainer `306cdaf`
