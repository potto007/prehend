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

All four primitives gain two keyword-only args. Existing positional callers are
unaffected (fully backward compatible).

```python
llm_query(prompt, model=None, *, context=None, reduce=None) -> str
llm_query_batched(prompts, model=None, *, context=None, reduce=None) -> list[str]
rlm_query(prompt, model=None, *, context=None, reduce=None) -> str
rlm_query_batched(prompts, model=None, *, context=None, reduce=None) -> list[str]
```

Batched semantics: `context`/`reduce` are scalars applied to EVERY prompt in the
batch (the common case is "run this same instruction over each of these prompts, each
with its own big context"). Per-prompt context is out of scope for v1; a caller needing
distinct contexts per prompt calls `llm_query`/`rlm_query` in a loop (or the model
composes its own). If `context` is a non-`str`, it is coerced via `str(context)` (mirrors
the existing `_subcall` coercion of `prompt`).

## Dispatch logic (per seam)

Let `fits(text)` mean `oversize_rejection(text, limit, model) is None` (ADR-0009
arithmetic). Let `compose(instr, data, label="Text")` = `f"{instr}\n\n{label}:\n{data}"`.

For each primitive, on a single (prompt, context) pair:

1. `context is None` -> UNCHANGED ADR-0009 behavior. `prompt` is sent as-is; if `prompt`
   itself is oversized it gets reject-with-hint. (This path is the ONLY guard against a
   bare oversized prompt; it must remain.)
2. `context` given and `fits(compose(prompt, context))` -> send the single composed
   prompt normally (one call). `reduce` is ignored (no reduction needed).
3. `context` given and NOT `fits(compose(prompt, context))` -> route to the map-reduce
   engine (below). Return the engine's combined answer string.

`subcall_context_limit is None` (guard disabled) -> `context` is still inlined via
`compose` (step 2 framing) but never triggers map-reduce; there is no limit to compare
against, so behavior collapses to "inline and send". (Auto-chunk requires a known
limit, consistent with ADR-0009.)

## Map-reduce engine

New module `prehend/utils/mapreduce.py`. Pure: all LM I/O is injected via `run_batch`.

```python
def map_reduce(
    prompt: str,
    context: str,
    *,
    run_batch: Callable[[list[str]], list[str]],
    fits: Callable[[str], bool],
    chunk_chars: int,
    reduce_prompt: str | None = None,
    max_reduce_depth: int = 3,
    compose: Callable[[str, str, str], str] = _compose,
) -> MapReduceResult: ...
```

- `run_batch(prompts) -> responses`: injected. The seam passes the real batched
  primitive (`_llm_query_batched` or `_rlm_query_batched`). Each call already enforces
  the ADR-0003 budget and the ADR-0009 per-prompt guard. Responses are positional.
- `fits(text) -> bool`: injected ADR-0009 check (so the engine has no token-accounting
  dependency of its own; testable with a char-length stub).
- `chunk_chars`: target chunk size = `recommended_chunk_chars(limit, model)` (b150371,
  ~88K @98304). Used for BOTH map chunking and reduce-group packing.
- `reduce_prompt`: defaults to `prompt` when None.
- `MapReduceResult`: `{answer: str, n_chunks: int, reduce_levels: int, truncated: bool}`.
  The seam returns `.answer`; the rest is for logging/metadata and assertions in tests.

### Algorithm

1. **Split**: cut `context` into consecutive slices each <= `chunk_chars` characters
   (no overlap). Slice on raw character index (`context[i:i+chunk_chars]`). At least one
   chunk; an empty context yields one empty chunk.
2. **Map**: `partials = run_batch([compose(prompt, c, "Text") for c in chunks])`.
3. **Reduce loop** (`level = 0`):
   - If `len(partials) == 1`: return `partials[0]` with `reduce_levels=level`.
   - If `level >= max_reduce_depth`: TRUNCATE -- join all partials, hard-cut the join to
     fit one chunk (`joined[:chunk_chars]`), append `"\n\n[note: reduce truncated at "
     "max depth; some partial results omitted]"` to the join BEFORE composing, run ONE
     final reduce, return its answer with `truncated=True`.
   - Otherwise GROUP: greedily pack consecutive partials into groups so that
     `fits(compose(reduce_prompt, join(group), "Partial results"))` holds for each
     group. `join` = `"\n\n".join`. A single partial that alone does not fit is hard-cut
     to `chunk_chars` and placed in its own group (bounded; partials should be small).
   - `partials = run_batch([compose(reduce_prompt, join(g), "Partial results") for g in
     groups])`; `level += 1`; loop.

Termination: each reduce level strictly reduces `len(partials)` (groups < partials
whenever len > 1, because each group holds >= 1 and at least one group holds >= 2 once
partials exceed what one group can hold; a group can always hold >= 1, and if every
partial were forced into its own group the join of any two adjacent small partials would
fit, so grouping makes progress) OR `max_reduce_depth` forces the single-shot truncated
exit. The depth bound is the hard backstop regardless.

### Why delegate to the existing batched primitives

- Parallelism: `_rlm_query_batched` uses a thread pool bounded by
  `max_concurrent_subcalls`; `_llm_query_batched` uses `send_lm_request_batched`. The map
  fan-out gets server-slot parallelism for free.
- Budget (ADR-0003): the batched primitives increment `_subcall_count` and short-circuit
  on the per-ask cap. A runaway map-reduce cannot blow the budget. No new accounting.
- Guard (ADR-0009): composed chunk/group prompts are sized to `recommended_chunk_chars`
  (< the safe ceiling), so the per-prompt guard inside the batched path never fires on
  them. Belt-and-suspenders: even if a group were mis-sized, the inner guard would
  reject that one prompt with a hint string (degraded, not overflow).

## Seam wiring

`prehend/environments/local_repl.py`:
- `_llm_query` / `_rlm_query`: add `context`/`reduce`; implement the 3-way dispatch.
  When map-reducing, inject `run_batch=self._llm_query_batched` (resp.
  `self._rlm_query_batched`) and `fits=lambda t: oversize_rejection(t, limit=..., model=
  ...) is None`. Return `result.answer`.
- `_llm_query_batched` / `_rlm_query_batched`: add `context`/`reduce`. When `context`
  is given, each prompt is independently run through the same single-pair dispatch
  (inline-if-fits / map-reduce-if-not), preserving order. Reuse a shared helper so the
  single and batched paths share dispatch logic.
- Keep the existing no-context path EXACTLY as today (placeholder fill, guard,
  circuit-breaker, send).

`prehend/core/rlm.py`:
- `_subcall(prompt, model=None, *, context=None, reduce=None)`. The subcall callback is
  what `_rlm_query`/`_rlm_query_batched` invoke. When `context` is given and oversized,
  `_subcall` does NOT map-reduce internally (the REPL seam owns the map-reduce loop and
  the fan-out); instead the REPL composes per-chunk prompts and calls back through
  `subcall_fn` per chunk. So `_subcall` only needs to accept and inline a `context` that
  FITS (compose into the prompt before the existing guard), and otherwise its ADR-0009
  reject-with-hint remains. (Rationale: the fan-out and tree-reduce live in ONE place,
  the REPL primitives; `_subcall` stays a single-shot leaf/child dispatch.)
  - Concretely: `_subcall` composes `prompt = compose(prompt, context)` when `context`
    is provided, then runs its existing guard/verifier/depth logic on the composed
    prompt. The REPL's `_rlm_query` is responsible for NOT handing `_subcall` an
    oversized composed prompt (it map-reduces first and calls `_subcall` per chunk).

Propagation: no new constructor params. `subcall_context_limit` already threads
Harness -> SRLM -> RLM -> LocalREPL (ADR-0009). `recommended_chunk_chars` is computed at
the seam from the already-threaded `subcall_context_limit` + `model_name`.

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
  chunk_chars before composing.
- `test_empty_context_yields_one_empty_chunk`: no crash; one map call.
- `test_run_batch_order_preserved`: partials map positionally to chunks.

Seam dispatch (LocalREPL with a fake LM handler / fake subcall_fn; assert no socket):
- `test_llm_query_context_fits_inlined_single_call`: small context -> one composed send,
  no map-reduce.
- `test_llm_query_oversized_context_triggers_mapreduce`: large context -> multiple
  underlying batched sends, combined answer returned.
- `test_llm_query_no_context_unchanged_reject_with_hint`: bare oversized prompt still
  returns the ADR-0009 hint (regression guard).
- `test_rlm_query_oversized_context_fans_out_to_children`: large context -> child RLM
  calls per chunk via subcall_fn.
- `test_batched_context_applies_to_every_prompt_order_preserved`.
- `test_context_coerced_to_str`.
- `test_guard_disabled_inlines_context_no_mapreduce`: `subcall_context_limit is None` ->
  composed and sent, never map-reduced.
- `test_subcall_inlines_fitting_context_then_guards`: `_subcall` composes a fitting
  context; an oversized composed prompt still returns reject-with-hint.

Prompt:
- `test_prompt_documents_context_arg_for_large_text`: prompt text mentions `context=`
  and `reduce=` as the large-text path.
- `test_prompt_keeps_manual_slicing_fallback`: the "sub-calls do NOT see your context"
  block is still present.

Regression: full suite (`uv run pytest`) -> 706+ green, 9 skipped unchanged.

## Validation (live, after green suite)

- Server-free smoke: construct LocalREPL with a fake handler; assert oversized `context=`
  map-reduces (combined answer) without hitting the socket.
- Timeout-tail subset (v13 :8080 ctx 98304, bge :8081, e4b :8083): run `multihop`
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
