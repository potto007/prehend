# Plan: Sub-call context-limit coherence (by-reference realignment)

Status: IN PROGRESS (autonomous overnight run, 2026-06-22)
Owner: Claude (autonomous), for Paul Otto

## Problem (diagnosed + doc-audited)
Harness is meant to be context-by-reference: large context lives in a `context` REPL
variable; the model slices/queries it via llm_query/rlm_query. On large plain-multihop
tasks the model inlines the ENTIRE ~317,768-char (~150K-token) context into one
`llm_query(f"...Context:\n{context}")` -> request ~150,261 tokens vs server window 98,304
-> 400 "exceeds available context size" -> spin -> 660s hard-kill timeout. Cold hit it too
(base path). Three architecture-audited DEVIATIONS from the documented design; this fix
realigns. New ADR-0009 required (first INPUT-axis guard; companions ADR-0002/0003/0008).

## What the specs/papers say (load-bearing, cited)
- RLM/SRLM premise N>>L: never feed full context to the net; offload to REPL var, decompose
  then query. A sub-call larger than the sub-model window defeats the premise.
  (recursive_language_models_mit_paper.md s1/s3.1; srlm_apple_paper.md s2.1)
- Chunking is the MODEL's job, taught via the orchestrator prompt. Papers favor AUTO-CHUNK
  or an ACTIONABLE reject-with-hint; a bare hard reject can derail trajectories.
- recursive-rlm-refactor-design/plan: "llm_query for SHORT text only", "NEVER pass large
  text chunks to llm_query()", "large chunks -> rlm_query()", "divide into chunks
  (e.g. 50000 chars each) and delegate each to rlm_query()". The CURRENT prompts.py
  contradicts this with "sub-LLM can handle around 500K chars / don't be afraid to put a lot
  of context". => prompt fix = realign to the refactor spec, not just edit a number.
- strategy-verifier-design: reject -> error-string "Strategy verifier rejected this call:
  <reason>" -> REPL -> orchestrator adapts. Verifier rejects whole-TASK rlm_query
  (containment/shingle) and EXEMPTS llm_query (output-token-capped). Our guard is a NEW
  input-SIZE RuleVerifier that MUST cover llm_query (break the exemption). Out-of-scope
  "future layers" (fan-out/size rules) explicitly leave this seam open.
- harness-api-design: Tier-B = explicit args, NO env var ("no env hack"). Runtime carries
  ctx but it is currently UNUSED. Probe is router-fragile (/props n_ctx 0). Ambiguous probe
  must fall back safely, never hard-fail. ctx-derived routing was deferred YAGNI -> this fills it.
- v13-sft trains delegation via llm_query over selected docs; if a selected doc is large the
  v13 model reproduces the overflow through llm_query. Guard sits in this gap.

## Live env facts
- v13 router :8080 ctx 98304; /props returns n_ctx=0 (probe -> None). So detect_runtime
  -> None -> runtime.ctx unavailable. EFFECTIVE limit must come from the explicit param
  (driver passes 98304) or get_context_limit fallback. The /v1/models meta DID report
  n_ctx=98304 (probe could be improved to read it; nice-to-have).
- bge-m3 :8081, e4b distiller :8083 up. Eval driver: rlm-trainer scripts/memory_cold_warm.py
  -> benchmark.run_benchmark. rlm-trainer .venv is runtime; prehend editable-installed.
- Durable run state ~/eval-runs/mem-v13-plain-multihop-e4b/ (cold 15/15; bank had bad entry
  exp_840d65c8 advising single-shot whole-context -> purge/regenerate).

## DESIGN (final)
### Unit A - token accounting (prehend/utils/token_utils.py)  [no deps; do FIRST]
- Add gemma keys to MODEL_CONTEXT_LIMITS: "gemma-4" and "gemma" -> 262144 (true trained
  window; honest model property). Fixes the silent 128000 default.
- count_tokens non-tiktoken path: keep char/4 estimate BUT apply a safety inflation so we do
  not UNDERcount gemma (gemma tokenizes denser than cl100k for structured text). Add
  `count_tokens_estimate(text, model)` helper returning a conservative token estimate; or add
  a margin constant. Tests assert gemma estimate >= char/4 and within sane bounds.
- New helper `resolve_subcall_limit(model, *, explicit=None, runtime_ctx=None) -> int`:
  first non-None of explicit, runtime_ctx, get_context_limit(model). Pure, unit-tested.

### Unit B - subcall_context_limit plumbing (Defaults + Harness + SRLM/RLM + LocalREPL)
- Add `subcall_context_limit: int | None = None` to harness.Defaults (Tier-A field) AND as a
  Harness.__init__ param (param wins over Defaults). NO env var in core (spec).
- Harness resolves effective limit = resolve_subcall_limit(model, explicit=param-or-default,
  runtime_ctx=self.runtime.ctx). Pass into srlm_kwargs -> RLM, and into environment_kwargs so
  LocalREPL can guard llm_query. Also available to RLM._subcall and to the prompt build.
- RLM.__init__ gains `subcall_context_limit: int | None`; threads to: (1) environment_kwargs
  for spawned LocalREPL, (2) _subcall guard, (3) build_rlm_system_prompt capacity wording.
- Safe fallback: if everything None, limit = get_context_limit(model). Never raise.

### Unit C - reject-with-hint input guard (the realignment)
- Shared pure helper (token_utils or a new prehend/utils/subcall_guard.py):
  `oversize_rejection(prompt, *, limit, model, margin_frac=0.15) -> str | None`.
  Returns None if count_tokens(prompt,model) <= floor(limit*(1-margin_frac)); else an
  ACTIONABLE rejection string naming the limit and instructing: split context into chunks of
  <= K chars and map-reduce via rlm_query_batched (K derived from limit*chars_per_token*safety).
  Reuse the verifier rejection-string convention so the orchestrator's existing adapt path
  handles it.
- Wire at BOTH seams:
  - local_repl._llm_query and _llm_query_batched: before send_lm_request, if oversize_rejection
    -> return the hint string instead of sending (covers llm_query - breaks verifier exemption).
  - rlm._subcall: before child/leaf send, if oversize_rejection -> return RLMChatCompletion with
    the hint as response (covers rlm_query leaf + child first-prompt).
- Margin covers the WHOLE request envelope (system+user wrapper+context) + decode headroom -
  use margin_frac generous enough (~15-20%) to absorb prompt envelope + tokenizer skew.
- Deterministic guard: does NOT fail open like the LM verifier; it is arithmetic.

### Unit D - prompt realignment (prehend/utils/prompts.py + build_rlm_system_prompt)
- Remove "sub-LLM can handle around 500K chars" (line ~12) and the "they can fit around 500K
  characters ... don't be afraid to put a lot of context" paragraph (line ~41).
- Replace with refactor-spec-aligned guidance, parameterized by the resolved limit:
  * llm_query is for SHORT text/extraction; do NOT pass large chunks to it.
  * For large context, chunk (~{chunk_chars} chars) and map-reduce via rlm_query_batched.
  * State the sub-LLM window as ~{subcall_char_budget} chars (computed from limit, with margin).
- build_rlm_system_prompt gains a `subcall_char_budget`/`chunk_chars` param; RLM passes the
  resolved value. Add a `{subcall_char_budget}` format field; keep a safe default if unset.
- Tests: no literal "500K"; capacity string reflects the resolved ctx; chunk guidance present.

### Unit E - ADR-0009 + memory hygiene
- docs/decisions/0009-*.md: input-axis sub-call context guard; refs 0002/0003/0008; records
  the llm_query-exemption break and the env-in-driver-only decision; consequences note the
  bad bank entry.
- Memory: purge exp_840d65c8 (and any single-shot-whole-context entries); regenerate bank in
  the cold phase after the fix.

## Driver wiring (rlm-trainer, kept out of prehend core)
- benchmark.run_benchmark + memory_cold_warm.py: accept/pass subcall_context_limit; default
  from a CLI flag/env (e.g. PREHEND_SUBCALL_CONTEXT_LIMIT) -> pass as the Harness param.
  Eval passes 98304 (the v13 server ctx). This is the operative correctness path (probe=None).

## Validation
- TDD each unit (test names from spec agents below). Full suite green (uv run pytest).
- Limited benchmark subset (a few plain-multihop tasks) -> confirm guard fires, model chunks,
  no overflow, tasks complete under timeout.
- Fresh-bank cold then warm on plain-multihop subset; compare. Regenerate bank.

## Test cases to author (from spec agents)
Guard: test_llm_query_input_over_ctx_limit_is_rejected_with_hint;
  test_subcall_guard_covers_llm_query_not_just_rlm_query; test_input_under_limit_passes_through;
  test_rejection_string_is_actionable_names_limit_and_chunking; test_token_count_margin_applied;
  test_rlm_query_leaf_oversize_rejected.
Limit plumbing: test_subcall_context_limit_defaults_to_resolved; test_explicit_param_overrides;
  test_falls_back_to_get_context_limit_when_ctx_unknown; test_runtime_ctx_threaded_into_limit.
token_utils: test_get_context_limit_gemma_not_default_128k; test_count_tokens_gemma_not_undercount;
  test_resolve_subcall_limit_precedence.
Prompt: test_no_hardcoded_500k_in_prompt; test_capacity_reflects_resolved_ctx;
  test_prompt_warns_llm_query_short_only_large_to_rlm_query.
Regression: test_max_output_chars_default_500; test_subcall_verifier_default_none_unchanged.

## Progress checklist
- [x] Spec-reader agents (2) -> invariants
- [x] Unit A: token accounting + tests (24 new tests green; gemma=262144; CONSERVATIVE_CHARS_PER_TOKEN=3.0; non-OpenAI models diverted off cl100k; resolve_subcall_limit/oversize_rejection/safe_chunk_chars)
- [~] Unit B+C+D: integration pass (subagent a834d45694832d01e RUNNING - threads limit, guards both seams, realigns prompt)
- [x] Unit E: ADR-0009 written + indexed (also added missing 0008 index row). Memory hygiene = fresh bank (don't reuse contaminated exp_840d65c8)
- [x] Driver wiring: benchmark.py + memory_cold_warm.py thread subcall_context_limit (CLI --subcall-context-limit + PREHEND_SUBCALL_CONTEXT_LIMIT env at DRIVER layer; both compile). Eval must pass 98304.
- [x] Full test suite green: 706 passed, 9 skipped (independently re-run). Guard verified server-free (returns rlm_query_batched hint without touching socket). Driver imports + threads 98304.
- [~] Limited benchmark subset validation - IN PROGRESS, see findings below
- [ ] Cold/warm run + compare (fresh bank) -- script staged at
      ~/eval-runs/mem-v13-plain-multihop-fix-2026-06-22/run-cold-warm.sh (cold->warm->compare,
      --subcall-context-limit 98304). Launch after subset validates.
- [ ] Morning report

## VALIDATION FINDINGS (live)
- v1 (3 plain-multihop tasks, off-mode, 300s, fix active):
  * OVERFLOW ELIMINATED: 0 new "exceeds context size" errors (count stable at 103). The core
    bug is FIXED - no more 150K-token requests, no spinning on rejected context.
  * BUT model OVER-CORRECTED: trajectory analysis (multihop_001/002) showed it calls
    rlm_query("find Dave...") BARE - no context slice passed - so the child gets nothing and
    returns "no information found"; model loops on reworded queries, emits no-op iterations,
    times out (progressing iter 4/9/6 of 12, not spinning). [my earlier "uses rlm_query_batched"
    grep was a FALSE POSITIVE matching the system-prompt text, not real calls.]
  * Root cause: the prompt never stated that a sub-call runs in a SEPARATE context with NO
    access to `context`; combined with a double "will be rejected" deterrent, the 12B model
    swung from inlining-everything to passing-nothing.
- FIX (prompts.py): added a prominent "CRITICAL - sub-calls do NOT see your context" block
  (after the function list) instructing to paste a SLICE into each sub-call prompt
  (e.g. context[:80000]), with the bare-call anti-example. Template still formats (706 tests
  green; budget field fills). 
- v2 (same 3 tasks, off, 600s, refined prompt): SUCCESS. 0 new overflow (103 stable).
  multihop_000 CORRECT (424s, 7 sub-calls all WITH context); multihop_001 completed
  in-code (wrong/absent answer, no loop); multihop_002 timeout-but-progressing (618s, 12
  sub-calls all WITH context). DECISIVE: 0 bare sub-calls across all trajectories (v1
  pathology gone) - the "sub-calls don't see context, pass a slice" block fixed the
  over-correction. 1/3 correct, up from 0/3. Harness now does real by-reference map-reduce.
- PERF TAIL (documented follow-up, NOT a blocker): multihop_002 made 12 large sequential-ish
  sub-calls and timed out. safe_chunk_chars(98304)=250674 chars => each sub-call ~53-83K
  tokens, slow to prefill. Lever: smaller chunk budget (more parallel via rlm_query_batched)
  and/or encourage batched over sequential. Correctness is fine; this is speed.

## COLD/WARM RESULT (COMPLETE) - the headline
- Cold 6/15 = 40.0% (4 wrong, 5 timeout). Warm 8/15 = 53.3% (3 wrong, 4 timeout).
- MEMORY ADVANTAGE: +13.3% (warm > cold). flipped wrong->correct: 004,005,006,011,014 (5);
  flipped correct->wrong: 007,012,013 (3); net +2. avg 320s cold / 391s warm.
- 0 context-overflow across ALL 30 solves (count stable at 103) - the fix held end to end.
- Bank: 5 correct-only entries. compare.json + raw trajectories under out/.
- This is the plain-multihop UPSIDE the eval was designed to surface (kb tasks saturated). The
  prior pre-fix run was break-even/contaminated; post-fix shows memory clearly helps here.
- Remaining timeouts (003,010 at 660s; 002/007 warm) = the perf tail (chunk-size lever), not
  overflow. multihop_007 flipped correct->wrong AND 97s->660s in warm = memory injection
  perturbed it into a timeout (a "memory hurt" case worth a look).

## STATUS: core fix COMMITTED (prehend ab59ed3, rlm-trainer 306cdaf). 706 tests green.

## CHUNK-BUDGET TUNE (perf tail) - IN VALIDATION
- DECOUPLED the recommended chunk size from the guard ceiling. subcall_guard.py:
  RECOMMENDED_CHUNK_FRAC=0.30 + recommended_chunk_chars(limit,model) (clamped < safe_chunk_chars).
  For 98304: ceiling safe_chunk_chars=250674 chars (unchanged, hard reject threshold); RECOMMENDED
  now 88473 chars (was 250674). rlm.py prompt budget + the rejection hint now use the SMALLER
  recommended value so the model makes several fast parallel chunks, not 1-2 giant slow ones.
  Guard still only rejects what genuinely won't fit (ceiling). 3 old tests updated +
  TestRecommendedChunkChars added; 706 green.
- Validating v3 (same 3 tasks vs v2 250K-char baseline: 000 424s ok / 001 543s wrong / 002 618s
  timeout) - expect 002 to complete faster / under 600s. Commit the tune after the result.

## (historical) COLD/WARM RUN: LAUNCHED (harness task bffjg91v1)
- Script: ~/eval-runs/mem-v13-plain-multihop-fix-2026-06-22/run-cold-warm.sh
  (cold 15 -> warm 15 -> compare, --subcall-context-limit 98304, fresh bank, reflect on e4b
  :8083). Log: same dir / cold-warm.log. Out: same dir / out/. Will take ~1-3h. Notifies on exit.
- Prior pre-fix cold baseline (for compare): 7/15 = 46.7% with overflow spins.
- NOTE: some of these tasks may have genuinely absent/single-hop answers ("Dave moved to
  Denver, no ownership"), so a give-up/conclude failure is partly model capability (a lever
  memory/warm may help). If v2 shows context-passing but still timeouts, that is task
  difficulty, not the fix; proceed to cold/warm regardless to measure memory's effect.

## Helper signatures (Unit A, DONE - integration consumes these)
- token_utils.resolve_subcall_limit(model, *, explicit=None, runtime_ctx=None) -> int
- subcall_guard.oversize_rejection(prompt, *, limit, model, margin_frac=0.15) -> str | None
- subcall_guard.safe_chunk_chars(limit, model, margin_frac=0.15) -> int
- token_utils.CONSERVATIVE_CHARS_PER_TOKEN = 3.0; MODEL_CONTEXT_LIMITS gemma-4/gemma=262144
- Harness/driver param name (committed): subcall_context_limit
