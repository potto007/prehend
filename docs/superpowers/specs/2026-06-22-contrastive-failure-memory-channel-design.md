# Spec: Contrastive failure memory channel

Status: APPROVED (design), pre-implementation
Date: 2026-06-22
Relates to: ADR-0005 (experience-memory layer). Builds on the correct-only distill
gate (`defer_collect` + `collect_pending`) and the `is_anti_give_up` write guard.
Source of truth for TDD. Where this doc and code disagree, this doc wins until amended.

## Problem

The memory layer today learns ONLY from correct solves: `MemoryHarness.collect_pending(
correct)` drops the pending solve when `correct is False`, so the `TraceDistiller` never
runs on a failure. That gate flipped plain-multihop from a wash to +13.3pp and is the
right policy for the IMITABLE-STRATEGY channel: a wrong/timeout trace, distilled as a
strategy, produces positively-framed "searched exhaustively, found nothing" lessons that
teach shallow-search-then-give-up (the bank-poisoning cascade, fixed in `9385017`).

But dropping failures entirely throws away the highest-value signal: WHY an approach
failed. The original defect was conflating two memory TYPES - an imitable recipe vs. a
cautionary guard rule - not a claim that failures are worthless. A failure's real lesson
is contrastive ("When <condition>, <action to avoid/change>"), which the entry schema
already represents (`polarity: "negative"`, `cautions`) and injection already renders
("apply negative guard rules when their condition fires"). The only reason failures don't
contribute is that the gate drops them before distillation.

## Goal

Add a CONTRASTIVE FAILURE CHANNEL: distill wrong solves into `negative`-polarity guard
entries (never imitable strategies), behind an opt-in flag that defaults OFF so the proven
correct-only behavior is unchanged until explicitly enabled.

## Non-goals (v1)

- Richer outcome enum (correct / wrong-partial / give-up / timeout). The `correct: bool`
  routing is enough to prove the channel; the enum is a follow-up if it proves out. (This
  enum is also the principled fix for the forced-negative mislabel tradeoff below.)
- The rlm-trainer driver `--learn-from-failure` flag (fast-follow in that repo).

## Decisions (settled in brainstorming + adversarial review)

- Failures become `negative` polarity ONLY - injected as guard rules, never imitable
  recipes. Forcing negative is an ACCEPTED LOSSY TRADEOFF, not an axiom: a wrong final
  answer can contain a correct sub-strategy that is now mislabeled as a guard. v1 accepts
  this (a wrong final answer is a weak basis for an imitable recipe); the richer-outcome
  enum is the documented follow-up that could rescue positive partials.
- Opt-in flag `learn_from_failure: bool = False`; default off preserves correct-only.
- `derived_from: "success" | "failure"` provenance on every entry for bank inspection.
- **Success supersedes failure for the same question** (review finding #1): entry id is
  `sha1(question)`, so a failure entry and a later success entry collide. The success must
  REPLACE the failure (we now know how to solve it); a failure must NEVER overwrite or
  duplicate a success.
- **Three-layer poisoning guard** (review findings #2/#3), because forced-negative only
  LABELS a harmful guard, it does not neutralize it, and `is_anti_give_up` was tuned for
  capitulation WORDING ("data missing") not the BEHAVIORAL premature-stop guards a failure
  distiller emits ("when chunks conflict, prefer the first and stop searching" - which the
  live filter does NOT catch):
  1. **Prompt constraint** - the failure prompt requires the corrective action to be
     something to do ADDITIONALLY or DIFFERENTLY (re-read, cross-check chunks, decompose
     differently, verify across chunks, widen search), and explicitly forbids stop-early /
     accept-partial / prefer-first / best-available / narrow-scope framing.
  2. **Content filter** - extend the write-time guard with premature-stop behavioral
     patterns (a heuristic, not complete).
  3. **Injection cap (NOT deferred)** - a polarity-aware cap so negative entries can never
     crowd positives out of an injected block. This is the STRUCTURAL backstop that bounds
     cascade blast radius regardless of what slips the content filter, and is required for
     v1 (the cascade would otherwise occur in the very bank the A/B measures).

## Design

### Unit A - TraceDistiller failure mode (`prehend/memory/distill.py`)

- `__call__(self, question, context, result, *, failed: bool = False) -> dict | None`.
- When `failed is False`: UNCHANGED behavior - `REFLECT_PROMPT`, polarity from the model
  (default "positive").
- When `failed is True`: use a new `FAILURE_REFLECT_PROMPT` (below) and FORCE
  `polarity = "negative"` regardless of what the model returns. Same JSON parse, same
  `is_anti_give_up` filtering of `key_insight`/`findings`/`cautions`, same "return None if
  nothing usable survives" guard.
- Every returned entry gains `"derived_from": "failure" if failed else "success"`.
- `FAILURE_REFLECT_PROMPT` (new constant) instructs, in substance:
  - "This solve attempt was INCORRECT. Extract the single most useful CORRECTIVE lesson a
    future agent should apply to AVOID this failure, as a guard rule phrased
    `When <condition>, <action>`."
  - "Return ONLY JSON with keys `key_insight` (the corrective guard rule, 50-70 words),
    `findings` (empty or short corrective strategies), `cautions` (short guard-rule
    strings)." (No `polarity` key needed; the distiller forces negative.)
  - **The corrective `<action>` MUST be something to do ADDITIONALLY or DIFFERENTLY**
    (re-read, cross-check chunks, decompose differently, verify across all chunks, widen
    the search, increase overlap). It must NOT be to stop early, accept a partial / first /
    best-available / most-frequent answer, prefer the first match, narrow scope to save
    time, or otherwise do LESS - those reproduce the shallow-search-then-give-up failure
    this channel exists to prevent.
  - "Do NOT produce a strategy to imitate. Do NOT conclude the data/information is missing,
    absent, unavailable, garbled, or unreadable - that is capitulation, not a lesson."
  - Generalize to the problem shape; do not restate specific numbers.
- Poisoning guard is THREE layers (forced-negative alone only LABELS a bad guard, it does
  not neutralize it - review finding #2): (1) the prompt constraint above; (2) the
  extended content filter (Unit A2); (3) the injection cap (Unit D). Forced-negative
  additionally ensures a surviving failure lesson is rendered as a guard rule, never an
  imitable recipe.

### Unit A2 - extend the write-time content filter (`prehend/memory/pruning_rules.py`)

`is_anti_give_up` catches capitulation WORDING but not BEHAVIORAL premature-stop guards.
Add a set of premature-stop patterns (applied to failure-channel `key_insight`/`findings`/
`cautions` exactly like `is_anti_give_up`) covering at least: "stop searching", "prefer
the first" / "first match", "partial result"/"partial answer", "best available"/"best
estimate", "narrow(ing) scope", "without verifying", "accept ... and stop", "simplest
interpretation ... stop". `PROTECTIVE_PATTERNS` (re-read/retry/verify) still overrides so
constructive guards survive. This is a HEURISTIC (a regex arms race cannot be complete);
the injection cap (Unit D) is the real structural backstop. Implement as a SEPARATE
function `is_premature_stop` applied ONLY to `failed=True` content - do NOT fold these
patterns into `is_anti_give_up` (review note): `is_anti_give_up` runs unconditionally on
the SUCCESS path too (`distill.py:78-81`), so folding would also drop legitimate positive
recipes (e.g. "index once, then prefer the first matching entity"), a regression on the
proven correct-only channel. The success path keeps using ONLY `is_anti_give_up`; the
failure path applies `is_anti_give_up` OR `is_premature_stop` (with `PROTECTIVE_PATTERNS`
overriding both). The distiller drops any failure finding/caution/insight that matches.

### Unit B - MemoryHarness routing (`prehend/memory/harness.py`)

- `__init__` gains `learn_from_failure: bool = False`; store `self.learn_from_failure`.
- `_collect(self, question, context, result, query_tags, *, failed: bool = False)`: pass
  `failed` through to `self.distiller(question, context, result, failed=failed)`, then
  PROVENANCE-AWARE collision handling (review finding #1, replacing today's plain
  dedupe-by-id). With `existing = bank.load()` and the new entry's `id`:
  - no entry with that id -> `bank.append(entry)` (`outcome="written"`).
  - an existing entry with that id is a SUCCESS (`derived_from != "failure"`):
    - new entry is also success -> drop (`outcome="duplicate"`); a known-good recipe is
      never re-distilled over itself.
    - new entry is a FAILURE -> drop (`outcome="superseded_skip"`); a failure must NEVER
      overwrite or shadow a success.
  - an existing entry with that id is a FAILURE (`derived_from == "failure"`):
    - new entry is a SUCCESS -> REPLACE: drop the old failure, append the success, write
      via `bank.save(updated)` (same length -> not a shrink -> `save` accepts it)
      (`outcome="superseded"`). Success supersedes the guard rule.
    - new entry is also a FAILURE -> drop (`outcome="duplicate"`); do not churn.
  - Tag merge (`if query_tags and not entry.get("tags")`) stays. `derived_from` is the
    provenance signal in the written entry; observer `on_collect` gains the new outcome
    strings above so failure writes/supersedes are distinguishable in telemetry.
- `collect_pending(self, correct: bool | None = True)`:
  - `correct is False` AND `self.learn_from_failure`: `self._collect(..., failed=True)`
    (NEW - distill as a failure instead of dropping).
  - `correct is False` AND NOT `self.learn_from_failure`: drop as today (observe
    `outcome="dropped"`).
  - `correct is True` or `None`: `self._collect(..., failed=False)` as today. (Note: a
    buggy scorer that returns `None` instead of `False` on a wrong solve learns it
    positively - a pre-existing risk, not introduced here; documented under Open risks.)
- The non-deferred `answer()` path (`defer_collect=False`) calls `_collect` with the
  default `failed=False`, so its behavior is unchanged (it has no outcome signal and only
  ever learns positively - acceptable; the failure channel requires deferred collection +
  a scoring caller, exactly like correct-only does).

### Unit D - polarity-aware injection cap (`prehend/memory/harness.py` + a helper)

The structural backstop (review finding #3): retrieval (`retrieve`) and rendering
(`render_memory_block`) have ZERO polarity awareness - top-k by cosine. A neighborhood of
similar failures (all embedded on the bare question) can fill top-k with negative guards
that crowd out the positive recipe. Cap it:

- Add a pure helper `select_for_injection(entries, *, max_negatives) -> list[dict]` that
  walks the cosine-RANKED retrieved entries IN ORDER and admits every positive but at most
  `max_negatives` negative entries (skipping further negatives), preserving relevance
  order. All-negative retrieval therefore injects at most `max_negatives`.
- `answer()` applies it between `_retrieve` and `render_memory_block`.
- **`bump_stats` only for entries actually INJECTED** (post-cap), not all retrieved. A
  negative repeatedly retrieved-but-capped-out keeps `use_count==0` and is eligible for
  `prune()` - so the cap also lets the bank self-clean dominated negatives. (Today
  `answer()` bumps every retrieved entry; this narrows it to the injected set.)
- `max_inject_negatives: int` is a `MemoryHarness` param (default 2), threaded from config
  (Unit C). Default ON - not deferred.

### Unit C - config + factory threading

- `MemoryConfig.learn_from_failure: bool = False` (`prehend/harness.py`), documented as
  "learn contrastive negative guard rules from WRONG solves too (requires `defer_collect`);
  default off preserves correct-only".
- `MemoryConfig.max_inject_negatives: int = 2` (the Unit D cap), documented as "max
  negative guard-rule entries injected per query so failure lessons cannot crowd out
  positive recipes".
- `Harness.__init__` passes `learn_from_failure=memory.learn_from_failure` and
  `max_inject_negatives=memory.max_inject_negatives` into `build_memory_harness_from_config`.
- `build_memory_harness` and `build_memory_harness_from_config`
  (`prehend/memory/factory.py`) gain `learn_from_failure: bool = False` and
  `max_inject_negatives: int = 2`, forwarded to `MemoryHarness(...)`.

## Test plan (TDD; author tests before code)

Distiller (fake `reflect_fn` returning canned JSON; fake embed backend):
- `test_failed_true_uses_failure_prompt`: the prompt passed to `reflect_fn` differs from
  the success prompt and contains the corrective-guard-rule framing.
- `test_failed_true_forces_negative_polarity`: even when the model returns
  `polarity:"positive"`, the entry is `negative`.
- `test_failed_false_unchanged_positive_default`: success path keeps model polarity / default.
- `test_failure_entry_has_derived_from_failure` / `test_success_entry_has_derived_from_success`.
- `test_failure_capitulation_still_filtered`: a failure whose only content is "data is
  missing" yields `None` (is_anti_give_up still drops it; no negative-capitulation entry).
- `test_failure_protective_guard_survives`: a failure caution "re-read and verify before
  concluding" survives (PROTECTIVE_PATTERNS override).

Content filter (Unit A2):
- `test_premature_stop_guards_filtered`: each of the review's six phrasings ("prefer the
  first match and stop searching", "return the best available estimate", "once a plausible
  candidate is found return it", "commit to the most frequent value", "narrow scope and
  answer with the partial result", "pick the simplest interpretation and stop") is dropped.
- `test_constructive_failure_guard_survives`: "When chunks conflict, re-read and
  cross-check across all chunks before concluding" survives (PROTECTIVE override).

Harness routing (fake distiller recording `failed`; in-memory bank):
- `test_collect_pending_false_with_flag_distills_failure`: `learn_from_failure=True`,
  `collect_pending(False)` -> distiller called with `failed=True`, entry written.
- `test_collect_pending_false_without_flag_drops`: default flag, `collect_pending(False)`
  -> distiller NOT called, nothing written (today's behavior; regression guard).
- `test_collect_pending_true_distills_success`: `collect_pending(True)` -> `failed=False`.
- `test_collect_pending_none_distills_success`: unscored -> `failed=False` (unchanged).
- `test_learn_from_failure_defaults_false`.

Provenance-aware collision (review finding #1 - the highest-value tests):
- `test_wrong_then_right_success_supersedes_failure`: same question, failure written then
  success collected -> bank holds ONE entry, the success (`derived_from="success"`,
  positive); `outcome="superseded"`.
- `test_right_then_wrong_failure_does_not_overwrite_success`: success then failure -> bank
  holds the SUCCESS unchanged; failure dropped (`outcome="superseded_skip"`).
- `test_failure_then_failure_dedup`: two failures same question -> one entry, no churn.
- `test_success_then_success_dedup`: unchanged dedupe behavior.

Injection cap (Unit D):
- `test_injection_caps_negatives`: retrieved set with 3 negatives + 2 positives,
  `max_inject_negatives=1` -> rendered block has 1 negative + both positives.
- `test_all_negative_retrieval_capped`: all retrieved negative -> at most
  `max_inject_negatives` injected.
- `test_bump_stats_only_for_injected`: a negative retrieved but capped out is NOT
  use_count-bumped (so prune can later drop it).
- `test_negative_entry_renders_as_caution_block`: a failure-derived (negative) entry
  injects as a guard rule, not a positive recipe.

Config/factory threading:
- `test_memoryconfig_learn_from_failure_defaults_false` + `test_memoryconfig_max_inject_negatives_default`.
- `test_factory_threads_learn_from_failure_and_cap_into_harness`.
- `test_harness_passes_failure_flags_from_config` (Harness -> build -> MemoryHarness).

Non-deferred path:
- `test_non_deferred_collect_defaults_failed_false` (regression: `answer()` with
  `defer_collect=False` still learns positively only).

Regression: full suite green (correct-only default path byte-unchanged when flag off and
the cap admits all positives + up to default negatives).

## Validation (live, after green suite)

- A/B on the same plain-multihop tasks: `learn_from_failure` OFF (correct-only baseline)
  vs ON, comparing warm accuracy and avg latency.
- Inspect `bank/meta.json`: failure-derived entries must be `polarity:"negative"`,
  `derived_from:"failure"`, and contain NO capitulation key_insights; check `use_count` of
  negative entries does not dominate (early-warning for a re-poisoning cascade).
- PASS = warm accuracy with the flag ON >= correct-only baseline AND no capitulation
  entries in the bank. If negatives dominate or accuracy drops, the held-back injection cap
  is the next lever.

## Open risks

- **Re-poisoning** (review #2/#3): the original cascade was capitulation-WORDED; this
  channel emits BEHAVIORAL premature-stop guards that a pure content filter cannot fully
  catch (regex arms race). The structural backstop is the Unit D injection cap (bounds how
  many negatives reach any block, regardless of bank composition) plus bump-only-injected
  (so capped-out negatives stay `use_count==0` and get pruned). The prompt constraint +
  extended filter are the first two layers; the cap is the one that does not depend on
  guessing phrasings. If the A/B still shows negatives dominating, lower
  `max_inject_negatives` (a config knob) before anything else.
- **Forced-negative mislabel** (review #4): a wrong final answer can carry a correct
  sub-strategy now rendered as a guard. ACCEPTED lossy tradeoff for v1 (a wrong answer is a
  weak basis for an imitable recipe); the richer-outcome enum (non-goal) is the principled
  fix that could route partial-success lessons to positive entries.
- **Buggy scorer** (review #5): `collect_pending(None)` learns positively, so a scorer that
  returns `None` instead of `False` on failure silently learns the failure as a positive
  recipe. Pre-existing (not introduced here); the driver must pass an explicit `False` on
  wrong solves.
- **Distiller honesty on failures**: the model may emit a vacuous "try harder" guard.
  Low-value but not harmful (negative, caution-framed, cap-bounded); the A/B + bank
  inspection (`derived_from`, `use_count`) will show if they accumulate.
- **Non-deferred path asymmetry**: with `defer_collect=False` the failure channel is inert
  (no outcome signal). Acceptable and consistent with correct-only, which also requires a
  scoring caller.

## References

- ADR-0005: `docs/decisions/0005-mnemex-experience-memory-layer.md`
- Distill quality bugs (the correct-only lever + cascade history):
  `~/.claude/projects/-home-potto-src-prehend/memory/project_memory-layer-distill-quality-bugs.md`
- Code: `prehend/memory/distill.py` (`TraceDistiller`, `REFLECT_PROMPT`),
  `prehend/memory/harness.py` (`collect_pending`, `_collect`),
  `prehend/memory/inject.py` (`render_memory_block`),
  `prehend/memory/pruning_rules.py` (`is_anti_give_up`, `PROTECTIVE_PATTERNS`),
  `prehend/harness.py` (`MemoryConfig`, `Harness.record_outcome`),
  `prehend/memory/factory.py` (`build_memory_harness*`)
- Prior art: Reflexion (verbal reinforcement / self-reflection on failures).
