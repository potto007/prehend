# Strategy Verifier

Date: 2026-06-11
Status: approved

## Problem

The orchestrator model decides unilaterally what to delegate to `llm_query` /
`rlm_query`. A model trained on the stock prompt will delegate an entire hard
question to `rlm_query`; the child re-runs the whole task from scratch and
consumes the run's budget (the 2026-06-11 kb ask: 600s spent, zero documents
read). Prompt steering is advisory and loses to training; `subcall_max_timeout`
is a blunt backstop that still burns the capped time on useless work.

A query engine needs an optimizer layer: something that reviews the proposed
strategy before paying for its execution.

## Design

Layered, like a DB optimizer: a free deterministic rewrite/reject pass on every
sub-call, plus a costlier adversarial LM pass reserved for the expensive call
kind. A rejection is a hard veto: the call never executes, and the REPL receives
`Strategy verifier rejected this call: <reason>` as the call's result string -
the same error-string channel the orchestrator already handles for budget and
timeout exhaustion. The orchestrator adapts on its next iteration.

### Components (`lm_repl/core/verifier.py`)

- `SubcallReview` - what gets judged: `kind` ("llm_query" | "rlm_query"),
  `prompt`, `root_prompt` (the task the calling RLM was given), `depth`.
- `Verdict` - `approved: bool`, `reason: str`.
- `SubcallVerifier` - protocol: `review(call) -> Verdict`. Anything
  implementing it can be plugged in.
- `RuleVerifier` - deterministic checks, applied to every kind:
  - **Whole-task delegation**: the sub-call prompt contains the root task
    (normalized containment), or shares the bulk of its 8-word shingles with
    it. Delegating your entire task to a copy of yourself is never a strategy.
- `LMVerifier` - devil's-advocate LM judgment, `rlm_query` only. Same backend
  as the run (a different model would thrash a `--models-max 1` router),
  output capped (default 256 tokens), JSON verdict parsed leniently.
  **Fails open**: any LM error, timeout, or unparseable output approves the
  call - the verifier must never be the thing that bricks a run.
- `TieredVerifier` - the composition, and the only stateful piece:
  1. Resubmission check: a prompt that was already vetoed (by any layer) is
     re-vetoed immediately with an escalating message - the second attempt is
     told it MUST change strategy. No LM re-review for resubmissions.
  2. `RuleVerifier` on every call.
  3. `LMVerifier` on `rlm_query` calls only (when configured).
  Records every veto in `.vetoes` (kind, prompt preview, reason, attempt) for
  telemetry. Thread-safe: `rlm_query_batched` reviews concurrently.

### Wiring

- `RLM(subcall_verifier=...)`. `completion()` stores the root prompt;
  `_spawn_completion_context` hands verifier + root prompt to `LMHandler`,
  which reviews in `_handle_single` / `_handle_batched` (kind "llm_query";
  batched requests review per-prompt - vetoed prompts get the rejection
  string, the rest execute). `_subcall` reviews before spawning the child or
  the leaf completion (kind "rlm_query").
- Children get the SAME verifier instance (propagated in `_subcall` like the
  runaway guards), so resubmission memory and veto telemetry span the
  recursion tree. A child's `root_prompt` is its delegated prompt: the rule
  becomes "don't re-delegate YOUR whole task" at every depth.
- A veto consumes an orchestrator iteration. Accepted: the escalating
  resubmission message is the counterweight, and `.vetoes` makes
  veto/adapt rates observable before any further tightening.
- Default `subcall_verifier=None`: behavior identical to today.

## Out of scope (future layers)

- Fan-out/batch-size sanity rules.
- Budget-aware cost model ("this batch cannot fit in the remaining 180s").
- Cross-run veto statistics.
