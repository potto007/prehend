---
status: "accepted"
date: "2026-06-17"
deciders: "potto"
---

# Runaway-generation guards (soft-budget, subcall caps, contention-retry)

## Context and Problem Statement

The RLM orchestrator's retrieval loop was **unbounded**: no hard cap on the number
of `llm_query` / `llm_query_batched` / `rlm_query` subcalls a single ask may issue.
The only guards were `max_iterations` (a *turn* count, not a *call* count) and an
advisory soft-timeout message the model could ignore. Evidence: in the 2026-06-17
eval, `ask36.r1` made **579 subcalls in 549s** on a question other reps answered in
0 calls. Unstable retrieval was the common root cause behind covered-ask refusals,
tail latency, and budget escapes. How do we bound search effort?

## Considered Options

- **Layered guards:** hard subcall circuit-breaker + soft-budget wrap-up +
  KV-contention retry.
- Time-budget nudge only (the existing advisory soft-timeout).
- Raise `max_iterations` / tune prompts (advisory only).

## Decision Outcome

Chosen option: **layered runaway-generation guards**:

1. **Hard per-ask subcall circuit-breaker.** Once the per-completion subcall count
   reaches `max_subcalls` (~40-50), further `llm_query`/`llm_query_batched` calls
   short-circuit and return a "retrieval budget exhausted; write your final answer
   or refuse" string instead of hitting the server; the completion loop force-REDUCEs
   at the next iteration boundary. Wired via `max_subcalls` on `RLM.__init__`
   (default `None` = off), exposed to the librarian as `KB_MAX_SUBCALLS`.
2. **Soft-budget early wrap-up.** A time-based nudge to wrap up before the deadline.
3. **KV-slot contention retry.** Detect the *recoverable* 500 (KV-slot contention)
   and retry; this complements, but cannot replace, the hard decode ceiling
   (ADR-0002) since the `ggml_abort` crash is non-recoverable.

### Consequences

- Good, because the 579-call runaway class is killed outright and several
  over-search refusals convert to answers (force-REDUCE from gathered context).
- Good, because tail latency and the INFRA budget-escapes the time-budget alone
  could not close are bounded.
- Bad, because `max_subcalls` is a blunt cap; set too low it truncates a legitimately
  deep multihop search.

## More Information

- rlm-trainer #4 (unbounded retrieval loop; circuit-breaker proposal + evidence).
- Companion: lm-repl ADR-0002 (hard decode-token ceiling), local-ai ADR-0007
  (unified KV serving).
- Commits: `42c275c` (circuit-breaker), `f0018c5` (soft-budget + 500 contention
  detection).
