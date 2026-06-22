---
status: "accepted"
date: "2026-06-22"
deciders: "potto"
---

# Contrastive failure memory channel (learn negative guard rules from wrong solves)

## Context and Problem Statement

The experience-memory layer (ADR-0005) learns ONLY from correct solves:
`MemoryHarness.collect_pending(correct)` drops a solve when `correct is False`, so the
distiller never runs on a failure. That gate flipped plain-multihop from a wash to
+13.3pp and is correct for the IMITABLE-STRATEGY channel: a wrong/timeout trace distilled
as a strategy yields positively-framed "searched exhaustively, found nothing" lessons that
teach shallow-search-then-give-up (the bank-poisoning cascade, fixed in `9385017`).

But dropping failures entirely discards the highest-value signal - WHY an approach failed.
The original defect was conflating two memory TYPES (imitable recipe vs. cautionary guard
rule), not a claim that failures are worthless. A failure's real lesson is contrastive
("When <condition>, <action to avoid/change>"), which the entry schema already represents
(`polarity:"negative"`, `cautions`) and injection already renders. Should the harness learn
from failures, and how, without re-triggering the cascade?

## Decision Drivers

- Survivorship bias: on hard banks the model is wrong >50% of the time; correct-only keeps
  the bank sparse and discards most experience.
- The prior cascade was failure-derived and self-reinforcing (top entry use_count=4, all
  capitulation) - any failure-learning must be structurally safe, not just filtered.
- Reflexion-style verbal-RL precedent: reflect on a failure, store a corrective note.
- Must not regress the proven correct-only +13.3pp path.

## Decision Outcome

Add a CONTRASTIVE FAILURE CHANNEL behind an opt-in `learn_from_failure` flag (default OFF,
so correct-only is unchanged until enabled). When on, `collect_pending(False)` distills the
wrong solve into a `negative`-polarity guard entry instead of dropping it.

- **Failure distillation** uses a dedicated `FAILURE_REFLECT_PROMPT` that demands a
  corrective `When <condition>, <action>` rule whose action is to do MORE/DIFFERENTLY
  (re-read, cross-check, decompose differently, verify, widen) and FORBIDS stop-early /
  accept-partial / prefer-first / capitulation framing. Polarity is forced `negative`
  (a failure can never become an imitable recipe). Entries carry
  `derived_from: "success"|"failure"` provenance.
- **Three-layer anti-poisoning guard** (forced-negative only LABELS, it does not
  neutralize a bad guard; `is_anti_give_up` was tuned for capitulation WORDING, not the
  BEHAVIORAL premature-stop guards a failure distiller emits):
  1. the failure-prompt constraint above;
  2. `is_premature_stop` content filter applied ONLY to failure content (NOT folded into
     `is_anti_give_up`, which runs on the success path and would then drop legitimate
     positive recipes);
  3. `select_for_injection` caps negative entries per injected block
     (`max_inject_negatives`, default 2) - the STRUCTURAL backstop that bounds cascade
     blast radius regardless of what slips the content filter. Only INJECTED entries get a
     `use_count` bump, so capped-out negatives stay `use_count==0` and get `prune()`d (the
     bank self-cleans dominated negatives).
- **Provenance-aware collision** (entry id is `sha1(question)`, so a failure and a later
  success collide): a SUCCESS supersedes a same-id FAILURE (we now know how to solve it); a
  FAILURE never overwrites or shadows a success; same-provenance collisions dedup. Without
  this, a later correct solve would be dropped as a duplicate of the earlier failure guard.

### Consequences

- Good: failures contribute contrastive guard rules; the bank is denser; the proven
  correct-only path is byte-unchanged when the flag is off; the cascade is structurally
  bounded by the injection cap + supersede + prune, not by enumerating phrasings.
- Bad / accepted tradeoffs: forcing `negative` can mislabel a genuine partial-success
  sub-strategy as a guard (a wrong final answer is a weak basis for an imitable recipe);
  the richer-outcome enum (correct / partial / give-up / timeout) is the documented
  follow-up that could route partial-success lessons to positive entries. A buggy scorer
  returning `None` instead of `False` learns a failure positively (pre-existing risk).
- Validation: A/B (`learn_from_failure` off vs on) on plain-multihop, comparing warm
  accuracy and inspecting `bank/meta.json` (`derived_from`, `use_count`) for capitulation.

## More Information

- Spec (2-round adversarial review, reviewer-approved):
  `docs/superpowers/specs/2026-06-22-contrastive-failure-memory-channel-design.md`
- Builds on ADR-0005; the correct-only gate + cascade history:
  `~/.claude/projects/-home-potto-src-prehend/memory/project_memory-layer-distill-quality-bugs.md`
- Code: `prehend/memory/distill.py` (`FAILURE_REFLECT_PROMPT`, `failed=`),
  `prehend/memory/pruning_rules.py` (`is_premature_stop`),
  `prehend/memory/harness.py` (`collect_pending` routing, provenance `_collect`,
  `select_for_injection`), `prehend/memory/factory.py`, `prehend/harness.py`
  (`MemoryConfig.learn_from_failure`/`max_inject_negatives`). Commit `4186a54`.
- Prior art: Reflexion (verbal reinforcement / self-reflection on failures).
