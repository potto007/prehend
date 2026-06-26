---
status: "accepted"
date: "2026-06-25"
deciders: "potto"
consulted: "overnight debugging session (clean cold/warm eval: all-positive bank root-cause)"
---

# Experience id keys on (question, provenance) so failure guards and success recipes coexist

## Context and Problem Statement

[ADR-0011](0011-contrastive-failure-memory-channel.md) added a contrastive failure
channel: wrong solves distill NEGATIVE guard rules (`derived_from="failure"`,
`polarity="negative"`). A clean N=30 cold/warm eval on plain-multihop with
`--learn-from-failure` ON nonetheless produced a bank of 5 entries, ALL positive -
ZERO negatives despite ~26 wrong solves. The contrastive channel never fired.

Root cause (reproduced byte-for-identically from first principles): `_entry_id` keyed
on the QUESTION TEXT ONLY (`sha1(question)[:12]`). Plain-multihop's 60 tasks use only
5 distinct questions ("What does {Alice,Bob,Carol,Dave,Eve} own?") - they vary the
offloaded CONTEXT, not the question - so all tasks collapse onto 5 ids. The collision
guard in `MemoryHarness._collect` then resolved every collision in favour of success:
a later success supersedes a prior failure of the same id, and a failure colliding with
an existing success is dropped (`outcome="superseded_skip"`, "a failure must never
shadow a known-good recipe"). Because each of the 5 questions is solved correctly at
least once, every id ended positive and every negative was discarded. The failure
channel is structurally untestable whenever questions repeat across differing contexts.

## Decision

Key the experience id on `(question, derived_from)`:
`_entry_id(question, derived_from) = "exp_" + sha1(f"{question}\x00{derived_from}")[:12]`
(`prehend/memory/distill.py`). A success recipe and a failure guard for the same
question now receive DISTINCT ids and COEXIST in the bank; same-`(question, provenance)`
entries still dedupe (one recipe + one guard per question). Retrieval embeds the bare
question, so a query surfaces BOTH the recipe and the guard (cosine ~1.0 to each), and
the existing injection cap (`select_for_injection`, `max_inject_negatives=2`) balances
how many negatives reach the solver.

This SUPERSEDES ADR-0011's id-collision resolution rule (success-shadows-failure). The
two cross-provenance branches in `_collect` (`superseded` / `superseded_skip`) are now
unreachable for freshly distilled entries and kept only as a defensive fallback for
legacy question-only banks. Anti-poisoning still rests on ADR-0011's CONTENT guards
(`is_anti_give_up` / `is_premature_stop`, which already strip capitulation framing at
distill time) plus the injection cap - not on dropping negatives wholesale by id.

## Consequences

- Good: the contrastive failure channel can finally be exercised and measured on
  context-varying task sets (plain multihop). One recipe + one guard per question is
  the intended retrieval shape for a contrastive prompt.
- Good: a wrong solve on context B is no longer silenced just because context A (same
  question) succeeded earlier - that silencing was wrong for offloaded-context tasks.
- Neutral: ids change for ALL entries (even successes: `sha1(q)` -> `sha1(q\x00success)`),
  so a pre-existing question-only bank will not dedupe against newly distilled entries.
  Acceptable - the memory evals rerun from fresh banks; no migration is performed.
- Risk: removing the success-shadows-failure id guard shifts the full anti-poisoning
  burden onto the content guards + injection cap. Re-validate on a rerun: inspect the
  bank's negative entries for capitulation framing and confirm warm does not regress.
