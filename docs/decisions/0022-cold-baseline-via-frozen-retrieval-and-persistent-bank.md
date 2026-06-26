---
status: "accepted"
date: "2026-06-26"
deciders: "potto"
consulted: "cold/warm measurement-correctness review (post-vLLM 60-task run prep)"
---

# Cold baseline = frozen retrieval (not an empty bank); the bank is persistent and versioned

## Context and Problem Statement

The cold/warm experience-memory eval (`rlm-trainer/scripts/memory_cold_warm.py`,
ADR-0005 validation) measured the lift from self-distilled memory by running the
SAME task set twice: `cold` (bank starts empty, each solve distills into it) then
`warm` (retrieve+inject against the now-populated bank).

Two coupled problems:

1. **Cold was not truly cold.** The bank grew DURING the cold run, so cold task N
   could retrieve experiences distilled from cold tasks 1..N-1. Later cold tasks
   were no longer first-exposure - they silently warmed themselves. This inflated
   the cold baseline and understated the measured cold->warm delta. The "cold must
   start empty" guard only ensured the bank was empty at the START of the run; it
   did nothing about within-run self-injection.

2. **A fresh bank every run.** Because cold's baseline integrity was assumed to
   rest on an empty bank (the guard `sys.exit`ed on a populated one), the bank was
   pinned under each run's `--out` dir and rebuilt from scratch every run. There
   was no way to accumulate a bank across runs, nor to keep separate "bank
   versions" when A/B-ing a memory-system change.

## Decision

**Cold's first-exposure property comes from FROZEN RETRIEVAL, not an empty bank.**
A new `freeze_retrieval` flag on the prehend memory layer
(`MemoryHarness`, threaded through `build_memory_harness[_from_config]` and
`MemoryConfig`) short-circuits `_retrieve` to empty so NO experience is injected,
while `_collect`/`collect_pending` still distill and write as usual. The cold/warm
driver sets `freeze_retrieval=(phase == "cold")`. Cold therefore injects nothing
on EVERY task - a true first-exposure baseline regardless of what the bank already
holds - yet still populates the bank for warm.

Because cold no longer reads the bank, its baseline integrity is independent of the
bank's contents. This makes the bank safely **persistent and reusable across runs**:

- The bank is decoupled from `--out` and lives at `--bank-dir` (default
  `runs/banks/default`, gitignored). `--out` holds only this run's
  `cold/warm/compare` results.
- The `cold must start empty` guard is REMOVED. cold appends to whatever the bank
  already holds; warm still requires a non-empty bank.
- Bank VERSIONS are distinct `--bank-dir` paths (no registry - YAGNI). A/B a
  memory-system change by giving each config its own `--bank-dir` so they never
  cross-contaminate.

The three phases are now clean: **off** = no bank (no distill, no retrieve) ·
**cold** = distill+write, retrieval frozen (true baseline, accumulates the bank) ·
**warm** = distill+write+retrieve against the full accumulated bank.

## Consequences

- Good: cold is a genuine floor. The cold->warm delta now measures the real lift
  from injecting the accumulated bank, not a half-warmed baseline.
- Good: the bank accumulates across runs (dedup by `(question, provenance)` id per
  ADR-0020 keeps re-runs of the same task set from bloating it), so warm's delta is
  measured against an increasingly rich bank instead of a one-pass bank.
- Good: `--bank-dir` versioning makes memory-system A/Bs reproducible and isolated.
- Change of behavior: the persistent bank is now the DEFAULT (not opt-in). Any
  caller that relied on a fresh bank under `--out` per run must pass an explicit
  `--bank-dir`. No live wrapper did (verified); only historical eval writeups
  reference the old layout.
- Partially revisits ADR-0020's "memory evals rerun from fresh banks; no migration"
  note: fresh-per-run is no longer forced, but banks remain disposable (gitignored)
  and a new `--bank-dir` is a clean slate, so still no migration machinery.
- Note: `freeze_retrieval` is a general prehend capability (write-only memory), not
  specific to the eval; the no-memory and standard read+write paths are unchanged
  when it is left at its default `False`.
