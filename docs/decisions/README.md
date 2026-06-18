# Architecture Decision Records

This directory holds [MADR](https://adr.github.io/madr/)-format Architecture
Decision Records for **lm-repl** (the patched RLM/SRLM orchestrator). Each ADR
captures one hard-won, non-obvious choice so it is found once and reversed only
with evidence. New decisions get the next number; accepted ADRs are immutable (to
change course, write a new ADR and mark the old one `superseded by ADR-NNNN`).

Cross-repo references are written `<repo> ADR-NNNN`. The cross-repo master index
lives in [rlm-trainer `docs/decisions/README.md`](https://github.com/ClearBridgeRIP/rlm-trainer/tree/main/docs/decisions).

| # | Title | Status |
|---|-------|--------|
| [0001](0001-lm-repl-as-patched-fork-of-rlms.md) | lm-repl as a patched fork of `rlms` | accepted |
| [0002](0002-hard-per-generation-decode-token-ceiling.md) | Hard per-generation decode-token ceiling (`max_decode_tokens`) | accepted |
| [0003](0003-runaway-generation-guards.md) | Runaway-generation guards (soft-budget, subcall caps, contention-retry) | accepted |

`0000-template.md` is the MADR template for new records.
