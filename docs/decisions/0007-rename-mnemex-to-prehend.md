---
status: "accepted"
date: "2026-06-21"
deciders: "potto"
---

# Full rename `mnemex` -> `prehend` (package, PyPI, repo)

## Context and Problem Statement

ADR-0006 renamed the whole project `lm-repl` -> `mnemex` to put the
learning/memory axis in the name. In practice `mnemex` proved hard to type and
spell, and the literal string was already taken on PyPI by an unrelated project
("Mnemex: temporal memory management", v0.6.0), so the distribution `name =
"mnemex"` could never actually publish (PyPI 403 on a name another project
owns). A name is only useful if it is both ownable on the registries and easy to
recall; `mnemex` was neither. What should the project be called, and is the new
name verified clean before another hard cut?

## Decision Drivers

- The PyPI distribution name MUST be claimable - `mnemex` is owned by someone
  else, so it is a hard blocker, not a preference.
- The name should still describe the project's nature: a harness that *learns*
  (memory) and *executes* (writes/runs programs to solve). "prehend" = to grasp,
  both mentally (comprehend) and physically (seize/act) - one root for both.
- Easy to type and spell; no Grok-style brand baggage; no collision with an
  active product or software trademark.
- Two repos import the package (`~/src/knowledge-base` kb-librarian,
  `~/src/rlm-trainer`); a rename must not silently break them.

## Considered Options

- **Full rename now to `prehend`, hard cut, no compat shim** - verify the name
  is clean across registries/domains first, then rename package/imports, PyPI
  name, repo, README; migrate both downstream consumers in the same effort.
- Keep `import mnemex`, publish under an available `mnemex-*` distribution name
  (import/dist mismatch).
- Drop PyPI publishing entirely and distribute via GitHub editable installs.
- Keep `mnemex` everywhere (status quo - cannot publish, hard to spell).

Name candidates were screened on the memory+execution axis; `prehend` was chosen
and then verified BEFORE committing (unlike `mnemex`, which collided): PyPI, npm,
crates.io, and the `github.com/prehend` namespace are all free; a web search for
"prehend" returns only dictionary entries; the sole trademark hit is PREHEND
INC., whose registered wordmark is "OICMLN" in Class 18 (leather goods/bags) -
a different field from software (Class 9/42), so no realistic conflict.

## Decision Outcome

Chosen option: **full rename, hard cut**. `mnemex` -> `prehend` across the
package and import paths; package dir `mnemex/` -> `prehend/`; `pyproject` `name`
`mnemex` -> `prehend`; project URLs and README reframed onto `prehend`; repo
`potto007/mnemex` -> `potto007/prehend`; PyPI distribution `prehend`. No
compatibility shim - the two downstream importers (`knowledge-base`,
`rlm-trainer`) are migrated in the same effort. The on-disk working directory is
also renamed `~/src/mnemex` -> `~/src/prehend` (ADR-0006 had deliberately left
the directory at its `lm-repl` name; this rename takes the directory too, so the
path matches the brand). Behavioral contracts consumers depend on are unchanged;
only the module path and name move.

Deliberately *not* renamed, because they are historical: accepted ADRs 0001-0006
(immutable; they keep the name they were accepted under) and the dated design
docs under `docs/superpowers/`.

### Consequences

- Good, because the distribution name is now actually claimable on PyPI and the
  name is easy to type and recall.
- Good, because the name was verified clean across registries, GitHub, and
  trademark BEFORE the rename - the failure mode of ADR-0006 (unpublishable
  name) cannot recur.
- Good, because "prehend" still carries the learn (comprehend) + execute (seize)
  duality the project embodies.
- Bad, because it is a second hard rename in quick succession; any reference to
  the short-lived `mnemex` PyPI/import name (none published externally) breaks.
- Bad, because old ADRs and historical design docs now reference two prior names
  the project no longer uses (accepted as the cost of ADR immutability).

## More Information

- Supersedes the naming decision of ADR-0006 (`lm-repl` -> `mnemex`).
- Name-availability verification (PyPI/npm/crates/GitHub/trademark/domains) was
  run via firecrawl on 2026-06-21; `prehend.io`/`.app`/`.sh` are available
  domains, `prehend.com`/`.ai` are already registered.
