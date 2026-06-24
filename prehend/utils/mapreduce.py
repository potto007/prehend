"""
Pure map-reduce engine for oversized sub-call DATA (ADR-0010, extends ADR-0009).

When a sub-call carries data too large to fit the sub-model window, the harness
splits the data into chunks, runs the instruction over each chunk (MAP), and
combines the per-chunk results with a hierarchical, depth-bounded tree REDUCE.
This shifts decomposition from advice (the model is told to chunk) to mechanism
(the harness chunks), killing the latency tail where the orchestrator made 1-2
giant chunks and timed out.

This module is PURE: all LM I/O is injected via ``run_batch`` (a context-free
batched send), and ``fits``/``compose``/``is_control`` are injectable so the
algorithm can be unit-tested with no token math and no sockets. See
docs/superpowers/specs/2026-06-22-auto-chunk-enforcement-design.md (source of truth).
"""

from collections.abc import Callable
from dataclasses import dataclass

# A partial that is actually a control/error string rather than a real answer.
# These must be filtered out of the reduce so guard/budget boilerplate never
# poisons the combined answer (review findings R1-8/R2-3).
_GUARD_PREFIX = "Sub-call input guard rejected this call:"
_BUDGET_MARKER = "retrieval budget exhausted"

# Sentinel a MAP chunk returns when it holds nothing relevant to the request.
# Dropped (like other control strings) BEFORE the reduce so the many no-signal
# chunks of a sparse/multihop context cannot statistically outvote the one chunk
# that holds the answer. Diagnosed on multihop_053: the answer-bearing partial
# ("Alice owns a golden key.") was drowned by 15 verbose "no information about
# Alice" partials and the reduce concluded "no mention of Alice" (reduce-loss).
_NO_INFO_SENTINEL = "NO_RELEVANT_INFO"

# Appended to the MAP (per-chunk) instruction only. Phrased to PRESERVE partial
# hops: a chunk holding an intermediate fact (e.g. "Alice moved to Chicago",
# which does not itself answer "what does Alice own?") must still report that
# fact so the reduce can chain hops - only a chunk with NO related fact at all
# emits the sentinel. Trails the instruction (data-first, ADR-0017) so it does
# not disturb the cacheable chunk prefix.
_MAP_SENTINEL_DIRECTIVE = (
    "\n\nExtract, do not answer. Quote any fact the text above states about a "
    "person, place, or thing named in the request - INCLUDING background facts "
    "such as where a named person lives, moves to, or works, which may be needed "
    "to reach the answer only indirectly through another fact. A fact that names "
    "any entity from the request is relevant even if it does not mention what the "
    f"request literally asks. Reply with exactly {_NO_INFO_SENTINEL} and nothing "
    "else ONLY if the text states no such fact about any entity in the request."
)

# Clean answer surfaced when EVERY chunk returned the sentinel (truly nothing
# found anywhere), so the raw token never leaks to the caller.
_NO_INFO_ANSWER = "No relevant information was found in the provided text."

# Appended to the truncated join when the tree hits max_reduce_depth so the loss
# is visible in the final reduce input and in logs.
_TRUNCATE_NOTE = "\n\n[note: reduce truncated at max depth; some partial results omitted]"


def _compose(instr: str, data: str, label: str = "Text") -> str:
    """Frame an instruction and a data blob into a single sub-call prompt.

    Data-first layout (ADR-0017): the large, stable data leads and the varying
    instruction trails. The served solver's radix/prefix cache matches from
    token 0, so leading with the chunk lets the SAME chunk be reused across
    sub-calls (only the short trailing instruction re-prefills). The old
    instruction-first layout diverged at token 0 and re-prefilled the whole
    chunk every query (~6.4x re-prefill measured on the multihop bench).
    """
    return f"{label}:\n{data}\n\n{instr}"


def _is_control(text: str) -> bool:
    """True for a non-answer control string (guard rejection, error, budget msg,
    or the no-info sentinel) - dropped before the reduce so it cannot dilute it."""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    return (
        stripped.startswith("Error:")
        or stripped.startswith(_GUARD_PREFIX)
        or stripped.upper().startswith(_NO_INFO_SENTINEL)
    )


def _is_no_info(text: str) -> bool:
    """True for the no-info sentinel specifically (a subset of _is_control)."""
    return isinstance(text, str) and text.strip().upper().startswith(_NO_INFO_SENTINEL)


@dataclass
class MapReduceResult:
    """Outcome of a map_reduce run. The seam returns ``answer``; the rest is
    metadata for logging and test assertions."""

    answer: str
    n_chunks: int
    reduce_levels: int
    truncated: bool
    dropped: int
    budget_exhausted: bool


def _greedy_groups(
    partials: list[str],
    reduce_prompt: str,
    fits: Callable[[str], bool],
    compose: Callable[[str, str, str], str],
) -> list[list[str]]:
    """Pack consecutive partials into groups, each composing to a prompt that
    ``fits``. Test-add-then-check: a partial joins the current group only if the
    composed candidate still fits; otherwise it opens a new group (a single
    partial that does not fit alone still gets its own group)."""
    groups: list[list[str]] = []
    current: list[str] = []
    for p in partials:
        candidate = current + [p]
        composed = compose(reduce_prompt, "\n\n".join(candidate), "Partial results")
        if current and not fits(composed):
            groups.append(current)
            current = [p]
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def map_reduce(
    prompt: str,
    context: str,
    *,
    run_batch: Callable[[list[str]], list[str]],
    fits: Callable[[str], bool],
    chunk_chars: int,
    reduce_prompt: str | None = None,
    max_reduce_depth: int = 3,
    overlap_chars: int = 0,
    is_control: Callable[[str], bool] = _is_control,
    compose: Callable[[str, str, str], str] = _compose,
) -> MapReduceResult:
    """Map ``prompt`` over chunks of ``context`` and tree-reduce the partials.

    Args:
        prompt: the per-chunk instruction (map step).
        context: the large data to chunk.
        run_batch: CONTEXT-FREE batched send; takes composed prompts, returns
            responses 1:1. Already enforces the budget/guard at the seam.
        fits: ADR-0009 ceiling check on a composed string (reduce-group packing).
        chunk_chars: data budget per chunk (the seam pre-subtracts the compose
            envelope so ``compose(prompt, chunk)`` fits). Also bounds partial size.
        reduce_prompt: combine instruction; defaults to ``prompt``.
        max_reduce_depth: tree depth bound; the unconditional termination backstop.
        is_control: predicate marking a response as a control/error string to drop.
        compose: instruction+data framing.
    """
    if reduce_prompt is None:
        reduce_prompt = prompt
    chunk_chars = max(1, chunk_chars)
    # Overlap must be < chunk_chars so the step is positive (else the split would
    # never advance). Negative overlap is meaningless -> 0.
    overlap_chars = max(0, min(overlap_chars, chunk_chars - 1))

    # 1. Split context into <= chunk_chars slices that OVERLAP by overlap_chars,
    #    so a span straddling a boundary appears in both neighbours (preserves
    #    cross-chunk links for multi-hop). overlap_chars=0 -> contiguous slices.
    #    Empty context yields one empty chunk; never zero chunks.
    step = chunk_chars - overlap_chars
    chunks = [context[i : i + chunk_chars] for i in range(0, max(len(context), 1), step)]
    n_chunks = len(chunks)

    dropped = 0
    budget_exhausted = False

    def _filter(responses: list[str]) -> list[str]:
        nonlocal dropped, budget_exhausted
        real: list[str] = []
        for r in responses:
            if isinstance(r, str) and _BUDGET_MARKER in r:
                budget_exhausted = True
            if is_control(r):
                dropped += 1
            else:
                real.append(r)
        return real

    # 2. Map: run the instruction over each chunk in one batch. The MAP
    #    instruction carries the no-info sentinel directive (the REDUCE prompt
    #    below does not) so a chunk with no related fact returns a droppable
    #    sentinel instead of a verbose "no information" answer that would dilute
    #    the reduce.
    map_instr = prompt + _MAP_SENTINEL_DIRECTIVE
    map_raw = run_batch([compose(map_instr, c, "Text") for c in chunks])
    partials = _filter(map_raw)

    # If every map partial was a control string, surface a useful answer rather
    # than an empty one. When they were all no-info sentinels (nothing relevant
    # anywhere) return a clean readable message - never the raw token. Otherwise
    # surface the first control string verbatim so the model sees the hint/error.
    if not partials:
        if map_raw and all(_is_no_info(r) for r in map_raw):
            answer = _NO_INFO_ANSWER
        else:
            answer = map_raw[0] if map_raw else ""
        return MapReduceResult(
            answer=answer,
            n_chunks=n_chunks,
            reduce_levels=0,
            truncated=False,
            dropped=dropped,
            budget_exhausted=budget_exhausted,
        )

    # 4. Reduce loop.
    level = 0
    while True:
        if len(partials) == 1:
            return MapReduceResult(
                answer=partials[0],
                n_chunks=n_chunks,
                reduce_levels=level,
                truncated=False,
                dropped=dropped,
                budget_exhausted=budget_exhausted,
            )

        # Truncate at the depth bound, or once budget is exhausted (stop the
        # multi-round fan-out and do a single consolidating reduce of what we have).
        if level >= max_reduce_depth or budget_exhausted:
            joined = "\n\n".join(partials)
            cut = joined[: max(0, chunk_chars - len(_TRUNCATE_NOTE))]
            final = compose(reduce_prompt, cut + _TRUNCATE_NOTE, "Partial results")
            final_raw = run_batch([final])
            real = _filter(final_raw)
            answer = real[0] if real else (final_raw[0] if final_raw else "")
            return MapReduceResult(
                answer=answer,
                n_chunks=n_chunks,
                reduce_levels=level + 1,
                truncated=True,
                dropped=dropped,
                budget_exhausted=budget_exhausted,
            )

        # Group: hard-cut each partial to the data budget (partials are model
        # answers with no inherent size bound), then greedily pack into groups.
        capped = [p[:chunk_chars] for p in partials]
        groups = _greedy_groups(capped, reduce_prompt, fits, compose)
        reduce_raw = run_batch(
            [compose(reduce_prompt, "\n\n".join(g), "Partial results") for g in groups]
        )
        partials = _filter(reduce_raw)
        level += 1

        if not partials:
            return MapReduceResult(
                answer=reduce_raw[0] if reduce_raw else "",
                n_chunks=n_chunks,
                reduce_levels=level,
                truncated=False,
                dropped=dropped,
                budget_exhausted=budget_exhausted,
            )
