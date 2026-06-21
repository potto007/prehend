"""Strategy verifier: the optimizer layer that reviews llm_query / rlm_query
calls before they execute.

Layered like a DB query optimizer: a free deterministic pass (RuleVerifier) on
every sub-call, plus a costlier adversarial LM pass (LMVerifier) reserved for
rlm_query - the call kind that can starve a run. A rejection is a hard veto:
the call never executes and the REPL receives the rejection string as the
call's result, the same error-string channel the orchestrator already handles
for budget and timeout exhaustion.

Spec: docs/superpowers/specs/2026-06-11-strategy-verifier-design.md
"""

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

REJECTION_PREFIX = "Strategy verifier rejected this call: "

# Roots shorter than this never trigger the whole-task rule: a terse question
# legitimately appears verbatim inside slice prompts ("given this excerpt,
# answer: <question>").
_MIN_ROOT_CHARS = 80

# Fraction of the root's word shingles that must appear in the sub-call prompt
# for it to count as the whole task lightly rephrased.
_SHINGLE_WORDS = 8
_SHINGLE_OVERLAP_VETO = 0.6

_VERIFIER_MAX_TOKENS = 256
_PROMPT_EXCERPT_CHARS = 1500


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _shingles(text: str, n: int = _SHINGLE_WORDS) -> set[tuple[str, ...]]:
    words = text.split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


@dataclass
class SubcallReview:
    """One proposed sub-call, as presented to a verifier."""

    kind: str  # "llm_query" | "rlm_query"
    prompt: str
    root_prompt: str | None  # the task the calling RLM was given
    depth: int = 0


@dataclass
class Verdict:
    approved: bool
    reason: str = ""


@runtime_checkable
class SubcallVerifier(Protocol):
    def review(self, call: SubcallReview) -> Verdict: ...


class RuleVerifier:
    """Deterministic checks. Free: no LM call, applied to every sub-call."""

    def review(self, call: SubcallReview) -> Verdict:
        # The whole-task veto targets rlm_query only: a child RLM re-running
        # the task is unbounded work, while an llm_query is already capped by
        # subcall_max_tokens - and legitimately quotes the full question for
        # manifest triage ("pick relevant doc ids for: <question>").
        if call.kind != "rlm_query":
            return Verdict(approved=True)
        root = _normalize(call.root_prompt or "")
        if len(root) < _MIN_ROOT_CHARS:
            return Verdict(approved=True)
        prompt = _normalize(call.prompt)

        contained = root in prompt
        overlap = 0.0
        if not contained:
            root_shingles = _shingles(root)
            if root_shingles:
                prompt_shingles = _shingles(prompt)
                overlap = len(root_shingles & prompt_shingles) / len(root_shingles)

        if contained or overlap >= _SHINGLE_OVERLAP_VETO:
            return Verdict(
                approved=False,
                reason=(
                    "this delegates your entire task to a sub-call. A sub-agent "
                    "re-running the whole task from scratch wastes the time "
                    "budget. Decompose instead: locate the relevant material "
                    "yourself, then delegate one well-scoped subtask (a "
                    "document slice, one sub-question) per call."
                ),
            )
        return Verdict(approved=True)


class LMVerifier:
    """Adversarial devil's-advocate review by an LM. Fails open: any backend
    error or unparseable output approves the call - the verifier must never
    be the thing that bricks a run."""

    def __init__(
        self,
        backend: str = "openai",
        backend_kwargs: dict[str, Any] | None = None,
        max_tokens: int = _VERIFIER_MAX_TOKENS,
    ):
        self.backend = backend
        self.backend_kwargs = backend_kwargs or {}
        self.max_tokens = max_tokens
        self._client = None  # lazy: created on first review

    def _get_client(self):
        if self._client is None:
            from prehend.clients import get_client

            self._client = get_client(self.backend, self.backend_kwargs)
        return self._client

    def _build_prompt(self, call: SubcallReview) -> str:
        root = (call.root_prompt or "")[:_PROMPT_EXCERPT_CHARS]
        prompt = call.prompt[:_PROMPT_EXCERPT_CHARS]
        return (
            "You are an adversarial strategy reviewer inside a recursive LM "
            "query engine. An orchestrator agent working on this task:\n"
            f"---\n{root}\n---\n"
            "proposes to spawn a recursive sub-agent (its own REPL and "
            "iterations; wall-clock cost is high) with this prompt:\n"
            f"---\n{prompt}\n---\n"
            "Approve only if this is a well-scoped, decomposed subtask the "
            "sub-agent can finish quickly. Reject if it re-asks most of the "
            "root task, is too vague to act on, or duplicates work the "
            "orchestrator should do itself by reading its materials directly. "
            'Reply ONLY with JSON: {"approve": true|false, "reason": "<one sentence>"}'
        )

    def review(self, call: SubcallReview) -> Verdict:
        try:
            text = self._get_client().completion(
                self._build_prompt(call), max_tokens=self.max_tokens
            )
            match = re.search(r'"approve"\s*:\s*(true|false)', text, re.IGNORECASE)
            if match is None:
                return Verdict(approved=True)  # unparseable: fail open
            if match.group(1).lower() == "true":
                return Verdict(approved=True)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
            reason = reason_match.group(1) if reason_match else "strategy rejected by reviewer"
            return Verdict(approved=False, reason=reason)
        except Exception:
            return Verdict(approved=True)  # backend error: fail open


@dataclass
class TieredVerifier:
    """The composition, and the only stateful piece.

    Order per review: (1) resubmission of an already-vetoed prompt is
    re-vetoed immediately with an escalating message - no LM re-review;
    (2) rules on every call; (3) LM devil's advocate on rlm_query only.
    Share one instance across a recursion tree so resubmission memory and
    veto telemetry span every depth.
    """

    rules: SubcallVerifier | None = None
    lm: SubcallVerifier | None = None
    vetoes: list[dict] = field(default_factory=list)

    def __post_init__(self):
        # Keyed by (kind, normalized prompt): downgrading a vetoed rlm_query
        # to a capped llm_query is compliance, not resubmission.
        self._vetoed: dict[tuple[str, str], tuple[int, str]] = {}
        self._lock = threading.Lock()

    def review(self, call: SubcallReview) -> Verdict:
        norm = (call.kind, _normalize(call.prompt))
        with self._lock:
            if norm in self._vetoed:
                count, first_reason = self._vetoed[norm]
                count += 1
                self._vetoed[norm] = (count, first_reason)
                reason = (
                    f"REJECTED AGAIN (attempt {count}): {first_reason} "
                    "You MUST change strategy: do this work yourself in the "
                    "REPL instead of re-submitting this delegation."
                )
                self._record(call, reason, count)
                return Verdict(approved=False, reason=reason)

        verdict = self.rules.review(call) if self.rules is not None else Verdict(approved=True)
        if verdict.approved and self.lm is not None and call.kind == "rlm_query":
            verdict = self.lm.review(call)

        if not verdict.approved:
            with self._lock:
                self._vetoed[norm] = (1, verdict.reason)
                self._record(call, verdict.reason, 1)
        return verdict

    def _record(self, call: SubcallReview, reason: str, attempt: int) -> None:
        self.vetoes.append(
            {
                "kind": call.kind,
                "prompt_preview": call.prompt[:120],
                "reason": reason,
                "attempt": attempt,
            }
        )
