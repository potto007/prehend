"""Trace-based experience distillation.

Turns a completed solve (an lm-repl ``RLMChatCompletion``) into a bank entry by
reflecting on the actual trajectory -- the cheap, honest distiller from the
mnemex plan (ADR-0005), distinct from FinAcumen's heavy re-solve cross-verify.

A reflect function (prompt -> raw LLM text) and an embedding backend are
injected, so the distiller stays decoupled from any specific client and is
unit-testable without network access.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from lm_repl.memory.embed import EmbeddingBackend
from lm_repl.memory.pruning_rules import is_anti_give_up

ReflectFn = Callable[[str], str]

_VALID_POLARITY = {"positive", "negative"}

REFLECT_PROMPT = """\
You are distilling a reusable lesson from a solved problem so a future agent can
reuse it. Read the question, the final answer, and the solving trajectory, then
return ONLY a JSON object with these keys:
  "polarity": "positive" (a guiding-path recipe that worked) or "negative"
              (a guard rule from a mistake, phrased "When <condition>, <action>").
  "key_insight": one distilled, reusable sentence (50-70 words max).
  "findings": list of short reusable strategy strings (may be empty).
  "cautions": list of short guard-rule strings (may be empty).
Do not restate the specific numbers; generalize to the problem shape.

<Question>
{question}
</Question>
<FinalAnswer>
{answer}
</FinalAnswer>
<Trajectory>
{trajectory}
</Trajectory>"""


class TraceDistiller:
    """Callable ``(question, context, result) -> entry dict | None``."""

    def __init__(
        self,
        reflect_fn: ReflectFn,
        backend: EmbeddingBackend,
        *,
        source: str = "lm-repl",
        trajectory_cap: int = 6000,
    ) -> None:
        self.reflect_fn = reflect_fn
        self.backend = backend
        self.source = source
        self.trajectory_cap = trajectory_cap

    def __call__(self, question: str, context: str, result: Any) -> dict | None:
        answer = self._final_answer(result)
        trajectory = self._trajectory(result)[: self.trajectory_cap]

        prompt = REFLECT_PROMPT.format(
            question=question, answer=answer, trajectory=trajectory
        )
        raw = self.reflect_fn(prompt) or ""
        parsed = _parse_json(raw)
        if not parsed:
            return None

        key_insight = str(parsed.get("key_insight", "")).strip()
        findings = [str(f) for f in (parsed.get("findings") or []) if not is_anti_give_up(str(f))]
        cautions = [str(c) for c in (parsed.get("cautions") or []) if not is_anti_give_up(str(c))]

        insight_ok = bool(key_insight) and not is_anti_give_up(key_insight)
        if not insight_ok and not findings and not cautions:
            return None
        if not insight_ok:
            key_insight = ""

        polarity = str(parsed.get("polarity", "")).strip().lower()
        if polarity not in _VALID_POLARITY:
            polarity = "positive"

        return {
            "id": _entry_id(question),
            "polarity": polarity,
            "key_insight": key_insight,
            "findings": findings,
            "cautions": cautions,
            "embedding": list(self.backend.embed(self._embed_text(question, context))),
            "stats": {"use_count": 0, "hit_count": 0},
            "source": self.source,
            "question": question,
            "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }

    @staticmethod
    def _embed_text(question: str, context: str) -> str:
        return question if not context else f"{question}\n\n{context}"

    @staticmethod
    def _final_answer(result: Any) -> str:
        for attr in ("response", "final_answer"):
            val = getattr(result, attr, None)
            if val:
                return str(val)
        return str(result)

    @staticmethod
    def _trajectory(result: Any) -> str:
        meta = getattr(result, "metadata", None)
        if meta:
            try:
                return json.dumps(meta, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(meta)
        return ""


def _entry_id(question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"exp_{digest}"


def _parse_json(raw: str) -> dict | None:
    """Extract the first JSON object from possibly-fenced, prose-wrapped text."""
    raw = raw.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
