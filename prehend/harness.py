"""High-level Harness API: owns orchestration strategy, runtime detection, and
memory composition so clients do not hand-assemble SRLM. See
docs/superpowers/specs/2026-06-21-prehend-harness-api-design.md and ADR-0008."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import urllib.request


@dataclass(frozen=True)
class Defaults:
    """Vetted Tier-A strategy/reliability defaults the Harness applies to SRLM."""
    max_output_chars: int = 500
    max_iterations: int = 10
    max_depth: int = 2
    max_errors: int = 3
    max_retries: int = 0
    stream: bool = False
    subcall_enable_thinking: bool = False
    max_concurrent_subcalls: int = 4
    soft_timeout_pct: float | None = None


VETTED = Defaults()


@dataclass(frozen=True)
class Runtime:
    """Resolved server facts (Tier B). slots drives map-reduce fan-out."""
    slots: int
    ctx: int | None = None


@dataclass(frozen=True)
class MemoryConfig:
    """ADR-0005 memory wiring, mapped to build_memory_harness_from_config."""
    bank_dir: str
    embed_model: str
    reflect_model: str
    embed_url: str | None = None
    embed_api_key: str | None = None
    k_max: int | None = None
    min_cosine: float | None = None


def _default_probe(base_url: str, api_key: str) -> Runtime | None:
    """Best-effort llama-server probe. Returns None if facts are unavailable."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        with urllib.request.urlopen(f"{root}/props", timeout=5) as r:
            props = json.loads(r.read())
        gen = props.get("default_generation_settings", {}) or {}
        ctx = gen.get("n_ctx") or None
        slots = props.get("total_slots") or gen.get("n_parallel") or 0
        if not slots or slots <= 0:
            return None
        return Runtime(slots=int(slots), ctx=int(ctx) if ctx else None)
    except Exception:
        return None


def detect_runtime(
    base_url: str,
    *,
    api_key: str = "not-needed",
    probe: Callable[[str, str], Runtime | None] | None = None,
) -> Runtime | None:
    """Hybrid Tier-B detection. None means 'ambiguous, caller should fall back'."""
    p = probe or _default_probe
    try:
        rt = p(base_url, api_key)
    except Exception:
        return None
    if rt is None or rt.slots <= 0:
        return None
    return rt
