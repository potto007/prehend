"""High-level Harness API: owns orchestration strategy, runtime detection, and
memory composition so clients do not hand-assemble SRLM. See
docs/superpowers/specs/2026-06-21-prehend-harness-api-design.md and ADR-0008."""
from __future__ import annotations

from dataclasses import dataclass


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
