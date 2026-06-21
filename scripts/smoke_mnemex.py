#!/usr/bin/env python3
"""Live smoke test of the mnemex experience-memory loop against a real LLM.

Proves the wiring end-to-end against the local llama-server (NOT a teacher):
solve -> distill the trajectory via a real reflect call -> write a bank entry to
disk -> retrieve + inject it on a second call. This is a WIRING test (does the
loop work live?), not a usefulness eval.

Tier 1: embeddings are a local deterministic ``HashingEmbeddingBackend`` because
the llama-server serves only chat GGUFs (no embedding endpoint). Consequence: the
hashing backend matches only byte-identical text, and retrieval embeds the bare
question while the distiller stores ``embed(question + "\\n\\n" + context)``. So
for retrieval to hit on call 2, the stored key must equal the query key, which
means ``context`` MUST be empty here. A real (semantic) embedder tolerates a
non-empty context; the hashing backend does not. Tier 2 swaps in a real embedder.

Usage:
    .venv/bin/python scripts/smoke_mnemex.py \\
        --base-url http://localhost:8080/v1 \\
        --model google/gemma-4-12b-it \\
        --bank-dir /tmp/mnemex_smoke_bank

While it runs, confirm the wire in another shell (the load-bearing gotcha):
    ss -tnp | grep $(pgrep -f smoke_mnemex)   # MUST show :8080
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv  # repo rule: scripts hitting the server load_dotenv()

from lm_repl.core.srlm import SRLM
from lm_repl.memory import (
    HashingEmbeddingBackend,
    OpenAIReflectFn,
    build_memory_harness,
)

load_dotenv()

DEFAULT_BASE_URL = "http://localhost:8080/v1"
# The actual loaded chat model on the local llama-server (google/gemma-4-12b-it
# has no GGUF on disk). Override with --model for a different one.
DEFAULT_MODEL = "gemma-4-12b-it-sft-kb-v13-sft"
DEFAULT_BANK_DIR = "/tmp/mnemex_smoke_bank"
QUESTION = "What is 6 multiplied by 7?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    p.add_argument("--embed-dim", type=int, default=64)
    p.add_argument("--min-cosine", type=float, default=0.5)
    p.add_argument(
        "--keep-bank",
        action="store_true",
        help="do not wipe the bank dir first (default: start fresh for reproducibility)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank_dir = Path(args.bank_dir)

    if not args.keep_bank and bank_dir.exists():
        shutil.rmtree(bank_dir)
        print(f"[setup] wiped existing bank dir {bank_dir}")

    print(f"[setup] base_url={args.base_url} model={args.model} bank_dir={bank_dir}")

    srlm = SRLM(
        backend="openai",
        backend_kwargs={
            "model_name": args.model,
            "base_url": args.base_url,
            "api_key": "EMPTY",
            # Bound the unbounded short-context direct path: disable gemma CoT and
            # cap tokens so the solve can't degenerate into a giant thinking trace.
            "default_extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 2048,
            },
        },
        direct_threshold=30_000,
    )
    reflect = OpenAIReflectFn.from_config(
        base_url=args.base_url, model=args.model,
        # Mechanical JSON extraction: disable gemma CoT (else it can run away into
        # an unbounded thinking trace) and cap tokens as a backstop.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        max_tokens=1024,
    )
    harness = build_memory_harness(
        srlm,
        bank_dir,
        embed_backend=HashingEmbeddingBackend(dim=args.embed_dim),
        reflect_fn=reflect,
        min_cosine=args.min_cosine,
    )

    # context MUST be empty: see module docstring (hashing backend keying).
    print(f"\n[call 1] learning -- {QUESTION!r}")
    r1 = harness.answer(context="", question=QUESTION)
    print(f"[call 1] answer: {_answer_text(r1)!r}")
    after1 = harness.bank.load()
    print(f"[call 1] bank entries: {len(after1)}")

    print(f"\n[call 2] retrieving -- {QUESTION!r}")
    r2 = harness.answer(context="", question=QUESTION)
    print(f"[call 2] answer: {_answer_text(r2)!r}")
    after2 = harness.bank.load()
    print(f"[call 2] bank entries: {len(after2)}")

    return _report(after1, after2)


def _answer_text(result: object) -> str:
    for attr in ("response", "final_answer"):
        val = getattr(result, attr, None)
        if val:
            return str(val)
    return str(result)


def _report(after1: list[dict], after2: list[dict]) -> int:
    """Check the pass criteria; return process exit code (0 = pass)."""
    failures: list[str] = []

    if len(after1) != 1:
        failures.append(f"call 1 should leave exactly 1 bank entry, got {len(after1)}")
    else:
        entry = after1[0]
        if not str(entry.get("key_insight", "")).strip():
            failures.append("call 1 entry has empty key_insight")
        if not entry.get("embedding"):
            failures.append("call 1 entry has no embedding")

    if len(after2) != 1:
        failures.append(f"call 2 should still have 1 entry (id dedup), got {len(after2)}")
    else:
        use_count = after2[0].get("stats", {}).get("use_count", 0)
        if use_count != 1:
            failures.append(
                f"call 2 should bump use_count to 1 (proves retrieval hit), got {use_count}"
            )

    print("\n" + "=" * 60)
    if failures:
        print("SMOKE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        if after1:
            print(f"\n  entry[0] keys: {sorted(after1[0])}")
            print(f"  entry[0] key_insight: {after1[0].get('key_insight')!r}")
            print(f"  entry[0] stats: {after2[0].get('stats') if after2 else None}")
        return 1

    print("SMOKE TEST PASSED:")
    print(f"  - 1 bank entry with key_insight + embedding")
    print(f"  - id dedup held (still 1 entry after call 2)")
    print(f"  - use_count bumped to 1 (retrieval hit on call 2)")
    print(f"  - key_insight: {after2[0].get('key_insight')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
