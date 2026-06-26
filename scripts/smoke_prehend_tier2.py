#!/usr/bin/env python3
"""Tier-2 live smoke test of prehend with a REAL semantic embedder (bge-m3).

Unlike Tier 1 (deterministic hashing backend, byte-identical match only), this
exercises the full value of the memory layer: semantic retrieval. Chat + reflect
go to the gemma chat model on the local llama-server; embeddings go to a separate
bge-m3 ``--embedding`` llama-server. Split endpoints, so we build the harness with
an explicit ``embed_backend`` + ``reflect_fn`` (not ``*_from_config``).

Flow (each call: retrieve -> inject -> solve -> distill -> collect):
  1. learn Q1 (with a real, non-empty context)            -> 1 entry
  2. re-ask Q1 exactly        -> exact hit, id-dedup       -> still 1 entry, use_count 1
  3. ask a PARAPHRASE of Q1   -> SEMANTIC hit on Q1's entry-> 2 entries, Q1.use_count 2

The call-3 bump on Q1's entry is the proof a hashing backend cannot produce: a
differently-worded question retrieved the original experience.

Prereqs:
  chat:  ~/src/local-ai/scripts/llama-server.sh load gemma-4-12b-it-sft-kb-v13-sft
  embed: llama-server --embedding -m <bge-m3.gguf> --pooling cls --port 8084
         (NOT :8081 - that port is the dual-context sub-call worker, ADR-0014)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv  # repo rule: scripts hitting the server load_dotenv()

from prehend.core.srlm import SRLM
from prehend.memory import (
    OpenAIEmbeddingBackend,
    OpenAIReflectFn,
    build_memory_harness,
)
from prehend.memory.retrieve import retrieve

load_dotenv()

DEFAULT_CHAT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_CHAT_MODEL = "gemma-4-12b-it-sft-kb-v13-sft"
DEFAULT_EMBED_BASE_URL = "http://localhost:8084/v1"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_BANK_DIR = "/tmp/prehend_tier2_bank"

Q1 = "What is 6 multiplied by 7?"
PARAPHRASE = "Compute the product of six and seven."
CONTEXT = "Basic arithmetic facts. Multiplication is repeated addition."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chat-base-url", default=DEFAULT_CHAT_BASE_URL)
    p.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    p.add_argument("--embed-base-url", default=DEFAULT_EMBED_BASE_URL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--bank-dir", default=DEFAULT_BANK_DIR)
    p.add_argument("--min-cosine", type=float, default=0.65)
    p.add_argument("--keep-bank", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank_dir = Path(args.bank_dir)
    if not args.keep_bank and bank_dir.exists():
        shutil.rmtree(bank_dir)
        print(f"[setup] wiped existing bank dir {bank_dir}")

    print(f"[setup] chat={args.chat_model} @ {args.chat_base_url}")
    print(f"[setup] embed={args.embed_model} @ {args.embed_base_url}")
    print(f"[setup] bank={bank_dir} min_cosine={args.min_cosine}")

    srlm = SRLM(
        backend="openai",
        backend_kwargs={
            "model_name": args.chat_model,
            "base_url": args.chat_base_url,
            "api_key": "EMPTY",
            # The short-context direct path is otherwise unbounded; on a thinking
            # model that lets the solve degenerate into a giant CoT trace. Disable
            # thinking + cap tokens for this mechanical smoke (wiring, not reasoning).
            "default_extra_body": {
                "chat_template_kwargs": {"enable_thinking": False},
                "max_tokens": 2048,
            },
        },
        direct_threshold=30_000,
    )
    embed_backend = OpenAIEmbeddingBackend.from_config(
        base_url=args.embed_base_url, model=args.embed_model
    )
    reflect = OpenAIReflectFn.from_config(
        base_url=args.chat_base_url, model=args.chat_model,
        # Distillation is mechanical JSON extraction - disable gemma's CoT so it
        # cannot degenerate into an unbounded thinking trace; cap as a backstop.
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        max_tokens=1024,
    )
    harness = build_memory_harness(
        srlm,
        bank_dir,
        embed_backend=embed_backend,
        reflect_fn=reflect,
        min_cosine=args.min_cosine,
    )

    print(f"\n[call 1] learn      -- {Q1!r}")
    harness.answer(context=CONTEXT, question=Q1)
    print(f"[call 1] bank entries: {len(harness.bank.load())}")

    print(f"\n[call 2] exact      -- {Q1!r}")
    harness.answer(context=CONTEXT, question=Q1)
    print(f"[call 2] bank entries: {len(harness.bank.load())}")

    # Diagnostic: show the semantic score the paraphrase gets against the bank.
    res = retrieve(PARAPHRASE, harness.bank, embed_backend, min_cosine=args.min_cosine)
    top = f"{res.scores[0]:.3f} -> {res.entries[0].get('question')!r}" if res.entries else "(no hit)"
    print(f"\n[call 3] paraphrase -- {PARAPHRASE!r}")
    print(f"[call 3] retrieve top: {top}")
    harness.answer(context=CONTEXT, question=PARAPHRASE)
    print(f"[call 3] bank entries: {len(harness.bank.load())}")

    return _report(harness.bank.load())


def _by_question(entries: list[dict], q: str) -> dict | None:
    return next((e for e in entries if e.get("question") == q), None)


def _report(entries: list[dict]) -> int:
    failures: list[str] = []

    e1 = _by_question(entries, Q1)
    ep = _by_question(entries, PARAPHRASE)

    if len(entries) != 2:
        failures.append(f"expected 2 entries (Q1 + paraphrase), got {len(entries)}")
    if e1 is None:
        failures.append("no entry for the original question Q1")
    else:
        if not str(e1.get("key_insight", "")).strip():
            failures.append("Q1 entry has empty key_insight")
        emb = e1.get("embedding") or []
        if len(emb) != 1024:
            failures.append(f"Q1 embedding dim should be 1024 (bge-m3), got {len(emb)}")
        uc = e1.get("stats", {}).get("use_count", 0)
        if uc != 2:
            failures.append(
                f"Q1 use_count should be 2 (exact hit + paraphrase hit), got {uc}"
            )
    if ep is None:
        failures.append("paraphrase was not distilled into its own entry")

    print("\n" + "=" * 60)
    if failures:
        print("TIER-2 SMOKE TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("TIER-2 SMOKE TEST PASSED (real semantic retrieval via bge-m3):")
    print("  - 2 entries (Q1 learned, paraphrase learned as distinct id)")
    print("  - Q1 embedding is 1024-dim (bge-m3)")
    print("  - Q1.use_count == 2: matched by an EXACT re-ask AND a paraphrase")
    print(f"  - Q1 key_insight: {e1.get('key_insight')!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
