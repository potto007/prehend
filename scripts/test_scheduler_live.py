"""Live integration test for the priority request scheduler against llama-server.

Requires llama-server running with --kv-unified and the model loaded
(POST /models/load - --no-models-autoload disables load-on-chat-request).
Restart the server between runs: prompt-cache prefix reuse from a previous
run masks KV pressure.

Tests:
  smoke      - single completion, scheduler admits immediately
  contention - unique concurrent requests that collectively exceed the unified
               KV pool. The server mass-fails every in-flight request with
               500 "Context size has been exceeded."; the client must retry
               each at p1 (solo execution) and all must succeed.
  toolarge   - single request bigger than the entire pool, verifying clean
               failure after exactly one p1 retry
"""

import asyncio
import hashlib
import logging
import multiprocessing
import os
import sys
import time

import openai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prehend.clients.openai import OpenAIClient, _is_context_contention
from prehend.clients.scheduler import RequestScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

BASE_URL = "http://127.0.0.1:8080/v1"
MODEL = "gemma-4-12b-it-cpt-sft-kb-v2-slerp"
PARALLEL = 8


def make_client(scheduler=None):
    return OpenAIClient(
        base_url=BASE_URL,
        model_name=MODEL,
        api_key="not-needed",
        scheduler=scheduler,
    )


def make_prompt(target_tokens, uniq=None):
    """Build a prompt of approximately target_tokens tokens.

    Each repetition of the filler is ~11 tokens. We add a short instruction
    at the end so the model generates a brief answer (not a verbatim repeat).

    uniq: if set, the prompt is made unique FROM THE FIRST TOKEN so that
    llama-server's prompt-cache prefix reuse cannot share KV across requests
    (identical prompts made earlier load tests meaningless - later requests
    reused the first request's KV and needed almost no pool space).
    """
    filler = "The quick brown fox jumps over the lazy dog. "
    reps = max(1, target_tokens // 11)
    body = filler * reps
    if uniq is not None:
        # unique header first + per-request salt sprinkled through the body
        words = body.split(". ")
        salted = ". ".join(f"{w} ({uniq}-{j})" if j % 20 == 0 else w for j, w in enumerate(words))
        return (
            f"Document {uniq}: read the following text carefully, then answer: what is 2+2?\n\n"
            + salted
            + "\n\nAnswer with just the number:"
        )
    return (
        "Read the following text carefully, then answer: what is 2+2?\n\n"
        + body
        + "\n\nAnswer with just the number:"
    )


def smoke_test():
    """Single completion with scheduler - verify normal operation is unchanged."""
    print("\n=== SMOKE TEST: single completion with scheduler ===")
    scheduler = RequestScheduler(max_concurrent=PARALLEL)
    client = make_client(scheduler)
    t0 = time.time()
    result = client.completion("What is 2+2? Answer in one word.")
    dt = time.time() - t0
    print(f"Response ({dt:.1f}s): {result[:200]}")
    print(f"Scheduler active: {scheduler.active} (should be 0)")
    assert scheduler.active == 0, "Slot leak!"
    print("PASSED\n")


async def contention_test(n_concurrent=12, tokens_per_prompt=12000, max_concurrent=None):
    """Blast concurrent requests to trigger context contention and p1 retries.

    With unified KV (pool = 65536 tokens) and prompts of ~12k tokens each,
    requests that collectively exceed the pool will get 400 contention errors.
    The scheduler should retry those at p1 (exclusive access) and they should
    succeed.

    Set max_concurrent > PARALLEL to intentionally over-subscribe and force
    contention, simulating a misconfiguration or bursty load.
    """
    mc = max_concurrent or PARALLEL
    print(f"\n=== CONTENTION TEST: {n_concurrent} unique requests, ~{tokens_per_prompt} tok each, max_concurrent={mc} ===")
    print("(exercises the shipped OpenAIClient retry path)")
    scheduler = RequestScheduler(max_concurrent=mc)
    client = make_client(scheduler)

    # Count the client's contention retries by watching its log line.
    retry_log_count = 0

    class _RetryCounter(logging.Handler):
        def emit(self, record):
            nonlocal retry_log_count
            if "retrying at p1" in record.getMessage():
                retry_log_count += 1

    logging.getLogger("prehend.clients.openai").addHandler(_RetryCounter())

    async def do_one(i):
        prompt = make_prompt(tokens_per_prompt, uniq=i)
        t0 = time.time()
        try:
            result = await client.acompletion(prompt)
            dt = time.time() - t0
            print(f"  [{i}] OK ({dt:.1f}s): {(result or '')[:60]!r}")
            return ("ok", dt)
        except Exception as e:
            dt = time.time() - t0
            print(f"  [{i}] FAIL ({dt:.1f}s): {type(e).__name__}: {str(e)[:120]}")
            return ("fail", dt)

    t0 = time.time()
    results = await asyncio.gather(*[do_one(i) for i in range(n_concurrent)])
    total = time.time() - t0

    oks = sum(1 for r, _ in results if r == "ok")
    fails = sum(1 for r, _ in results if r == "fail")

    print(f"\nResults: {oks} ok, {fails} failed")
    print(f"Contention retries at p1 (from client log): {retry_log_count}")
    print(f"Total wall time: {total:.1f}s")
    print(f"Scheduler active: {scheduler.active} (should be 0)")
    assert scheduler.active == 0, "Slot leak!"
    print()


async def mixed_test(n_holders=4, holder_tokens=12000, big_tokens=35000):
    """Reproduce the original incident: a big request arrives while other slots
    hold most of the unified KV pool.

    Holders: n_holders requests of ~holder_tokens each, generating long outputs
    so they occupy KV for a sustained window. Big request: ~big_tokens prompt,
    fits in the total ctx (65536) but NOT in the free remainder while holders
    are active -> server 400s with exceed_context_size_error -> client retries
    at p1, which drains the pool and succeeds.
    """
    print(f"\n=== MIXED TEST: {n_holders} holders ~{holder_tokens} tok + 1 big ~{big_tokens} tok ===")
    scheduler = RequestScheduler(max_concurrent=PARALLEL)
    client = make_client(scheduler)

    holder_prompt = (
        make_prompt(holder_tokens)
        + "\nAfter answering, write a detailed 2000-word essay about foxes and dogs."
    )
    big_prompt = make_prompt(big_tokens)

    async def holder(i):
        t0 = time.time()
        try:
            result = await client.acompletion(holder_prompt)
            print(f"  [holder {i}] OK ({time.time()-t0:.1f}s): {len(result or '')} chars")
            return "ok"
        except Exception as e:
            print(f"  [holder {i}] FAIL ({time.time()-t0:.1f}s): {str(e)[:100]}")
            return "fail"

    async def big():
        # Let holders get admitted and start prefill first
        await asyncio.sleep(8)
        t0 = time.time()
        try:
            result = await client.acompletion(big_prompt)
            print(f"  [big] OK ({time.time()-t0:.1f}s): {(result or '')[:80]}")
            return "ok"
        except Exception as e:
            print(f"  [big] FAIL ({time.time()-t0:.1f}s): {str(e)[:150]}")
            return "fail"

    t0 = time.time()
    results = await asyncio.gather(*([holder(i) for i in range(n_holders)] + [big()]))
    print(f"\nTotal wall time: {time.time()-t0:.1f}s")
    print(f"Results: {results}")
    print(f"Scheduler active: {scheduler.active} (should be 0)")
    assert scheduler.active == 0, "Slot leak!"
    print("Watch the log above for 'context contention (...), retrying at p1'")
    print()


async def toolarge_test():
    """Single request bigger than the entire pool - should fail cleanly after one p1 retry."""
    print("\n=== TOO-LARGE TEST: single request > pool size ===")
    scheduler = RequestScheduler(max_concurrent=PARALLEL)
    client = make_client(scheduler)
    prompt = make_prompt(80000)
    t0 = time.time()
    try:
        result = await client.acompletion(prompt)
        dt = time.time() - t0
        print(f"  Unexpected success ({dt:.1f}s): {result[:80]}")
        print("UNEXPECTED - should have failed")
    except openai.BadRequestError as e:
        dt = time.time() - t0
        is_contention = _is_context_contention(e)
        print(f"  Failed as expected ({dt:.1f}s), contention={is_contention}")
        print(f"  Error: {str(e)[:120]}")
    print(f"Scheduler active: {scheduler.active} (should be 0)")
    assert scheduler.active == 0, "Slot leak!"
    print("PASSED\n")


def _multiproc_worker(idx, n_per_proc, toks, mc, coord_dir):
    """One OS process: own gate-equipped scheduler, unique prompts."""
    from prehend.clients.coordination import CrossProcessGate

    key = hashlib.sha256(BASE_URL.encode()).hexdigest()[:16]
    gate = CrossProcessGate(coord_dir, key)
    scheduler = RequestScheduler(max_concurrent=mc, gate=gate)
    client = make_client(scheduler)

    async def run():
        async def do_one(i):
            # Per-process uniq offset: prompts must be unique across ALL
            # processes or prefix-cache reuse masks KV pressure.
            uniq = idx * 10000 + i
            prompt = make_prompt(toks, uniq=uniq)
            t0 = time.time()
            try:
                result = await client.acompletion(prompt)
                print(f"  [p{idx}:{i}] OK ({time.time() - t0:.1f}s): {(result or '')[:40]!r}")
                return "ok"
            except Exception as e:
                print(f"  [p{idx}:{i}] FAIL ({time.time() - t0:.1f}s): {type(e).__name__}: {str(e)[:100]}")
                return "fail"

        results = await asyncio.gather(*[do_one(i) for i in range(n_per_proc)])
        assert scheduler.active == 0, "Slot leak!"
        return results.count("fail")

    sys.exit(min(asyncio.run(run()), 250))


def multiproc_test(n_procs=4, n_per_proc=4, toks=12000, mc=2,
                   coord_dir="/tmp/lm-repl-coord-live"):
    """N OS processes share one llama-server through the cross-process gate.

    Collectively oversubscribes the unified KV pool so the server mass-kills
    in-flight requests; every process's p1 retries must drain GLOBALLY (the
    other processes' traffic included) and succeed. Success: zero failures.
    """
    print(f"\n=== MULTIPROC TEST: {n_procs} procs x {n_per_proc} reqs, "
          f"~{toks} tok, mc={mc}/proc, dir={coord_dir} ===")
    os.makedirs(coord_dir, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=_multiproc_worker, args=(i, n_per_proc, toks, mc, coord_dir))
        for i in range(n_procs)
    ]
    t0 = time.time()
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    total_fails = sum(p.exitcode or 0 for p in procs)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")
    print(f"Total failures across processes: {total_fails}")
    print("PASSED" if total_fails == 0 else "FAILED")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if mode == "smoke":
        smoke_test()
    elif mode == "contention":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        toks = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
        mc = int(sys.argv[4]) if len(sys.argv) > 4 else None
        asyncio.run(contention_test(n, toks, mc))
    elif mode == "toolarge":
        asyncio.run(toolarge_test())
    elif mode == "multiproc":
        np_ = int(sys.argv[2]) if len(sys.argv) > 2 else 4
        npp = int(sys.argv[3]) if len(sys.argv) > 3 else 4
        toks = int(sys.argv[4]) if len(sys.argv) > 4 else 12000
        mc = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        multiproc_test(np_, npp, toks, mc)
    elif mode == "all":
        smoke_test()
        asyncio.run(contention_test())
        asyncio.run(toolarge_test())
    else:
        print(f"Usage: {sys.argv[0]} [smoke|contention|toolarge|multiproc|all] [n_procs|n_concurrent] [n_per_proc] [tokens_per_prompt] [max_concurrent]")
