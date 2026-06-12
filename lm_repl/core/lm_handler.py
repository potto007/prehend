"""
LMHandler - Routes LLM requests from the RLM process and environment subprocesses.

Uses a multi-threaded socket server. Protocol: 4-byte length prefix + JSON payload.
"""

import asyncio
import hashlib
import time
from pathlib import Path
from socketserver import StreamRequestHandler, ThreadingTCPServer
from threading import Thread

from lm_repl.clients.base_lm import BaseLM
from lm_repl.clients.coordination import CrossProcessGate
from lm_repl.clients.scheduler import RequestScheduler
from lm_repl.core.comms_utils import LMRequest, LMResponse, socket_recv, socket_send
from lm_repl.core.types import RLMChatCompletion, UsageSummary
from lm_repl.core.verifier import REJECTION_PREFIX, SubcallReview, SubcallVerifier


class LMRequestHandler(StreamRequestHandler):
    """Socket handler for LLM completion requests."""

    def handle(self):
        try:
            request_data = socket_recv(self.connection)
            if not isinstance(request_data, dict):
                response = LMResponse.error_response("Request must be a JSON object")
                self._safe_send(response)
                return

            request = LMRequest.from_dict(request_data)
            handler: LMHandler = self.server.lm_handler  # type: ignore

            if request.is_batched:
                # Batched request: process multiple prompts concurrently
                response = self._handle_batched(request, handler)
            elif request.prompt:
                # Single request: process one prompt
                response = self._handle_single(request, handler)
            else:
                response = LMResponse.error_response("Missing 'prompt' or 'prompts' in request.")

            self._safe_send(response)

        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            # Client disconnected - this is expected during parallel execution
            # when workers complete and close their sockets. Silently ignore.
            pass

        except Exception as e:
            # Try to send error response, but don't fail if socket is broken
            response = LMResponse.error_response(str(e))
            self._safe_send(response)

    def _safe_send(self, response: LMResponse) -> bool:
        """Send response, returning False if the socket is broken."""
        try:
            socket_send(self.connection, response.to_dict())
            return True
        except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError):
            # Client disconnected - silently ignore
            return False

    def _rejection_completion(
        self, prompt: str | dict, rejection: str, handler: "LMHandler", model: str | None
    ) -> RLMChatCompletion:
        """A vetoed call's result: the rejection string IS the response, so the
        orchestrator reads it in the REPL and adapts on its next iteration."""
        return RLMChatCompletion(
            root_model=model or handler.default_client.model_name,
            prompt=prompt,
            response=rejection,
            usage_summary=UsageSummary(model_usage_summaries={}),
            execution_time=0.0,
        )

    def _handle_single(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        """Handle a single prompt request."""
        rejection = handler.review_subcall(request.prompt)
        if rejection is not None:
            return LMResponse.success_response(
                chat_completion=self._rejection_completion(
                    request.prompt, rejection, handler, request.model
                )
            )

        client = handler.get_client(request.model, request.depth)

        start_time = time.perf_counter()
        content = client.completion(
            request.prompt, priority=request.priority, **handler.subcall_kwargs()
        )
        end_time = time.perf_counter()

        model_usage = client.get_last_usage()
        root_model = request.model or client.model_name
        usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})
        return LMResponse.success_response(
            chat_completion=RLMChatCompletion(
                root_model=root_model,
                prompt=request.prompt,
                response=content,
                usage_summary=usage_summary,
                execution_time=end_time - start_time,
            )
        )

    def _handle_batched(self, request: LMRequest, handler: "LMHandler") -> LMResponse:
        """Handle a batched prompts request using async for concurrency."""
        client = handler.get_client(request.model, request.depth)

        # Review per prompt: vetoed prompts get the rejection string as their
        # result, the rest execute normally.
        rejections = [handler.review_subcall(p) for p in request.prompts]
        to_run = [p for p, r in zip(request.prompts, rejections, strict=True) if r is None]

        start_time = time.perf_counter()

        subcall_kwargs = handler.subcall_kwargs()
        if getattr(client, "scheduler", None) is not None:
            # The client's RequestScheduler already bounds concurrency (and adds priority
            # ordering), so a semaphore on top would only fight its queue.
            async def run_one(prompt: str):
                return await client.acompletion(
                    prompt, priority=request.priority, **subcall_kwargs
                )
        else:
            sem = asyncio.Semaphore(handler.batch_max_concurrent)

            async def run_one(prompt: str):
                async with sem:
                    return await client.acompletion(
                        prompt, priority=request.priority, **subcall_kwargs
                    )

        async def run_all():
            tasks = [run_one(prompt) for prompt in to_run]
            return await asyncio.gather(*tasks)

        executed = iter(asyncio.run(run_all()) if to_run else [])
        results = [r if r is not None else next(executed) for r in rejections]
        end_time = time.perf_counter()

        total_time = end_time - start_time
        model_usage = client.get_last_usage()
        root_model = request.model or client.model_name
        usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})

        chat_completions = [
            RLMChatCompletion(
                root_model=root_model,
                prompt=prompt,
                response=content,
                usage_summary=usage_summary,
                execution_time=total_time / len(request.prompts),  # approximate per-prompt time
            )
            for prompt, content in zip(request.prompts, results, strict=True)
        ]

        return LMResponse.batched_success_response(chat_completions=chat_completions)


class ThreadingLMServer(ThreadingTCPServer):
    """Multi-threaded TCP server for LM requests."""

    daemon_threads = True
    allow_reuse_address = True


class LMHandler:
    """
    Handles all LM calls from the RLM main process and environment subprocesses.

    Uses a multi-threaded socket server for concurrent requests.
    Protocol: 4-byte big-endian length prefix + JSON payload.
    """

    def __init__(
        self,
        client: BaseLM,
        host: str = "127.0.0.1",
        port: int = 0,  # auto-assign available port
        other_backend_client: BaseLM | None = None,
        batch_max_concurrent: int = 16,
        scheduler_max_concurrent: int | None = None,
        scheduler_aging_interval: float | None = 30.0,
        scheduler_coordination_dir: str | Path | None = None,
        subcall_max_tokens: int | None = None,
        subcall_extra_body: dict | None = None,
        root_max_tokens: int | None = None,
        verifier: SubcallVerifier | None = None,
        verifier_root: str | None = None,
    ):
        self.default_client = client
        self.other_backend_client = other_backend_client
        self.clients: dict[str, BaseLM] = {}
        self.host = host
        self._server: ThreadingLMServer | None = None
        self._thread: Thread | None = None
        self._port = port
        self.batch_max_concurrent = batch_max_concurrent
        # Generation cap applied to every SUB-call served over the socket
        # (llm_query / llm_query_batched). Bounds runaway generations - a
        # degenerate greedy loop otherwise generates until the context fills.
        # Root orchestrator calls (LMHandler.completion) are NOT capped by
        # subcall_max_tokens. The client's completion() must accept max_tokens
        # when this is set.
        self.subcall_max_tokens = subcall_max_tokens
        # Request-body extras for every SUB-call, e.g. {"chat_template_kwargs":
        # {"enable_thinking": False}} so gemma sub-calls skip the thought
        # channel (2026-06-12: thinking-mode sub-calls ruminated in-channel to
        # the token cap and returned empty content). Root calls are unaffected
        # - the orchestrator reasons better with thinking on. The client's
        # completion() must accept extra_body when this is set.
        self.subcall_extra_body = subcall_extra_body
        # Generation cap for ROOT orchestrator calls. The forced final REDUCE
        # after iteration exhaustion is also a root call - uncapped, it ran
        # away to ~50K tokens on 2026-06-11 (n_tokens 65024 at deadline
        # cancel), eating 9 of a 10-minute ask. Set generously: real final
        # answers run a few thousand tokens; only runaways hit this.
        self.root_max_tokens = root_max_tokens
        # Strategy verifier: reviews every llm_query sub-call served over the
        # socket before it executes. verifier_root is the task the calling RLM
        # was given, so the whole-task-delegation rule has something to
        # compare against. None disables review (previous behavior).
        self.verifier = verifier
        self.verifier_root = verifier_root

        # One scheduler shared by every client that targets the same server, so the
        # priority queue (and p1 exclusivity) spans all traffic. Match
        # scheduler_max_concurrent to the server's slot count (llama-server --parallel).
        # None disables scheduling entirely (previous behavior).
        # scheduler_aging_interval: seconds of queue wait worth one priority level
        # (anti-starvation); None disables aging.
        # scheduler_coordination_dir: opt-in cross-process gate extending p1
        # exclusivity to other OS processes targeting the same base_url
        # (lock files keyed by sha256(base_url)). Requires the scheduler.
        if scheduler_coordination_dir is not None and scheduler_max_concurrent is None:
            raise ValueError(
                "scheduler_coordination_dir requires scheduler_max_concurrent "
                "(no scheduler, no cross-process gate)"
            )
        self.scheduler: RequestScheduler | None = None
        if scheduler_max_concurrent is not None:
            gate = None
            if scheduler_coordination_dir is not None:
                key_src = str(getattr(client, "base_url", None) or "default")
                server_key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
                gate = CrossProcessGate(scheduler_coordination_dir, server_key)
            self.scheduler = RequestScheduler(
                max_concurrent=scheduler_max_concurrent,
                aging_interval=scheduler_aging_interval,
                gate=gate,
            )
            for c in (client, other_backend_client):
                if c is not None and hasattr(c, "scheduler"):
                    c.scheduler = self.scheduler

        self.register_client(client.model_name, client)

    def register_client(self, model_name: str, client: BaseLM) -> None:
        """Register a client for a specific model name."""
        self.clients[model_name] = client

    def get_client(self, model: str | None = None, depth: int = 0) -> BaseLM:
        """Get client by model name or depth, or return default.

        Routing logic:
        - depth=0: use default_client (main backend)
        - depth=1: use other_backend_client if it exists, otherwise default_client
        - If model is specified and exists in clients, use that (overrides depth routing)
        """
        if model and model in self.clients:
            return self.clients[model]

        # Route based on depth
        if depth == 1 and self.other_backend_client is not None:
            return self.other_backend_client

        return self.default_client

    @property
    def port(self) -> int:
        """Get the actual port (useful when auto-assigned)."""
        if self._server:
            return self._server.server_address[1]
        return self._port

    @property
    def address(self) -> tuple[str, int]:
        """Get (host, port) tuple for connecting."""
        return (self.host, self.port)

    def start(self) -> tuple[str, int]:
        """Start the socket server in a background thread. Returns (host, port)."""
        if self._server is not None:
            return self.address

        self._server = ThreadingLMServer((self.host, self._port), LMRequestHandler)
        self._server.lm_handler = self  # type: ignore

        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

        return self.address

    def stop(self):
        """Stop the socket server."""
        if self._server:
            self._server.shutdown()
            self._server = None
            self._thread = None

    def subcall_kwargs(self) -> dict:
        """Extra completion() kwargs applied to socket-served sub-calls."""
        kwargs: dict = {}
        if self.subcall_max_tokens is not None:
            kwargs["max_tokens"] = self.subcall_max_tokens
        if self.subcall_extra_body is not None:
            kwargs["extra_body"] = self.subcall_extra_body
        return kwargs

    def _all_clients(self) -> list[BaseLM]:
        seen: dict[int, BaseLM] = {}
        for c in (self.default_client, self.other_backend_client, *self.clients.values()):
            if c is not None:
                seen[id(c)] = c
        return list(seen.values())

    def set_run_deadline(self, max_timeout: float | None) -> None:
        """Arm a wall-clock deadline on every client that supports one."""
        for c in self._all_clients():
            if hasattr(c, "set_deadline"):
                c.set_deadline(max_timeout)

    def cancel_inflight(self) -> None:
        """Abort in-flight and queued LM calls on every cancellable client.

        stop() only shuts the socket server down; request threads already blocked
        in a client call are daemon threads that would otherwise keep generating
        (and retrying) long after the run ended."""
        for c in self._all_clients():
            event = getattr(c, "cancel_event", None)
            if event is not None:
                event.set()

    def review_subcall(self, prompt: str | dict | None) -> str | None:
        """Run the strategy verifier over one llm_query sub-call. Returns the
        rejection string if vetoed, None if approved (or no verifier)."""
        if self.verifier is None or prompt is None:
            return None
        verdict = self.verifier.review(
            SubcallReview(
                kind="llm_query",
                prompt=prompt if isinstance(prompt, str) else str(prompt),
                root_prompt=self.verifier_root,
            )
        )
        if verdict.approved:
            return None
        return REJECTION_PREFIX + verdict.reason

    def completion(self, prompt: str, model: str | None = None) -> str:
        """Direct completion call (for main process use)."""
        if self.root_max_tokens is not None:
            return self.get_client(model).completion(prompt, max_tokens=self.root_max_tokens)
        return self.get_client(model).completion(prompt)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    def get_usage_summary(self) -> UsageSummary:
        """Get the usage summary for all clients, merged into a single dict."""
        merged = {}
        # Include default client
        default_summary = self.default_client.get_usage_summary()
        merged.update(default_summary.model_usage_summaries)
        # Include other backend client if it exists
        if self.other_backend_client is not None:
            other_summary = self.other_backend_client.get_usage_summary()
            merged.update(other_summary.model_usage_summaries)
        # Include all registered clients
        for client in self.clients.values():
            client_summary = client.get_usage_summary()
            merged.update(client_summary.model_usage_summaries)
        return UsageSummary(model_usage_summaries=merged)
