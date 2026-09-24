"""Backend abstraction for Mem0 Platform and OSS modes."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from contextlib import closing, suppress
from typing import Any, Callable

logger = logging.getLogger(__name__)

# A hop restart (host reboot, OOM kill, `docker restart`, a Coolify recreate of that one service)
# opens a window the start-order dependency cannot cover: the API is still serving, mem0 accepts an
# `infer=true` POST /memories, the dial to the extraction hop fails, and the server answers
# `502 provider_unavailable` in ~0.1 s. Nothing retried it, so the turn was lost outright.
# Retry the extraction write in place (it already runs on the background sync thread, so the
# user-facing turn is never blocked); the `infer=false` fast path is single-shot and unchanged.
_SYNC_RETRY_ATTEMPTS = 3  # 1 try + 2 retries
_SYNC_RETRY_BACKOFF_SECS = (2.0, 5.0)
_TRANSIENT_STATUS = (502, 503)
_PROVIDER_UNAVAILABLE = "provider_unavailable"

# The extraction (`infer=true`) write is the one mem0 call whose latency is set by an LLM hop
# (app -> go-rotation -> the model), and it is the one call that could silently lose a turn: the flat
# 30 s client timeout sat INSIDE the live distribution, so extractions the server was still
# completing were cut off client-side (measured 2026-09-23: median 3.7 s, p90 9.8 s, p99 27.6 s, max
# 35.8 s over 375 calls in 2 h 49 m from go-rotation's own log; 8 turns lost to `timeout` that day).
# The server's own budget for that leg is MEM0_LLM_TIMEOUT=90, so any client budget below 90 s can
# only ever cut off work the server would have finished. 120 s = that 90 s server budget + 30 s for
# the extraction semaphore queue, the embedding and the store write; ~3.4x the observed max, >12x
# p90. Reads/searches and the exact `infer=false` fast path keep the flat 30 s budget: they are
# user-facing and were never this loss class. The wait is on the background sync thread, never the
# turn loop, so a slow extraction still costs the user nothing.
_EXTRACT_READ_TIMEOUT_SECS = 120.0
_EXTRACT_TIMEOUT_SECS = 30.0  # connect budget for the extraction write; unchanged
_EXTRACT_TIMEOUT_RETRIES = 1  # a read timeout retries exactly once (see _extraction_retryable)


def _add_kwargs(user_id: str, agent_id: str, infer: bool, metadata: dict | None) -> dict[str, Any]:
    return {"user_id": user_id, "agent_id": agent_id, "infer": infer, **({"metadata": metadata} if metadata else {})}


def _unwrap_results(response: Any) -> list:
    """Normalize API response — extract results list from dict or pass through."""
    return response.get("results", []) if isinstance(response, dict) else response if isinstance(response, list) else []


def _is_transient_upstream(exc: BaseException, *, allow_timeout: bool = False) -> bool:
    """True only for a failure that provably did NOT store anything and may self-heal.

    Two cases, both narrow on purpose:
      * the server answered 502/503 — its own ``provider_unavailable`` classification, or a proxy's
        plain bad-gateway while a container is recreated: the request was answered, nothing stored;
      * the socket never connected (dead hop, connect timeout): nothing reached the server.
    A read timeout is deliberately NOT retried — the request may already have been stored, and a
    retry would then duplicate the memory. A 4xx is the caller's problem: never retried.

    ``allow_timeout`` exists for the EXTRACTION write only (``_extraction_retryable``): its 120 s
    budget is far outside the live distribution, so a timeout there means something is genuinely
    wrong upstream, and one bounded retry is worth more than a lost turn. It is never enabled for
    the ``infer=false`` fast path or a read.
    """
    import httpx

    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        if response.status_code not in _TRANSIENT_STATUS:
            return False
        try:
            code = (response.json() or {}).get("code")
        except Exception:  # a proxy's HTML error page carries no classified body
            return True
        return code in (None, _PROVIDER_UNAVAILABLE)
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return True
    return bool(allow_timeout and isinstance(exc, httpx.TimeoutException))


def _extraction_retryable() -> Callable[[BaseException], bool]:
    """Predicate for the extraction write: transient upstream errors as before, plus ONE timeout.

    A timed-out extraction may already have landed server-side, so the timeout retry is bounded to a
    single attempt (a duplicate is recoverable — the replay worker checks the tenant first — while a
    silently dropped turn is not). Every attempt that reaches the server is still counted by
    ``GET /metrics``, and the caller records the turn if the retry also fails.
    """
    timeouts_allowed = _EXTRACT_TIMEOUT_RETRIES

    def predicate(exc: BaseException) -> bool:
        nonlocal timeouts_allowed
        if _is_transient_upstream(exc):
            return True
        if _is_transient_upstream(exc, allow_timeout=True) and timeouts_allowed > 0:
            timeouts_allowed -= 1
            return True
        return False

    return predicate


def _retry_transient(call, *, attempts: int | None = None, backoff: tuple | None = None, sleep=time.sleep, log=logger, retryable=None):
    """Run ``call`` while it fails transiently, then re-raise the last error.

    The final error still propagates, so the memory plugin's circuit breaker and its
    ``Mem0 sync failed`` warning keep firing exactly as before: a client-side retry hides no
    outage — each attempt that reaches the server is still counted by ``GET /metrics``.
    """
    attempts = _SYNC_RETRY_ATTEMPTS if attempts is None else max(1, attempts)
    backoff = _SYNC_RETRY_BACKOFF_SECS if backoff is None else backoff
    retryable = retryable or _is_transient_upstream
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts or not retryable(exc):
                # Stamp the failure with how many attempts it took, so the drop record the caller
                # writes can say whether the one bounded retry was used (or even issued) - an
                # operator reading the ledger wants "the retry failed too" vs "no retry was made".
                with suppress(Exception):
                    setattr(exc, "mem0_attempts", attempt)
                raise
            delay = backoff[min(attempt - 1, len(backoff) - 1)]
            log.info("Mem0 extraction write hit a transient upstream error (%s); retrying in %.1fs (%d of %d retries)", exc, delay, attempt, attempts - 1)
            sleep(delay)
    raise RuntimeError("unreachable: _retry_transient with attempts >= 1 always returns or raises")



class Mem0Backend(ABC):
    """Unified interface over Platform (MemoryClient), self-hosted (HTTP) and OSS (Memory) backends.
    update()/delete() are template methods: subclasses implement raw ``_update``/``_delete``."""

    @abstractmethod
    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]: ...
    @abstractmethod
    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict: ...
    @abstractmethod
    def _update(self, memory_id: str, text: str) -> None: ...
    @abstractmethod
    def _delete(self, memory_id: str) -> None: ...

    def update(self, memory_id: str, text: str) -> dict:
        self._update(memory_id, text)
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id: str) -> dict:
        self._delete(memory_id)
        return {"result": "Memory deleted.", "memory_id": memory_id}

    def close(self) -> None:
        pass


class PlatformBackend(Mem0Backend):
    """Wraps mem0.MemoryClient for Mem0 Platform (cloud API)."""

    def __init__(self, api_key: str):
        from mem0 import MemoryClient
        self._client = MemoryClient(api_key=api_key)

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        return _unwrap_results(self._client.search(query, filters=filters, top_k=top_k, rerank=rerank))

    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict:
        return self._client.add(messages, **_add_kwargs(user_id, agent_id, infer, metadata))

    def _update(self, memory_id: str, text: str) -> None:
        self._client.update(memory_id=memory_id, text=text)

    def _delete(self, memory_id: str) -> None:
        self._client.delete(memory_id=memory_id)


class SelfHostedBackend(Mem0Backend):
    """Direct HTTP backend for a self-hosted Mem0 server (the FastAPI ``server/``).
    mem0.MemoryClient is hardwired to the cloud API (``Authorization: Token``, ``GET /v1/ping/`` in ``__init__``),
    so this speaks the server's real contract: ``X-API-Key`` auth and the ``/memories`` / ``/search`` routes."""

    def __init__(self, api_key: str, host: str, transport=None):
        import httpx
        headers = {"Content-Type": "application/json", **({"X-API-Key": api_key} if api_key else {})}  # key omitted only for AUTH_DISABLED servers
        # Connect-level retries keep one dropped SYN from counting toward the breaker. ``transport`` is injectable for tests.
        self._client = httpx.Client(base_url=host.rstrip("/"), headers=headers, timeout=30.0, transport=transport or httpx.HTTPTransport(retries=2))
        # The extraction write gets its own READ budget (see _EXTRACT_READ_TIMEOUT_SECS); everything
        # else - search, update, delete, the exact `infer=false` add - keeps the flat 30s above.
        self._extract_timeout = httpx.Timeout(_EXTRACT_TIMEOUT_SECS, read=_EXTRACT_READ_TIMEOUT_SECS)

    def _json(self, method: str, path: str, *, timeout=None, **kwargs) -> Any:
        resp = self._client.request(method, path, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        # rerank is forwarded: the server's /search accepts it and reranks when a reranker is
        # configured (a server without one treats it as a no-op). user_id belongs in filters
        # (top-level is deprecated).
        payload = {"query": query, "top_k": top_k, **({"filters": filters} if filters else {})}
        if rerank:
            payload["rerank"] = True
        return _unwrap_results(self._json("POST", "/search", json=payload))

    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict:
        payload = {"messages": messages, **_add_kwargs(user_id, agent_id, infer, metadata)}
        if not infer:
            return self._json("POST", "/memories", json=payload)  # explicit write: single-shot, unchanged
        return _retry_transient(
            lambda: self._json("POST", "/memories", json=payload, timeout=self._extract_timeout),
            retryable=_extraction_retryable(),
        )

    def _update(self, memory_id: str, text: str) -> None:
        self._json("PUT", f"/memories/{memory_id}", json={"text": text})

    def _delete(self, memory_id: str) -> None:
        self._json("DELETE", f"/memories/{memory_id}")

    def close(self) -> None:
        with suppress(Exception):
            self._client.close()


_DIRECT_OPENAI_PROVIDER = "hermes_openai"
_DIRECT_OPENAI_CLASS_PATH = "plugins.memory.mem0._openai_llm.DirectOpenAILLM"


def _register_direct_openai_provider() -> None:
    """Register Hermes' OpenAI-only Mem0 LLM provider once per factory."""
    from mem0.configs.llms.openai import OpenAIConfig
    from mem0.utils.factory import LlmFactory
    provider_map = getattr(LlmFactory, "provider_to_class", None)
    register_provider = getattr(LlmFactory, "register_provider", None)
    if not isinstance(provider_map, dict) or not callable(register_provider):
        raise RuntimeError("mem0 LlmFactory does not support the provider registration required for the Hermes OpenAI OSS backend")
    if provider_map.get(_DIRECT_OPENAI_PROVIDER) != (_DIRECT_OPENAI_CLASS_PATH, OpenAIConfig):
        register_provider(_DIRECT_OPENAI_PROVIDER, _DIRECT_OPENAI_CLASS_PATH, OpenAIConfig)


class OSSBackend(Mem0Backend):
    """Wraps mem0.Memory for self-hosted (OSS) mode."""

    def __init__(self, oss_config: dict):
        import os
        from mem0 import Memory
        from ._oss_providers import EMBEDDER_PROVIDERS, KNOWN_DIMS, LLM_PROVIDERS

        def _provider_block(name: str, registry: dict) -> dict:
            """Copy of oss_config[name] with the legacy ``api_base`` key mapped to the provider's canonical base-URL key."""
            block = dict(oss_config[name])
            provider_config = dict(block.get("config", {}))
            legacy_base = provider_config.pop("api_base", None)
            canonical_key = registry.get(str(block.get("provider") or "").strip().lower(), {}).get("base_url_key")
            if legacy_base and canonical_key:
                provider_config.setdefault(canonical_key, legacy_base)
            block["config"] = provider_config
            return block

        vector_store = dict(oss_config["vector_store"])
        vs_config = dict(vector_store.get("config", {}))
        if "path" in vs_config:
            vs_config["path"] = os.path.expanduser(vs_config["path"])
        embedder_config = oss_config.get("embedder", {}).get("config", {})
        dims = embedder_config.get("embedding_dims") or KNOWN_DIMS.get(embedder_config.get("model", ""))
        if dims:
            vs_config["embedding_model_dims"] = dims
            self._recreate_collection_if_dims_changed(vector_store.get("provider", "qdrant"), vs_config, dims)
        vector_store["config"] = vs_config
        config = {"vector_store": vector_store, "llm": _provider_block("llm", LLM_PROVIDERS), "embedder": _provider_block("embedder", EMBEDDER_PROVIDERS), "version": "v1.1"}
        if str(config["llm"].get("provider") or "").strip().lower() == "openai":
            # mem0 validates LlmConfig.provider before its factory lookup: build the supported OpenAI config, then swap the provider.
            _register_direct_openai_provider()
            from mem0.configs.base import MemoryConfig
            memory_config = MemoryConfig(**config)
            try:
                memory_config.llm.provider = _DIRECT_OPENAI_PROVIDER
            except (AttributeError, TypeError) as exc:
                raise RuntimeError("mem0 MemoryConfig does not expose a mutable llm.provider for the Hermes OpenAI OSS backend") from exc
            self._memory = Memory(memory_config)
        else:
            self._memory = Memory.from_config(config)

    @staticmethod
    def _recreate_collection_if_dims_changed(provider: str, vs_config: dict, expected_dims: int) -> None:
        """Delete stale vector collection when embedding dimensions change."""
        collection_name = vs_config.get("collection_name", "mem0")
        with suppress(Exception):
            if provider == "qdrant":
                from qdrant_client import QdrantClient
                path, url = vs_config.get("path"), vs_config.get("url")
                if path:
                    client = QdrantClient(path=path)
                elif url:
                    client = QdrantClient(url=url, api_key=vs_config.get("api_key"))
                else:
                    return
                with closing(client):
                    if not client.collection_exists(collection_name):
                        return
                    vectors = client.get_collection(collection_name).config.params.vectors
                    # Named-vector collections expose a dict; unnamed expose an object with .size.
                    if isinstance(vectors, dict):
                        vectors = next(iter(vectors.values()), None)
                    current_dims = getattr(vectors, "size", None)
                    if current_dims is not None and current_dims != expected_dims:
                        client.delete_collection(collection_name)
            elif provider == "pgvector":
                import psycopg2
                from psycopg2 import sql as pgsql
                conn_params = {k: vs_config[k] for k in ("host", "port", "user", "password", "dbname", "sslmode") if vs_config.get(k)}
                with closing(psycopg2.connect(**conn_params)) as conn:
                    conn.autocommit = True
                    with closing(conn.cursor()) as cur:
                        cur.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'vector'", (collection_name,))
                        row = cur.fetchone()
                        if row and row[0] > 0 and row[0] != expected_dims:
                            cur.execute(pgsql.SQL("DROP TABLE IF EXISTS {}").format(pgsql.Identifier(collection_name)))

    def search(self, query: str, *, filters: dict, top_k: int = 10, rerank: bool = False) -> list[dict]:
        # OSS runs in-process, so Memory.search takes the flag directly.
        return _unwrap_results(self._memory.search(query, filters=filters, top_k=top_k, rerank=rerank))

    def add(self, messages: list, *, user_id: str, agent_id: str, infer: bool = False, metadata: dict | None = None) -> dict:
        return self._memory.add(messages, **_add_kwargs(user_id, agent_id, infer, metadata))

    def _update(self, memory_id: str, text: str) -> None:
        self._memory.update(memory_id, data=text)

    def _delete(self, memory_id: str) -> None:
        self._memory.delete(memory_id)

    def close(self):
        with suppress(Exception):
            telemetry = getattr(self._memory, "telemetry", None)
            if telemetry and hasattr(telemetry, "posthog"):
                with suppress(Exception):
                    telemetry.posthog.shutdown()
            vs = getattr(self._memory, "vector_store", None)
            # Memory, then its vector store, then the store's raw client; the first failure aborts the chain.
            for obj in filter(None, (self._memory, vs, getattr(vs, "client", None))):
                if hasattr(obj, "close"):
                    obj.close()
