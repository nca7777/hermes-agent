"""HTTP server that forwards OpenAI-compatible requests to a configured upstream.

A credential-attaching forwarder: request/response bodies are never mediated, logged, or rewritten.
The one shim: after a *clean* upstream EOF, a ``text/event-stream`` response that carried a terminal
``finish_reason`` or ``lastOne: true`` but omitted ``data: [DONE]`` gets a single ``[DONE]`` frame.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import secrets
import signal
import socket
from typing import Optional

try:
    import aiohttp
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential
from hermes_cli.proxy.sse_done import DONE_SSE_FRAME, SseDoneTracker, content_type_is_sse

logger = logging.getLogger(__name__)

# Stripped when forwarding upstream: ``host``/``content-length`` are recomputed by aiohttp,
# ``authorization`` is replaced with our bearer; adapters may also replace provider identity
# headers, and everything else passes through.
_HOP_BY_HOP_HEADERS = frozenset({
    "host", "content-length", "connection", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade", "authorization",
})
# aiohttp recomputes Content-Encoding/Content-Length on stream — let it.
_RESPONSE_DROP_HEADERS = _HOP_BY_HOP_HEADERS | {"content-encoding", "content-length"}

DEFAULT_PORT = 8645
DEFAULT_HOST = "127.0.0.1"
DOWNSTREAM_BEARER_ENV = "SUBSCRIPTION_PROXY_KEY"
MIN_DOWNSTREAM_BEARER_LENGTH = 32
# Mirrors api_server's MAX_REQUEST_BYTES (10 MB); client_max_size bounds every read path,
# including chunked bodies.
MAX_REQUEST_BYTES = 10_000_000


def _require_aiohttp() -> None:
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp is required for `hermes proxy`. Run `hermes setup` to install it.")


def _json_error(status: int, message: str, code: str = "proxy_error") -> "web.Response":
    """OpenAI-style error JSON response."""
    body = {"error": {"message": message, "type": code, "code": code}}
    return web.json_response(body, status=status)


def _filter_headers(headers, drop: frozenset = _HOP_BY_HOP_HEADERS) -> dict:
    """Strip hop-by-hop (+ auth) headers; ``drop`` widens the set for upstream responses."""
    return {key: value for key, value in headers.items() if key.lower() not in drop}


def _valid_downstream_bearer(request: "web.Request", expected: Optional[str]) -> bool:
    """Constant-time check of an optional downstream bearer secret."""
    if not expected:
        return True
    parts = request.headers.get("Authorization", "").split(" ")
    if len(parts) != 2:
        return False
    scheme, provided = parts
    if scheme.lower() != "bearer" or not provided or any(char.isspace() for char in provided):
        return False
    return secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _is_loopback_address(address: str) -> bool:
    """Classify an IPv4/IPv6 address, including IPv4-mapped IPv6."""
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    mapped = getattr(parsed, "ipv4_mapped", None)
    return bool(mapped.is_loopback if mapped is not None else parsed.is_loopback)


def _loopback_bind_target(host: str) -> Optional[str]:
    """Return a numeric loopback bind target, or ``None`` for any unsafe answer.

    Hostnames are resolved and accepted only when all of their IPv4/IPv6 results
    are loopback. Returning a numeric address lets ``run_server`` bind without a
    second DNS lookup that could change after validation.
    """
    candidate = str(host or "").strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    if _is_loopback_address(candidate):
        return candidate
    try:
        resolved = socket.getaddrinfo(candidate, None, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return None
    addresses = [
        sockaddr[0]
        for family, _socktype, _proto, _canonname, sockaddr in resolved
        if family in {socket.AF_INET, socket.AF_INET6}
    ]
    if not addresses or not all(_is_loopback_address(address) for address in addresses):
        return None
    return addresses[0]


def is_loopback_bind_host(host: str) -> bool:
    """Return whether every address represented by a bind host is loopback."""
    return _loopback_bind_target(host) is not None


def validate_bind_security(host: str, downstream_bearer: Optional[str]) -> str:
    """Validate bind auth and return the host that may safely be passed to aiohttp."""
    loopback_target = _loopback_bind_target(host)
    if loopback_target is not None:
        return loopback_target
    bearer = downstream_bearer or ""
    if len(bearer) < MIN_DOWNSTREAM_BEARER_LENGTH or any(char.isspace() for char in bearer):
        raise ValueError(
            f"Refusing non-loopback subscription proxy bind {host!r}: "
            f"{DOWNSTREAM_BEARER_ENV} must contain at least "
            f"{MIN_DOWNSTREAM_BEARER_LENGTH} non-whitespace characters."
        )
    return host


async def _open_upstream(
    request: "web.Request",
    rel_path: str,
    body: bytes,
    cred: UpstreamCredential,
    adapter: UpstreamAdapter,
):
    """Send the request upstream with ``cred``; returns ``(session, response)`` or
    ``(error_response, None)``."""
    upstream_url = f"{cred.base_url.rstrip('/')}{rel_path}"
    upstream_url_for_log = upstream_url.partition("?")[0]
    if request.query_string:  # preserved verbatim
        upstream_url = f"{upstream_url}?{request.query_string}"
    fwd_headers = _filter_headers(
        request.headers,
        _HOP_BY_HOP_HEADERS | adapter.replaced_request_headers,
    )
    fwd_headers.update(adapter.get_upstream_headers(cred))
    fwd_headers["Authorization"] = f"{cred.token_type} {cred.bearer}"
    logger.debug(
        "proxy: forwarding %s %s -> %s (body=%d bytes)",
        request.method,
        rel_path,
        upstream_url_for_log,
        len(body),
    )
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=300))
    except Exception as exc:  # pragma: no cover - aiohttp setup issue
        return _json_error(500, f"proxy session init failed: {exc}"), None
    try:
        upstream_resp = await session.request(
            request.method, upstream_url, data=body if body else None, headers=fwd_headers, allow_redirects=False
        )
    except RuntimeError as exc:
        await session.close()
        return _json_error(500, str(exc)), None
    except aiohttp.ClientError as exc:
        await session.close()
        logger.warning("proxy: upstream connection failed: %s", exc)
        return _json_error(502, f"upstream connection failed: {exc}", code="upstream_unreachable"), None
    except asyncio.TimeoutError:
        await session.close()
        return _json_error(504, "upstream request timed out", code="upstream_timeout"), None
    except Exception:
        await session.close()
        raise
    return session, upstream_resp


async def _stream_back(request: "web.Request", session, upstream_resp) -> "web.StreamResponse":
    """Relay status + filtered headers, then the body chunk-by-chunk, appending a missing SSE
    ``[DONE]`` only after a clean EOF."""
    resp = web.StreamResponse(
        status=upstream_resp.status, headers=_filter_headers(upstream_resp.headers, _RESPONSE_DROP_HEADERS)
    )
    await resp.prepare(request)
    done_tracker: Optional[SseDoneTracker] = None
    if content_type_is_sse(upstream_resp.headers):
        done_tracker = SseDoneTracker()
    try:
        async for chunk in upstream_resp.content.iter_any():
            if chunk:
                if done_tracker is not None:
                    done_tracker.feed(chunk)
                await resp.write(chunk)
        if done_tracker is not None and done_tracker.should_append_done():
            try:
                await resp.write(DONE_SSE_FRAME)
            except Exception as exc:  # client hung up at EOF — harmless
                logger.debug("proxy: DONE append skipped: %s", exc)
    except (aiohttp.ClientError, asyncio.CancelledError, OSError) as exc:
        if done_tracker is not None:
            done_tracker.mark_interrupted()
        logger.warning("proxy: streaming interrupted: %s", exc)
    finally:
        upstream_resp.release()
        await session.close()
    await resp.write_eof()
    return resp


def create_app(
    adapter: UpstreamAdapter,
    downstream_bearer: Optional[str] = None,
) -> "web.Application":
    """Build the aiohttp application bound to a specific upstream adapter.

    Every credential method is synchronous and blocking (Nous takes the 15s cross-process
    ``_auth_store_lock()`` and may POST a token refresh; Codex/xAI rotate key pools under locks),
    so they run via ``asyncio.to_thread`` — a contended lock or refresh must never freeze the
    single loop and every other in-flight streaming completion.
    """
    _require_aiohttp()
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    # AppKey: forward-compat with aiohttp versions that strip bare-string keys.
    app[web.AppKey("adapter", UpstreamAdapter)] = adapter

    async def handle_health(request: "web.Request") -> "web.Response":
        authenticated = await asyncio.to_thread(adapter.is_authenticated)
        return web.json_response({"status": "ok", "upstream": adapter.display_name, "authenticated": authenticated})

    async def handle_proxy(request: "web.Request") -> "web.StreamResponse":
        if not _valid_downstream_bearer(request, downstream_bearer):
            response = _json_error(
                401,
                "Missing or invalid downstream bearer token.",
                code="invalid_downstream_auth",
            )
            response.headers["WWW-Authenticate"] = "Bearer"
            return response
        rel_path = "/" + request.match_info.get("tail", "").lstrip("/")
        if rel_path not in adapter.allowed_paths:
            allowed = ", ".join(sorted(adapter.allowed_paths))
            return _json_error(
                404, f"Path /v1{rel_path} is not forwarded by this proxy. Allowed: {allowed}", code="path_not_allowed"
            )
        allowed_methods = adapter.allowed_methods(rel_path)
        if request.method not in allowed_methods:
            response = _json_error(
                405,
                f"Method {request.method} is not allowed for /v1{rel_path}.",
                code="method_not_allowed",
            )
            response.headers["Allow"] = ", ".join(sorted(allowed_methods))
            return response
        # Enforce aiohttp's application body cap before a potentially blocking credential
        # refresh. Invalid or oversized requests must not touch provider auth state.
        body = await request.read()
        try:
            cred = await asyncio.to_thread(adapter.get_credential)
        except Exception as exc:
            logger.warning("proxy: credential resolution failed: %s", exc)
            return _json_error(401, str(exc), code="upstream_auth_failed")
        session, upstream_resp = await _open_upstream(request, rel_path, body, cred, adapter)
        if upstream_resp is None:
            return session
        if upstream_resp.status in {401, 429}:
            # One-shot retry with a refreshed/rotated credential (Nous: unconditional refresh
            # POST under the auth lock; Codex/xAI: pool refresh/rotation).
            try:
                retry_cred = await asyncio.to_thread(
                    adapter.get_retry_credential, failed_credential=cred, status_code=upstream_resp.status
                )
            except Exception as exc:
                logger.warning("proxy: retry credential resolution failed: %s", exc)
                retry_cred = None
            if retry_cred is not None:
                upstream_resp.release()
                await session.close()
                session, upstream_resp = await _open_upstream(request, rel_path, body, retry_cred, adapter)
                if upstream_resp is None:
                    return session
        return await _stream_back(request, session, upstream_resp)

    app.router.add_get("/health", handle_health)  # never goes upstream
    app.router.add_route("*", "/v1/{tail:.*}", handle_proxy)  # forwards if the path is allowed
    return app


async def run_server(
    adapter: UpstreamAdapter,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    downstream_bearer: Optional[str] = None,
    shutdown_event: Optional[asyncio.Event] = None,
) -> None:
    """Run the proxy in the current event loop until shutdown_event is set."""
    _require_aiohttp()
    bind_host = validate_bind_security(host, downstream_bearer)
    app = create_app(adapter, downstream_bearer=downstream_bearer)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host=bind_host, port=port)
    await site.start()
    logger.info("proxy: listening on http://%s:%d/v1 -> %s", bind_host, port, adapter.display_name)
    stop_event = shutdown_event or asyncio.Event()
    if shutdown_event is None:  # we own the loop's lifetime → wire signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)  # windows-footgun: ok
            except NotImplementedError:
                pass  # Windows / restricted envs — Ctrl+C still raises KeyboardInterrupt
    try:
        await stop_event.wait()
    finally:
        logger.info("proxy: shutting down")
        await runner.cleanup()


__all__ = [
    "create_app",
    "run_server",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DOWNSTREAM_BEARER_ENV",
    "MIN_DOWNSTREAM_BEARER_LENGTH",
    "is_loopback_bind_host",
    "validate_bind_security",
    "AIOHTTP_AVAILABLE",
]
