"""Tests for the `hermes proxy` subcommand and its upstream adapters."""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.proxy.adapters import ADAPTERS, get_adapter
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential
from hermes_cli.proxy.adapters.nous_portal import NousPortalAdapter
from hermes_cli.proxy.adapters.openai_codex import OpenAICodexAdapter
from hermes_cli.proxy.adapters.xai import XAIGrokAdapter
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


def _codex_jwt(account_id: str, marker: str) -> str:
    payload = {
        "exp": 4_102_444_800,
        "https://api.openai.com/auth": {
            "chatgpt_account_id": account_id,
            "chatgpt_data_residency": "us",
        },
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"e30.{encoded}.{marker}"


def _write_codex_auth_store(home: Path, access_token: str, refresh_token: str) -> bytes:
    payload = {
        "version": 1,
        "providers": {
            "openai-codex": {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                },
            },
        },
        "credential_pool": {
            "openai-codex": [{
                "id": f"codex-{refresh_token}",
                "label": "Codex",
                "auth_type": "oauth",
                "priority": 0,
                "source": "manual:device_code",
                "access_token": access_token,
                "refresh_token": refresh_token,
            }],
        },
    }
    raw = json.dumps(payload, indent=2).encode()
    (home / "auth.json").write_bytes(raw)
    return raw


def test_codex_adapter_always_resolves_default_home_pool(tmp_path, monkeypatch):
    """A→B→A profile scopes share the root proxy without reading or copying their tokens."""
    root = tmp_path / "default-home"
    profile_a = root / "profiles" / "a"
    profile_b = root / "profiles" / "b"
    for home in (root, profile_a, profile_b):
        home.mkdir(parents=True)
    root_access = _codex_jwt("acct-root", "root")
    _write_codex_auth_store(root, root_access, "root-refresh")
    profile_a_before = _write_codex_auth_store(
        profile_a, _codex_jwt("acct-a", "a"), "profile-a-refresh"
    )
    profile_b_before = _write_codex_auth_store(
        profile_b, _codex_jwt("acct-b", "b"), "profile-b-refresh"
    )
    monkeypatch.setenv("HERMES_HOME", str(profile_a))

    assert ADAPTERS["openai-codex"] is OpenAICodexAdapter
    adapter = get_adapter("openai-codex")
    observed = []
    for home in (profile_a, profile_b, profile_a):
        token = set_hermes_home_override(home)
        try:
            credential = adapter.get_credential()
            observed.append(credential.bearer)
            headers = adapter.get_upstream_headers(credential)
            assert headers["ChatGPT-Account-ID"] == "acct-root"
            assert headers["originator"] == "hermes-agent"
        finally:
            reset_hermes_home_override(token)

    assert observed == [root_access, root_access, root_access]
    assert (profile_a / "auth.json").read_bytes() == profile_a_before
    assert (profile_b / "auth.json").read_bytes() == profile_b_before


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# NousPortalAdapter
# ---------------------------------------------------------------------------


def _write_auth_store(hermes_home: Path, nous_state: Dict[str, Any]) -> Path:
    """Write an auth.json with the given nous state into a hermetic HERMES_HOME."""
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(json.dumps({
        "version": 1,
        "providers": {"nous": nous_state},
    }))
    return auth_path




def test_nous_adapter_concurrent_refresh_serialized(tmp_path, monkeypatch):
    """Two parallel get_credential() calls must serialize through the lock."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth_store(tmp_path, {
        "access_token": "a", "refresh_token": "r",
    })

    call_log: list = []
    in_flight = threading.Event()
    overlap_detected = threading.Event()
    counter = [0]
    counter_lock = threading.Lock()

    def serializing_refresh(**kwargs):
        # If another thread is already inside refresh, the lock is broken.
        if in_flight.is_set():
            overlap_detected.set()
        in_flight.set()
        try:
            call_log.append(threading.current_thread().ident)
            # Simulate refresh latency so any race window is exposed.
            import time
            time.sleep(0.05)
            with counter_lock:
                counter[0] += 1
                idx = counter[0]
            return {
                "api_key": f"key-{idx}",
                "expires_at": "2099-01-01T00:00:00Z",
                "base_url": "https://inference-api.nousresearch.com/v1",
            }
        finally:
            in_flight.clear()

    adapter = NousPortalAdapter()
    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(adapter.get_credential().bearer)
        except Exception as exc:  # pragma: no cover - shouldn't happen
            errors.append(exc)

    with patch(
        "hermes_cli.proxy.adapters.nous_portal.resolve_nous_runtime_credentials",
        side_effect=serializing_refresh,
    ):
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, f"workers errored: {errors}"
    assert len(results) == 3
    assert len(call_log) == 3
    assert not overlap_detected.is_set(), "refresh calls overlapped — lock is broken"
    assert all(r.startswith("key-") for r in results)


# ---------------------------------------------------------------------------
# XAIGrokAdapter
# ---------------------------------------------------------------------------


def _write_xai_pool_entry(
    hermes_home: Path,
    *,
    access_token: str = "xai-access-token",
    refresh_token: str = "xai-refresh-token",
    base_url: str = "https://api.x.ai/v1",
    source: str = "manual:xai_pkce",
) -> Path:
    """Write an xai-oauth pool entry into a hermetic HERMES_HOME."""
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {
            "xai-oauth": [
                {
                    "id": "xai123",
                    "label": "xai-test",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": source,
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "base_url": base_url,
                }
            ]
        },
    }))
    return auth_path


def test_xai_adapter_not_authenticated_when_no_pool_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {},
    }))
    assert not XAIGrokAdapter().is_authenticated()


def test_xai_adapter_retry_rotates_pool_entry_on_429(tmp_path, monkeypatch):
    """429 from xAI must rotate to the next pool entry, not attempt refresh.

    Pre-fix (#28932) ``get_retry_credential`` only fired on 401, so a 429
    rate-limit response flowed back to the client unchanged AND the
    rate-limited bearer stayed active for the next request — defeating
    the whole point of pool rotation.

    Post-fix: 429 lands on ``mark_exhausted_and_rotate`` (no refresh —
    that's irrelevant for rate limits), stamps the 1-hour cooldown
    via ``EXHAUSTED_TTL_429_SECONDS`` on the offending key, and
    returns the next available credential.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Two pool entries so rotation has somewhere to go.
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {
            "xai-oauth": [
                {
                    "id": "xai-first",
                    "label": "xai-first",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "manual:xai_pkce",
                    "access_token": "first-access-token",
                    "refresh_token": "first-refresh-token",
                    "base_url": "https://api.x.ai/v1",
                },
                {
                    "id": "xai-second",
                    "label": "xai-second",
                    "auth_type": "oauth",
                    "priority": 1,
                    "source": "manual:xai_pkce",
                    "access_token": "second-access-token",
                    "refresh_token": "second-refresh-token",
                    "base_url": "https://api.x.ai/v1",
                },
            ]
        },
    }))

    # Refresh must NOT be called on the 429 path — guard against
    # the fix accidentally trying to refresh-on-rate-limit.
    def _refresh_must_not_run(*args, **kwargs):
        raise AssertionError("refresh_xai_oauth_pure must not run on 429")

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _refresh_must_not_run)

    adapter = XAIGrokAdapter()
    failed = adapter.get_credential()
    assert failed.bearer == "first-access-token", "starting bearer should be the first entry"

    retry = adapter.get_retry_credential(
        failed_credential=failed,
        status_code=429,
    )

    assert retry is not None, "429 must rotate to next pool entry"
    assert retry.bearer == "second-access-token", (
        f"expected rotation to second entry, got {retry.bearer!r}"
    )


# ---------------------------------------------------------------------------
# Server: path filtering + forwarding
#
# We run the proxy AND a fake upstream as real aiohttp servers on ephemeral
# ports. Avoids pytest-aiohttp's fixtures (extra dependency for one test file).
# ---------------------------------------------------------------------------

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

from hermes_cli.proxy.server import (  # noqa: E402
    MAX_REQUEST_BYTES,
    MIN_DOWNSTREAM_BEARER_LENGTH,
    create_app,
    is_loopback_bind_host,
    run_server,
    validate_bind_security,
)


class FakeAdapter(UpstreamAdapter):
    """A test adapter that returns a fixed credential without touching disk."""

    def __init__(self, base_url: str, bearer: str = "test-bearer",
                 allowed=None, raise_on_credential=False,
                 retry_bearer: str | None = None):
        self._base_url = base_url
        self._bearer = bearer
        self._allowed = frozenset(allowed or ["/chat/completions"])
        self._raise = raise_on_credential
        self._retry_bearer = retry_bearer
        self.calls = 0
        self.retry_calls = 0

    @property
    def name(self): return "fake"

    @property
    def display_name(self): return "Fake Provider"

    @property
    def allowed_paths(self): return self._allowed

    def is_authenticated(self): return True

    def get_credential(self):
        self.calls += 1
        if self._raise:
            raise RuntimeError("simulated auth failure")
        return UpstreamCredential(
            bearer=self._bearer, base_url=self._base_url,
            expires_at="2099-01-01T00:00:00Z",
        )

    def get_retry_credential(self, *, failed_credential, status_code):
        _ = failed_credential
        self.retry_calls += 1
        if status_code != 401 or not self._retry_bearer:
            return None
        return UpstreamCredential(
            bearer=self._retry_bearer,
            base_url=self._base_url,
            expires_at="2099-01-01T00:00:00Z",
        )


class FakeCodexAdapter(FakeAdapter):
    @property
    def replaced_request_headers(self):
        return frozenset({
            "user-agent",
            "originator",
            "chatgpt-account-id",
            "x-openai-internal-codex-residency",
        })

    def get_upstream_headers(self, credential):
        from agent.codex_headers import codex_cloudflare_headers

        return codex_cloudflare_headers(credential.bearer)


async def _start_runner(app: "web.Application"):
    """Spin up an aiohttp app on an ephemeral localhost port. Returns (runner, base_url)."""
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    sockets = list(site._server.sockets)  # type: ignore[union-attr]
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _build_fake_upstream(captured: Dict[str, Any]) -> "web.Application":
    async def echo(request):
        body = await request.read()
        captured["requests"].append({
            "method": request.method,
            "path": request.path,
            "query": request.query_string,
            "auth": request.headers.get("Authorization"),
            "body": body.decode("utf-8") if body else "",
        })
        return web.json_response({"echoed": True, "path": request.path})

    async def sse(request):
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        for chunk in [b"data: hello\n\n", b"data: world\n\n", b"data: [DONE]\n\n"]:
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_route("*", "/v1/chat/completions", echo)
    app.router.add_route("*", "/v1/embeddings", echo)
    app.router.add_route("*", "/v1/models", echo)
    app.router.add_route("*", "/v1/sse", sse)
    return app


def _build_retrying_fake_upstream(captured: Dict[str, Any]) -> "web.Application":
    async def maybe_unauthorized(request):
        body = await request.read()
        auth = request.headers.get("Authorization")
        captured["requests"].append({
            "method": request.method,
            "path": request.path,
            "auth": auth,
            "body": body.decode("utf-8") if body else "",
        })
        if auth == "Bearer jwt-bearer":
            return web.json_response({"error": "bad token"}, status=401)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_route("*", "/v1/chat/completions", maybe_unauthorized)
    return app


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.42.0.9", "::1", "[::1]", "::ffff:127.0.0.1"],
)
def test_loopback_bind_classifier_accepts_ipv4_and_ipv6_literals(host):
    assert is_loopback_bind_host(host)


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.168.1.10", "::", "2001:db8::1", "::ffff:192.168.1.10"],
)
def test_loopback_bind_classifier_rejects_non_loopback_literals(host):
    assert not is_loopback_bind_host(host)


def test_loopback_bind_classifier_resolves_hostnames_fail_closed(monkeypatch):
    answers = {
        "localhost.test": ["127.0.0.1", "::1"],
        "remote.test": ["203.0.113.8"],
        "mixed.test": ["127.0.0.1", "203.0.113.8"],
    }

    def fake_getaddrinfo(host, _port, *, type):
        assert type == socket.SOCK_STREAM
        if host == "missing.test":
            raise socket.gaierror("not found")
        return [
            (
                socket.AF_INET6 if ":" in address else socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (address, 0),
            )
            for address in answers[host]
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert is_loopback_bind_host("localhost.test")
    assert validate_bind_security("localhost.test", None) == "127.0.0.1"
    assert not is_loopback_bind_host("remote.test")
    assert not is_loopback_bind_host("mixed.test")
    assert not is_loopback_bind_host("missing.test")


def test_non_loopback_bind_requires_strong_downstream_bearer():
    with pytest.raises(ValueError, match="Refusing non-loopback"):
        validate_bind_security("0.0.0.0", None)
    with pytest.raises(ValueError, match="at least 32 non-whitespace"):
        validate_bind_security("0.0.0.0", "short")
    with pytest.raises(ValueError, match="at least 32 non-whitespace"):
        validate_bind_security("0.0.0.0", "x" * MIN_DOWNSTREAM_BEARER_LENGTH + " ")

    validate_bind_security("0.0.0.0", "x" * MIN_DOWNSTREAM_BEARER_LENGTH)
    validate_bind_security("127.0.0.1", None)


def test_server_boundary_refuses_unsafe_remote_bind():
    adapter = FakeAdapter("http://127.0.0.1:1/v1")
    with pytest.raises(ValueError, match="Refusing non-loopback"):
        asyncio.run(run_server(adapter, host="::", downstream_bearer=None))






def test_server_strips_client_auth_header():
    """The client's Authorization header MUST NOT reach the upstream."""
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={},
                    headers={"Authorization": "Bearer SHOULD_NOT_LEAK"},
                ) as resp:
                    await resp.read()
            assert captured["requests"][0]["auth"] == "Bearer ours"
            assert "SHOULD_NOT_LEAK" not in captured["requests"][0]["auth"]
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_downstream_bearer_auth_gates_forwarding_and_keeps_health_open(caplog):
    """Missing/wrong bearers never resolve credentials; the exact bearer forwards once."""
    downstream_secret = "container-only-secret"
    caplog.set_level("DEBUG", logger="hermes_cli.proxy.server")

    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="upstream-secret")
        proxy_runner, proxy_base = await _start_runner(
            create_app(adapter, downstream_bearer=downstream_secret)
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{proxy_base}/health") as response:
                    assert response.status == 200
                    assert (await response.json())["status"] == "ok"

                for headers in (
                    {},
                    {"Authorization": "Bearer wrong-secret"},
                    {"Authorization": f"Basic {downstream_secret}"},
                    {"Authorization": f"Bearer{downstream_secret}"},
                    {"Authorization": f"Bearer  {downstream_secret}"},
                    {"Authorization": f"Bearer\t{downstream_secret}"},
                ):
                    async with session.post(
                        f"{proxy_base}/v1/chat/completions",
                        json={"must_not_forward": True},
                        headers=headers,
                    ) as response:
                        body = await response.json()
                        assert response.status == 401
                        assert response.headers["WWW-Authenticate"] == "Bearer"
                        assert body["error"]["code"] == "invalid_downstream_auth"

                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"forward": True},
                    headers={"Authorization": f"Bearer {downstream_secret}"},
                ) as response:
                    assert response.status == 200
                    assert await response.json() == {
                        "echoed": True,
                        "path": "/v1/chat/completions",
                    }

            assert adapter.calls == 1
            assert len(captured["requests"]) == 1
            assert captured["requests"][0]["auth"] == "Bearer upstream-secret"
            assert json.loads(captured["requests"][0]["body"]) == {"forward": True}
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())
    assert downstream_secret not in caplog.text


def test_path_methods_are_enforced_before_credential_resolution():
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(
            f"{upstream_base}/v1",
            allowed=["/responses", "/models"],
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                for method, path, allow in (
                    ("GET", "/responses", "POST"),
                    ("POST", "/models", "GET, HEAD"),
                    ("DELETE", "/models", "GET, HEAD"),
                ):
                    async with session.request(method, f"{proxy_base}/v1{path}") as response:
                        assert response.status == 405
                        assert response.headers["Allow"] == allow
                        assert (await response.json())["error"]["code"] == "method_not_allowed"
                assert adapter.calls == 0
                assert captured["requests"] == []

                for method in ("GET", "HEAD"):
                    async with session.request(method, f"{proxy_base}/v1/models") as response:
                        assert response.status == 200
                        await response.read()
                assert adapter.calls == 2
                assert [request["method"] for request in captured["requests"]] == ["GET", "HEAD"]
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_oversized_body_is_rejected_before_credential_resolution():
    async def run():
        adapter = FakeAdapter("http://127.0.0.1:1/v1")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    data=b"x" * (MAX_REQUEST_BYTES + 1),
                ) as response:
                    assert response.status == 413
                    await response.read()
            assert adapter.calls == 0
        finally:
            await proxy_runner.cleanup()

    asyncio.run(run())


def test_debug_log_omits_forwarded_query_string(caplog):
    query_secret = "must-not-appear-in-logs"
    caplog.set_level("DEBUG", logger="hermes_cli.proxy.server")

    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(f"{upstream_base}/v1")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions?api_key={query_secret}",
                    json={},
                ) as response:
                    assert response.status == 200
                    await response.read()
            assert captured["requests"][0]["query"] == f"api_key={query_secret}"
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())
    assert query_secret not in caplog.text
    assert "?" not in next(
        record.getMessage()
        for record in caplog.records
        if record.name == "hermes_cli.proxy.server" and "forwarding" in record.getMessage()
    )


def test_codex_responses_tools_and_stream_pass_through_with_server_identity():
    async def run():
        captured: Dict[str, Any] = {}
        frames = [
            b'data: {"type":"response.output_item.done","item":{"type":"function_call",'
            b'"call_id":"call_1","name":"lookup","arguments":"{\\"q\\":\\"test\\"}"}}\n\n',
            b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n',
            b"data: [DONE]\n\n",
        ]

        async def responses(request):
            captured["body"] = await request.json()
            captured["headers"] = dict(request.headers)
            response = web.StreamResponse(
                status=200,
                headers={"Content-Type": "text/event-stream"},
            )
            await response.prepare(request)
            for frame in frames:
                await response.write(frame)
            await response.write_eof()
            return response

        upstream = web.Application()
        upstream.router.add_post("/v1/responses", responses)
        upstream_runner, upstream_base = await _start_runner(upstream)
        root_token = _codex_jwt("acct-proxy-root", "proxy")
        adapter = FakeCodexAdapter(
            f"{upstream_base}/v1",
            bearer=root_token,
            allowed=["/responses"],
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        payload = {
            "model": "gpt-test",
            "stream": True,
            "input": [{"role": "user", "content": "use the tool"}],
            "tools": [{
                "type": "function",
                "name": "lookup",
                "description": "Lookup a value",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/responses",
                    json=payload,
                    headers={
                        "Authorization": "Bearer CLIENT-MUST-NOT-LEAK",
                        "User-Agent": "spoofed-client",
                        "originator": "spoofed-origin",
                        "ChatGPT-Account-ID": "acct-spoofed",
                        "x-openai-internal-codex-residency": "spoofed-region",
                    },
                ) as response:
                    body = await response.read()

            assert captured["body"] == payload
            assert captured["headers"]["Authorization"] == f"Bearer {root_token}"
            assert captured["headers"]["ChatGPT-Account-ID"] == "acct-proxy-root"
            assert captured["headers"]["originator"] == "hermes-agent"
            assert captured["headers"]["User-Agent"].startswith("HermesAgent/")
            assert captured["headers"]["x-openai-internal-codex-residency"] == "us"
            assert body == b"".join(frames)
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def _build_sse_upstream(
    frames: list[bytes],
    *,
    path: str = "/v1/chat/completions",
) -> "web.Application":
    async def sse(request):
        _ = await request.read()
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        for chunk in frames:
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_route("*", path, sse)
    return app


def test_proxy_appends_done_when_upstream_omits_sentinel():
    """#90848: complete Portal-shaped SSE without [DONE] gets one appended."""
    async def run():
        frames = [
            b'data: {"choices":[{"delta":{"content":"LONGCAT_OK"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: {"choices":[],"lastOne":true,"usage":{"prompt_tokens":1}}\n\n',
        ]
        upstream_runner, upstream_base = await _start_runner(
            _build_sse_upstream(frames)
        )
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"stream": True},
                ) as resp:
                    body = await resp.read()
            text = body.decode("utf-8")
            assert 'data: {"choices":[{"delta":{"content":"LONGCAT_OK"}}]}' in text
            assert '"finish_reason":"stop"' in text
            assert '"lastOne":true' in text
            assert text.count("data: [DONE]") == 1
            assert text.rstrip().endswith("data: [DONE]")
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_proxy_does_not_duplicate_existing_done():
    async def run():
        frames = [
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        upstream_runner, upstream_base = await _start_runner(
            _build_sse_upstream(frames)
        )
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"stream": True},
                ) as resp:
                    body = await resp.read()
            assert body.decode("utf-8").count("data: [DONE]") == 1
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_proxy_does_not_append_done_after_error_event():
    async def run():
        frames = [
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            b'data: {"error":{"message":"boom","type":"api_error"}}\n\n',
        ]
        upstream_runner, upstream_base = await _start_runner(
            _build_sse_upstream(frames)
        )
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"stream": True},
                ) as resp:
                    body = await resp.read()
            assert "data: [DONE]" not in body.decode("utf-8")
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_proxy_does_not_append_done_after_malformed_trailing_frame():
    async def run():
        frames = [
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: {"choices": [MALFORMED]}\n\n',
        ]
        upstream_runner, upstream_base = await _start_runner(
            _build_sse_upstream(frames)
        )
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"stream": True},
                ) as resp:
                    body = await resp.read()
            assert "data: [DONE]" not in body.decode("utf-8")
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------


def test_cli_refuses_unsafe_remote_bind_before_credential_lookup(monkeypatch, capsys):
    from hermes_cli.proxy import cli as proxy_cli

    adapter = MagicMock()
    monkeypatch.setattr(proxy_cli, "get_adapter", lambda _provider: adapter)
    monkeypatch.delenv("SUBSCRIPTION_PROXY_KEY", raising=False)

    result = proxy_cli.cmd_proxy_start(
        SimpleNamespace(provider="nous", host="0.0.0.0", port=8645)
    )

    assert result == 2
    adapter.is_authenticated.assert_not_called()
    assert "Refusing non-loopback" in capsys.readouterr().err


def test_cli_propagates_downstream_bearer_from_environment(monkeypatch, capsys):
    from hermes_cli.proxy import cli as proxy_cli

    downstream_bearer = "0123456789abcdef" * 2
    adapter = MagicMock()
    adapter.is_authenticated.return_value = True
    adapter.display_name = "Fake Provider"
    captured = {}

    async def fake_run_server(
        actual_adapter,
        *,
        host,
        port,
        downstream_bearer,
    ):
        captured.update(
            adapter=actual_adapter,
            host=host,
            port=port,
            downstream_bearer=downstream_bearer,
        )

    monkeypatch.setattr(proxy_cli, "get_adapter", lambda _provider: adapter)
    monkeypatch.setattr(proxy_cli, "run_server", fake_run_server)
    monkeypatch.setenv("SUBSCRIPTION_PROXY_KEY", downstream_bearer)

    result = proxy_cli.cmd_proxy_start(
        SimpleNamespace(provider="nous", host="0.0.0.0", port=9864)
    )

    assert result == 0
    assert captured == {
        "adapter": adapter,
        "host": "0.0.0.0",
        "port": 9864,
        "downstream_bearer": downstream_bearer,
    }
    output = capsys.readouterr().err
    assert "Downstream auth: required (SUBSCRIPTION_PROXY_KEY)" in output
    assert downstream_bearer not in output




