"""An ``infer=true`` extraction write survives a hop restart — card t_da57983b.

`t_9d4a4c97` made the go-rotation hop a *start* precondition of the API, but `depends_on` governs
start order only: a hop that dies after the API is serving (host reboot, OOM, `docker restart`, a
Coolify recreate of that one service) still opens a window in which mem0 answers
``502 provider_unavailable`` in ~0.1 s and the turn is gone. These tests pin the writer-side retry
that closes it, and pin the two things the retry must NOT break: the ``infer=false`` fast path, and
the fact that a real outage stays visible (the error still propagates, and each attempt that reaches
the server is still counted).
"""

from __future__ import annotations

import time

import httpx
import pytest

import plugins.memory.mem0 as mem0_plugin
import plugins.memory.mem0._backend as backend_mod
from plugins.memory.mem0 import Mem0MemoryProvider
from plugins.memory.mem0._backend import (
    SelfHostedBackend,
    _is_transient_upstream,
    _retry_transient,
)

OK_BODY = {"results": [{"id": "m1", "memory": "fact", "event": "ADD"}]}
UNAVAILABLE_BODY = {
    "detail": "Provider is unreachable or returned a server error.",
    "code": "provider_unavailable",
    "request_id": "abc12345",
}


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch):
    """Timing is asserted in its own test; no other test may actually sleep."""
    monkeypatch.setattr(backend_mod, "_SYNC_RETRY_BACKOFF_SECS", (0.0, 0.0))


def _backend(responses):
    """SelfHostedBackend over a stub transport answering `responses` in order (last repeats)."""
    seen = []

    def handler(request):
        seen.append(request)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    return SelfHostedBackend("k", "http://sh:8888", transport=httpx.MockTransport(handler)), seen


def _add(backend, *, infer=True, user="u1"):
    return backend.add([{"role": "user", "content": "Nick likes teal"}], user_id=user, agent_id="hermes", infer=infer)


class TestTransientClassification:
    """Only failures that provably stored nothing and may self-heal are retried."""

    def _error(self, status, body):
        request = httpx.Request("POST", "http://sh:8888/memories")
        response = httpx.Response(status, json=body, request=request)
        return httpx.HTTPStatusError("boom", request=request, response=response)

    def test_the_server_s_own_provider_unavailable_is_transient(self):
        assert _is_transient_upstream(self._error(502, UNAVAILABLE_BODY)) is True

    def test_a_proxy_502_with_no_classified_body_is_transient(self):
        request = httpx.Request("POST", "http://sh:8888/memories")
        response = httpx.Response(502, text="<html>502 Bad Gateway</html>", request=request)
        assert _is_transient_upstream(httpx.HTTPStatusError("boom", request=request, response=response)) is True

    def test_a_rate_limit_is_not_transient(self):
        assert _is_transient_upstream(self._error(502, {"code": "provider_rate_limited"})) is False

    def test_a_bad_request_is_not_transient(self):
        assert _is_transient_upstream(self._error(400, {"detail": "malformed"})) is False

    def test_a_read_timeout_is_not_transient(self):
        # The request may already have been stored: retrying would duplicate the memory.
        assert _is_transient_upstream(httpx.ReadTimeout("slow")) is False

    def test_a_dead_socket_is_transient(self):
        assert _is_transient_upstream(httpx.ConnectError("refused")) is True


class TestWriterSideRetry:
    def test_a_502_provider_unavailable_is_retried_and_the_write_lands(self):
        backend, seen = _backend([httpx.Response(502, json=UNAVAILABLE_BODY), httpx.Response(200, json=OK_BODY)])
        assert _add(backend)["results"][0]["id"] == "m1"
        assert len(seen) == 2  # the failed attempt plus the one that landed

    def test_a_dead_socket_is_retried(self):
        seen = []

        def handler(request):
            seen.append(request)
            if len(seen) == 1:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json=OK_BODY)

        backend = SelfHostedBackend("k", "http://sh:8888", transport=httpx.MockTransport(handler))
        assert _add(backend)["results"][0]["id"] == "m1"
        assert len(seen) == 2

    def test_the_infer_false_fast_path_is_untouched(self):
        """mem0_add is synchronous and user-facing: a single shot, never a retry."""
        backend, seen = _backend([httpx.Response(502, json=UNAVAILABLE_BODY)])
        with pytest.raises(httpx.HTTPStatusError):
            _add(backend, infer=False)
        assert len(seen) == 1

    def test_a_persistent_outage_still_raises_after_the_budget(self):
        """A retry must not swallow a real outage: the plugin's breaker and warning still see it."""
        backend, seen = _backend([httpx.Response(502, json=UNAVAILABLE_BODY)])
        with pytest.raises(httpx.HTTPStatusError):
            _add(backend)
        assert len(seen) == backend_mod._SYNC_RETRY_ATTEMPTS

    def test_a_bad_request_is_not_retried_through_the_backend(self):
        backend, seen = _backend([httpx.Response(400, json={"detail": "malformed"})])
        with pytest.raises(httpx.HTTPStatusError):
            _add(backend)
        assert len(seen) == 1

    def test_the_backoff_is_waited_between_attempts(self):
        slept = []
        calls = []

        def call():
            calls.append(1)
            raise httpx.HTTPStatusError(
                "502",
                request=httpx.Request("POST", "http://sh:8888/memories"),
                response=httpx.Response(502, json=UNAVAILABLE_BODY),
            )

        with pytest.raises(httpx.HTTPStatusError):
            _retry_transient(call, attempts=3, backoff=(2.0, 5.0), sleep=slept.append)
        assert len(calls) == 3
        assert slept == [2.0, 5.0]


class TestTheNextTurnIsNotLost:
    """A retrying write holds the plugin's sync thread; the next turn must not be dropped for it."""

    def test_the_join_window_covers_the_retry_budget(self):
        # sync_turn waits this long for the previous sync before skipping the turn entirely, so it
        # has to outlast a write that is spending its retry budget.
        assert mem0_plugin._SYNC_JOIN_SECS > sum(backend_mod._SYNC_RETRY_BACKOFF_SECS) + 3.0

    def test_a_retrying_sync_still_ingests_the_next_turn(self, monkeypatch):
        monkeypatch.setattr(backend_mod, "_SYNC_RETRY_BACKOFF_SECS", (0.3, 0.0))
        seen = []

        def handler(request):
            seen.append(request)
            if len(seen) == 1:  # the window: the hop is down for this write only
                return httpx.Response(502, json=UNAVAILABLE_BODY)
            return httpx.Response(200, json=OK_BODY)

        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._backend = SelfHostedBackend("k", "http://sh:8888", transport=httpx.MockTransport(handler))

        provider.sync_turn("turn one", "reply one")
        time.sleep(0.05)
        provider.sync_turn("turn two", "reply two")

        provider._sync_thread.join(timeout=10)
        assert provider._sync_thread.is_alive() is False
        # turn one: 502 then the retry that landed. turn two: its own write. Nothing skipped.
        assert len(seen) == 3
