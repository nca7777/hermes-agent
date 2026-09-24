"""The extraction path must be lossless-or-loud — card t_240ddbcf.

Two halves, and the second is load-bearing:

* the ``infer=true`` write gets a read budget justified by the measured latency distribution
  (``_EXTRACT_READ_TIMEOUT_SECS``), so an extraction the server is still completing is no longer cut
  off client-side; a read timeout retries exactly once, and is never allowed into the ``infer=false``
  fast path or a read;
* every way a turn can be dropped — the write failed, the previous sync was still in flight, the
  circuit breaker was open, there is no backend — now leaves a DURABLE record under the home
  (``mem0_extraction_losses.jsonl``), instead of a log line nobody reads without a scan.

A healthy turn must leave NO record: a loss ledger that fills up on success is worse than none.
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest

import plugins.memory.mem0 as mem0_plugin
import plugins.memory.mem0._backend as backend_mod
import plugins.memory.mem0._loss_record as loss_record
from plugins.memory.mem0 import Mem0MemoryProvider
from plugins.memory.mem0._backend import SelfHostedBackend
from plugins.memory.mem0._loss_record import record

OK_BODY = {"results": [{"id": "m1", "memory": "fact", "event": "ADD"}]}


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch):
    monkeypatch.setattr(backend_mod, "_SYNC_RETRY_BACKOFF_SECS", (0.0, 0.0))


def _backend(handler):
    return SelfHostedBackend("k", "http://sh:8888", transport=httpx.MockTransport(handler))


def _read_timeout(request):
    return request.extensions["timeout"]["read"]


class TestTheExtractionWriteHasItsOwnBudget:
    def test_the_extraction_write_gets_the_long_read_budget(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=OK_BODY)

        _backend(handler).add([{"role": "user", "content": "hi"}], user_id="u", agent_id="a", infer=True)
        assert _read_timeout(seen[0]) == backend_mod._EXTRACT_READ_TIMEOUT_SECS
        assert backend_mod._EXTRACT_READ_TIMEOUT_SECS > 90.0  # the server's own MEM0_LLM_TIMEOUT

    def test_the_exact_write_keeps_the_short_budget(self):
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, json=OK_BODY)

        backend = _backend(handler)
        backend.add([{"role": "user", "content": "hi"}], user_id="u", agent_id="a", infer=False)
        # No per-request override: the client's own flat 30s budget applies, unchanged.
        assert backend._client.timeout.read == 30.0
        assert _read_timeout(seen[0]) is None


    def test_a_search_timeout_is_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout("too slow", request=request)

        with pytest.raises(httpx.ReadTimeout):
            _backend(handler).search("q", filters={"user_id": "u"})
        assert len(calls) == 1


class TestTheExtractionTimeoutRetriesOnce:
    def test_a_read_timeout_retries_exactly_once_and_can_succeed(self):
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout("too slow", request=request)
            return httpx.Response(200, json=OK_BODY)

        result = _backend(handler).add([{"role": "user", "content": "hi"}], user_id="u", agent_id="a", infer=True)
        assert result["results"][0]["id"] == "m1"
        assert len(calls) == 2

    def test_a_second_read_timeout_propagates(self):
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout("too slow", request=request)

        with pytest.raises(httpx.ReadTimeout):
            _backend(handler).add([{"role": "user", "content": "hi"}], user_id="u", agent_id="a", infer=True)
        assert len(calls) == 2  # one bounded retry, then the error is raised for the caller to record

    def test_the_failed_attempt_count_is_stamped_on_the_error(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        with pytest.raises(httpx.ReadTimeout) as caught:
            _backend(handler).add([{"role": "user", "content": "hi"}], user_id="u", agent_id="a", infer=True)
        assert getattr(caught.value, "mem0_attempts", None) == 2


class TestTheRecord:
    def test_a_dropped_turn_is_one_json_line_with_the_turn(self, tmp_path):
        rec = record(event="lost", exc=RuntimeError("Failed to store: timed out"), session="s1",
                     home=str(tmp_path), turn=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}])
        assert rec is not None
        path = tmp_path / loss_record.FILE_NAME
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        got = json.loads(lines[0])
        assert got["event"] == "lost" and got["class"] == "timeout" and got["loss"] is True
        assert got["turn"] == {"user": "q", "assistant": "a"} and got["session"] == "s1"
        assert os.stat(path).st_mode & 0o777 == 0o600

    def test_a_secret_shaped_turn_is_stored_redacted(self, tmp_path):
        secret = "a" * 64
        record(event="lost", exc=RuntimeError("502 Bad Gateway"), session="s1", home=str(tmp_path),
               turn=[{"role": "user", "content": f"my key is {secret}"},
                     {"role": "assistant", "content": "noted"}])
        got = json.loads((tmp_path / loss_record.FILE_NAME).read_text().splitlines()[0])
        assert got["redacted"] is True
        assert secret not in json.dumps(got)
        assert got["class"] == "502_bad_gateway"

    def test_a_skip_is_recorded_with_its_own_class(self, tmp_path):
        record(event="skipped_busy", session="s1", home=str(tmp_path),
               turn=[{"role": "assistant", "content": "a"}])
        got = json.loads((tmp_path / loss_record.FILE_NAME).read_text().splitlines()[0])
        assert got["class"] == "skipped_busy" and got["event"] == "skipped_busy"

    def test_record_never_raises_when_the_home_is_unwritable(self, tmp_path):
        blocked = tmp_path / "not-a-dir"
        blocked.write_text("x")
        assert record(event="lost", exc=RuntimeError("boom"), home=str(blocked)) is None


class _RaisingBackend:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.calls += 1
        raise self.exc


class _OkBackend:
    def __init__(self):
        self.calls = 0

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.calls += 1
        return {"results": []}


def _provider(backend):
    provider = Mem0MemoryProvider()
    provider.initialize("test-session")
    provider._user_id, provider._agent_id = "u123", "hermes"
    provider._backend = backend
    return provider


def _records(tmp_path):
    path = tmp_path / loss_record.FILE_NAME
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class TestTheDroppedTurnReachesTheRecord:
    def test_a_failed_extraction_is_recorded_and_still_warned(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(loss_record, "_home_dir", lambda: str(tmp_path))
        backend = _RaisingBackend(httpx.ReadTimeout("timed out"))
        provider = _provider(backend)
        with caplog.at_level("WARNING"):
            provider.sync_turn("remember teal", "teal noted", session_id="s1")
            provider._sync_thread.join(timeout=10)
        assert backend.calls == 1
        assert "Mem0 sync failed" in caplog.text  # the log line the operator already had
        got = _records(tmp_path)
        assert len(got) == 1
        assert got[0]["event"] == "lost" and got[0]["class"] == "timeout"
        assert got[0]["turn"]["user"] == "remember teal" and got[0]["session"] == "s1"

    def test_the_record_says_how_many_attempts_the_drop_took(self, tmp_path, monkeypatch):
        """The ledger must distinguish "the one retry failed too" from "no retry was made"."""
        monkeypatch.setattr(loss_record, "_home_dir", lambda: str(tmp_path))

        def handler(request):
            raise httpx.ReadTimeout("timed out", request=request)

        provider = _provider(_backend(handler))
        provider.sync_turn("remember teal", "teal noted", session_id="s1")
        provider._sync_thread.join(timeout=10)
        got = _records(tmp_path)
        assert len(got) == 1
        assert got[0]["attempts"] == 2

    def test_a_healthy_extraction_leaves_no_record(self, tmp_path, monkeypatch):
        """NEGATIVE CONTROL: success must not write a loss line."""
        monkeypatch.setattr(loss_record, "_home_dir", lambda: str(tmp_path))
        backend = _OkBackend()
        provider = _provider(backend)
        provider.sync_turn("remember teal", "teal noted", session_id="s1")
        provider._sync_thread.join(timeout=10)
        assert backend.calls == 1
        assert _records(tmp_path) == []
        assert not (tmp_path / loss_record.FILE_NAME).exists()

    def test_a_turn_skipped_while_the_previous_sync_runs_is_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loss_record, "_home_dir", lambda: str(tmp_path))
        monkeypatch.setattr(mem0_plugin, "_SYNC_JOIN_SECS", 0.05)
        release = __import__("threading").Event()

        class _SlowBackend:
            def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
                release.wait(timeout=10)
                return {"results": []}

        provider = _provider(_SlowBackend())
        provider.sync_turn("first question", "slow answer", session_id="s1")
        time.sleep(0.05)
        provider.sync_turn("second question", "second answer", session_id="s1")
        got = _records(tmp_path)
        release.set()
        provider._sync_thread.join(timeout=10)
        assert len(got) == 1
        assert got[0]["event"] == "skipped_busy"
        assert got[0]["turn"]["user"] == "second question"

    def test_a_turn_skipped_by_the_open_breaker_is_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loss_record, "_home_dir", lambda: str(tmp_path))
        provider = _provider(_OkBackend())
        provider._consecutive_failures = mem0_plugin._BREAKER_THRESHOLD
        provider._breaker_open_until = time.monotonic() + 60
        provider.sync_turn("question during the outage", "answer", session_id="s1")
        got = _records(tmp_path)
        assert len(got) == 1 and got[0]["event"] == "skipped_breaker"

    def test_a_turn_with_no_backend_is_recorded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(loss_record, "_home_dir", lambda: str(tmp_path))
        provider = _provider(None)
        provider.sync_turn("question with no store", "answer", session_id="s1")
        got = _records(tmp_path)
        assert len(got) == 1 and got[0]["event"] == "skipped_no_backend"
