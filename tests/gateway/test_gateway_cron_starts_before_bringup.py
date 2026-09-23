"""The gateway starts its cron ticker BEFORE adapter/plugin bring-up (t_661d3766).

One gateway process per host owns every profile's cron store. ``runner.start()`` (adapter connect
plus per-profile plugin bring-up — credential guard, HMC plugin init, telemetry budget watcher, one
home at a time) blocks for minutes on a many-home host: measured ~5 s/home, ~13 minutes across 97
homes, on top of MCP discovery's 120 s ceiling. While the ticker was started LAST, nothing on the
host fired from the predecessor's final tick until bring-up finished — 14-18 minutes of fleet-wide
cron silence after EVERY gateway restart, including the 5-minute mem0 watchdog (the 502 alarm path)
and the ``*/20`` spend guard.

The invariant these tests express: the ticker is running before adapter bring-up begins, so the
first tick is NOT serialized behind it — and an early ticker is never leaked when startup aborts.
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest

from gateway.config import GatewayConfig

# Stand-in for the minutes of adapter/plugin bring-up across 97 homes. Any value that is comfortably
# larger than the scheduling jitter of starting one thread expresses the ordering.
BRINGUP_SECONDS = 0.5


class _RecordingTicker:
    """Stands in for ``SupervisedTickerThread``; records its start on the shared event log."""

    events: list = []
    captured: dict = {}
    instances: list = []
    started_at: float | None = None

    def __init__(self, target, *, args=(), kwargs=None, stop_event=None, name="cron"):
        _RecordingTicker.captured = dict(kwargs or {})
        _RecordingTicker.instances.append(self)
        self.stop_event = stop_event
        self.restarts = 0
        self.started = False

    def start(self):
        self.started = True
        _RecordingTicker.started_at = time.monotonic()
        _RecordingTicker.events.append("ticker.start")

    def restart_if_dead(self):
        return False

    def is_alive(self):
        # Mirrors SupervisedTickerThread: False once the stop request lands, which is what the
        # abort paths wait on.
        return self.started and not (self.stop_event is not None and self.stop_event.is_set())


@pytest.fixture
def isolated_profiles(tmp_path, monkeypatch):
    """A scratch HOME so profile enumeration can never read or write the live install."""
    fake_home = tmp_path / "fakehome"
    hermes_home = fake_home / ".hermes"
    (hermes_home / "profiles" / "secondary").mkdir(parents=True)
    (hermes_home / "profiles" / "secondary" / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    from hermes_cli import profiles as profiles_mod

    assert str(profiles_mod._get_profiles_root()).startswith(str(tmp_path))
    _RecordingTicker.events = []
    _RecordingTicker.captured = {}
    _RecordingTicker.instances = []
    _RecordingTicker.started_at = None
    return hermes_home


def _patch_gateway_startup(monkeypatch, tmp_path, runner_cls):
    """Everything ``start_gateway`` touches up to and past the cron start, minus the real system."""
    from cron import scheduler_thread

    monkeypatch.setattr(scheduler_thread, "SupervisedTickerThread", _RecordingTicker)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda *a, **kw: None)
    monkeypatch.setattr("gateway.status.acquire_gateway_runtime_lock", lambda *a, **kw: True)
    monkeypatch.setattr("gateway.status.write_pid_file", lambda *a, **kw: None)
    monkeypatch.setattr("gateway.status.remove_pid_file", lambda *a, **kw: None)
    monkeypatch.setattr("gateway.status.release_gateway_runtime_lock", lambda *a, **kw: None)
    monkeypatch.setattr("tools.skills_sync.sync_skills", lambda quiet=True: None)
    monkeypatch.setattr("hermes_logging.setup_logging", lambda hermes_home, mode: None)
    monkeypatch.setattr("gateway.run.GatewayRunner", runner_cls)

    async def _no_host_attach(*_a, **_kw):
        return None

    monkeypatch.setattr("gateway.run._host_attach_or_none", _no_host_attach)
    monkeypatch.setattr("gateway.run._start_gateway_configure_logging", lambda *a, **kw: None)
    monkeypatch.setattr("gateway.run._enable_multiplex_log_routing", lambda *a, **kw: False)

    async def _no_control_socket(*_a, **_kw):
        return None

    monkeypatch.setattr("gateway.run._start_gateway_start_control_socket", _no_control_socket)
    monkeypatch.setattr("gateway.run._refresh_host_gateway_record", lambda *a, **kw: None)
    monkeypatch.setattr("gateway.run._start_gateway_claim_pid_file", lambda *a, **kw: True)
    monkeypatch.setattr(
        "gateway.run._start_gateway_make_shutdown_signal_handler",
        lambda runner, state: (lambda received_signal=None: None))
    monkeypatch.setattr(
        "gateway.run._start_gateway_make_restart_signal_handler",
        lambda runner: (lambda received_signal=None: None))
    monkeypatch.setattr("gateway.run._start_gateway_housekeeping", lambda *_a, **_kw: None)

    async def _no_mcp_shutdown(*_a, **_kw):
        return True

    monkeypatch.setattr("gateway.run._shutdown_mcp_servers_nonblocking", _no_mcp_shutdown)
    monkeypatch.setattr("gateway.shutdown_flush.recover_pending_to_db", lambda **_kw: 0)
    monkeypatch.setattr("hermes_cli.nous_auth_keepalive.start_nous_auth_keepalive", lambda: None)
    monkeypatch.setattr("hermes_cli.nous_auth_keepalive.stop_nous_auth_keepalive", lambda: None)


class _BringUpRunner:
    """A minimal runner whose ``start()`` stands in for the real bring-up and asserts liveness."""

    def __init__(self, config, *, start_result=True):
        self.config = config
        self.adapters = {}
        self._profile_adapters = {}
        self._primary_profile_name = "default"
        self._running = False
        self._draining = False
        self._external_drain_active = False
        self.should_exit_cleanly = False
        self.should_exit_with_failure = False
        self.exit_reason = None
        self.exit_code = None
        self._restart_requested = False
        self._restart_via_service = False
        self._gateway_health_export_runtime_shutdowns = 0
        self._gateway_health_export_runtime = types.SimpleNamespace(
            shutdown=self._health_export_shutdown)
        self._start_result = start_result

    def _health_export_shutdown(self):
        self._gateway_health_export_runtime_shutdowns += 1

    async def start(self):
        _RecordingTicker.events.append("runner.start:begin")
        assert _RecordingTicker.started_at is not None, (
            "the cron ticker must already be running when adapter/plugin bring-up begins")
        await asyncio.sleep(BRINGUP_SECONDS)
        _RecordingTicker.events.append("runner.start:end")
        self._running = bool(self._start_result)
        return self._start_result

    async def wait_for_shutdown(self):
        return None

    def _start_systemd_watchdog(self):
        return None


@pytest.mark.asyncio
async def test_cron_ticker_starts_before_adapter_bringup(isolated_profiles, monkeypatch, tmp_path):
    """The ticker is live before bring-up starts, so the first tick is not serialized behind it."""
    import gateway.run as gateway_run

    _patch_gateway_startup(monkeypatch, tmp_path, _BringUpRunner)
    discovered: list = []

    async def _discover(config):
        discovered.append(True)
        _RecordingTicker.events.append("mcp.discover")

    monkeypatch.setattr(gateway_run, "_discover_gateway_mcp_tools", _discover)

    boot_started_at = time.monotonic()
    result = await gateway_run.start_gateway(
        config=GatewayConfig(), replace=False, verbosity=None)

    assert result is True
    events = _RecordingTicker.events
    assert "ticker.start" in events, "the gateway must start its cron ticker during startup"
    assert events.index("ticker.start") < events.index("runner.start:begin"), (
        "the cron ticker must start BEFORE adapter/plugin bring-up, not after it: "
        f"observed order {events}")

    # The interval the card measures: boot -> first tick opportunity. Today the ticker starts only
    # after bring-up returns, so this offset is >= bring-up; with the fix it is orders of magnitude
    # smaller and independent of how long bring-up takes.
    assert _RecordingTicker.started_at is not None
    ticker_start_offset = _RecordingTicker.started_at - boot_started_at
    assert ticker_start_offset < BRINGUP_SECONDS, (
        "the first tick waited out adapter bring-up "
        f"({ticker_start_offset:.2f}s >= {BRINGUP_SECONDS}s of simulated bring-up)")

    # Delivery semantics: the ticker holds the LIVE adapter maps, which bring-up mutates in place —
    # so a job due mid-bring-up delivers through whatever is already connected, and fails closed for
    # a target that is not up yet instead of being skipped.
    assert _RecordingTicker.captured["adapters"] is not None
    assert _RecordingTicker.captured["profile_adapters"] is not None


@pytest.mark.asyncio
async def test_early_cron_ticker_is_stopped_when_bringup_fails(isolated_profiles, monkeypatch, tmp_path):
    """A startup that aborts before serving must not leak the early-started ticker thread."""
    import gateway.run as gateway_run

    runner_cls = type("FailingBringUpRunner", (_BringUpRunner,), {})
    _patch_gateway_startup(monkeypatch, tmp_path,
                           lambda config: runner_cls(config, start_result=False))
    stopped: list = []
    monkeypatch.setattr(gateway_run, "_stop_cron_provider", lambda provider: stopped.append(provider))

    result = await gateway_run.start_gateway(
        config=GatewayConfig(), replace=False, verbosity=None)

    assert result is False
    assert _RecordingTicker.events[:2] == ["ticker.start", "runner.start:begin"]
    ticker = _RecordingTicker.instances[-1]
    assert ticker.stop_event.is_set(), "the early ticker must be asked to stop on an aborted startup"
    assert ticker.is_alive() is False, "the early ticker thread must not outlive a failed startup"
    assert stopped, "the cron provider must be stopped on an aborted startup"


@pytest.mark.asyncio
async def test_early_cron_ticker_is_stopped_on_clean_early_exit(isolated_profiles, monkeypatch, tmp_path):
    """``should_exit_cleanly`` after bring-up still tears the early ticker down before exiting."""
    import gateway.run as gateway_run

    from gateway.restart import GATEWAY_SERVICE_RESTART_EXIT_CODE

    class CleanExitRunner(_BringUpRunner):
        def __init__(self, config):
            super().__init__(config)
            self.should_exit_cleanly = True
            self.exit_code = GATEWAY_SERVICE_RESTART_EXIT_CODE

    _patch_gateway_startup(monkeypatch, tmp_path, CleanExitRunner)

    with pytest.raises(SystemExit) as exc:
        await gateway_run.start_gateway(
            config=GatewayConfig(), replace=False, verbosity=None)

    assert exc.value.code == GATEWAY_SERVICE_RESTART_EXIT_CODE
    ticker = _RecordingTicker.instances[-1]
    assert ticker.stop_event.is_set()
    assert ticker.is_alive() is False
