"""A worker killed by its OWN cgroup's OOM killer is booked, counted and escalated.

The worker scope carries ``MemoryMax``; when the work inside it needs more than that
the kernel OOM-kills it and systemd SIGKILLs the rest of the unit, so the card leaves
no exit trailer and no board call. Booked as a bare ``pid N not alive`` crash it was
indistinguishable from a broken card, and the dispatcher re-spawned identical work
onto the same wall for hours (see ``/home/axwel/audits/axwel-server-20260920/``).

These tests lock: the scope journal is asked (and only trusted for a real
``oom-kill`` result), the death gets its own run outcome + evidence, and the card is
escalated on a durable count that an operator unblock cannot restart.
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

TASK_ID = "t_oom0001"
RUN_ID = 7
UNIT = "hermes-worker-kanban-t_oom0001-run-7.scope"

# The shape systemd actually writes for an OOM-killed worker scope on this host.
JOURNAL = (
    f"{UNIT}: Started [systemd-run] /usr/bin/python -m hermes_cli.main -p a --cli chat -q \"work\".\n"
    f"{UNIT}: The kernel OOM killer killed some processes in this unit.\n"
    f"{UNIT}: Killing process 3199500 (hermes) with signal SIGKILL.\n"
    f"{UNIT}: Failed with result 'oom-kill'.\n"
    f"{UNIT}: Consumed 5min 10.520s CPU time over 7min 36.309s wall clock time, "
    "4G memory peak, 32.6M memory swap peak.\n"
)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _claim_dead_worker(conn, tid: str, pid: int, log_tail: str = "") -> None:
    """Claim ``tid`` for a worker that died without writing an exit trailer."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()
    if log_tail:
        log = kb.worker_log_path(tid)
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(log_tail + "\n")


def _oom_evidence(*_args, **_kwargs) -> dict:
    return {"unit": UNIT, "memory_peak": "4G", "lines": [f"{UNIT}: Failed with result 'oom-kill'."]}


def _latest_run(conn, tid: str):
    return conn.execute(
        "SELECT outcome, error, metadata FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()


def _latest_gave_up(conn, tid: str):
    """The ``gave_up`` event ``_record_task_failure`` appends when the breaker or the OOM
    escalation trips. The crash RUN carries the death itself; this payload carries the
    escalation bookkeeping (``oom_deaths`` / ``sticky`` / ``effective_limit``)."""
    return conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id=? AND kind='gave_up' ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()


def test_scope_journal_probe_reads_only_a_real_oom_kill(monkeypatch: pytest.MonkeyPatch):
    """The probe answers with the unit + memory peak on ``Failed with result 'oom-kill'``,
    and with ``None`` for every other shape (no journal, no entries, no run id)."""
    seen: list[list[str]] = []

    def _run(argv, **_kwargs):
        seen.append(list(argv))
        return SimpleNamespace(returncode=0, stdout=JOURNAL, stderr="")

    monkeypatch.setattr(kbd.subprocess, "run", _run)
    evidence = kbd.read_worker_oom_kill_evidence(TASK_ID, RUN_ID)
    assert evidence is not None
    assert evidence["unit"] == UNIT
    assert evidence["memory_peak"] == "4G"
    assert seen[0][:4] == ["journalctl", "--user", "--unit", UNIT]
    assert seen[0][1] == "--user"

    # A journal without the oom-kill result is not evidence (a timeout kill, a
    # degraded host with no scope at all).
    monkeypatch.setattr(
        kbd.subprocess, "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="nothing to see\n", stderr=""),
    )
    assert kbd.read_worker_oom_kill_evidence(TASK_ID, RUN_ID) is None
    # journalctl failing (no user journal, no permission) must not raise.
    monkeypatch.setattr(
        kbd.subprocess, "run",
        lambda *_a, **_k: SimpleNamespace(returncode=1, stdout="", stderr="no journal"),
    )
    assert kbd.read_worker_oom_kill_evidence(TASK_ID, RUN_ID) is None
    # No run id -> no scope name -> no probe at all.
    assert kbd.read_worker_oom_kill_evidence(TASK_ID, None) is None


def test_oom_death_gets_its_own_outcome_and_evidence(kanban_home, monkeypatch: pytest.MonkeyPatch):
    """A generic "pid N not alive" crash becomes ``outcome='oom_killed'`` carrying the
    unit, the memory peak and the action that works (split the card), while the event
    kind stays ``crashed`` so every existing watcher/notifier still fires on it."""
    monkeypatch.setattr(kbd, "read_worker_oom_kill_evidence", _oom_evidence)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="heavy", assignee="a")
        _claim_dead_worker(conn, tid, 80001)

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        metadata = kb._json_dict(run["metadata"])
        assert run["outcome"] == "oom_killed"
        assert metadata.get("oom_killed") is True
        assert metadata.get("systemd_unit") == UNIT
        assert metadata.get("memory_peak") == "4G"
        assert "MemoryMax" in run["error"]
        assert "Split the card" in run["error"]
        event = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,),
        ).fetchone()
        assert event["kind"] == "crashed"
        assert kb._json_dict(event["payload"]).get("oom_killed") is True
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 1
        assert "OOM-killed" in (task.last_failure_error or "")


def test_oom_kill_escalates_on_a_count_an_unblock_cannot_reset(
    kanban_home, monkeypatch: pytest.MonkeyPatch,
):
    """The real incident's loop, step for step.

    Two OOM deaths trip the ordinary ``failure_limit`` and the card blocks; the operator
    unblocks it (the action that restarted the loop in the wild) and the identical death
    happens a third time. That third death must escalate on the DURABLE count — the
    unblock reset ``consecutive_failures`` to 0, and the escalation has to hold anyway.
    """
    monkeypatch.setattr(kbd, "read_worker_oom_kill_evidence", _oom_evidence)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="loop", assignee="a")

        _claim_dead_worker(conn, tid, 81001)
        kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, tid).status == "ready"  # below failure_limit

        _claim_dead_worker(conn, tid, 81002)
        kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, tid).status == "blocked"  # ordinary breaker
        assert kb._json_dict(_latest_gave_up(conn, tid)["payload"]).get("oom_deaths") == 2

        # Operator unblock: the counter resets by design, the OOM count must not.
        kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).consecutive_failures == 0

        _claim_dead_worker(conn, tid, 81003)
        kbd.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        run = _latest_run(conn, tid)
        metadata = kb._json_dict(_latest_gave_up(conn, tid)["payload"])
        assert kbd._oom_kill_deaths(conn, tid) == kbd.OOM_KILL_ESCALATION_LIMIT == 3
        assert run["outcome"] == "oom_killed"
        assert metadata.get("oom_deaths") == 3
        assert metadata.get("oom_escalation_limit") == kbd.OOM_KILL_ESCALATION_LIMIT
        assert metadata.get("sticky") is True
        # Escalated at ONE counted failure: the durable OOM count did the blocking,
        # not ``consecutive_failures``.
        assert task.status == "blocked"
        assert task.consecutive_failures == 1
        # Held for a human: neither a raised failure_limit nor the next tick releases it.
        kb.recompute_ready(conn, failure_limit=10)
        assert kb.get_task(conn, tid).status == "blocked"

        # Pressing the same button again does not restart the loop.
        kb.unblock_task(conn, tid)
        _claim_dead_worker(conn, tid, 81004)
        kbd.detect_crashed_workers(conn)
        assert kb.get_task(conn, tid).status == "blocked"
        assert kbd._oom_kill_deaths(conn, tid) == 4


def test_plain_crash_booking_is_unchanged(kanban_home, monkeypatch: pytest.MonkeyPatch):
    """No OOM evidence: the death keeps the generic crash outcome, so a genuinely broken
    card is still diagnosed as one."""
    monkeypatch.setattr(kbd, "read_worker_oom_kill_evidence", lambda *_a, **_k: None)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="broken", assignee="a")
        _claim_dead_worker(conn, tid, 82001)

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        assert run["outcome"] == "crashed"
        assert "oom_killed" not in (run["metadata"] or "")
        assert kbd._oom_kill_deaths(conn, tid) == 0


def test_oom_error_text_is_not_mistaken_for_an_auth_wall(
    kanban_home, monkeypatch: pytest.MonkeyPatch,
):
    """The stored error carries the worker's last output, which can contain words the
    quota/auth blocker regex matches (``claude auth status``, #117097). An OOM death must
    not be parked as ``blocker_auth`` because of it — the card is memory-bound, and the
    escalation above is its route to a human."""
    monkeypatch.setattr(kbd, "read_worker_oom_kill_evidence", _oom_evidence)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="heavy", assignee="a")
        _claim_dead_worker(conn, tid, 83001, log_tail="  💻 $  claude auth status --json")

        kbd.detect_crashed_workers(conn)

        assert "auth" in (kb.get_task(conn, tid).last_failure_error or "")
        assert kbd.check_respawn_guard(conn, tid) is None


# ---------------------------------------------------------------------------
# A stale exit trailer must not be able to hide a real OOM kill
# ---------------------------------------------------------------------------
# ``_worker_log_exit_code`` reads the LAST trailer of the 4 KB tail of an
# APPEND-MODE log shared by every run of the card, so the trailer it finds can
# belong to a PREVIOUS run. Trusting it before asking the journal booked an
# own-cgroup OOM death as that earlier run's clean-exit protocol violation: the
# durable OOM count never moved and the escalation never armed.

def _append_worker_log(tid: str, text: str) -> None:
    """Append to the card's own log, the way a re-run does (``_open_worker_log``)."""
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(text)


_STALE_TRAILER = f"previous run: wrote its epilogue\n{KANBAN_WORKER_EXIT_TRAILER}0\n"
_OWN_OUTPUT = "build output of the run that died\n"


@pytest.mark.parametrize("shape", ["trailer_after_own_output", "trailer_before_own_output"])
def test_a_stale_exit_trailer_cannot_hide_an_oom_kill(
    kanban_home, monkeypatch: pytest.MonkeyPatch, shape: str,
):
    """Prior run's ``rc=0`` trailer in the same log + journal OOM evidence ⇒ the death
    is still booked ``oom_killed``, in both append orders (the trailer last is the
    harsher shape: nothing follows it to give it away)."""
    monkeypatch.setattr(kbd, "read_worker_oom_kill_evidence", _oom_evidence)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="heavy re-run", assignee="a")
        _claim_dead_worker(conn, tid, 84001)
        _append_worker_log(
            tid,
            _OWN_OUTPUT + _STALE_TRAILER if shape == "trailer_after_own_output"
            else _STALE_TRAILER + _OWN_OUTPUT,
        )
        assert kbd._worker_log_exit_code(tid) == 0  # the trailer IS what gets read

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        metadata = kb._json_dict(run["metadata"])
        assert run["outcome"] == "oom_killed"
        assert metadata.get("oom_killed") is True
        assert metadata.get("systemd_unit") == UNIT
        assert metadata.get("memory_peak") == "4G"
        assert "MemoryMax" in run["error"]
        # The stale trailer's code is not passed off as THIS death's exit code.
        assert metadata.get("exit_code") is None
        assert metadata.get("exit_kind") is None
        assert "worker exited cleanly" not in (run["error"] or "")
        # The durable count moved, so the escalation can arm.
        assert kbd._oom_kill_deaths(conn, tid) == 1


def test_a_stale_trailer_without_oom_evidence_keeps_its_protocol_violation(
    kanban_home, monkeypatch: pytest.MonkeyPatch,
):
    """The other side of the same coin: the journal is consulted, and when it does NOT
    vouch for an OOM kill the trailer keeps its ordinary booking — a genuinely sloppy
    worker is not relabelled as memory-bound."""
    monkeypatch.setattr(kbd, "read_worker_oom_kill_evidence", lambda *_a, **_k: None)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="sloppy", assignee="a")
        _claim_dead_worker(conn, tid, 84501)
        _append_worker_log(tid, _OWN_OUTPUT + _STALE_TRAILER)

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        assert run["outcome"] == "crashed"
        assert kb._json_dict(run["metadata"]).get("protocol_violation") is True
        assert "worker exited cleanly" in run["error"]
        assert kbd._oom_kill_deaths(conn, tid) == 0
        # A protocol violation is counted by its own streak, not the crash breaker —
        # the point here is only that the death kept its ordinary booking.
        assert "oom_killed" not in (run["metadata"] or "")


def test_a_stale_rate_limit_trailer_cannot_requeue_an_oom_killed_card(
    kanban_home, monkeypatch: pytest.MonkeyPatch,
):
    """A stale trailer naming the quota sentinel would requeue the card WITHOUT counting
    a failure — a silent, endless loop for a card that is really memory-bound. The
    journal outranks it, so the death is counted like any other OOM kill."""
    monkeypatch.setattr(kbd, "read_worker_oom_kill_evidence", _oom_evidence)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="heavy", assignee="a")
        _claim_dead_worker(conn, tid, 84601)
        _append_worker_log(
            tid, _OWN_OUTPUT + f"{KANBAN_WORKER_EXIT_TRAILER}{kb.KANBAN_RATE_LIMIT_EXIT_CODE}\n",
        )

        kbd.detect_crashed_workers(conn)

        run = _latest_run(conn, tid)
        assert run["outcome"] == "oom_killed"
        assert kb.get_task(conn, tid).consecutive_failures == 1


def test_the_scope_journal_is_asked_once_per_death(kanban_home, monkeypatch: pytest.MonkeyPatch):
    """One bounded ``journalctl`` per reclaimed death, whichever path classified it —
    the trailer probe and the no-trailer probe are never both run for one death."""
    calls: list[tuple] = []

    def _counted(*args, **_kwargs):
        calls.append(args)
        return None

    monkeypatch.setattr(kbd, "read_worker_oom_kill_evidence", _counted)
    with kbc.connect() as conn:
        trailer_tid = kb.create_task(conn, title="trailer death", assignee="a")
        _claim_dead_worker(conn, trailer_tid, 85001)
        _append_worker_log(conn and trailer_tid, _OWN_OUTPUT + _STALE_TRAILER)

        silent_tid = kb.create_task(conn, title="no trailer at all", assignee="a")
        _claim_dead_worker(conn, silent_tid, 85002)

        kbd.detect_crashed_workers(conn)

        assert len(calls) == 2
        assert [args[0] for args in calls] == [trailer_tid, silent_tid]
