"""The ``timed_out`` notice must name the failure it actually has, and never invent a duration.

Two defects on this surface, both seen on live cards:

* A card whose worker ran out of ITERATIONS (``agent/turn_finalizer`` records it as ``timed_out``)
  was announced as a card that ran too long — a different failure with a different remedy.
* A card with ``max_runtime_seconds = NULL`` was announced as ``timed out (max_runtime=0s)``: the
  NULL formatted as a number, i.e. a zero-second cap the card never had.

These tests drive the real emitters and the real formatters, so a wording change that only looks
right in the source cannot pass.
"""

import json

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_event_wording import (
    CAUSE_ITERATION_BUDGET,
    CAUSE_RUNTIME_LIMIT,
    CAUSE_UNKNOWN,
    iteration_budget_clause,
    timed_out_cause,
    timed_out_iteration_budget,
    timed_out_limit_seconds,
)


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


# --------------------------------------------------------------------------- the shared facts
class TestTimedOutFacts:
    def test_missing_cap_is_none_never_zero(self):
        """Absence of ``limit_seconds`` is an unset cap, not a zero-second one."""
        assert timed_out_limit_seconds({}) is None
        assert timed_out_limit_seconds({"limit_seconds": None}) is None
        assert timed_out_limit_seconds(None) is None
        assert timed_out_limit_seconds({"limit_seconds": "not-a-number"}) is None
        assert timed_out_limit_seconds({"limit_seconds": 0}) == 0

    def test_cause_comes_from_the_stamp_then_the_emitters_own_fields(self):
        assert timed_out_cause({"cause": CAUSE_ITERATION_BUDGET, "budget_max": 80}) == CAUSE_ITERATION_BUDGET
        assert timed_out_cause({"budget_used": 80, "budget_max": 80}) == CAUSE_ITERATION_BUDGET  # pre-stamp
        assert timed_out_cause({"limit_seconds": 900}) == CAUSE_RUNTIME_LIMIT
        assert timed_out_cause({"cause": CAUSE_RUNTIME_LIMIT}) == CAUSE_RUNTIME_LIMIT
        assert timed_out_cause({}) == CAUSE_UNKNOWN
        assert timed_out_cause(None) == CAUSE_UNKNOWN

    def test_pre_stamp_event_is_read_from_its_own_error_sentence(self):
        """The stored events already on the board carry no cause field — only the emitter's sentence."""
        live = {"error": "Iteration budget exhausted (80/80) — task could not complete within the "
                         "allowed iterations", "failures": 1, "retry_status": "ready"}
        assert timed_out_cause(live) == CAUSE_ITERATION_BUDGET
        assert timed_out_iteration_budget(live) == (80, 80)
        assert iteration_budget_clause(live) == "iteration budget exhausted (80/80)"
        # A cap stop's sentence never matches, and a stop with no cause stays unknown.
        assert timed_out_cause({"error": "elapsed 300s > limit 60s", "limit_seconds": 60}) == CAUSE_RUNTIME_LIMIT
        assert timed_out_cause({"error": "something else entirely"}) == CAUSE_UNKNOWN

    def test_iteration_budget_pair_and_clause(self):
        assert timed_out_iteration_budget({"budget_used": 80, "budget_max": 80}) == (80, 80)
        assert timed_out_iteration_budget({"budget_max": 12}) == (12, 12)
        assert timed_out_iteration_budget({"limit_seconds": 60}) is None
        assert iteration_budget_clause({"budget_used": 80, "budget_max": 80}) == "iteration budget exhausted (80/80)"
        assert iteration_budget_clause({"cause": CAUSE_ITERATION_BUDGET}) == "iteration budget exhausted"


# --------------------------------------------------------------------------- the TUI/Desktop notice
class TestTuiNotice:
    SUB = {"task_id": "t_abc123"}
    TASK = type("T", (), {"title": "build the thing", "assignee": "worker", "result": None})()

    def _text(self, payload):
        from types import SimpleNamespace

        from tui_gateway.server import _format_kanban_event_text

        ev = SimpleNamespace(kind="timed_out", payload=payload)
        return _format_kanban_event_text(self.SUB, self.TASK, ev, "")

    def test_iteration_budget_is_named_not_called_a_timeout(self):
        text = self._text({"cause": CAUSE_ITERATION_BUDGET, "budget_used": 80, "budget_max": 80})
        assert "iteration budget exhausted (80/80); will retry" in text
        assert "max_runtime" not in text and "0s" not in text

    def test_no_cap_prints_no_duration(self):
        text = self._text({"error": "x", "failures": 1, "retry_status": "ready"})
        assert text.endswith("timed out; will retry")
        assert "0s" not in text and "max_runtime" not in text

    def test_the_live_card_that_reported_the_bug_now_reads_its_cause(self):
        """The exact payload stored on the card that produced ``timed out (max_runtime=0s)``."""
        text = self._text({
            "error": "Iteration budget exhausted (80/80) — task could not complete within the "
                     "allowed iterations",
            "failures": 1,
            "retry_status": "ready",
        })
        assert "iteration budget exhausted (80/80); will retry" in text
        assert "0s" not in text and "max_runtime" not in text

    def test_real_cap_keeps_its_number(self):
        assert "max_runtime=900s)" in self._text({"cause": CAUSE_RUNTIME_LIMIT, "limit_seconds": 900})

    def test_bad_payload_still_renders_without_a_duration(self):
        text = self._text({"limit_seconds": "not-a-number"})
        assert "timed out" in text and "0s" not in text


# --------------------------------------------------------------------------- the Telegram notice
class TestGatewayNotice:
    def _text(self, payload):
        from types import SimpleNamespace

        from gateway.kanban_watchers_notifier import _fmt_timed_out

        head = SimpleNamespace(head="Kanban t_abc123")
        ev = SimpleNamespace(kind="timed_out", payload=payload)
        return _fmt_timed_out(ev, head)[0]

    def test_iteration_budget_names_the_remedy(self):
        text = self._text({"cause": CAUSE_ITERATION_BUDGET, "budget_used": 80, "budget_max": 80})
        assert "iteration budget exhausted (80/80)" in text
        assert "time limit" not in text and "minute" not in text

    def test_no_cap_is_not_narrated_as_a_time_limit(self):
        text = self._text({"error": "x", "failures": 1})
        assert "was stopped by the dispatcher" in text
        assert "no runtime cap set" in text and "0s" not in text and "minute" not in text

    def test_real_cap_keeps_the_existing_wording(self):
        text = self._text({"cause": CAUSE_RUNTIME_LIMIT, "limit_seconds": 900})
        assert "ran past its 15-minute limit and was stopped" in text


# --------------------------------------------------------------------------- the event that carries the cause
class TestEventCarriesItsCause:
    def test_non_tripping_failure_keeps_the_callers_event_fields(self, board):
        """``event_payload_extra`` reaches the event on BOTH failure branches.

        It used to be merged only when the breaker tripped, so a worker that exhausted its
        iterations — the ordinary, below-threshold case — emitted an event with no cause at all
        and the notice had nothing to name.
        """
        conn = board
        tid = kb.create_task(conn, title="capless budget exhaustion", assignee="worker")
        kb.claim_task(conn, tid)
        kbd._record_task_failure(
            conn, tid,
            error="Iteration budget exhausted (80/80) — task could not complete within the allowed iterations",
            outcome="timed_out",
            release_claim=True,
            end_run=True,
            event_payload_extra={"cause": CAUSE_ITERATION_BUDGET, "budget_used": 80, "budget_max": 80},
        )
        row = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'timed_out'", (tid,)
        ).fetchone()
        payload = json.loads(row["payload"])
        assert payload["cause"] == CAUSE_ITERATION_BUDGET
        assert timed_out_iteration_budget(payload) == (80, 80)
        assert kb.get_task(conn, tid).status == "ready"  # below the breaker: retried, not blocked

        # The real event, through the real notice: the operator reads the cause.
        from types import SimpleNamespace

        from tui_gateway.server import _format_kanban_event_text

        task = kb.get_task(conn, tid)
        text = _format_kanban_event_text(
            {"task_id": tid}, task, SimpleNamespace(kind="timed_out", payload=payload), ""
        )
        assert "iteration budget exhausted (80/80)" in text
        assert "max_runtime" not in text
