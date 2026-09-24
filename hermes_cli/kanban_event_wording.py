"""How a kanban terminal event reads to an operator — the facts, shared by both notifier surfaces.

The gateway watcher (``gateway/kanban_watchers_notifier.py``) and the TUI/Desktop/dashboard
poller (``tui_gateway/session_notifications.py``) render the same board events for different
transports. The FACTS a notice asserts are derived here once, so the two surfaces cannot drift
into telling an operator different things about one card; only the sentence around them differs.

Two defects this module exists to prevent, both observed on live cards:

* **A NULL optional field rendered as a number.** ``tasks.max_runtime_seconds`` is nullable and
  means "no cap was set". Formatting the NULL with ``... or 0`` emitted
  ``timed out (max_runtime=0s)`` — a claim that the card had a zero-second cap, which is
  nonsense and points the operator at the wrong remedy. Absence is not zero:
  :func:`timed_out_limit_seconds` returns ``None`` and the caller must print no duration at all.
* **A cause label hardcoded to one of two unrelated causes.** ``timed_out`` is emitted for two
  different failures: a worker that exceeded its per-task runtime cap
  (``kanban_db_dispatch.enforce_max_runtime``) and a worker that ran out of ITERATIONS
  (``agent/turn_finalizer._record_kanban_budget_exhausted``). The remedies differ — scope or
  checkpoint the card versus give it more seconds — so the notice must name the cause it has.

Emitters stamp ``cause`` on the event payload; :func:`timed_out_cause` prefers that declaration and
otherwise infers it from the emitter's own fields — and, for events already on the board (written
before the stamp existed), from the emitter's own error sentence, so a stored card reads correctly
too.

Any NEW optional field rendered into a notice must follow the same rule: a missing value gets no
rendered value, never a stand-in ``0``.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Tuple

# ``timed_out`` causes. A worker out of iterations is not a worker out of time.
CAUSE_ITERATION_BUDGET = "iteration_budget"
CAUSE_RUNTIME_LIMIT = "runtime_limit"
# The event records neither cause (a pre-stamp event, or a stop with no cap and no budget).
CAUSE_UNKNOWN = "unknown"
_KNOWN_CAUSES = (CAUSE_ITERATION_BUDGET, CAUSE_RUNTIME_LIMIT)

# The budget emitter's own error sentence (``agent/turn_finalizer._record_kanban_budget_exhausted``),
# matched only as a FALLBACK for events written before the cause stamp existed. The counts are
# carried inside it, so a stored event from before this change can still be named correctly.
_BUDGET_ERROR_RE = re.compile(r"iteration\s+budget\s+exhausted\s*\(\s*(\d+)\s*/\s*(\d+)\s*\)", re.I)
_BUDGET_ERROR_BARE_RE = re.compile(r"iteration\s+budget\s+exhausted", re.I)


def _as_int(value: Any) -> Optional[int]:
    """``int`` for a numeric value; ``None`` for unset, blank or non-numeric (never raises).

    Payload fields arrive from SQLite, from JSON, and from worker processes, so a value can be a
    string, a float, or junk. ``bool`` is rejected rather than read as 0/1 — a flag is not a number.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def timed_out_limit_seconds(payload: Any) -> Optional[int]:
    """The runtime cap a ``timed_out`` event stopped at, or ``None`` when no cap was recorded.

    ``None`` means the card had no cap (or the field is unreadable) — NOT a cap of zero — and the
    caller must omit the duration instead of rendering ``0s``. A cap that is genuinely present is
    returned as-is, ``0`` included, because then the number is a fact about the card.
    """
    if not isinstance(payload, Mapping) or "limit_seconds" not in payload:
        return None
    return _as_int(payload.get("limit_seconds"))


def timed_out_iteration_budget(payload: Any) -> Optional[Tuple[int, int]]:
    """``(used, max)`` when the event records an iteration-budget exhaustion, else ``None``.

    ``budget_max`` is the marker: the budget emitter stamps it, the runtime-cap emitter never does.
    A missing ``budget_used`` falls back to the max so the rendered pair is never ``None/N``. For an
    event written before the stamp existed, the counts are recovered from the emitter's own error
    sentence instead — the only record that event carries.
    """
    if not isinstance(payload, Mapping):
        return None
    total = _as_int(payload.get("budget_max"))
    if total is not None:
        used = _as_int(payload.get("budget_used"))
        return (total if used is None else used, total)
    match = _BUDGET_ERROR_RE.search(str(payload.get("error") or ""))
    return (int(match.group(1)), int(match.group(2))) if match else None


def _error_names_iteration_budget(payload: Any) -> bool:
    """True when the event's own error sentence names an iteration-budget exhaustion (no counts)."""
    return bool(isinstance(payload, Mapping)
                and _BUDGET_ERROR_BARE_RE.search(str(payload.get("error") or "")))


def timed_out_cause(payload: Any) -> str:
    """Which failure a ``timed_out`` event actually records: one of the ``CAUSE_*`` values.

    A declared ``cause`` wins; otherwise the emitter's own fields decide, and an event written
    before the stamp existed is read from its own error sentence. An event carrying a runtime cap
    AND budget evidence is a contradiction the emitters cannot produce, so the first match wins and
    the rest is only ever a fallback.
    """
    if not isinstance(payload, Mapping):
        return CAUSE_UNKNOWN
    declared = str(payload.get("cause") or "").strip().lower()
    if declared in _KNOWN_CAUSES:
        return declared
    if timed_out_iteration_budget(payload) is not None or _error_names_iteration_budget(payload):
        return CAUSE_ITERATION_BUDGET
    if timed_out_limit_seconds(payload) is not None:
        return CAUSE_RUNTIME_LIMIT
    return CAUSE_UNKNOWN


def iteration_budget_clause(payload: Any) -> str:
    """``iteration budget exhausted (80/80)`` — counts omitted only when the event lacks them."""
    budget = timed_out_iteration_budget(payload)
    return (
        f"iteration budget exhausted ({budget[0]}/{budget[1]})" if budget
        else "iteration budget exhausted"
    )
