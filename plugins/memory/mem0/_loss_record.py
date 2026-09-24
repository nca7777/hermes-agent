"""Durable record of a turn whose fact extraction was dropped (kanban t_240ddbcf).

Why this exists
---------------
The `infer=true` path (``sync_turn`` -> server-side fact extraction) is the ONE memory write whose
latency is set by an LLM hop, and it is also the only write whose failure had no durable trace: a
dropped turn produced one ``Mem0 sync failed: ...`` warning in that home's log and nothing else, so
finding it required scanning every home's logs (``mem0_extraction_losses.py``) and then correlating
the turn back out of ``state.db`` by timestamp. A client-side timeout made the turn vanish silently.

This module makes the drop a FIRST-CLASS, machine-readable event:

    <HERMES_HOME>/mem0_extraction_losses.jsonl      (one JSON object per line, mode 600)

It is written by the plugin itself, at the moment the drop is known, so the ledger/audit reads it
without a log scan - and the record carries the turn's own text, so the replay worker no longer has
to reconstruct it from a neighbouring timestamp.

Contract
--------
* ``record`` NEVER raises and never wedges the caller: a failure to record is reported by a
  ``False`` return, and the warning log line is still written by the caller either way.
* A turn's text is a credential risk: any message containing a secret-shaped token is stored
  REDACTED (the whole message becomes a marker, not a partial mask) and flagged ``redacted: true``.
  No key, token or password may reach this file even though it sits beside ``state.db``.
* Events other than ``lost`` exist because a dropped turn has more than one shape: a turn skipped
  because the previous sync was still in flight (``skipped_busy``), one skipped while the circuit
  breaker is open (``skipped_breaker``), and one skipped because there is no backend at all
  (``skipped_no_backend``). All four lose the turn's facts; only the first was ever logged.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

FILE_NAME = "mem0_extraction_losses.jsonl"
SCHEMA_VERSION = 1
_MAX_BYTES = 5 * 1024 * 1024

# A drop class vocabulary that matches mem0_extraction_losses.py exactly, so a record and the log
# line for the same event classify identically and the harvester can dedupe them.
_CLASSES = (
    (re.compile(r"502|Bad Gateway", re.I), "502_bad_gateway"),
    (re.compile(r"401|Unauthorized", re.I), "auth_401"),
    (re.compile(r"timed out|timeout", re.I), "timeout"),
    (re.compile(r"Connection refused|Errno 111", re.I), "connection_refused"),
    (re.compile(r"Connection reset|Errno 104", re.I), "connection_reset"),
    (re.compile(r"Server disconnected", re.I), "server_disconnected"),
    (re.compile(r"Bad file descriptor|Errno 9", re.I), "bad_file_descriptor"),
)
_DEFAULT_CLASS = "other_error"

# Secret shapes worth refusing to persist. Deliberately coarse: a false positive costs one
# unreadable turn in a loss ledger, a false negative writes a live credential to disk.
_SECRET_SHAPES = (
    re.compile(r"\b[0-9a-fA-F]{64}\b"),                       # the fleet's mem0/admin keys
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),                  # OpenAI-style
    re.compile(r"\b(?:gh[pousr]|xox[baprs])-[A-Za-z0-9_\-]{10,}"),  # GitHub / Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                      # AWS access key id
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

_write_lock = threading.Lock()


def classify_error(exc: BaseException | str | None) -> str:
    """Drop class for an exception/string; same vocabulary as the log harvest."""
    body = "" if exc is None else (exc if isinstance(exc, str) else str(exc))
    for rx, label in _CLASSES:
        if rx.search(body):
            return label
    return _DEFAULT_CLASS


def _contains_secret(text: str) -> bool:
    return any(rx.search(text or "") for rx in _SECRET_SHAPES)


def _redact(text: str) -> tuple[str, bool]:
    """(text-to-store, redacted?) - the WHOLE message goes when any secret shape is present."""
    if text and _contains_secret(text):
        return "[redacted: message contained a secret-shaped token]", True
    return (text or ""), False


def _home_dir() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home())


def _rotate(path: str) -> None:
    """Keep the ledger bounded: one generation, oldest dropped. Called with the lock held."""
    try:
        if os.path.getsize(path) > _MAX_BYTES:
            os.replace(path, path + ".1")
    except OSError:
        pass


def record(*, event: str, exc: BaseException | None = None, session: str | None = None,
           turn: list[dict] | None = None, attempts: int | None = None,
           reason: str | None = None, home: str | None = None) -> dict[str, Any] | None:
    """Append one drop record. Returns the record written, or ``None`` if it could not be written.

    ``event`` is one of ``lost``/``skipped_busy``/``skipped_breaker``/``skipped_no_backend``.
    ``turn`` is the message list actually handed to the backend (truncated as sent).
    """
    try:
        user = asst = ""
        redacted = False
        for m in turn or []:
            content = str(m.get("content") or "")
            if m.get("role") == "user":
                user, r = _redact(content)
                redacted = redacted or r
            elif m.get("role") == "assistant":
                asst, r = _redact(content)
                redacted = redacted or r
        now = time.time()
        cls = classify_error(exc) if event == "lost" else event
        rec: dict[str, Any] = {
            "v": SCHEMA_VERSION,
            "at": datetime.fromtimestamp(now, timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "ts": now,
            "event": event,
            "class": cls,
            "loss": True,
            "session": session,
            "attempts": attempts,
            "reason": reason,
            "error": None if exc is None else str(exc)[:500],
            "redacted": redacted,
            "turn": {"user": user, "assistant": asst},
            "chars": len(asst) if asst else len(user),
        }
        base = home or _home_dir()
        path = os.path.join(base, FILE_NAME)
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with _write_lock:
            _rotate(path)
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
        return rec
    except Exception as e:  # never let the ledger cost a turn: the warning line still fires
        logger.debug("mem0 drop record could not be written: %s", e)
        return None
