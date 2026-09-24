"""The auxiliary client must not keep sending a credential the pool has benched.

A route whose ``api_key`` was resolved outside the pool (the session's runtime key, an aux
slot's ``api_key``) does not follow pool rotation: after a rotation the retry re-sends the
credential the pool just benched, and the rejection that comes back is attributed to the
entry the pool rotated TO. That benches a live account for the provider's whole reset
window — an OpenCode Go weekly cap benched a fresh account until Sunday's reset while it
answered 200. ``_get_cached_client`` now prefers the pool's live entry whenever the
resolved key is one the pool has already benched.
"""

from __future__ import annotations

import json
import logging
import os

import agent.auxiliary_client as aux


def _entry(idx: int, *, token: str, benched: bool = False) -> dict:
    row = {
        "id": f"go-{idx}",
        "label": f"go-account-{idx}",
        "auth_type": "api_key",
        "priority": idx,
        "source": "manual",
        "access_token": token,
        "base_url": "https://go.example.test/v1",
    }
    if benched:
        row.update(
            {
                "last_status": "exhausted",
                "last_status_at": 1.0,
                "last_error_code": 429,
                "last_error_reason": "usage_limit_reached",
                "last_error_reset_at": "2026-09-27T20:00:00Z",
            }
        )
    return row


def _home(tmp_path, monkeypatch, entries) -> None:
    home = tmp_path / "hermes"
    home.mkdir(parents=True, exist_ok=True)
    (home / "auth.json").write_text(
        json.dumps({"version": 1, "credential_pool": {"opencode-go": entries}})
    )
    # Env seeding would add a live row of its own and hide what the rows say.
    for name in [k for k in os.environ if k.startswith("OPENCODE_GO_API_KEY")]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(aux, "_client_cache", {})


def _capture(monkeypatch) -> dict:
    seen: dict = {}

    def _fake(provider, model, async_mode, **kwargs):
        seen.update(kwargs)
        seen["provider"] = provider
        return None, None

    monkeypatch.setattr(aux, "resolve_provider_client", _fake)
    return seen


def test_benched_resolved_key_is_replaced_by_the_pool_live_entry(tmp_path, monkeypatch, caplog):
    """The reported defect: the retry after a rotation kept the benched key."""
    _home(
        tmp_path, monkeypatch,
        [_entry(0, token="key-a", benched=True), _entry(1, token="key-b")],
    )
    seen = _capture(monkeypatch)

    with caplog.at_level(logging.INFO):
        aux._get_cached_client("opencode-go", "deepseek-v4.1-flash", api_key="key-a")

    assert seen["explicit_api_key"] == "key-b"
    assert "benched pool entry" in caplog.text


def test_live_resolved_key_is_left_alone(tmp_path, monkeypatch, caplog):
    """A key the pool has not benched must not be swapped for another entry."""
    _home(
        tmp_path, monkeypatch,
        [_entry(0, token="key-a", benched=True), _entry(1, token="key-b")],
    )
    seen = _capture(monkeypatch)

    with caplog.at_level(logging.INFO):
        aux._get_cached_client("opencode-go", "deepseek-v4.1-flash", api_key="key-b")

    assert seen["explicit_api_key"] == "key-b"
    assert "benched pool entry" not in caplog.text


def test_resolved_key_outside_the_pool_is_left_alone(tmp_path, monkeypatch):
    """An aux-slot key that belongs to no pool row says nothing about the pool."""
    _home(
        tmp_path, monkeypatch,
        [_entry(0, token="key-a", benched=True), _entry(1, token="key-b")],
    )
    seen = _capture(monkeypatch)

    aux._get_cached_client("opencode-go", "deepseek-v4.1-flash", api_key="an-aux-slot-key")

    assert seen["explicit_api_key"] == "an-aux-slot-key"


def test_no_resolved_key_still_derives_from_the_pool(tmp_path, monkeypatch):
    """Pre-existing behaviour: with no key of its own the client follows the pool."""
    _home(
        tmp_path, monkeypatch,
        [_entry(0, token="key-a", benched=True), _entry(1, token="key-b")],
    )
    seen = _capture(monkeypatch)

    aux._get_cached_client("opencode-go", "deepseek-v4.1-flash")

    assert seen["explicit_api_key"] == "key-b"


def test_all_entries_benched_leaves_the_resolved_key_untouched(tmp_path, monkeypatch):
    """Nothing live to prefer: the resolved key stands and the caller sees the rejection."""
    _home(
        tmp_path, monkeypatch,
        [_entry(0, token="key-a", benched=True), _entry(1, token="key-b", benched=True)],
    )
    seen = _capture(monkeypatch)

    aux._get_cached_client("opencode-go", "deepseek-v4.1-flash", api_key="key-b")

    assert seen["explicit_api_key"] == "key-b"
