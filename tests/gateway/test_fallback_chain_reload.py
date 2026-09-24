"""Regression tests for #60955: gateway must not freeze fallback_providers.

Cron reloads ``fallback_providers`` from disk on every job. The gateway used to
freeze ``self._fallback_model`` at process start, so a chain configured (or
edited) after ``hermes gateway`` was already running never reached messaging
sessions — even though cron in the same process fell back correctly.

These tests pin the reload + cached-agent apply helpers without driving the
full Feishu session path.
"""

from __future__ import annotations

import time
from types import SimpleNamespace


def test_refresh_fallback_model_rereads_config(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(
        _fallback_model=None,
    )
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    chain = bound()

    assert chain == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    assert runner._fallback_model == chain

    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: anthropic/claude-sonnet-4.6\n"
    )
    updated = bound()
    assert updated == [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}
    ]
    assert runner._fallback_model == updated


def test_apply_fallback_chain_skips_while_cooldown_holds_fallback():
    """Do not clobber a live fallback activation during its cooldown window."""
    from gateway.run import GatewayRunner

    live = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    agent = SimpleNamespace(
        _fallback_chain=live,
        _fallback_model=live[0],
        _fallback_index=1,
        _fallback_activated=True,
        _rate_limited_until=time.monotonic() + 30,
    )
    GatewayRunner._apply_fallback_chain_to_agent(
        agent,
        [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}],
    )

    assert agent._fallback_chain == live
    assert agent._fallback_index == 1
    assert agent._fallback_activated is True


def test_apply_fallback_chain_adopts_a_chain_removed_under_a_live_activation(caplog):
    """A config edit that REMOVES the chain beats a cooldown armed for that chain.

    The pause guard empties ``fallback_providers`` while the OpenCode Go cap window is engaged.
    Before this, an agent that had already activated the metered Fireworks fallback ignored the
    edit for its whole cooldown — up to days when the cooldown came from the provider's own
    ``reset_at`` — so it kept paying with ``fallback_providers: []`` on disk.
    """
    import logging

    from gateway.run import GatewayRunner

    live = [{"provider": "fireworks", "model": "accounts/fireworks/models/deepseek-v4p1-flash"}]
    agent = SimpleNamespace(
        _fallback_chain=live,
        _fallback_model=live[0],
        _fallback_index=1,
        _fallback_activated=True,
        provider="fireworks",
        _rate_limited_until=time.monotonic() + 93 * 3600,   # provider-reset-sourced: days
        _rate_limit_backoff_count=4,
    )
    with caplog.at_level(logging.INFO):
        GatewayRunner._apply_fallback_chain_to_agent(agent, [])

    assert agent._fallback_chain == []
    assert agent._fallback_model is None
    assert agent._rate_limited_until == 0
    assert agent._rate_limit_backoff_count == 0
    # Activation flag stays: restore_primary_runtime owns the return trip and clears it itself.
    # Clearing it here would hit that function's own early return and strand the session on the
    # metered provider with an empty chain.
    assert agent._fallback_activated is True
    assert "clearing the cooldown so the session returns to the primary" in caplog.text


def test_apply_fallback_chain_keeps_chain_when_config_is_untouched():
    """No regression to #60955/#95066: a configured chain still does not clobber an activation."""
    from gateway.run import GatewayRunner

    live = [{"provider": "fireworks", "model": "accounts/fireworks/models/deepseek-v4p1-flash"}]
    agent = SimpleNamespace(
        _fallback_chain=["old"],
        _fallback_model="old",
        _fallback_index=0,
        _fallback_activated=True,
        provider="fireworks",
        _rate_limited_until=time.monotonic() + 93 * 3600,
        _rate_limit_backoff_count=4,
    )
    GatewayRunner._apply_fallback_chain_to_agent(agent, live)

    assert agent._fallback_chain == ["old"]
    assert agent._rate_limited_until > 0      # the cooldown is untouched on this path
    assert agent._rate_limit_backoff_count == 4




def test_load_fallback_model_static_unchanged_contract(tmp_path, monkeypatch):
    """_load_fallback_model remains a pure static reader used by refresh."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
        "fallback_model:\n"
        "  provider: nous\n"
        "  model: Hermes-4\n"
    )

    chain = GatewayRunner._load_fallback_model()
    assert chain == [
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "nous", "model": "Hermes-4"},
    ]
