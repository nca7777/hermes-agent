from __future__ import annotations

import json
import pytest

from agent import image_gen_registry
from agent.image_gen_provider import ImageGenProvider


@pytest.fixture(autouse=True)
def _reset_registry():
    image_gen_registry._reset_for_tests()
    yield
    image_gen_registry._reset_for_tests()


class _FakeCodexProvider(ImageGenProvider):
    @property
    def name(self) -> str:
        return "codex"

    def generate(self, prompt, aspect_ratio="landscape", **kwargs):
        return {
            "success": True,
            "image": "/tmp/codex-test.png",
            "model": "gpt-5.2-codex",
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "provider": "codex",
        }


class TestPluginDispatch:
    def test_dispatch_routes_to_codex_provider(self, monkeypatch, tmp_path):
        from tools import image_generation_tool
        from agent import image_gen_registry as registry_module
        from hermes_cli import plugins as plugins_module

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / "config.yaml").write_text("image_gen:\n  provider: codex\n")
        image_gen_registry.register_provider(_FakeCodexProvider())

        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")
        monkeypatch.setattr(plugins_module, "_ensure_plugins_discovered", lambda: None)
        monkeypatch.setattr(registry_module, "get_provider", lambda name: _FakeCodexProvider() if name == "codex" else None)

        dispatched = image_generation_tool._dispatch_to_plugin_provider("draw cat", "square")
        payload = json.loads(dispatched)

        assert payload["success"] is True
        assert payload["provider"] == "codex"
        assert payload["image"] == "/tmp/codex-test.png"
        assert payload["aspect_ratio"] == "square"


    def test_deepinfra_key_alone_does_not_select_image_backend(self, monkeypatch):
        """DeepInfra chat credentials do not imply consent to image billing."""
        from tools import image_generation_tool

        monkeypatch.setenv("DEEPINFRA_API_KEY", "«redacted:sk-…»")
        monkeypatch.delenv("FAL_KEY", raising=False)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: None)
        assert image_generation_tool._dispatch_to_plugin_provider("a cat", "square") is None

    def test_requirements_ignore_unselected_paid_plugin(self, monkeypatch):
        from tools import image_generation_tool

        monkeypatch.setattr(image_generation_tool, "check_fal_api_key", lambda: False)
        monkeypatch.setattr(
            image_generation_tool, "_read_configured_image_provider", lambda: None
        )
        assert image_generation_tool.check_image_generation_requirements() is False

    def test_requirements_force_refresh_on_registry_miss(self, monkeypatch):
        """Plugin discovery is once-per-process, so a long-lived session (desktop/dashboard
        serve) that resolved plugins before a backend was installed, enabled, or selected must
        still answer the gate truthfully — otherwise the avatar picker reads `available: false`
        and never dispatches, even though generation itself would have found the backend."""
        from tools import image_generation_tool

        calls: list[bool] = []

        class _AvailableProvider(_FakeCodexProvider):
            def is_available(self) -> bool:
                return True

        def _lookup(name, *, force: bool = False):
            calls.append(force)
            return _AvailableProvider() if force else None

        monkeypatch.setattr(image_generation_tool, "check_fal_api_key", lambda: False)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")
        monkeypatch.setattr(image_generation_tool, "_get_plugin_provider", _lookup)

        assert image_generation_tool.check_image_generation_requirements() is True
        assert calls == [False, True]

    def test_requirements_registered_provider_is_not_force_refreshed(self, monkeypatch):
        """A registry hit stays on the cheap path: no forced plugin reload per probe."""
        from tools import image_generation_tool

        calls: list[bool] = []

        class _AvailableProvider(_FakeCodexProvider):
            def is_available(self) -> bool:
                return True

        def _lookup(name, *, force: bool = False):
            calls.append(force)
            return _AvailableProvider() if not force else None

        monkeypatch.setattr(image_generation_tool, "check_fal_api_key", lambda: False)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "codex")
        monkeypatch.setattr(image_generation_tool, "_get_plugin_provider", _lookup)

        assert image_generation_tool.check_image_generation_requirements() is True
        assert calls == [False]

    def test_requirements_false_when_provider_absent_after_refresh(self, monkeypatch):
        """A genuinely missing backend still reports False after the refresh — no false positive."""
        from tools import image_generation_tool

        monkeypatch.setattr(image_generation_tool, "check_fal_api_key", lambda: False)
        monkeypatch.setattr(image_generation_tool, "_read_configured_image_provider", lambda: "nope")
        monkeypatch.setattr(
            image_generation_tool, "_get_plugin_provider", lambda name, force=False: None
        )
        assert image_generation_tool.check_image_generation_requirements() is False
