"""
The four classifier modes, profile resolution, and profile-switch integrity.

Everything here is offline. The one test that must prove real routing
(`upstream_reused`) points the gateway's classifier at the mock upstream server,
which serves classifier replies too -- so credential reuse is demonstrated over
real HTTP with no key and no spend.

Two invariants this file exists to protect:

* `heuristics` issues zero network calls, whatever else is configured.
* Switching profiles never moves `DATA_DIR` / `SHM_CACHE_DIR`, so the SQLite
  graph and the spilled artifacts survive a profile change untouched.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import replace

import httpx
import pytest

from src.artifacts import ArtifactStore
from src.classifier import Classifier, decide_route
from src.config import (
    CLASSIFIER_MODE_AUTO,
    CLASSIFIER_MODE_EXTERNAL_JEV,
    CLASSIFIER_MODE_HEURISTICS,
    CLASSIFIER_MODE_LOCAL_OLLAMA,
    CLASSIFIER_MODE_UPSTREAM_REUSED,
    CLASSIFIER_MODES,
    available_profiles,
    find_profile,
    load_settings,
)
from src.gateway import main
from src.memory import GraphMemory

SHIPPED_PROFILES = ("antigravity", "claude", "codex", "hermes", "openrouter")
TOOL_SCHEMA = [
    {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}
]


@pytest.fixture
def clean_env():
    """`load_env_file` writes into `os.environ`; put it back afterwards.

    Without this, loading a profile file would leak `GATEWAY_PORT` and friends
    into every later test in the session.
    """
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def header(headers: dict, name: str) -> str:
    """Case-insensitive header lookup (httpx lowercases, the mock may not)."""
    for key, value in (headers or {}).items():
        if key.lower() == name.lower():
            return value
    return ""


def is_classifier_body(body: dict) -> bool:
    for message in (body or {}).get("messages") or []:
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            if "binary classifier" in message["content"].lower():
                return True
    return False


# ------------------------------------------------------------------------------
# Mode resolution
# ------------------------------------------------------------------------------
def test_auto_mode_keeps_the_original_behaviour():
    """No CLASSIFIER_MODE at all must behave exactly as it did before."""
    offline = load_settings(env={})
    assert offline.effective_classifier_mode == CLASSIFIER_MODE_HEURISTICS
    assert offline.classifier_enabled is False

    configured = load_settings(
        env={
            "CLASSIFIER_API_URL": "https://jev.test/v1/chat/completions",
            "CLASSIFIER_API_KEY": "k",
        }
    )
    assert configured.effective_classifier_mode == CLASSIFIER_MODE_EXTERNAL_JEV
    assert configured.classifier_enabled is True


def test_heuristics_mode_blanks_the_endpoint_even_when_one_is_supplied():
    settings = load_settings(
        env={
            "CLASSIFIER_MODE": "heuristics",
            "CLASSIFIER_API_URL": "https://not-for-heuristics.test/v1/chat/completions",
            "CLASSIFIER_API_KEY": "should-be-ignored",
        }
    )
    assert settings.effective_classifier_mode == CLASSIFIER_MODE_HEURISTICS
    assert settings.classifier_api_url == ""
    assert settings.classifier_api_key == ""
    assert settings.classifier_enabled is False


def test_unknown_mode_degrades_to_auto_instead_of_crashing():
    settings = load_settings(env={"CLASSIFIER_MODE": "typo_not_a_mode"})
    assert settings.effective_classifier_mode in CLASSIFIER_MODES
    assert settings.effective_classifier_mode == CLASSIFIER_MODE_HEURISTICS


def test_upstream_reused_borrows_the_upstream_endpoint_and_credential():
    settings = load_settings(
        env={
            "CLASSIFIER_MODE": "upstream_reused",
            "UPSTREAM_BASE_URL": "https://openrouter.ai/api/v1",
            "UPSTREAM_API_KEY": "subscription-key",
            "CLASSIFIER_MODEL": "gemini-2.5-flash",
        }
    )
    assert settings.effective_classifier_mode == CLASSIFIER_MODE_UPSTREAM_REUSED
    assert settings.classifier_api_url == "https://openrouter.ai/api/v1/chat/completions"
    assert settings.classifier_api_key == "subscription-key"
    assert settings.classifier_model == "gemini-2.5-flash"
    # No second provider, no second key: classification is free.
    assert settings.classifier_enabled is True


def test_upstream_reused_works_without_a_distinct_classifier_key():
    """A local bridge that ignores auth must not silently disable routing."""
    settings = load_settings(
        env={
            "CLASSIFIER_MODE": "upstream_reused",
            "UPSTREAM_BASE_URL": "http://127.0.0.1:8080/v1",
            "ALLOW_LEGACY_UPSTREAM_PORT": "1",
        }
    )
    assert settings.classifier_api_key == ""
    assert settings.classifier_enabled is True


def test_local_ollama_defaults_to_a_local_runner():
    settings = load_settings(env={"CLASSIFIER_MODE": "local_ollama"})
    assert settings.effective_classifier_mode == CLASSIFIER_MODE_LOCAL_OLLAMA
    assert settings.classifier_api_url == "http://127.0.0.1:11434/v1/chat/completions"
    assert settings.classifier_model == "qwen2.5:0.5b"
    assert settings.classifier_enabled is True
    # 11434 must never be mistaken for a forwarding loop.
    assert settings.is_loop_upstream() is False


def test_local_ollama_base_url_and_model_are_configurable():
    settings = load_settings(
        env={
            "CLASSIFIER_MODE": "local_ollama",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11435/v1/",
            "CLASSIFIER_MODEL": "llama-3.2:1b",
        }
    )
    assert settings.classifier_api_url == "http://127.0.0.1:11435/v1/chat/completions"
    assert settings.classifier_model == "llama-3.2:1b"


def test_external_jev_keeps_the_dedicated_endpoint_and_indirection():
    settings = load_settings(
        env={
            "CLASSIFIER_MODE": "external_jev",
            "JEV_API_URL": "https://jev.test/v1",
            "JEV_API_KEY_ENV": "OPENROUTER_API_KEY",
            "OPENROUTER_API_KEY": "sk-or-borrowed",
            "CLASSIFIER_PROTOCOL": "blueprint",
        }
    )
    assert settings.effective_classifier_mode == CLASSIFIER_MODE_EXTERNAL_JEV
    assert settings.classifier_api_url == "https://jev.test/v1"
    assert settings.classifier_api_key == "sk-or-borrowed"
    assert settings.classifier_protocol == "blueprint"


def test_blueprint_protocol_is_ignored_outside_external_jev():
    """`blueprint` is a JEV-specific shape; the other modes are OpenAI-shaped."""
    settings = load_settings(
        env={
            "CLASSIFIER_MODE": "local_ollama",
            "CLASSIFIER_PROTOCOL": "blueprint",
        }
    )
    assert settings.classifier_protocol == "openai"


# ------------------------------------------------------------------------------
# heuristics: provably zero network
# ------------------------------------------------------------------------------
async def test_heuristics_mode_issues_zero_network_calls(monkeypatch):
    settings = load_settings(
        env={
            "CLASSIFIER_MODE": "heuristics",
            "CLASSIFIER_API_URL": "https://must-never-be-called.invalid/v1/chat/completions",
            "CLASSIFIER_API_KEY": "leaked-key",
        }
    )
    classifier = Classifier(settings)

    def explode(*_args, **_kwargs):
        raise AssertionError("heuristics mode opened a network connection")

    # Both the client class and the transport seam are poisoned: any attempt to
    # classify would fail loudly rather than silently pass.
    monkeypatch.setattr(httpx, "AsyncClient", explode)

    assert await classifier.needs_tools("summarise the architecture decision") is None
    assert classifier.calls == 0


async def test_heuristics_mode_fails_open_and_retains_tools(settings):
    offline = replace(settings, classifier_mode=CLASSIFIER_MODE_HEURISTICS)
    classifier = Classifier(offline)

    decision = await decide_route(
        offline, classifier, "some-model", "an ambiguous medium length request", True, False
    )

    assert decision.route == "Classifier-FailOpen"
    assert decision.tool_action == "Retained-Classifier"
    assert decision.stripped is False


# ------------------------------------------------------------------------------
# upstream_reused: real HTTP against the mock
# ------------------------------------------------------------------------------
async def test_upstream_reused_routes_through_the_mock_upstream(settings, mock_state):
    resolved = load_settings(
        env={
            "CLASSIFIER_MODE": "upstream_reused",
            "UPSTREAM_BASE_URL": settings.upstream_base_url,
            "UPSTREAM_API_KEY": "test-upstream-key",
        }
    )
    classifier = Classifier(resolved)

    verdict = await classifier.needs_tools("explain how the relay works")

    assert verdict is True
    assert classifier.calls == 1

    hits = [entry for entry in mock_state.requests if is_classifier_body(entry["body"])]
    assert len(hits) == 1, "the classifier query should reach the upstream"
    assert hits[0]["path"].endswith("/chat/completions")
    assert hits[0]["body"]["model"] == resolved.classifier_model


async def test_upstream_reused_sends_the_upstream_bearer_token(settings, mock_state):
    """Credential reuse, not a second credential: the upstream's own key travels."""
    resolved = load_settings(
        env={
            "CLASSIFIER_MODE": "upstream_reused",
            "UPSTREAM_BASE_URL": settings.upstream_base_url,
            "UPSTREAM_API_KEY": "test-upstream-key",
        }
    )
    await Classifier(resolved).needs_tools("explain how the relay works")

    hits = [entry for entry in mock_state.requests if is_classifier_body(entry["body"])]
    assert header(hits[0]["headers"], "authorization") == "Bearer test-upstream-key"


async def test_upstream_reused_mode_classifies_the_escape_path(settings, store, memory, mock_state):
    """End to end: a reused classifier can still drive prune -> escape -> replay."""
    from src.gateway import create_app

    resolved = load_settings(
        env={
            "CLASSIFIER_MODE": "upstream_reused",
            "UPSTREAM_BASE_URL": settings.upstream_base_url,
            "UPSTREAM_API_KEY": "test-upstream-key",
        }
    )
    resolved = replace(
        resolved,
        data_dir=settings.data_dir,
        log_dir=settings.log_dir,
        shm_cache_dir=settings.shm_cache_dir,
        memory_injection=False,
    )
    app = create_app(settings=resolved, store=store, memory=memory)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw", timeout=30.0
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "stream": True,
                "messages": [{"role": "user", "content": "force_escape please"}],
                "tools": TOOL_SCHEMA,
            },
        )

    assert "EarlyEscapeAbort" in response.headers["x-proxy-route"]
    assert response.headers["x-proxy-tool-action"] == "Reverted-To-Full"


# ------------------------------------------------------------------------------
# local_ollama: payload formatting
# ------------------------------------------------------------------------------
def test_local_ollama_payload_is_openai_shaped():
    settings = load_settings(env={"CLASSIFIER_MODE": "local_ollama"})
    payload = Classifier(settings)._build_payload(
        "Does this need tools?", "some user state", "needs_tools"
    )

    assert payload["model"] == "qwen2.5:0.5b"
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 32
    assert payload["stream"] is False, "a streamed verdict would break the parser"
    assert payload["messages"][0]["role"] == "system"
    assert "binary classifier" in payload["messages"][0]["content"]
    assert "some user state" in payload["messages"][1]["content"]
    # JSON-serialisable, which is what the HTTP layer needs.
    assert json.loads(json.dumps(payload))


async def test_local_ollama_posts_to_the_local_runner(monkeypatch):
    settings = load_settings(env={"CLASSIFIER_MODE": "local_ollama"})
    classifier = Classifier(settings)
    seen = {}

    async def fake_post(url, payload):
        seen["url"] = url
        seen["payload"] = payload
        return {"choices": [{"message": {"content": '{"needs_tools": false}'}}]}

    monkeypatch.setattr(classifier, "_post", fake_post)

    assert await classifier.needs_tools("what is the capital of France") is False
    assert seen["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert seen["payload"]["model"] == "qwen2.5:0.5b"


# ------------------------------------------------------------------------------
# external_jev: legacy shape retained
# ------------------------------------------------------------------------------
def test_external_jev_blueprint_payload_is_unchanged():
    settings = load_settings(
        env={
            "CLASSIFIER_MODE": "external_jev",
            "CLASSIFIER_API_URL": "https://jev.test/v1",
            "CLASSIFIER_API_KEY": "k",
            "CLASSIFIER_PROTOCOL": "blueprint",
        }
    )
    payload = Classifier(settings)._build_payload("instruction", "state", "needs_tools")

    assert payload["model"] == settings.classifier_model
    assert payload["state"] == "state"
    assert payload["questions"]["needs_tools"]["type"] == "no"


def test_blueprint_probability_response_still_parses():
    settings = load_settings(
        env={
            "CLASSIFIER_MODE": "external_jev",
            "CLASSIFIER_API_URL": "https://jev.test/v1",
            "CLASSIFIER_API_KEY": "k",
            "CLASSIFIER_PROTOCOL": "blueprint",
            "CLASSIFIER_NEEDS_TOOLS_THRESHOLD": "0.15",
        }
    )
    classifier = Classifier(settings)
    verdict = classifier._interpret_probability(
        {"answers": {"needs_tools": {"probability": 0.9}}}, "needs_tools", 0.15
    )
    assert verdict is True


# ------------------------------------------------------------------------------
# Loop guard relaxation
# ------------------------------------------------------------------------------
def test_legacy_port_relaxation_is_opt_in_and_does_not_weaken_the_real_guard():
    default = load_settings(env={"GATEWAY_PORT": "8091"})
    relaxed = load_settings(
        env={"GATEWAY_PORT": "8091", "ALLOW_LEGACY_UPSTREAM_PORT": "1"}
    )

    # 8080 is the Antigravity bridge, so it must be reachable as an upstream...
    assert default.is_loop_upstream("http://127.0.0.1:8080/v1") is True
    assert relaxed.is_loop_upstream("http://127.0.0.1:8080/v1") is False
    # ...while pointing at our own port is still fatal, flag or no flag.
    assert relaxed.is_loop_upstream("http://127.0.0.1:8091/v1") is True
    assert relaxed.is_loop_upstream("http://localhost:8091/v1") is True


# ------------------------------------------------------------------------------
# Profiles
# ------------------------------------------------------------------------------
def test_all_shipped_profiles_are_discoverable():
    assert set(SHIPPED_PROFILES).issubset(set(available_profiles()))


def test_unknown_profile_exits_with_an_error(capsys):
    assert main(["--profile", "definitely-not-a-profile"]) == 2
    assert "unknown profile" in capsys.readouterr().err


def test_list_profiles_exits_cleanly(capsys):
    assert main(["--list-profiles"]) == 0
    assert "antigravity" in capsys.readouterr().out


@pytest.mark.parametrize("name", SHIPPED_PROFILES)
def test_each_shipped_profile_loads_and_has_a_valid_mode(name, clean_env):
    path = find_profile(name)
    assert path is not None, f"profile {name!r} is missing"

    settings = load_settings(env_file=path)

    assert settings.upstream_base_url.startswith("http")
    assert settings.effective_classifier_mode in CLASSIFIER_MODES
    assert settings.is_loop_upstream() is False, f"{name} would forward to itself"


def test_antigravity_profile_relaxes_the_legacy_port_and_reuses_gemini():
    settings = load_settings(env_file=find_profile("antigravity"))

    assert settings.port == 8091, "must not collide with Antigravity on 8080"
    assert settings.upstream_base_url == "http://127.0.0.1:8080/v1"
    assert settings.allow_legacy_upstream is True
    assert settings.effective_classifier_mode == CLASSIFIER_MODE_UPSTREAM_REUSED
    assert settings.classifier_model == "gemini-2.5-flash"
    assert settings.classifier_api_url == "http://127.0.0.1:8080/v1/chat/completions"


@pytest.mark.parametrize("name", SHIPPED_PROFILES)
def test_profiles_never_override_storage_directories(name):
    """The invariant that makes switching profiles safe.

    A profile that moved DATA_DIR or SHM_CACHE_DIR would strand the SQLite graph
    and orphan every spilled artifact on the next switch.
    """
    text = find_profile(name).read_text(encoding="utf-8")

    for key in ("DATA_DIR", "LOG_DIR", "SHM_CACHE_DIR"):
        assert re.search(rf"^\s*{key}=", text, re.M) is None, (
            f"profile {name!r} sets {key}, which breaks cross-profile state"
        )


def test_profile_switch_keeps_graph_and_artifacts_intact(tmp_path):
    """Switch provider AND classifier mode; state on disk must be untouched."""
    shared = {
        "DATA_DIR": str(tmp_path / "data"),
        "LOG_DIR": str(tmp_path / "logs"),
        "SHM_CACHE_DIR": str(tmp_path / "shm"),
    }

    before = load_settings(
        env={**shared, "CLASSIFIER_MODE": "heuristics", "GATEWAY_PORT": "8091"}
    )
    graph = GraphMemory(before.db_path)
    graph.init_schema()
    store = ArtifactStore(before.shm_cache_dir, 200)
    store.ensure_dir()

    assert graph.ingest_sync("s1", "The billing service uses PostgreSQL 16.", "") == 1
    payload = "B" * 500
    handle, _path = store.spill(payload, "s1")

    # Now switch to a completely different provider and classifier mode.
    after = load_settings(
        env={
            **shared,
            "CLASSIFIER_MODE": "upstream_reused",
            "UPSTREAM_BASE_URL": "https://openrouter.ai/api/v1",
            "UPSTREAM_API_KEY": "k",
            "GATEWAY_PORT": "8090",
        }
    )

    assert after.db_path == before.db_path
    assert after.shm_cache_dir == before.shm_cache_dir
    assert after.effective_classifier_mode != before.effective_classifier_mode

    # The artifact is still readable by its handle: handles are content hashes,
    # so they are deliberately profile-independent.
    assert payload in store.fetch(handle, 0, 5000)

    # The graph survived and is still in WAL mode.
    connection = sqlite3.connect(str(after.db_path))
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        rows = connection.execute("SELECT source, predicate, target FROM relations").fetchall()
    finally:
        connection.close()
    assert ("service", "uses", "PostgreSQL") in rows


def test_mode_change_is_reflected_by_the_singleton(monkeypatch):
    """`get_classifier` must rebuild when the mode changes, not cache the first."""
    from src.classifier import get_classifier

    offline = load_settings(env={})
    reused = load_settings(
        env={
            "CLASSIFIER_MODE": "upstream_reused",
            "UPSTREAM_BASE_URL": "http://127.0.0.1:9099/v1",
            "UPSTREAM_API_KEY": "k",
        }
    )

    first = get_classifier(offline)
    assert first.mode == CLASSIFIER_MODE_HEURISTICS

    second = get_classifier(reused)
    assert second.mode == CLASSIFIER_MODE_UPSTREAM_REUSED
    assert second is not first, "a stale classifier would route with the wrong policy"

    # And back again.
    assert get_classifier(offline).mode == CLASSIFIER_MODE_HEURISTICS


def test_auto_mode_constant_is_not_a_real_mode():
    assert CLASSIFIER_MODE_AUTO not in CLASSIFIER_MODES
