"""
Heuristics and fast-path routing.

These are the sub-millisecond decisions taken before any network call, so they
are tested directly as pure functions and then again end-to-end through the
ASGI app to prove the decision actually reaches the upstream request.
"""

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

from src.classifier import (
    Classifier,
    decide_route,
    evaluate_fast_path,
    inject_escape_instruction,
    is_reasoning_model,
    parse_classifier_bool,
    strip_escape_instruction,
)
from src.config import ESCAPE_TOKEN, load_settings, sanitize_url
from src.gateway import create_app

TOOL_SCHEMA = [
    {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}
]


def body_for(prompt: str, model: str = "mock-model", **extra):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOL_SCHEMA,
        **extra,
    }


# ------------------------------------------------------------------------------
# Greeting / conversational detection
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt",
    ["hi", "Hello", "hey", "yo", "thanks", "thank you", "ok", "okay", "cool",
     "understood", "great", "good morning", "bye", "مرحبا", "أهلا", "شكرا", "تمام"],
)
def test_greetings_are_stripped(prompt):
    assert evaluate_fast_path(prompt) == "strip_tools"


@pytest.mark.parametrize("prompt", ["hi!", "Hello.", "thanks!", "OK?", "hey!!"])
def test_greeting_punctuation_is_tolerated(prompt):
    assert evaluate_fast_path(prompt) == "strip_tools"


def test_long_greeting_like_text_is_not_stripped():
    """"hi" buried in a sentence is not a greeting."""
    assert evaluate_fast_path("hi, could you summarise this document for me please") is None


# ------------------------------------------------------------------------------
# Action detection
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt",
    [
        "run the test suite",
        "please build the project",
        "git commit these changes",
        "cat src/main.py",
        "grep -r TODO .",
        "docker compose up",
        "pip install requests",
        "npm run dev",
        "what does src/config.py do?",
        "look at ./README.md",
        "```python\nprint(1)\n```",
        "refactor the auth module",
        "debug this traceback",
        "migrate the database",
    ],
)
def test_action_prompts_keep_tools(prompt):
    assert evaluate_fast_path(prompt) == "keep_tools"


@pytest.mark.parametrize(
    "prompt",
    [
        "what is the capital of France?",
        "explain how a transformer works in general terms",
        "tell me a story about a lighthouse keeper",
    ],
)
def test_ambiguous_prompts_need_a_decision(prompt):
    assert evaluate_fast_path(prompt) is None


def test_empty_and_missing_prompt_is_not_a_verdict():
    assert evaluate_fast_path("") is None
    assert evaluate_fast_path("   ") is None
    assert evaluate_fast_path(None) is None


# ------------------------------------------------------------------------------
# Reasoning model detection
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model",
    [
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini",
        "deepseek/deepseek-r1", "deepseek-ai/deepseek-r1", "deepseek-r1-0528",
        "some/reasoner-model", "vendor/thinking-32b", "qwen3-r1-distill",
    ],
)
def test_reasoning_models_are_detected(model):
    assert is_reasoning_model(model) is True


@pytest.mark.parametrize(
    "model", ["gpt-4o", "gpt-4o-mini", "claude-sonnet-4", "meta/llama-3.1-8b", "mock-model"]
)
def test_standard_models_are_not_reasoning(model):
    assert is_reasoning_model(model) is False


def test_extra_reasoning_regex_is_honoured():
    assert is_reasoning_model("acme/custom-cot", ("custom-cot",)) is True
    assert is_reasoning_model("acme/custom-cot") is False
    # A malformed pattern must not raise.
    assert is_reasoning_model("acme/custom-cot", ("[unclosed",)) is False


# ------------------------------------------------------------------------------
# Classifier reply parsing
# ------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"needs_tools": true}', True),
        ('{"needs_tools": false}', False),
        ('```json\n{"needs_tools": true}\n```', True),
        ('{"needs_tools": "false"}', False),
        ('{"needs_tools": 1}', True),
        ("Yes, this needs tools.", True),
        ("No.", False),
        ("true", True),
        ("false", False),
    ],
)
def test_classifier_reply_parsing(text, expected):
    assert parse_classifier_bool(text, "needs_tools") is expected


@pytest.mark.parametrize("text", ["", "I am not sure, maybe?", "¯\\_(ツ)_/¯", "perhaps"])
def test_unparseable_replies_are_not_verdicts(text):
    assert parse_classifier_bool(text, "needs_tools") is None


# ------------------------------------------------------------------------------
# Escape instruction plumbing
# ------------------------------------------------------------------------------
def test_escape_instruction_is_injected_and_removable():
    messages = [{"role": "user", "content": "hello"}]
    created = inject_escape_instruction(messages)
    assert created is True
    assert messages[0]["role"] == "system"
    assert ESCAPE_TOKEN in messages[0]["content"]

    messages[0]["content"] = strip_escape_instruction(messages[0]["content"])
    assert messages[0]["content"].strip() == ""


def test_escape_instruction_appends_to_existing_system_message():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
    ]
    created = inject_escape_instruction(messages)
    assert created is False
    assert len(messages) == 2
    assert ESCAPE_TOKEN in messages[0]["content"]
    assert strip_escape_instruction(messages[0]["content"]) == "You are helpful."


# ------------------------------------------------------------------------------
# Routing decisions (no network)
# ------------------------------------------------------------------------------
async def test_route_reasoning_model_always_retains_tools(settings, classifier):
    decision = await decide_route(
        settings, classifier, "deepseek/deepseek-r1", "hello", True, False
    )
    assert decision.tool_action == "Retained-Full"
    assert decision.route == "Reasoning-Passthrough"
    assert decision.stripped is False


async def test_route_without_tools_is_unmodified(settings, classifier):
    decision = await decide_route(settings, classifier, "m", "hello", False, False)
    assert decision.tool_action == "Unmodified"


async def test_route_mid_tool_loop_is_unmodified(settings, classifier):
    decision = await decide_route(settings, classifier, "m", "hello", True, True)
    assert decision.tool_action == "Unmodified"
    assert decision.route == "ToolLoop"


async def test_route_fails_open_when_classifier_disabled(settings, classifier):
    """No classifier configured -> keep the tools, never silently drop them."""
    decision = await decide_route(
        settings, classifier, "m", "explain how a transformer works in general terms", True, False
    )
    assert decision.tool_action == "Retained-Classifier"
    assert decision.route == "Classifier-FailOpen"
    assert decision.stripped is False


# ------------------------------------------------------------------------------
# End-to-end: the decision must reach the upstream request
# ------------------------------------------------------------------------------
async def test_greeting_strips_tools_upstream(api, mock_state):
    response = await api.post("/v1/chat/completions", json=body_for("hello"))
    assert response.status_code == 200
    assert response.headers["x-proxy-tool-action"] == "Stripped-ZeroTokens"
    assert response.headers["x-proxy-route"] == "FastPath-Strip"
    assert mock_state.bodies, "gateway did not reach the upstream"
    assert "tools" not in mock_state.bodies[0]


async def test_action_prompt_forwards_tools_upstream(api, mock_state):
    response = await api.post("/v1/chat/completions", json=body_for("run the test suite"))
    assert response.status_code == 200
    assert response.headers["x-proxy-route"] == "FastPath-Keep"
    assert "tools" in mock_state.bodies[0]


async def test_reasoning_model_keeps_tools_even_for_a_greeting(api, mock_state):
    response = await api.post(
        "/v1/chat/completions", json=body_for("hello", model="deepseek/deepseek-r1")
    )
    assert response.headers["x-proxy-route"] == "Reasoning-Passthrough"
    assert response.headers["x-proxy-tool-action"] == "Retained-Full"
    assert "tools" in mock_state.bodies[0]


async def test_synthetic_fetch_log_is_registered(api, mock_state):
    await api.post("/v1/chat/completions", json=body_for("run the tests"))
    names = [tool["function"]["name"] for tool in mock_state.bodies[0]["tools"]]
    assert "fetch_log" in names


async def test_unknown_fields_are_forwarded_transparently(api, mock_state):
    """Invariant: protocol transparency -- this is an OpenAI-compatible proxy."""
    payload = body_for(
        "run the test suite",
        temperature=0.42,
        top_p=0.9,
        seed=7,
        response_format={"type": "json_object"},
        some_future_field={"nested": [1, 2, 3]},
    )
    await api.post("/v1/chat/completions", json=payload)
    forwarded = mock_state.bodies[0]
    assert forwarded["temperature"] == 0.42
    assert forwarded["top_p"] == 0.9
    assert forwarded["seed"] == 7
    assert forwarded["response_format"] == {"type": "json_object"}
    assert forwarded["some_future_field"] == {"nested": [1, 2, 3]}


async def test_malformed_body_is_rejected_cleanly(api):
    response = await api.post(
        "/v1/chat/completions",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


# ------------------------------------------------------------------------------
# Classifier integration
# ------------------------------------------------------------------------------
async def test_classifier_can_strip_an_ambiguous_prompt(settings, store, memory):
    tuned = replace(
        settings,
        classifier_api_url="https://classifier.test/v1/chat/completions",
        classifier_api_key="test-key",
        upstream_api_key="k",
    )
    classifier = Classifier(tuned)

    async def fake_post(url, payload):
        return {"choices": [{"message": {"content": '{"needs_tools": false}'}}]}

    classifier._post = fake_post  # type: ignore[assignment]

    app = create_app(settings=tuned, classifier=classifier, store=store, memory=memory)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        response = await client.post(
            "/v1/chat/completions",
            json=body_for("explain how a transformer works in general terms"),
        )
    assert response.headers["x-proxy-route"] == "Classifier-Strip"
    assert response.headers["x-proxy-tool-action"] == "Stripped-ZeroTokens"


async def test_classifier_failure_fails_open_and_keeps_tools(settings, store, memory):
    tuned = replace(
        settings,
        classifier_api_url="https://classifier.test/v1/chat/completions",
        classifier_api_key="test-key",
    )
    classifier = Classifier(tuned)

    async def failing_post(url, payload):
        return None  # simulates timeout / non-200 / unparseable

    classifier._post = failing_post  # type: ignore[assignment]

    app = create_app(settings=tuned, classifier=classifier, store=store, memory=memory)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        response = await client.post(
            "/v1/chat/completions",
            json=body_for("explain how a transformer works in general terms"),
        )
    assert response.headers["x-proxy-tool-action"] == "Retained-Classifier"
    assert response.headers["x-proxy-route"] == "Classifier-FailOpen"


async def test_classifier_decisions_are_cached(settings):
    tuned = replace(
        settings,
        classifier_api_url="https://classifier.test/v1/chat/completions",
        classifier_api_key="test-key",
    )
    classifier = Classifier(tuned)
    calls = {"n": 0}

    async def counting_post(url, payload):
        calls["n"] += 1
        return {"choices": [{"message": {"content": '{"needs_tools": true}'}}]}

    classifier._post = counting_post  # type: ignore[assignment]
    assert await classifier.needs_tools("a distinctive prompt") is True
    assert await classifier.needs_tools("a distinctive prompt") is True
    assert calls["n"] == 1


async def test_blueprint_protocol_probability_is_supported(settings):
    tuned = replace(
        settings,
        classifier_api_url="https://classifier.test/v1",
        classifier_api_key="k",
        classifier_protocol="blueprint",
        classifier_needs_tools_threshold=0.15,
    )
    classifier = Classifier(tuned)

    async def probability_post(url, payload):
        assert "questions" in payload, "blueprint protocol must send questions"
        return {"answers": {"needs_tools": {"probability": 0.02}}}

    classifier._post = probability_post  # type: ignore[assignment]
    assert await classifier.needs_tools("anything at all") is False


# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
def test_loop_guard_detects_self_reference():
    settings = load_settings(
        env={
            "GATEWAY_HOST": "127.0.0.1",
            "GATEWAY_PORT": "8090",
            "UPSTREAM_BASE_URL": "http://127.0.0.1:8090/v1",
        }
    )
    assert settings.is_loop_upstream() is True


def test_loop_guard_detects_legacy_port_and_localhost():
    settings = load_settings(env={"GATEWAY_PORT": "8090"})
    assert settings.is_loop_upstream("http://localhost:8080/v1") is True
    assert settings.is_loop_upstream("http://0.0.0.0:8090/v1") is True


def test_loop_guard_allows_real_providers_and_other_local_ports():
    settings = load_settings(env={"GATEWAY_PORT": "8090"})
    assert settings.is_loop_upstream("https://api.openai.com/v1") is False
    assert settings.is_loop_upstream("http://127.0.0.1:11434/v1") is False


def test_markdown_wrapped_url_is_sanitised():
    assert (
        sanitize_url("[https://api.openai.com/v1](https://api.openai.com/v1)")
        == "https://api.openai.com/v1"
    )


def test_env_defaults_are_portable():
    """Nothing may default to a vendor-specific directory."""
    settings = load_settings(env={})
    assert settings.host == "127.0.0.1"
    assert settings.port == 8090
    assert "hermes" not in str(settings.data_dir).lower()
    assert "hermes" not in str(settings.log_dir).lower()
    assert str(settings.data_dir).endswith(".agent-gateway/data")


def test_env_aliases_allow_migration():
    settings = load_settings(
        env={
            "HERMES_PROXY_PORT": "7777",
            "REAL_UPSTREAM_BASE_URL": "https://openrouter.ai/api/v1",
            "JEV_API_URL": "https://classifier.test/v1",
            "JEV_API_KEY": "abc",
        }
    )
    assert settings.port == 7777
    assert settings.upstream_base_url == "https://openrouter.ai/api/v1"
    assert settings.classifier_enabled is True


def test_classifier_key_indirection_avoids_duplicating_a_secret():
    settings = load_settings(
        env={
            "CLASSIFIER_API_URL": "https://classifier.test/v1",
            "CLASSIFIER_API_KEY_ENV": "UPSTREAM_API_KEY",
            "UPSTREAM_API_KEY": "shared-secret",
        }
    )
    assert settings.classifier_api_key == "shared-secret"


async def test_audit_log_records_structured_json(api, settings):
    await api.post("/v1/chat/completions", json=body_for("hello"))
    log_path = settings.audit_log_path
    assert log_path.exists()
    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    assert lines
    entry = json.loads(lines[-1])
    assert entry["route"] == "FastPath-Strip"
    assert "latency_ms" in entry
