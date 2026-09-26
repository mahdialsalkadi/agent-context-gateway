"""
Semantic Tool Routing via Jev (Local Qwen3.5-2B on Vulkan).

Validates the Semantic Tool Router architecture:
- BM25 lexical ranking and arbitrary hard caps (<= 5) are eliminated.
- Jev determines the exact tool subset required for each turn.
- Single tool, multi-tool (e.g. 7 tools without caps), and conversational (0 tools) selections.
- Multi-turn execution-aware context routing.
- Diagnostic fail-open with explicit error logging when Jev is unavailable.
- Gateway integration and audit telemetry across OpenAI and Anthropic surfaces.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Dict, List

import httpx
import pytest

from src.analytics import PRUNED_ROUTES, Analytics
from src.classifier import (
    RoutedToolList,
    _tool_description,
    _tool_name,
    parse_jev_tool_selection,
    route_tools_via_jev,
)
from src.config import Settings, load_settings
from src.dashboard import recent_requests
from src.gateway import (
    apply_selective_pruning,
    create_app,
    extract_execution_context,
    prune_for_tool_loop,
)


def tool(name: str, description: str, properties: int = 2) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {f"arg{i}": {"type": "string"} for i in range(properties)},
            },
        },
    }


def anthropic_tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}},
    }


def build_toolset(count: int = 22) -> list:
    return [
        tool("bash", "Run a shell command on the host"),
        tool("read_file", "Read a file from disk"),
        tool("write_file", "Write content to a file"),
        tool("web_search", "Search the web for information"),
        tool("web_fetch", "Fetch a URL and return its text"),
        tool("send_email", "Send an email to a recipient"),
        tool("calendar_create", "Create a calendar event"),
        tool("sql_query", "Run a query against the SQL database"),
        tool("image_generate", "Generate an image from a prompt"),
        tool("image_edit", "Edit an image with instructions"),
        tool("translate_text", "Translate text between languages"),
        tool("screenshot", "Take a screenshot of the screen"),
        tool("git_commit", "Commit changes with git"),
        tool("docker_build", "Build a docker image"),
        tool("spotify_play", "Play music on Spotify"),
        tool("smart_home", "Control smart home devices"),
        tool("weather", "Get the weather forecast"),
        tool("stock_price", "Get a stock price quote"),
        tool("reminder_set", "Set a reminder"),
        tool("map_route", "Get a driving route on a map"),
        tool("pdf_extract", "Extract text from a PDF"),
        tool("fetch_log", "Read a truncated log slice"),
    ][:count]


# ------------------------------------------------------------------------------
# Phase 1: Jev output parsing
# ------------------------------------------------------------------------------
def test_parse_jev_tool_selection_json_array():
    candidates = {"bash", "read_file", "web_search", "write_file"}
    assert parse_jev_tool_selection('["bash", "read_file"]', candidates) == ["bash", "read_file"]


def test_parse_jev_tool_selection_markdown_fences():
    candidates = {"bash", "read_file", "web_search"}
    raw = '```json\n["web_search"]\n```'
    assert parse_jev_tool_selection(raw, candidates) == ["web_search"]


def test_parse_jev_tool_selection_json_object():
    candidates = {"bash", "read_file", "web_search"}
    raw = '{"tools": ["bash", "web_search"]}'
    assert parse_jev_tool_selection(raw, candidates) == ["bash", "web_search"]


def test_parse_jev_tool_selection_comma_separated():
    candidates = {"bash", "read_file", "web_search"}
    raw = "bash, web_search"
    assert parse_jev_tool_selection(raw, candidates) == ["bash", "web_search"]


def test_parse_jev_tool_selection_empty_array_or_none():
    candidates = {"bash", "read_file"}
    assert parse_jev_tool_selection("[]", candidates) == []
    assert parse_jev_tool_selection("none", candidates) == []
    assert parse_jev_tool_selection("", candidates) == []


def test_parse_jev_tool_selection_discards_hallucinated_tools():
    candidates = {"bash", "read_file"}
    raw = '["bash", "imaginary_tool_42"]'
    assert parse_jev_tool_selection(raw, candidates) == ["bash"]


# ------------------------------------------------------------------------------
# Phase 1: Semantic tool router (route_tools_via_jev)
# ------------------------------------------------------------------------------
async def test_route_tools_via_jev_selects_exact_subset():
    tools = build_toolset(10)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(["bash", "read_file"])}}
                ]
            },
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-jev") as client:
        routed = await route_tools_via_jev(
            prompt="check the code and run tests",
            tools=tools,
            client=client,
        )

    assert isinstance(routed, RoutedToolList)
    assert getattr(routed, "selected_names") == ["bash", "read_file"]
    assert len(routed) == 2
    assert [_tool_name(t) for t in routed] == ["bash", "read_file"]
    # Full schemas preserved
    assert routed[0] == tools[0]
    assert routed[1] == tools[1]


async def test_route_tools_via_jev_no_arbitrary_cap_allows_7_tools():
    """Validates Invariant 2: No arbitrary <= 5 cap."""
    tools = build_toolset(20)
    seven_tools = [
        "bash", "read_file", "write_file", "git_commit",
        "web_search", "web_fetch", "docker_build"
    ]

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(seven_tools)}}]},
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-jev") as client:
        routed = await route_tools_via_jev(
            prompt="full dev pipeline task",
            tools=tools,
            client=client,
        )

    assert len(routed) == 7
    assert getattr(routed, "selected_names") == seven_tools


async def test_route_tools_via_jev_empty_for_conversational():
    tools = build_toolset(10)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "[]"}}]},
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-jev") as client:
        routed = await route_tools_via_jev(
            prompt="write a haiku about trees",
            tools=tools,
            client=client,
        )

    assert len(routed) == 0
    assert getattr(routed, "selected_names") == []


async def test_route_tools_via_jev_passes_execution_context():
    """Validates Invariant 3: Execution-aware context routing."""
    tools = build_toolset(10)
    seen_prompts = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        user_msg = body["messages"][1]["content"]
        seen_prompts.append(user_msg)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '["write_file"]'}}]},
        )

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-jev") as client:
        context = "Tool Result (call_1): Found relevant weather API keys in config"
        routed = await route_tools_via_jev(
            prompt="save this to disk",
            tools=tools,
            client=client,
            context=context,
        )

    assert len(seen_prompts) == 1
    assert "Found relevant weather API keys" in seen_prompts[0]
    assert [_tool_name(t) for t in routed] == ["write_file"]


async def test_route_tools_via_jev_fails_open_on_error(capsys):
    """Validates Diagnostic Fail-Open requirement: logs [JEV-ROUTER-ERROR]."""
    tools = build_toolset(5)

    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-jev") as client:
        routed = await route_tools_via_jev(
            prompt="do something",
            tools=tools,
            client=client,
        )

    # Returns full catalog unmodified
    assert len(routed) == len(tools)
    assert getattr(routed, "error") is not None
    assert "HTTP 500" in getattr(routed, "error")

    captured = capsys.readouterr()
    assert "[JEV-ROUTER-ERROR]" in captured.err


# ------------------------------------------------------------------------------
# Backwards compatibility shims
# ------------------------------------------------------------------------------
def test_deprecated_shims():
    tools = build_toolset(5)
    settings = load_settings(env={})
    pruned, dropped = apply_selective_pruning(settings, "test", tools)
    assert pruned == tools
    assert dropped == 0

    loop_pruned, loop_dropped = prune_for_tool_loop(tools, ["bash"])
    assert loop_pruned == tools
    assert loop_dropped == 0


# ------------------------------------------------------------------------------
# Phase 2: Gateway Integration (OpenAI Surface)
# ------------------------------------------------------------------------------
async def test_gateway_jev_routed_three_tools(settings, store, memory, mock_state):
    """Gateway in local_jev mode forwards Jev's exact selection."""
    from unittest.mock import AsyncMock, patch

    tuned = replace(settings, classifier_mode="local_jev")
    app = create_app(settings=tuned, classifier=None, store=store, memory=memory)

    payload_tools = build_toolset(20)
    selected = [payload_tools[0], payload_tools[1], payload_tools[2]]
    routed_result = RoutedToolList(selected)
    routed_result.selected_names = ["bash", "read_file", "write_file"]

    transport = httpx.ASGITransport(app=app)
    with patch("src.gateway.route_tools_via_jev", new=AsyncMock(return_value=routed_result)):
        async with httpx.AsyncClient(transport=transport, base_url="http://gw", timeout=30.0) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "refactor the code and run tests"}],
                    "tools": payload_tools,
                },
            )

    assert response.status_code == 200
    assert response.headers["x-proxy-route"] == "Jev-Routed (3 tools)"
    assert response.headers["x-proxy-tool-action"] == "Retained-Semantic"
    after = int(response.headers["x-proxy-tools-after"])
    assert after == 3
    sent = [_tool_name(t) for t in mock_state.bodies[-1]["tools"]]
    assert sent == ["bash", "read_file", "write_file"]


async def test_gateway_jev_routed_seven_tools_no_arbitrary_cap(
    settings, store, memory, mock_state
):
    """Gateway forwards all 7 tools without any 5-tool truncation."""
    from unittest.mock import AsyncMock, patch

    seven_names = [
        "bash", "read_file", "write_file", "git_commit",
        "web_search", "web_fetch", "docker_build"
    ]
    tuned = replace(settings, classifier_mode="local_jev")
    app = create_app(settings=tuned, classifier=None, store=store, memory=memory)

    payload_tools = build_toolset(20)
    selected = [t for t in payload_tools if _tool_name(t) in seven_names]
    routed_result = RoutedToolList(selected)
    routed_result.selected_names = seven_names

    transport = httpx.ASGITransport(app=app)
    with patch("src.gateway.route_tools_via_jev", new=AsyncMock(return_value=routed_result)):
        async with httpx.AsyncClient(transport=transport, base_url="http://gw", timeout=30.0) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "complete task"}],
                    "tools": payload_tools,
                },
            )

    assert response.status_code == 200
    assert response.headers["x-proxy-route"] == "Jev-Routed (7 tools)"
    sent = [_tool_name(t) for t in mock_state.bodies[-1]["tools"]]
    assert len(sent) == 7
    assert set(sent) == set(seven_names)


async def test_gateway_jev_strips_conversational_turn(settings, store, memory, mock_state):
    """Jev returning [] results in Jev-Strip and 0 tools."""
    from unittest.mock import AsyncMock, patch

    tuned = replace(settings, classifier_mode="local_jev")
    app = create_app(settings=tuned, classifier=None, store=store, memory=memory)

    routed_result = RoutedToolList([])
    routed_result.selected_names = []

    transport = httpx.ASGITransport(app=app)
    with patch("src.gateway.route_tools_via_jev", new=AsyncMock(return_value=routed_result)):
        async with httpx.AsyncClient(transport=transport, base_url="http://gw", timeout=30.0) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "write an essay on stars"}],
                    "tools": build_toolset(10),
                },
            )

    assert response.status_code == 200
    assert response.headers["x-proxy-route"] == "Jev-Strip"
    assert response.headers["x-proxy-tool-action"] == "Stripped-ZeroTokens"
    assert "tools" not in mock_state.bodies[-1]


async def test_gateway_jev_fail_open_on_jev_crash(settings, store, memory, mock_state):
    """When Jev is down, gateway fails open with Jev-Router-FailOpen."""
    from unittest.mock import AsyncMock, patch

    tuned = replace(settings, classifier_mode="local_jev")
    app = create_app(settings=tuned, classifier=None, store=store, memory=memory)

    payload_tools = build_toolset(10)
    fail_open_result = RoutedToolList(payload_tools)
    fail_open_result.error = "Connection Refused"

    transport = httpx.ASGITransport(app=app)
    with patch("src.gateway.route_tools_via_jev", new=AsyncMock(return_value=fail_open_result)):
        async with httpx.AsyncClient(transport=transport, base_url="http://gw", timeout=30.0) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "stream": True,
                    "messages": [{"role": "user", "content": "run task"}],
                    "tools": payload_tools,
                },
            )

    assert response.status_code == 200
    assert response.headers["x-proxy-route"] == "Jev-Router-FailOpen"
    assert response.headers["x-proxy-tool-action"] == "Retained-Classifier"
    assert len(mock_state.bodies[-1]["tools"]) >= 10


# ------------------------------------------------------------------------------
# Phase 2: Gateway Integration (Anthropic Surface)
# ------------------------------------------------------------------------------
async def test_anthropic_surface_jev_routing(settings, store, memory, mock_state):
    from unittest.mock import AsyncMock, patch

    tuned = replace(settings, classifier_mode="local_jev")
    app = create_app(settings=tuned, classifier=None, store=store, memory=memory)

    input_tools = [
        anthropic_tool("bash", "Run command"),
        anthropic_tool("read_file", "Read file"),
        anthropic_tool("spotify", "Music"),
    ]
    routed_result = RoutedToolList([input_tools[0], input_tools[1]])
    routed_result.selected_names = ["bash", "read_file"]

    transport = httpx.ASGITransport(app=app)
    with patch("src.gateway.route_tools_via_jev", new=AsyncMock(return_value=routed_result)):
        async with httpx.AsyncClient(transport=transport, base_url="http://gw", timeout=30.0) as client:
            response = await client.post(
                "/v1/messages",
                headers={"anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-3-opus",
                    "messages": [{"role": "user", "content": "inspect codebase"}],
                    "tools": input_tools,
                },
            )

    assert response.status_code == 200
    assert response.headers["x-proxy-route"] == "Jev-Routed (2 tools)"
    assert response.headers["x-proxy-tool-action"] == "Retained-Semantic"
    sent = [_tool_name(t) for t in mock_state.bodies[-1]["tools"]]
    assert sent == ["bash", "read_file"]


# ------------------------------------------------------------------------------
# Phase 3: Dashboard & Telemetry formatting
# ------------------------------------------------------------------------------
def test_dashboard_recent_requests_formats_selected_tools(tmp_path):
    log_file = tmp_path / "audit.jsonl"
    with log_file.open("w") as f:
        f.write(
            json.dumps(
                {
                    "ts": 1727330000,
                    "route": "Jev-Routed (2 tools)",
                    "tools_before": 20,
                    "tools_after": 2,
                    "selected_tools": ["bash", "read_file"],
                    "latency_ms": 35.4,
                }
            )
            + "\n"
        )

    analytics = Analytics(log_file)
    feed = recent_requests(analytics)
    assert len(feed) == 1
    assert feed[0]["route"] == "Jev-Routed (2 tools)"
    assert feed[0]["tools"] == "20 → 2 [bash, read_file]"
    assert feed[0]["latency"] == "35ms"


def test_pruned_routes_includes_jev_strip():
    assert "Jev-Strip" in PRUNED_ROUTES
