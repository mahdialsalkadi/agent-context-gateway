"""
Selective sub-tool pruning.

The point of this file: when an agent offers 20+ tools, most turns need two or
three of them, and sending the whole schema is pure waste. These tests pin the
ranker's precision (irrelevant tools are gone), its recall guarantees
(mission-critical I/O survives; Anthropic shapes work), and its honesty (when
there is no evidence, nothing is dropped).
"""

from __future__ import annotations

import time

from src.classifier import (
    ALWAYS_KEEP_TOOLS,
    ARABIC_INTENT_MAP,
    CORE_TOOLS,
    INTENT_KEYWORDS,
    intent_boosts,
    rank_tools,
)
from src.config import (
    DEFAULT_SELECTIVE_TOOL_LIMIT,
    SELECTIVE_PRUNING_MIN_TOOLS,
    load_settings,
)
from src.gateway import apply_selective_pruning, create_app

import httpx
import pytest


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


def names(tools):
    """Tool names in either payload shape."""
    result = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if isinstance(t.get("function"), dict):
            result.append(t["function"]["name"])
        else:
            result.append(t.get("name"))
    return result


ARABIC_PROMPT = "افحص ملف الكود وشغل التيست"  # "inspect the code file and run the test"


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


def big_toolset(count: int = 25) -> list:
    """A realistic wide catalog: the base set, padded above `count`."""
    tools = build_toolset()
    while len(tools) < count:
        tools.append(
            tool(f"aux_{len(tools)}", f"auxiliary capability number {len(tools)}")
        )
    return tools


# ------------------------------------------------------------------------------
# Ranker precision
# ------------------------------------------------------------------------------
def test_web_prompt_keeps_the_search_tool_and_drops_the_rest():
    tools = build_toolset()
    kept = rank_tools("search the web for rust async tutorials", tools, 5)

    assert "web_search" in names(kept)
    assert len(kept) < len(tools)
    # Irrelevant tools must be gone, not merely ranked lower.
    assert "spotify_play" not in names(kept)
    assert "smart_home" not in names(kept)
    assert "stock_price" not in names(kept)


def test_shell_prompt_keeps_bash_and_read_helpers():
    tools = build_toolset()
    kept = names(rank_tools("run the test suite in bash", tools, 5))

    assert "bash" in kept


def test_email_prompt_ranks_send_email_first():
    tools = build_toolset()
    kept = names(rank_tools("send an email to alice about the meeting", tools, 5))

    assert "send_email" in kept


def test_stopwords_do_not_rank_irrelevant_tools():
    """'search the web for tutorials' contains 'the'; screenshot must not ride in."""
    tools = build_toolset()
    kept = names(rank_tools("search the web for tutorials", tools, 5))

    assert "screenshot" not in kept


def test_kept_tools_never_exceed_the_limit():
    tools = build_toolset(25)
    kept = rank_tools("search the web and read files and run git commits", tools, 5)
    assert len(kept) <= 5 + len(ALWAYS_KEEP_TOOLS)


def test_payload_order_is_preserved():
    """Some providers reuse prompt caches keyed on schema order."""
    tools = build_toolset()
    kept = rank_tools("send an email about the weather", tools, 5)
    original_positions = [tools.index(t) for t in kept]
    assert original_positions == sorted(original_positions)


# ------------------------------------------------------------------------------
# Recall guarantees
# ------------------------------------------------------------------------------
def test_mission_critical_tools_survive_even_when_unmentioned():
    tools = build_toolset()
    kept = names(rank_tools("generate an image of a sunset", tools, 3))

    assert "bash" in kept, "the escape-replay contract promises a shell"
    assert "fetch_log" in kept, "the gateway's own retrieval tool"
    assert "read_file" in kept, "how an agent recovers context after a strip"


def test_anthropic_tool_shape_is_supported():
    tools = [
        anthropic_tool("web_search", "Search the web for information"),
        anthropic_tool("send_email", "Send an email to a recipient"),
        anthropic_tool("weather", "Get the weather forecast"),
        anthropic_tool("stock_price", "Get a stock price quote"),
    ]
    kept = names(rank_tools("search the web for tutorials", tools, 2))

    assert "web_search" in kept
    assert "stock_price" not in kept


# ------------------------------------------------------------------------------
# Honesty: no evidence, no pruning
# ------------------------------------------------------------------------------
def test_prompt_matching_nothing_keeps_everything():
    tools = build_toolset()
    kept = rank_tools("tell me a joke about penguins", tools, 5)
    assert len(kept) == len(tools)


def test_stopword_only_prompt_keeps_everything():
    tools = build_toolset()
    kept = rank_tools("to be or not to be", tools, 5)
    assert len(kept) == len(tools)


def test_empty_prompt_keeps_everything():
    tools = build_toolset()
    assert len(rank_tools("", tools, 5)) == len(tools)


def test_garbage_tools_do_not_crash_the_ranker():
    tools = [None, "not-a-dict", {"function": "broken"}, *build_toolset(10)]
    kept = rank_tools("search the web for tutorials", tools, 5)
    assert isinstance(kept, list)


# ------------------------------------------------------------------------------
# Latency budget: the ranker rides on the hot path
# ------------------------------------------------------------------------------
def test_ranking_a_30_tool_payload_is_sub_2ms():
    tools = build_toolset(25) + [
        tool(f"extra_{i}", f"description number {i} for tool {i}") for i in range(5)
    ]
    prompt = "refactor the auth module and run the whole test suite with bash"

    started = time.perf_counter()
    for _ in range(200):
        rank_tools(prompt, tools, 5)
    per_call_ms = (time.perf_counter() - started) / 200 * 1000

    assert per_call_ms < 2.0, f"ranker took {per_call_ms:.3f}ms per call on 30 tools"


def test_bm25_scores_a_large_catalog_quickly():
    tools = build_toolset(22)
    prompt = "summarise the deployment status and fetch the build logs"
    started = time.perf_counter()
    for _ in range(200):
        rank_tools(prompt, tools, 5)
    per_call_ms = (time.perf_counter() - started) / 200 * 1000
    assert per_call_ms < 2.0


# ------------------------------------------------------------------------------
# Route integration
# ------------------------------------------------------------------------------
def test_apply_selective_pruning_gates_on_threshold_and_toggle():
    settings = load_settings(env={})
    tools = build_toolset(22)
    prompt = "search the web for tutorials"

    # Disabled: untouched.
    disabled = type(settings)(**{**settings.__dict__, "enable_selective_pruning": False})
    assert apply_selective_pruning(disabled, prompt, tools)[1] == 0

    # Below the threshold: untouched.
    small = tools[: SELECTIVE_PRUNING_MIN_TOOLS]
    assert apply_selective_pruning(settings, prompt, small)[1] == 0

    # Above it: pruned.
    kept, dropped = apply_selective_pruning(settings, prompt, tools)
    assert dropped > 0
    assert len(kept) < len(tools)


async def test_gateway_route_prunes_a_20_tool_schema(settings, store, memory, mock_state):
    """End to end: 20 tools in, few tools out, telemetry headers set."""
    tuned = type(settings)(**{**settings.__dict__, "selective_tool_limit": 5})
    app = create_app(settings=tuned, classifier=None, store=store, memory=memory)

    payload_tools = build_toolset(20)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw", timeout=30.0
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "stream": True,
                "messages": [{"role": "user", "content": "search the web for tutorials"}],
                "tools": payload_tools,
            },
        )

    assert response.status_code == 200
    before = int(response.headers["x-proxy-tools-before"])
    after = int(response.headers["x-proxy-tools-after"])
    assert before == 21, "the synthetic fetch_log tool is appended before counting"
    assert after < before, "the schema must shrink"

    # And the upstream actually received the smaller schema.
    assert len(mock_state.bodies[-1]["tools"]) == after


async def test_forced_tool_choice_is_never_pruned(settings, store, memory):
    tuned = type(settings)(**{**settings.__dict__, "selective_tool_limit": 5})
    app = create_app(settings=tuned, classifier=None, store=store, memory=memory)

    payload_tools = build_toolset(20)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw", timeout=30.0
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "stream": True,
                "tool_choice": {"type": "function", "function": {"name": "web_search"}},
                "messages": [{"role": "user", "content": "search the web for tutorials"}],
                "tools": payload_tools,
            },
        )

    after = int(response.headers["x-proxy-tools-after"])
    assert after == 21, "an explicit tool_choice means every tool may be needed"


async def test_stripped_route_does_not_rank(settings, store, memory):
    """All-or-nothing and selective pruning must not both fire on one turn."""
    from src.classifier import Classifier
    from dataclasses import replace as dc_replace

    tuned = dc_replace(
        type(settings)(
            **{
                **settings.__dict__,
                "classifier_api_url": "https://c.test/v1/chat/completions",
                "classifier_api_key": "k",
            }
        ),
    )
    stripper = Classifier(tuned)

    async def always_false(_prompt):
        return False

    stripper.needs_tools = always_false
    app = create_app(settings=tuned, classifier=stripper, store=store, memory=memory)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw", timeout=30.0
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "stream": True,
                "messages": [{"role": "user", "content": "search the web for tutorials"}],
                "tools": build_toolset(20),
            },
        )

    assert response.headers["x-proxy-tool-action"] == "Stripped-ZeroTokens"
    after = int(response.headers["x-proxy-tools-after"])
    assert after == 0, "a stripped turn sends no schema at all"


# ------------------------------------------------------------------------------
# Hard cap: the gateway never forwards the whole catalog
# ------------------------------------------------------------------------------
def test_core_tools_fit_inside_the_default_limit():
    assert len(CORE_TOOLS) <= DEFAULT_SELECTIVE_TOOL_LIMIT


def test_intent_map_boosts_core_tools_for_an_arabic_prompt():
    boosts = intent_boosts(ARABIC_PROMPT)
    assert boosts.get("bash")
    assert boosts.get("read_file")


def test_hard_cap_falls_back_to_only_core_tools_when_there_is_no_evidence():
    tools = big_toolset(25)
    kept = rank_tools("tell me a joke about penguins", tools, 5, hard_cap=True)

    assert 0 < len(kept) <= 5
    assert set(names(kept)).issubset(CORE_TOOLS)
    # The legacy default is untouched -- the gateway opts in explicitly.
    assert len(rank_tools("tell me a joke about penguins", tools, 5)) == len(tools)


def test_hard_cap_keeps_bash_and_read_file_for_arabic_within_the_limit():
    kept = names(rank_tools(ARABIC_PROMPT, big_toolset(25), 5, hard_cap=True))

    assert len(kept) <= DEFAULT_SELECTIVE_TOOL_LIMIT
    assert "bash" in kept
    assert "read_file" in kept


def test_arabic_intent_map_alias_shares_one_source_of_truth():
    assert ARABIC_INTENT_MAP is INTENT_KEYWORDS
    assert "شغل" in ARABIC_INTENT_MAP["bash"]
    assert "افحص" in ARABIC_INTENT_MAP["read_file"]


def test_mission_arabic_prompt_caps_a_49_tool_catalog():
    """The blueprint's headline case: 49 tools in, core trio out."""
    kept = names(rank_tools("افحص الملف وشغل التيست", big_toolset(49), 5, hard_cap=True))

    assert len(kept) <= DEFAULT_SELECTIVE_TOOL_LIMIT
    assert {"bash", "read_file", "fetch_log"}.issubset(set(kept))


def test_hard_cap_shrinks_a_prompt_with_no_english_overlap():
    """Non-English intent must not fall back to 'keep everything'."""
    kept = rank_tools("ترجم هذا النص إلى الإنجليزية", big_toolset(25), 5, hard_cap=True)
    assert len(kept) <= DEFAULT_SELECTIVE_TOOL_LIMIT
    assert len(kept) < 25


def test_apply_selective_pruning_hard_caps_an_arabic_prompt():
    settings = load_settings(env={})
    kept, dropped = apply_selective_pruning(settings, ARABIC_PROMPT, big_toolset(25))

    assert dropped > 0
    assert len(kept) <= DEFAULT_SELECTIVE_TOOL_LIMIT
    assert {"bash", "read_file"}.issubset(set(names(kept)))


async def test_gateway_caps_an_arabic_prompt_to_five_tools(
    settings, store, memory, mock_state
):
    tuned = type(settings)(**{**settings.__dict__, "selective_tool_limit": 5})
    app = create_app(settings=tuned, classifier=None, store=store, memory=memory)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw", timeout=30.0
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "stream": True,
                "messages": [{"role": "user", "content": ARABIC_PROMPT}],
                "tools": big_toolset(25),
            },
        )

    assert response.status_code == 200
    after = int(response.headers["x-proxy-tools-after"])
    assert after <= 5, "the hard cap must hold for a non-English prompt"

    sent = [tool["function"]["name"] for tool in mock_state.bodies[-1]["tools"]]
    assert len(sent) == after
    assert "bash" in sent and "read_file" in sent


async def test_tool_turn_does_not_expand_back_to_the_full_catalog(
    settings, store, memory, mock_state
):
    """A mid-loop turn must not silently re-expand the schema."""
    tuned = type(settings)(**{**settings.__dict__, "selective_tool_limit": 5})
    app = create_app(settings=tuned, classifier=None, store=store, memory=memory)

    messages = [
        {"role": "user", "content": "search the web for tutorials"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "results"},
    ]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw", timeout=30.0
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-model",
                "stream": True,
                "messages": messages,
                "tools": big_toolset(25),
            },
        )

    assert response.status_code == 200
    after = int(response.headers["x-proxy-tools-after"])
    assert after <= 5, "a tool turn must not re-expand the catalog"
