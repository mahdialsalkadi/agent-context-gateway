"""
Streaming behaviour and the Anthropic bridge.

The load-bearing assertions here are about *not* buffering: the original
implementation called `aread()` to check for the escape sentinel, which pulls
the entire response into memory before the client sees a single byte. These
tests pin the bounded behaviour so it cannot regress.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace

import httpx
import pytest

from src.bridge import (
    AnthropicStreamTranslator,
    anthropic_to_openai_request,
    estimate_tokens,
    openai_to_anthropic_response,
    translate_stream,
)
from src.config import ESCAPE_TOKEN
from src.classifier import Classifier
from src.gateway import create_app, head_could_be_escape, sniff_for_escape
from tests.mock_upstream import (
    DEFAULT_TEXT,
    done,
    escape_stream,
    finish_delta,
    openai_stream,
    sse,
    text_delta,
    tool_delta,
)

TOOL_SCHEMA = [
    {"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}
]

ANTHROPIC_TOOL = {
    "name": "terminal",
    "description": "Run a shell command",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}


class FakeStreamResponse:
    """Minimal stand-in exposing only what `sniff_for_escape` uses."""

    status_code = 200

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.yielded = 0
        self.closed = False

    async def aiter_raw(self):
        for chunk in self._chunks:
            self.yielded += 1
            yield chunk

    async def aread(self):
        self.yielded = len(self._chunks)
        return b"".join(self._chunks)

    async def aclose(self):
        self.closed = True


def openai_body(prompt: str, **extra):
    payload = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOL_SCHEMA,
        "stream": True,
    }
    payload.update(extra)
    return payload


class AlwaysStrip(Classifier):
    """Classifier stub that always prunes, so the escape path is reachable.

    The local heuristics only strip whole-prompt greetings, which cannot also
    carry the mock's escape marker -- going through the classifier is how a real
    ambiguous prompt gets pruned.
    """

    async def needs_tools(self, prompt):
        return False

    async def value_supersedes(self, source, predicate, old_target, new_target):
        return True


@pytest.fixture
async def stripping_api(settings, store, memory):
    app = create_app(
        settings=settings, classifier=AlwaysStrip(settings), store=store, memory=memory
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gateway", timeout=30.0
    ) as client:
        yield client


def parse_sse(body: bytes):
    """Split an SSE byte stream into (event, data) pairs."""
    events = []
    for block in body.decode("utf-8", "replace").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        events.append((name, data))
    return events


# ------------------------------------------------------------------------------
# Bounded sniffing
# ------------------------------------------------------------------------------
async def test_sniff_is_bounded_and_never_reads_the_whole_stream():
    chunk = text_delta("x" * 64)
    response = FakeStreamResponse([chunk] * 4000)

    buffered, escaped = await sniff_for_escape(response.aiter_raw(), 512, 192, 64)

    assert escaped is False
    assert sum(len(part) for part in buffered) <= 512 + 192
    assert response.yielded < 4000, "sniffing consumed the entire stream"


async def test_sniff_exits_early_when_the_head_is_unambiguous():
    chunks = [text_delta(DEFAULT_TEXT)] * 5
    response = FakeStreamResponse(chunks)

    buffered, escaped = await sniff_for_escape(response.aiter_raw(), 512, 192, 64)

    assert escaped is False
    assert response.yielded == 1
    assert b"".join(buffered) == chunks[0]


async def test_sniff_aborts_promptly_on_the_sentinel():
    response = FakeStreamResponse(escape_stream())

    started = time.perf_counter()
    buffered, escaped = await sniff_for_escape(response.aiter_raw(), 512, 192, 64)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert escaped is True
    assert elapsed_ms < 50.0, f"escape abort took {elapsed_ms:.1f}ms, expected < 50ms"
    assert response.yielded == 1, "should stop at the chunk containing the sentinel"


async def test_sniff_handles_sentinel_split_across_chunks():
    head = ESCAPE_TOKEN[:-3].encode()
    tail = ESCAPE_TOKEN[-3:].encode()
    response = FakeStreamResponse([b"data: ", head, tail, b"\n\n"])
    _, escaped = await sniff_for_escape(response.aiter_raw(), 512, 192, 64)
    assert escaped is True


async def test_head_heuristic_rejects_a_non_escape_opening():
    assert head_could_be_escape("Hello there") is False
    assert head_could_be_escape(ESCAPE_TOKEN) is True
    assert head_could_be_escape(ESCAPE_TOKEN[:5]) is True
    assert head_could_be_escape("") is True


# ------------------------------------------------------------------------------
# Escape interception end to end
# ------------------------------------------------------------------------------
async def test_escape_triggers_replay_with_tools_restored(stripping_api, mock_state):
    payload = openai_body("force_escape please")
    # The pruned attempt has no tools, so the model asks for them back.
    response = await stripping_api.post("/v1/chat/completions", json=payload)

    assert response.headers["x-proxy-route"] == "Classifier-Strip->EarlyEscapeAbort"
    assert response.headers["x-proxy-tool-action"] == "Reverted-To-Full"

    # Two dispatches: the pruned attempt, then the replay with tools.
    bodies = mock_state.bodies
    assert len(bodies) == 2, f"expected a replay, saw {len(bodies)} upstream calls"
    assert "tools" not in bodies[0]
    assert "tools" in bodies[1]
    assert ESCAPE_TOKEN not in json.dumps(bodies[1])

    # And the client receives the replay's content, not the escape text.
    assert "terminal" in response.text or "tool_calls" in response.text


async def test_escape_instruction_is_removed_from_the_replay(stripping_api, mock_state):
    await stripping_api.post("/v1/chat/completions", json=openai_body("force_escape please"))
    assert len(mock_state.bodies) == 2
    replay = json.dumps(mock_state.bodies[1])
    assert "SYSTEM INSTRUCTION" not in replay
    assert ESCAPE_TOKEN not in replay


# ------------------------------------------------------------------------------
# Passthrough fidelity
# ------------------------------------------------------------------------------
async def test_reasoning_route_is_not_sniffed_or_replayed(api, mock_state):
    response = await api.post(
        "/v1/chat/completions",
        json=openai_body("force_escape please", model="deepseek/deepseek-r1"),
    )
    assert response.headers["x-proxy-route"] == "Reasoning-Passthrough"
    assert len(mock_state.bodies) == 1, "a reasoning model must never be re-dispatched"


async def test_streamed_text_is_relayed_to_the_client(api):
    response = await api.post("/v1/chat/completions", json=openai_body("run the tests"))
    assert response.status_code == 200
    assert "[DONE]" in response.text

    # The mock deliberately splits mid-word, so reassemble from the deltas
    # rather than expecting one contiguous string in the raw stream.
    text = "".join(
        json.loads(data)["choices"][0]["delta"].get("content", "")
        for _, data in parse_sse(response.content)
        if data and data != "[DONE]"
    )
    assert text == DEFAULT_TEXT


async def test_upstream_error_is_relayed_with_telemetry(api):
    response = await api.post("/v1/chat/completions", json=openai_body("please fail now"))
    assert response.status_code == 500
    # An ambiguous prompt with no classifier configured keeps tools and fails open.
    assert response.headers["x-proxy-tool-action"] == "Retained-Classifier"
    assert "Upstream500" in response.headers["x-proxy-route"]


async def test_bypass_header_skips_all_modification(api, mock_state):
    response = await api.post(
        "/v1/chat/completions",
        json=openai_body("hello"),
        headers={"x-agent-gateway-bypass": "true"},
    )
    assert response.status_code == 200
    assert response.headers["x-proxy-route"] == "Bypass"
    # Untouched: the client's own tool list goes through as-is.
    assert "tools" in mock_state.bodies[0]


async def test_gateway_api_key_is_enforced_when_configured(settings, store, memory):
    guarded = replace(settings, gateway_api_key="s3cret")
    app = create_app(settings=guarded, store=store, memory=memory)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        unauthorised = await client.post("/v1/chat/completions", json=openai_body("hello"))
        assert unauthorised.status_code == 401

        bearer = await client.post(
            "/v1/chat/completions",
            json=openai_body("hello"),
            headers={"Authorization": "Bearer s3cret"},
        )
        assert bearer.status_code == 200

        # Claude Code authenticates with x-api-key instead of a bearer token.
        api_key = await client.post(
            "/v1/chat/completions",
            json=openai_body("hello"),
            headers={"x-api-key": "s3cret"},
        )
        assert api_key.status_code == 200


async def test_client_authorization_is_not_forwarded_upstream(api, mock_state, settings):
    await api.post(
        "/v1/chat/completions",
        json=openai_body("run the tests"),
        headers={"Authorization": "Bearer client-secret"},
    )
    headers = {k.lower(): v for k, v in mock_state.requests[0]["headers"].items()}
    assert headers.get("authorization") == f"Bearer {settings.upstream_api_key}"
    assert "client-secret" not in headers.get("authorization", "")


# ------------------------------------------------------------------------------
# Anthropic request translation
# ------------------------------------------------------------------------------
def test_anthropic_request_translation_core():
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 256,
        "temperature": 0.5,
        "system": "You are terse.",
        "stop_sequences": ["STOP"],
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [ANTHROPIC_TOOL],
        "tool_choice": {"type": "auto"},
    }
    out = anthropic_to_openai_request(body)

    assert out["model"] == "claude-sonnet-4"
    assert out["max_tokens"] == 256
    assert out["temperature"] == 0.5
    assert out["stop"] == ["STOP"]
    assert out["messages"][0] == {"role": "system", "content": "You are terse."}
    assert out["messages"][1] == {"role": "user", "content": "hello"}
    assert out["tools"][0]["function"]["name"] == "terminal"
    assert out["tools"][0]["function"]["parameters"] == ANTHROPIC_TOOL["input_schema"]
    assert out["tool_choice"] == "auto"


def test_anthropic_system_blocks_are_flattened():
    out = anthropic_to_openai_request(
        {
            "model": "m",
            "system": [
                {"type": "text", "text": "First.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Second."},
            ],
            "messages": [{"role": "user", "content": "hi"}],
        }
    )
    assert out["messages"][0]["content"] == "First.\nSecond."


def test_anthropic_tool_result_becomes_a_tool_message():
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "On it."},
                    {"type": "tool_use", "id": "toolu_1", "name": "terminal", "input": {"command": "ls"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "file.txt"}],
            },
        ],
    }
    out = anthropic_to_openai_request(body)
    roles = [message["role"] for message in out["messages"]]
    assert roles == ["user", "assistant", "tool"]

    assistant = out["messages"][1]
    assert assistant["tool_calls"][0]["function"]["name"] == "terminal"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"command": "ls"}

    tool_message = out["messages"][2]
    assert tool_message["tool_call_id"] == "toolu_1"
    assert tool_message["content"] == "file.txt"


def test_anthropic_tool_result_error_flag_is_preserved():
    out = anthropic_to_openai_request(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_9",
                            "content": "command not found",
                            "is_error": True,
                        }
                    ],
                }
            ],
        }
    )
    assert out["messages"][0]["content"].startswith("ERROR:")


def test_anthropic_images_become_data_urls():
    out = anthropic_to_openai_request(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
                        },
                    ],
                }
            ],
        }
    )
    parts = out["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["image_url"]["url"] == "data:image/png;base64,AAAA"


@pytest.mark.parametrize(
    "choice,expected",
    [
        ({"type": "auto"}, "auto"),
        ({"type": "any"}, "required"),
        ({"type": "none"}, "none"),
        ({"type": "tool", "name": "terminal"}, {"type": "function", "function": {"name": "terminal"}}),
    ],
)
def test_anthropic_tool_choice_mapping(choice, expected):
    out = anthropic_to_openai_request(
        {"model": "m", "messages": [{"role": "user", "content": "x"}], "tools": [ANTHROPIC_TOOL], "tool_choice": choice}
    )
    assert out["tool_choice"] == expected


# ------------------------------------------------------------------------------
# Anthropic response translation
# ------------------------------------------------------------------------------
def test_non_streaming_response_translation_with_tool_use():
    payload = {
        "id": "chatcmpl-1",
        "model": "mock-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Running it.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": '{"command":"ls"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }
    out = openai_to_anthropic_response(payload, "claude-sonnet-4")
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "claude-sonnet-4"
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0] == {"type": "text", "text": "Running it."}
    assert out["content"][1]["type"] == "tool_use"
    assert out["content"][1]["input"] == {"command": "ls"}
    assert out["usage"] == {"input_tokens": 12, "output_tokens": 5}


def test_malformed_tool_arguments_are_preserved_not_dropped():
    out = openai_to_anthropic_response(
        {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"id": "c", "function": {"name": "t", "arguments": "{not json"}}
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
    )
    assert out["content"][0]["input"] == {"_raw_arguments": "{not json"}


def test_response_with_no_content_still_has_a_block():
    out = openai_to_anthropic_response({"choices": [{"message": {"content": ""}}]})
    assert out["content"] == [{"type": "text", "text": ""}]


# ------------------------------------------------------------------------------
# Anthropic streaming translation
# ------------------------------------------------------------------------------
async def test_stream_translation_emits_a_valid_event_sequence():
    source = openai_stream()

    async def gen():
        for frame in source:
            yield frame

    frames = [frame async for frame in translate_stream(gen(), "claude-sonnet-4", 11)]
    events = parse_sse(b"".join(frames))
    names = [name for name, _ in events]

    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert "content_block_delta" in names
    assert names[-1] == "message_stop"
    assert names.count("message_start") == 1
    assert names.count("message_stop") == 1

    start = json.loads(dict(events)["message_start"])
    assert start["message"]["model"] == "claude-sonnet-4"
    assert start["message"]["usage"]["input_tokens"] == 11

    text = "".join(
        json.loads(data)["delta"]["text"]
        for name, data in events
        if name == "content_block_delta" and json.loads(data)["delta"]["type"] == "text_delta"
    )
    assert text == DEFAULT_TEXT

    stop = [json.loads(data) for name, data in events if name == "message_delta"][0]
    assert stop["delta"]["stop_reason"] == "end_turn"


async def test_stream_translation_handles_tool_calls():
    source = openai_stream(tool_call=True)

    async def gen():
        for frame in source:
            yield frame

    frames = [frame async for frame in translate_stream(gen(), "m")]
    events = parse_sse(b"".join(frames))
    names = [name for name, _ in events]

    assert "content_block_start" in names
    starts = [json.loads(d) for n, d in events if n == "content_block_start"]
    assert starts[-1]["content_block"]["type"] == "tool_use"
    assert starts[-1]["content_block"]["name"] == "terminal"

    partial = "".join(
        json.loads(d)["delta"]["partial_json"]
        for n, d in events
        if n == "content_block_delta" and json.loads(d)["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(partial) == {"command": "ls -la"}

    stop = [json.loads(d) for n, d in events if n == "message_delta"][0]
    assert stop["delta"]["stop_reason"] == "tool_use"


def test_translator_handles_chunks_split_mid_frame():
    """A chunk boundary is not an SSE frame boundary."""
    translator = AnthropicStreamTranslator("m")
    frame = text_delta("Hello world")
    out = translator.feed(frame[: len(frame) // 2])
    out += translator.feed(frame[len(frame) // 2:])
    out += translator.finish()
    events = parse_sse(b"".join(out))
    text = "".join(
        json.loads(d)["delta"]["text"]
        for n, d in events
        if n == "content_block_delta" and json.loads(d)["delta"]["type"] == "text_delta"
    )
    assert text == "Hello world"


def test_translator_finish_is_idempotent():
    translator = AnthropicStreamTranslator("m")
    translator.feed(text_delta("hi"))
    first = translator.finish()
    second = translator.finish()
    assert first
    assert second == []


def test_translator_emits_a_block_for_an_empty_response():
    translator = AnthropicStreamTranslator("m")
    frames = translator.feed(done())
    names = [name for name, _ in parse_sse(b"".join(frames))]
    assert names[0] == "message_start"
    assert "content_block_start" in names
    assert names[-1] == "message_stop"


def test_estimate_tokens_is_positive():
    assert estimate_tokens("") >= 1
    assert estimate_tokens("a" * 400) == 100


# ------------------------------------------------------------------------------
# Anthropic surface end to end
# ------------------------------------------------------------------------------
async def test_anthropic_route_streams_translated_events(api, mock_state):
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 128,
        "stream": True,
        "messages": [{"role": "user", "content": "run the test suite"}],
        "tools": [ANTHROPIC_TOOL],
    }
    response = await api.post("/v1/messages", json=body)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    events = parse_sse(response.content)
    names = [name for name, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    assert any(name == "content_block_delta" for name in names)

    # The upstream must receive an OpenAI-shaped request, not an Anthropic one.
    upstream = mock_state.bodies[0]
    assert isinstance(upstream["messages"], list)
    assert upstream["messages"][0]["role"] == "user"
    assert upstream["tools"][0]["type"] == "function"


async def test_anthropic_route_non_streaming(api):
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 128,
        "messages": [{"role": "user", "content": "run the test suite"}],
    }
    response = await api.post("/v1/messages", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "message"
    assert payload["content"][0]["type"] == "text"
    assert payload["content"][0]["text"] == DEFAULT_TEXT


async def test_anthropic_route_streams_tool_use(api):
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 128,
        "stream": True,
        "messages": [{"role": "user", "content": "use_tool to list files"}],
        "tools": [ANTHROPIC_TOOL],
    }
    response = await api.post("/v1/messages", json=body)
    events = parse_sse(response.content)
    names = [name for name, _ in events]
    assert "content_block_start" in names
    starts = [json.loads(d) for n, d in events if n == "content_block_start"]
    assert any(block["content_block"]["type"] == "tool_use" for block in starts)


async def test_anthropic_route_never_prunes_a_forced_tool_choice(api, mock_state):
    body = {
        "model": "claude-sonnet-4",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [ANTHROPIC_TOOL],
        "tool_choice": {"type": "tool", "name": "terminal"},
    }
    response = await api.post("/v1/messages", json=body)
    assert "ForcedToolChoice" in response.headers["x-proxy-route"]
    assert response.headers["x-proxy-tool-action"] == "Retained-Forced"
    assert "tools" in mock_state.bodies[0]


async def test_classifier_over_real_http_prunes_then_escapes(settings, store, memory, mock_state):
    """Full path over real HTTP: classifier says strip, model escapes, replay.

    The mock upstream doubles as a classifier endpoint, so this exercises the
    whole chain without a stub.
    """
    tuned = replace(
        settings,
        classifier_api_url=f"{settings.upstream_base_url}/chat/completions",
        classifier_api_key="test",
    )
    app = create_app(settings=tuned, store=store, memory=memory)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://gw", timeout=30.0
    ) as client:
        response = await client.post(
            "/v1/chat/completions", json=openai_body("force_escape please")
        )

    assert "EarlyEscapeAbort" in response.headers["x-proxy-route"]
    assert response.headers["x-proxy-tool-action"] == "Reverted-To-Full"

    # First real upstream call is the pruned attempt; the last one is the replay.
    upstream = [
        body for body in mock_state.bodies if body.get("stream") is not None
    ]
    assert "tools" not in upstream[0]
    assert "tools" in upstream[-1]


async def test_anthropic_model_override_is_applied(settings, store, memory, mock_state):
    """Claude Code sends `claude-*` ids; a plain OpenAI upstream needs a mapping."""
    tuned = replace(settings, anthropic_model_override="gpt-4o-mini")
    app = create_app(settings=tuned, store=store, memory=memory)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        await client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "run the tests"}],
            },
        )
    assert mock_state.bodies[0]["model"] == "gpt-4o-mini"


async def test_count_tokens_endpoint(api):
    body = {
        "model": "claude-sonnet-4",
        "messages": [{"role": "user", "content": "a" * 400}],
    }
    response = await api.post("/v1/messages/count_tokens", json=body)
    assert response.status_code == 200
    assert response.json()["input_tokens"] > 0


async def test_anthropic_route_uses_x_api_key_authentication(settings, store, memory):
    guarded = replace(settings, gateway_api_key="s3cret")
    app = create_app(settings=guarded, store=store, memory=memory)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as client:
        denied = await client.post(
            "/v1/messages",
            json={"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert denied.status_code == 401
        assert denied.json()["type"] == "error"

        allowed = await client.post(
            "/v1/messages",
            json={"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
            headers={"x-api-key": "s3cret"},
        )
        assert allowed.status_code == 200


async def test_models_endpoint_is_proxied(api):
    response = await api.get("/v1/models")
    assert response.status_code == 200
    ids = [model["id"] for model in response.json()["data"]]
    assert "mock-model" in ids


async def test_health_endpoint_reports_redacted_config(api):
    response = await api.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["upstream_key"] is True
    # A redacted snapshot must never carry the secret itself.
    assert "test-upstream-key" not in json.dumps(payload)
    assert "classifier_enabled" in payload
    assert payload["classifier_mode"] in (
        "heuristics",
        "upstream_reused",
        "local_ollama",
        "external_jev",
    )


# ------------------------------------------------------------------------------
# Opt-in Anthropic thinking passthrough
# ------------------------------------------------------------------------------
def reasoning_delta(text: str, key: str = "reasoning_content") -> bytes:
    """An OpenAI chunk carrying reasoning text in one of the shapes seen live."""
    return sse(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": "mock-model",
            "choices": [{"index": 0, "delta": {key: text}, "finish_reason": None}],
        }
    )


def reasoning_then_text(key: str = "reasoning_content") -> list:
    return [
        reasoning_delta("step one. ", key),
        reasoning_delta("step two.", key),
        text_delta("The answer."),
        finish_delta("stop"),
        done(),
    ]


async def translate(frames, thinking: bool = False):
    async def gen():
        for frame in frames:
            yield frame

    return [
        frame
        async for frame in translate_stream(
            gen(), "claude-sonnet-4", 11, None, thinking
        )
    ]


def block_starts(events):
    return [json.loads(data) for name, data in events if name == "content_block_start"]


def deltas_of(events, delta_type: str) -> str:
    return "".join(
        json.loads(data)["delta"].get("text") or json.loads(data)["delta"].get("thinking") or ""
        for name, data in events
        if name == "content_block_delta"
        and json.loads(data)["delta"]["type"] == delta_type
    )


async def test_reasoning_is_dropped_by_default():
    """Default must stay unchanged: unsigned thinking blocks can break clients."""
    events = parse_sse(b"".join(await translate(reasoning_then_text())))

    starts = block_starts(events)
    assert starts, "the response should still carry content blocks"
    assert all(start["content_block"]["type"] == "text" for start in starts)
    assert deltas_of(events, "thinking_delta") == ""
    assert deltas_of(events, "text_delta") == "The answer."


async def test_thinking_passthrough_emits_a_thinking_block_first():
    events = parse_sse(b"".join(await translate(reasoning_then_text(), thinking=True)))

    starts = block_starts(events)
    assert starts[0]["content_block"]["type"] == "thinking"
    assert starts[0]["index"] == 0, "thinking must precede text, as Anthropic requires"
    assert any(start["content_block"]["type"] == "text" for start in starts)

    assert deltas_of(events, "thinking_delta") == "step one. step two."
    assert deltas_of(events, "text_delta") == "The answer."
    # Block order is observable through the index sequence.
    assert [start["content_block"]["type"] for start in starts] == ["thinking", "text"]


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning", "thinking"])
async def test_thinking_passthrough_handles_every_reasoning_key(key):
    events = parse_sse(
        b"".join(await translate(reasoning_then_text(key), thinking=True))
    )
    assert deltas_of(events, "thinking_delta") == "step one. step two."


async def test_thinking_passthrough_handles_a_part_list():
    frames = [
        sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning": [{"type": "text", "text": "part a"}]},
                        "finish_reason": None,
                    }
                ]
            }
        ),
        text_delta("done"),
        finish_delta("stop"),
        done(),
    ]
    events = parse_sse(b"".join(await translate(frames, thinking=True)))
    assert deltas_of(events, "thinking_delta") == "part a"


def test_non_streaming_thinking_passthrough_is_opt_in():
    payload = {
        "choices": [{"message": {"content": "Answer.", "reasoning_content": "because"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
    }

    default = openai_to_anthropic_response(payload, "m")
    assert [block["type"] for block in default["content"]] == ["text"]

    enabled = openai_to_anthropic_response(payload, "m", True)
    assert enabled["content"][0] == {"type": "thinking", "thinking": "because"}
    assert enabled["content"][1]["type"] == "text"
    assert enabled["content"][1]["text"] == "Answer."


def test_anthropic_profile_exposes_thinking_passthrough_setting():
    from src.config import load_settings

    assert load_settings(env={}).anthropic_thinking_passthrough is False
    assert (
        load_settings(env={"ANTHROPIC_THINKING_PASSTHROUGH": "1"}).anthropic_thinking_passthrough
        is True
    )
