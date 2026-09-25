"""
Anthropic <-> OpenAI translation.

Claude Code does not speak `/v1/chat/completions`; it speaks the Anthropic
Messages API at `/v1/messages` (and probes `/v1/messages/count_tokens`). Pointing
`ANTHROPIC_BASE_URL` at this gateway therefore requires a real translation layer,
not a passthrough -- otherwise the tool-pruning logic, which reasons about
OpenAI-style tool schemas, could never apply.

This module is pure translation with no environment access and no I/O, so it is
cheap to unit test. Coverage targets the shapes agents actually send:

* system as a string or a list of text blocks
* user/assistant content as a string or a list of text/image/tool_use blocks
* `tool_result` blocks, which must become separate OpenAI `tool` messages
* `tool_choice` in all four Anthropic forms
* streaming: OpenAI `delta` frames -> Anthropic `content_block_*` events

Not translated (documented rather than silently dropped): extended-thinking
blocks, `cache_control` hints, and audio/document content blocks.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Callable, Dict, Iterable, List, Optional, Tuple

TOOL_USE_ID_PREFIX = "toolu_"

# OpenAI finish_reason -> Anthropic stop_reason
STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
    "error": "end_turn",
}


def estimate_tokens(text: str) -> int:
    """Rough character-based token estimate.

    Used only for the `usage` fields, which Claude Code reads for display. It is
    deliberately approximate: requesting exact usage would mean sending
    `stream_options` to the upstream, which several OpenAI-compatible servers
    reject outright.
    """
    return max(1, len(text or "") // 4)


# ------------------------------------------------------------------------------
# Request: Anthropic -> OpenAI
# ------------------------------------------------------------------------------
def _text_from_blocks(blocks: Any) -> str:
    if isinstance(blocks, str):
        return blocks
    if not isinstance(blocks, list):
        return ""
    parts: List[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "\n".join(part for part in parts if part)


def _image_to_openai(block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source = block.get("source")
    if not isinstance(source, dict):
        return None
    kind = source.get("type")
    if kind == "base64" and source.get("data"):
        media = source.get("media_type", "image/png")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media};base64,{source['data']}"},
        }
    if kind == "url" and source.get("url"):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_content(block: Dict[str, Any]) -> str:
    """Flatten an Anthropic tool_result into OpenAI tool message content."""
    inner = block.get("content")
    if isinstance(inner, str):
        text = inner
    elif isinstance(inner, list):
        chunks: List[str] = []
        for part in inner:
            if isinstance(part, dict) and part.get("type") == "text":
                chunks.append(str(part.get("text", "")))
            elif isinstance(part, str):
                chunks.append(part)
            else:
                chunks.append(json.dumps(part))
        text = "\n".join(chunks)
    elif inner is None:
        text = ""
    else:
        text = json.dumps(inner)

    # OpenAI has no error flag on tool messages, so surface it in the text.
    if block.get("is_error"):
        text = f"ERROR: {text}"
    return text


def anthropic_tools_to_openai(tools: Any) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    if not isinstance(tools, list):
        return converted
    for tool in tools:
        if not isinstance(tool, dict) or not tool.get("name"):
            continue
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return converted


def anthropic_tool_choice_to_openai(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return None
    kind = choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "none":
        return "none"
    if kind == "tool" and choice.get("name"):
        return {"type": "function", "function": {"name": choice["name"]}}
    return None


def anthropic_messages_to_openai(messages: Iterable[Any]) -> Tuple[List[Dict[str, Any]], int]:
    """Convert messages, returning (openai_messages, tool_results_emitted)."""
    converted: List[Dict[str, Any]] = []
    tool_results = 0

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")

        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            converted.append({"role": role, "content": ""})
            continue

        # --- assistant: text + tool_use -> content + tool_calls -------------
        if role == "assistant":
            text = _text_from_blocks(content)
            tool_calls: List[Dict[str, Any]] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id") or f"{TOOL_USE_ID_PREFIX}{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
            entry: Dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
                # OpenAI requires a non-empty content or tool_calls; keep "" for
                # providers that reject null on an assistant turn.
                entry["content"] = text or ""
            converted.append(entry)
            continue

        # --- user: text + image + tool_result -------------------------------
        parts: List[Dict[str, Any]] = []
        text = _text_from_blocks(content)
        if text:
            parts.append({"type": "text", "text": text})

        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                image = _image_to_openai(block)
                if image:
                    parts.append(image)
            elif block.get("type") == "tool_result":
                # Each result becomes its own `tool` message, immediately after
                # the assistant turn that requested it.
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _tool_result_content(block),
                    }
                )
                tool_results += 1

        if parts:
            # Single text part alone is normalised to a plain string.
            if len(parts) == 1 and parts[0].get("type") == "text":
                converted.append({"role": "user", "content": parts[0]["text"]})
            else:
                converted.append({"role": "user", "content": parts})

    return converted, tool_results


def anthropic_to_openai_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """Translate an Anthropic Messages request into a Chat Completions request."""
    messages: List[Dict[str, Any]] = []

    system = body.get("system")
    system_text = _text_from_blocks(system)
    if system_text:
        messages.append({"role": "system", "content": system_text})

    converted, _ = anthropic_messages_to_openai(body.get("messages"))
    messages.extend(converted)

    out: Dict[str, Any] = {"model": body.get("model", ""), "messages": messages}

    # Forward the sampling knobs that exist in both APIs.
    for src, dst in (
        ("max_tokens", "max_tokens"),
        ("temperature", "temperature"),
        ("top_p", "top_p"),
    ):
        if body.get(src) is not None:
            out[dst] = body[src]

    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]

    metadata = body.get("metadata")
    if isinstance(metadata, dict) and metadata.get("user_id"):
        out["user"] = metadata["user_id"]

    tools = anthropic_tools_to_openai(body.get("tools"))
    if tools:
        out["tools"] = tools
        tool_choice = anthropic_tool_choice_to_openai(body.get("tool_choice"))
        if tool_choice is not None:
            out["tool_choice"] = tool_choice

    return out


# ------------------------------------------------------------------------------
# Response: OpenAI -> Anthropic (non-streaming)
# ------------------------------------------------------------------------------
def _tool_use_block(call: Dict[str, Any]) -> Dict[str, Any]:
    function = call.get("function") or {}
    raw = function.get("arguments")
    if isinstance(raw, str):
        try:
            arguments = json.loads(raw) if raw.strip() else {}
        except Exception:
            # Anthropic requires an object. Preserve the raw text rather than
            # discarding it, so the model can still see what it produced.
            arguments = {"_raw_arguments": raw}
    elif isinstance(raw, dict):
        arguments = raw
    else:
        arguments = {}

    return {
        "type": "tool_use",
        "id": call.get("id") or f"{TOOL_USE_ID_PREFIX}{uuid.uuid4().hex[:24]}",
        "name": function.get("name", ""),
        "input": arguments,
    }


def openai_to_anthropic_response(payload: Dict[str, Any], model: str = "") -> Dict[str, Any]:
    """Convert a non-streaming OpenAI completion into an Anthropic message."""
    choices = payload.get("choices") or [{}]
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}

    content: List[Dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                content.append({"type": "text", "text": part["text"]})

    for call in message.get("tool_calls") or []:
        if isinstance(call, dict):
            content.append(_tool_use_block(call))

    legacy = message.get("function_call")
    if isinstance(legacy, dict) and legacy.get("name"):
        content.append(_tool_use_block({"id": None, "function": legacy}))

    if not content:
        content.append({"type": "text", "text": ""})

    stop_reason = STOP_REASON_MAP.get(choice.get("finish_reason") or "stop", "end_turn")
    usage = payload.get("usage") or {}

    return {
        "id": payload.get("id") or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model or payload.get("model", ""),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


# ------------------------------------------------------------------------------
# Response: OpenAI SSE -> Anthropic SSE (streaming)
# ------------------------------------------------------------------------------
def sse_frame(event: str, data: Dict[str, Any]) -> bytes:
    """Render one Anthropic SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode()


def _text_delta_frame(index: int, text: str) -> bytes:
    return sse_frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        },
    )


def _json_delta_frame(index: int, partial: str) -> bytes:
    return sse_frame(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial},
        },
    )


class AnthropicStreamTranslator:
    """Stateful OpenAI-SSE -> Anthropic-SSE converter.

    Feed it raw upstream byte chunks; it returns the Anthropic frames to emit.
    State is required because Anthropic models a stream as explicit content
    blocks, while OpenAI just emits an undifferentiated delta sequence.
    """

    def __init__(self, model: str, input_tokens: int = 0) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self._buffer = ""
        self._started = False
        self._finished = False
        self._block_index = -1
        self._open_kind: Optional[str] = None  # "text" | "tool"
        self._tool_slots: Dict[int, int] = {}  # openai tool index -> an block index
        self._text_chars = 0
        self._stop_reason = "end_turn"
        self._message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # --- framing -----------------------------------------------------------
    def _message_start(self) -> bytes:
        self._started = True
        return sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": self._message_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
                },
            },
        )

    def _open_block(self, kind: str, block: Dict[str, Any]) -> bytes:
        self._block_index += 1
        self._open_kind = kind
        return sse_frame(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": self._block_index,
                "content_block": block,
            },
        )

    def _close_block(self) -> bytes:
        if self._open_kind is None:
            return b""
        self._open_kind = None
        return sse_frame(
            "content_block_stop",
            {"type": "content_block_stop", "index": self._block_index},
        )

    # --- deltas ------------------------------------------------------------
    def _on_text(self, text: str) -> List[bytes]:
        frames: List[bytes] = []
        if self._open_kind != "text":
            frames.append(self._close_block())
            frames.append(self._open_block("text", {"type": "text", "text": ""}))
        self._text_chars += len(text)
        frames.append(_text_delta_frame(self._block_index, text))
        return frames

    def _on_tool_call(self, call: Dict[str, Any]) -> List[bytes]:
        frames: List[bytes] = []
        slot = int(call.get("index", 0) or 0)
        function = call.get("function") or {}

        if slot not in self._tool_slots:
            frames.append(self._close_block())
            frames.append(
                self._open_block(
                    "tool",
                    {
                        "type": "tool_use",
                        "id": call.get("id")
                        or f"{TOOL_USE_ID_PREFIX}{uuid.uuid4().hex[:24]}",
                        "name": function.get("name", ""),
                        "input": {},
                    },
                )
            )
            self._tool_slots[slot] = self._block_index

        arguments = function.get("arguments")
        if isinstance(arguments, str) and arguments:
            frames.append(_json_delta_frame(self._tool_slots[slot], arguments))
        return frames

    def _on_choice(self, choice: Dict[str, Any]) -> List[bytes]:
        frames: List[bytes] = []
        delta = choice.get("delta") or {}

        if choice.get("finish_reason"):
            self._stop_reason = STOP_REASON_MAP.get(choice["finish_reason"], "end_turn")

        text = delta.get("content")
        if isinstance(text, str) and text:
            frames.extend(self._on_text(text))
        elif isinstance(text, list):
            for part in text:
                if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                    frames.extend(self._on_text(part["text"]))

        for call in delta.get("tool_calls") or []:
            if isinstance(call, dict):
                frames.extend(self._on_tool_call(call))

        legacy = delta.get("function_call")
        if isinstance(legacy, dict) and legacy.get("name"):
            frames.extend(self._on_tool_call({"index": 0, "id": None, "function": legacy}))

        return frames

    # --- public API --------------------------------------------------------
    def feed(self, chunk: bytes) -> List[bytes]:
        """Consume a raw upstream chunk; return Anthropic frames ready to send.

        Chunk boundaries do not align with SSE frame boundaries, so partial
        lines are retained until completed.
        """
        frames: List[bytes] = []
        if self._finished:
            return frames

        try:
            self._buffer += chunk.decode("utf-8", errors="ignore")
        except Exception:
            return frames

        while "\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition("\n")
            frames.extend(self._on_line(line.strip()))

        # Empty frames are dropped so the client never receives zero-length
        # SSE writes from a no-op block transition.
        return [frame for frame in frames if frame]

    def _on_line(self, line: str) -> List[bytes]:
        frames: List[bytes] = []
        if self._finished or not line.startswith("data:"):
            return frames

        payload = line[5:].strip()
        if not payload:
            return frames

        if payload == "[DONE]":
            frames.extend(self.finish())
            return frames

        try:
            event = json.loads(payload)
        except Exception:
            return frames

        if not self._started:
            frames.append(self._message_start())

        for choice in event.get("choices") or []:
            if isinstance(choice, dict):
                frames.extend(self._on_choice(choice))

        # Some providers emit usage only on a trailing frame.
        usage = event.get("usage") if isinstance(event, dict) else None
        if isinstance(usage, dict) and usage.get("prompt_tokens"):
            self.input_tokens = int(usage["prompt_tokens"])

        return frames

    def finish(self) -> List[bytes]:
        """Emit block close, message_delta and message_stop exactly once."""
        if self._finished:
            return []
        self._finished = True

        frames: List[bytes] = []
        if not self._started:
            frames.append(self._message_start())

        # Anthropic always returns at least one content block.
        if self._open_kind is None and not self._tool_slots and self._block_index < 0:
            frames.append(self._open_block("text", {"type": "text", "text": ""}))

        frames.append(self._close_block())
        frames.append(
            sse_frame(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": self._stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": estimate_tokens("x" * self._text_chars)},
                },
            )
        )
        frames.append(sse_frame("message_stop", {"type": "message_stop"}))
        return [frame for frame in frames if frame]


async def translate_stream(
    source: AsyncIterator[bytes],
    model: str,
    input_tokens: int = 0,
    on_upstream_chunk: Optional[Callable[[bytes], None]] = None,
) -> AsyncIterator[bytes]:
    """Adapt an OpenAI SSE byte stream into an Anthropic SSE byte stream.

    `on_upstream_chunk` sees each raw upstream chunk, which lets the caller tap
    the original OpenAI deltas for memory ingestion while the client receives
    fully translated Anthropic events.
    """
    translator = AnthropicStreamTranslator(model, input_tokens)
    async for chunk in source:
        if on_upstream_chunk is not None:
            try:
                on_upstream_chunk(chunk)
            except Exception:
                pass
        for frame in translator.feed(chunk):
            yield frame
    for frame in translator.finish():
        yield frame
