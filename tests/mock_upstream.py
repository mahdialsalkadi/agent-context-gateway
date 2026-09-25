"""
Offline mock upstream.

Lets the entire test suite -- and a human trying the gateway out -- run with
zero API keys and zero cost. Two ways to use it:

1. As a real HTTP server (default), which exercises the true network path::

       python -m tests.mock_upstream --port 9099
       UPSTREAM_BASE_URL=http://127.0.0.1:9099/v1 agent-gateway

2. As an in-process `httpx.MockTransport`, for deterministic byte-level tests::

       transport = httpx.MockTransport(scripted_handler([...]))

Response shape is chosen from the conversation, so a test just writes a prompt:

* contains ``force_escape``  -> emits the escape sentinel in the *first* delta
* contains ``use_tool``      -> emits a streamed tool call
* contains ``fail``          -> returns HTTP 500
* otherwise                  -> streams a short text answer

Every received request is recorded on the shared state, which is how tests
assert what the gateway actually forwarded (e.g. whether the tool schema
survived pruning).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_TEXT = "Hello! This is the offline mock upstream replying."

ESCAPE_TOKEN = "[ESCAPE_NEED_TOOLS]"

TOOL_CALL_NAME = "terminal"
TOOL_CALL_ARGUMENTS = '{"command": "ls -la"}'


# ------------------------------------------------------------------------------
# SSE construction
# ------------------------------------------------------------------------------
def sse(chunk: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode()


def text_delta(text: str, model: str = "mock-model") -> bytes:
    return sse(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
    )


def tool_delta(name: str, arguments: str, call_id: str = "call_mock", model: str = "mock-model") -> bytes:
    return sse(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
    )


def finish_delta(reason: str = "stop", model: str = "mock-model") -> bytes:
    return sse(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
        }
    )


def done() -> bytes:
    return b"data: [DONE]\n\n"


def openai_stream(text: str = DEFAULT_TEXT, tool_call: bool = False) -> List[bytes]:
    """A complete scripted OpenAI SSE response, split into realistic chunks."""
    frames: List[bytes] = []

    if tool_call:
        frames.append(tool_delta(TOOL_CALL_NAME, TOOL_CALL_ARGUMENTS))
        frames.append(finish_delta("tool_calls"))
    else:
        # Deliberately split mid-word to prove partial-line handling works.
        third = max(1, len(text) // 3)
        frames.append(text_delta(text[:third]))
        frames.append(text_delta(text[third:third * 2]))
        frames.append(text_delta(text[third * 2:]))
        frames.append(finish_delta("stop"))

    frames.append(done())
    return frames


def escape_stream(follow_up: str = " I need the terminal to do that.") -> List[bytes]:
    """Escape sentinel in the very first delta, then ordinary text."""
    return [
        text_delta(ESCAPE_TOKEN),
        text_delta(follow_up),
        finish_delta("stop"),
        done(),
    ]


def plan_for(messages: Sequence[Dict[str, Any]], tools: Optional[list] = None) -> Dict[str, Any]:
    """Pick a behaviour from the latest user turn."""
    prompt = ""
    for message in reversed(list(messages or [])):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                prompt = content
            elif isinstance(content, list):
                prompt = " ".join(
                    str(part.get("text", "")) for part in content if isinstance(part, dict)
                )
            break

    lowered = prompt.lower()
    has_tools = bool(tools)

    # A real model only says "I need tools" when it has none. Modelling that here
    # makes the escape-replay path observable: the pruned first attempt escapes,
    # and the replay -- which has tools -- produces a tool call instead.
    if "force_escape" in lowered:
        return {"kind": "tool" if has_tools else "escape", "prompt": prompt}
    if "fail" in lowered:
        return {"kind": "error", "status": 500}
    if "use_tool" in lowered or "call the tool" in lowered:
        return {"kind": "tool", "prompt": prompt}
    return {"kind": "text", "prompt": prompt}


# ------------------------------------------------------------------------------
# Anthropic-shaped SSE (kept for protocol reference / extension testing)
# ------------------------------------------------------------------------------
def anthropic_sse(event: str, data: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def anthropic_stream(text: str = DEFAULT_TEXT) -> List[bytes]:
    frames = [
        anthropic_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_mock",
                    "type": "message",
                    "role": "assistant",
                    "model": "mock-model",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
        ),
        anthropic_sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    for piece in (text[: len(text) // 2], text[len(text) // 2:]):
        frames.append(
            anthropic_sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}},
            )
        )
    frames.append(anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(
        anthropic_sse(
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 8}},
        )
    )
    frames.append(anthropic_sse("message_stop", {"type": "message_stop"}))
    return frames


# ------------------------------------------------------------------------------
# Shared request log
# ------------------------------------------------------------------------------
class MockState:
    """Records every request the gateway forwards."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: List[Dict[str, Any]] = []

    def record(self, path: str, body: Optional[dict], headers: Dict[str, str]) -> Dict[str, Any]:
        entry = {"path": path, "body": body, "headers": headers, "ts": time.time()}
        with self._lock:
            self.requests.append(entry)
        return entry

    @property
    def bodies(self) -> List[dict]:
        with self._lock:
            return [entry["body"] for entry in self.requests]

    def reset(self) -> None:
        with self._lock:
            self.requests.clear()


# ------------------------------------------------------------------------------
# HTTP server
# ------------------------------------------------------------------------------
def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # keep test output clean
            return

        # --- helpers -------------------------------------------------------
        def _read_json(self) -> Optional[dict]:
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return None
            try:
                return json.loads(raw)
            except Exception:
                return None

        def _send_json(self, status: int, payload: dict) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _send_stream(self, frames: Iterable[bytes]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for frame in frames:
                self.wfile.write(frame)
                self.wfile.flush()

        # --- routes --------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/v1/models"):
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": "mock-model", "object": "model", "owned_by": "mock"},
                            {"id": "deepseek/deepseek-r1", "object": "model", "owned_by": "mock"},
                            {"id": "mock-tool-model", "object": "model", "owned_by": "mock"},
                        ],
                    },
                )
                return
            self._send_json(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:  # noqa: N802
            body = self._read_json()
            state.record(self.path, body, dict(self.headers))
            body = body or {}

            if self.path.rstrip("/").endswith("/v1/messages"):
                self._send_stream(anthropic_stream())
                return

            if not self.path.rstrip("/").endswith("/chat/completions"):
                self._send_json(404, {"error": {"message": "not found"}})
                return

            plan = plan_for(body.get("messages") or [], body.get("tools"))
            model = str(body.get("model") or "mock-model")

            if plan["kind"] == "error":
                self._send_json(
                    plan["status"],
                    {"error": {"message": "mock upstream failure", "type": "mock"}},
                )
                return

            if not body.get("stream"):
                if plan["kind"] == "tool":
                    message: Dict[str, Any] = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_mock",
                                "type": "function",
                                "function": {"name": TOOL_CALL_NAME, "arguments": TOOL_CALL_ARGUMENTS},
                            }
                        ],
                    }
                    finish = "tool_calls"
                elif plan["kind"] == "escape":
                    message = {"role": "assistant", "content": f"{ESCAPE_TOKEN} I need tools."}
                    finish = "stop"
                else:
                    message = {"role": "assistant", "content": DEFAULT_TEXT}
                    finish = "stop"
                self._send_json(
                    200,
                    {
                        "id": "chatcmpl-mock",
                        "object": "chat.completion",
                        "model": model,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
                    },
                )
                return

            if plan["kind"] == "escape":
                self._send_stream(escape_stream())
            elif plan["kind"] == "tool":
                self._send_stream(openai_stream(tool_call=True))
            else:
                self._send_stream(openai_stream())

    return Handler


def start_mock_upstream(host: str = "127.0.0.1", port: int = 0) -> Tuple[ThreadingHTTPServer, MockState, str]:
    """Start the mock upstream on a background thread.

    Returns (server, state, base_url) where base_url already ends in ``/v1``.
    """
    state = MockState()
    server = ThreadingHTTPServer((host, port), make_handler(state))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{server.server_address[1]}/v1"
    return server, state, base_url


# ------------------------------------------------------------------------------
# In-process transport
# ------------------------------------------------------------------------------
def scripted_handler(frames: Sequence[bytes]) -> Any:
    """Build an httpx handler that replays pre-built SSE frames.

    Useful when a test needs byte-exact control, including deliberately awkward
    chunk boundaries.
    """

    def handler(request: "httpx.Request") -> "httpx.Response":
        import httpx

        payload = json.loads(request.content) if request.content else {}
        plan = plan_for(payload.get("messages") or [], payload.get("tools"))
        if plan["kind"] == "error":
            return httpx.Response(500, json={"error": {"message": "mock failure"}}, request=request)
        if not payload.get("stream"):
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "model": payload.get("model", "mock-model"),
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": DEFAULT_TEXT},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 4},
                },
                request=request,
            )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=iter(frames), request=request)

    return handler


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-mock-upstream",
        description="Offline OpenAI/Anthropic mock upstream for agent-context-gateway.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9099)
    args = parser.parse_args(argv)

    server, _, base_url = start_mock_upstream(args.host, args.port)
    print(f"[mock-upstream] listening on {base_url}")
    print(f"[mock-upstream] try: UPSTREAM_BASE_URL={base_url} UPSTREAM_API_KEY=test agent-gateway")
    print(
        "[mock-upstream] prompt keywords: 'use_tool' -> tool call, "
        "'force_escape' -> escape sentinel, 'fail' -> HTTP 500"
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[mock-upstream] stopping.")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
