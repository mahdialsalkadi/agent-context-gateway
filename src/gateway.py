"""
FastAPI application: routing, streaming and the two protocol surfaces.

Exposed routes
--------------
``GET  /health``                     liveness plus a redacted config snapshot
``GET  /v1/models``                  passthrough so clients can probe
``POST /v1/chat/completions``        OpenAI-compatible surface
``POST /v1/messages``                Anthropic surface (Claude Code)
``POST /v1/messages/count_tokens``   Anthropic token probe

Invariants enforced here, and covered by the test suite:

1. **No forwarding loop.** Startup aborts if the upstream resolves to this
   gateway's own address or a legacy proxy port.
2. **Fail-open.** A broken classifier keeps the tools; it never removes them.
3. **Transparent protocol.** Unknown JSON fields are forwarded untouched.
4. **No full-response buffering.** Streaming is chunk-in/chunk-out; the escape
   look-ahead is bounded and exits as early as the head becomes unambiguous.

The app is built by `create_app`, which accepts injected collaborators so tests
can run entirely offline against a mock upstream.
"""

from __future__ import annotations

import json
import argparse
import os
import socket
import sys
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from . import __version__
from .artifacts import FETCH_LOG_TOOL, ArtifactStore, get_store
from .dashboard import (
    dashboard_response,
    memory_relations,
    forget_relation,
    profile_payload,
    stats_payload,
)
from .bridge import (
    anthropic_to_openai_request,
    estimate_tokens,
    openai_to_anthropic_response,
    translate_stream,
)
from .classifier import (
    CORE_TOOLS,
    ESCAPE_TOKEN,
    Classifier,
    decide_route,
    get_classifier,
    inject_escape_instruction,
    rank_tools,
    strip_escape_instruction,
)
from .config import (
    SELECTIVE_PRUNING_MIN_TOOLS,
    Settings,
    available_profiles,
    find_profile,
    load_settings,
)
from .memory import GraphMemory, get_memory
from .messages import (
    iter_tool_call_refs,
    last_user_text,
    normalize_messages,
    text_of,
)

# ------------------------------------------------------------------------------
# Upstream auth
# ------------------------------------------------------------------------------
# Values that mean "no real credential of our own". A flat-rate subscription
# bridge (Antigravity, a Claude Code session) or a local engine authenticates the
# *client's* session, so the gateway must forward the client's Authorization
# header verbatim instead of replacing it with a placeholder bearer token.
_PLACEHOLDER_UPSTREAM_KEYS = frozenset(
    {"", "dummy", "none", "null", "changeme", "replace_me", "placeholder"}
)


def upstream_auth_header(api_key: str, client_auth: str = "") -> str:
    """The `Authorization` value to send upstream, or "" for none.

    A real `UPSTREAM_API_KEY` always wins: the gateway owns the credential and
    injects it, so the agent never learns it. A placeholder (or absent) key means
    the upstream rides the client's own subscription session, so the client's
    header is forwarded untouched -- that is the whole point of bridging. As a
    last resort the placeholder itself is re-sent, so an upstream that insists on
    a non-empty bearer still gets one.
    """
    key = (api_key or "").strip()
    if key and key.lower() not in _PLACEHOLDER_UPSTREAM_KEYS:
        return f"Bearer {key}"
    if client_auth:
        return client_auth
    return f"Bearer {key}" if key else ""


def client_upstream_auth(request: Request, gateway_api_key: str = "") -> str:
    """The client's own credential, for bridging a subscription session.

    Claude Code sends its session token as a bearer *and* as `x-api-key`; either
    is forwarded. A credential that matches `GATEWAY_API_KEY` authenticates the
    client to *this* gateway, not to the upstream, so it is never leaked.
    """
    header = (request.headers.get("authorization") or "").strip()
    if not header:
        token = (request.headers.get("x-api-key") or "").strip()
        header = f"Bearer {token}" if token else ""
    if not header:
        return ""
    presented = header[7:].strip() if header.lower().startswith("bearer ") else header
    if gateway_api_key and presented == gateway_api_key:
        return ""
    return header


# ------------------------------------------------------------------------------
# Streaming helpers
# ------------------------------------------------------------------------------
class SSETap:
    """Accumulates assistant text from an OpenAI SSE byte stream.

    Buffers partial lines across chunk boundaries: a chunk is not guaranteed to
    end on an SSE frame boundary, and naive per-chunk JSON parsing drops tokens.
    """

    MAX_BUFFER = 1_000_000

    def __init__(self) -> None:
        self._buffer = ""
        self.text_parts: List[str] = []

    def feed(self, chunk: bytes) -> None:
        try:
            self._buffer += chunk.decode("utf-8", errors="ignore")
        except Exception:
            return
        if len(self._buffer) > self.MAX_BUFFER:
            self._buffer = self._buffer[-65536:]
        while "\n" in self._buffer:
            line, _, self._buffer = self._buffer.partition("\n")
            self._on_line(line.strip())

    def _on_line(self, line: str) -> None:
        if not line.startswith("data:"):
            return
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return
        try:
            event = json.loads(payload)
        except Exception:
            return
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or choice.get("message") or {}
            if not isinstance(delta, dict):
                continue
            for key in ("content", "reasoning_content", "reasoning"):
                value = delta.get(key)
                if isinstance(value, str) and value:
                    self.text_parts.append(value)

    @property
    def text(self) -> str:
        return "".join(self.text_parts).strip()


async def _close_quietly(
    response: Optional[httpx.Response], client: Optional[httpx.AsyncClient]
) -> None:
    for closer in (response, client):
        if closer is None:
            continue
        try:
            await closer.aclose()
        except Exception:
            pass


def apply_selective_pruning(
    settings: Settings,
    prompt: str,
    tools: List[Dict[str, Any]],
    must_keep: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """Shrink a large tool schema to the tools this turn can actually need.

    Returns (tools, dropped_count). Only fires above
    SELECTIVE_PRUNING_MIN_TOOLS, but from there it is an *absolute* cap:
    `rank_tools` is called in hard-cap mode, so no prompt -- including one in a
    language BM25 cannot read -- can return the whole catalog, and the result is
    always `<= settings.selective_tool_limit`.

    `must_keep` names tools the conversation is already mid-call on (a tool
    turn). They are guaranteed inside the ceiling -- displacing the weakest
    ranked pick, never overflowing it -- so the model always receives the schema
    it is about to be answered for.
    """
    if (
        not settings.enable_selective_pruning
        or not isinstance(tools, list)
        or len(tools) <= SELECTIVE_PRUNING_MIN_TOOLS
    ):
        return tools, 0
    try:
        kept = rank_tools(
            prompt, tools, settings.selective_tool_limit, hard_cap=True
        )
        if must_keep:
            kept = _ensure_present(kept, tools, must_keep, settings.selective_tool_limit)
    except Exception:
        return tools, 0
    if len(kept) >= len(tools):
        return tools, 0
    return kept, len(tools) - len(kept)


def _tool_name(tool: Any) -> str:
    """The tool's name in either the OpenAI or Anthropic payload shape."""
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")


def _ensure_present(
    kept: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    must_keep: List[str],
    limit: int,
) -> List[Dict[str, Any]]:
    """Fold `must_keep` into `kept` without ever exceeding `limit` entries.

    A pending call the upstream is mid-answer on outranks every heuristic: if it
    was displaced by the ranker, the weakest non-required tool gives way. Order
    follows the original payload so prompt caches stay warm.
    """
    required = {name for name in must_keep if name}
    if not required:
        return kept
    kept_names = {_tool_name(tool) for tool in kept}
    missing = [name for name in must_keep if name not in kept_names]
    if not missing:
        return kept

    result = list(kept)
    for name in missing:
        if any(_tool_name(tool) == name for tool in result):
            continue
        source = next(
            (tool for tool in tools if _tool_name(tool) == name), None
        )
        if source is None:
            continue
        if len(result) < limit:
            result.append(source)
            continue
        # Displace the weakest entry that is not itself required.
        for index in reversed(range(len(result))):
            if _tool_name(result[index]) not in required:
                result[index] = source
                break
    # Restore payload order for the returned subset.
    kept_order = {_tool_name(tool): index for index, tool in enumerate(tools)}
    result.sort(key=lambda tool: kept_order.get(_tool_name(tool), len(tools)))
    return result


def prune_for_tool_loop(
    tools: List[Dict[str, Any]],
    active_tools: List[str],
    limit: int = 5,
) -> Tuple[List[Dict[str, Any]], int]:
    """Hard-cap the tools in ToolLoop strictly to <= 5 tools (active tool + CORE_TOOLS).

    Ensures requests in a multi-turn tool cycle (`role: "tool"`) NEVER leak the full
    catalog (49 tools).
    """
    if not isinstance(tools, list) or not tools:
        return tools, 0
    if len(tools) <= limit:
        return tools, 0

    tool_by_name = {_tool_name(t): t for t in tools}

    # 1. Active tool(s) first
    selected_names: List[str] = []
    for name in active_tools:
        if name in tool_by_name and name not in selected_names:
            selected_names.append(name)
            if len(selected_names) >= limit:
                break

    # 2. CORE_TOOLS to fill up to limit
    if len(selected_names) < limit:
        for tool in tools:
            name = _tool_name(tool)
            if name in CORE_TOOLS and name not in selected_names:
                selected_names.append(name)
                if len(selected_names) >= limit:
                    break

    # 3. Fallback to remaining tools if still fewer than limit
    if len(selected_names) < limit:
        for tool in tools:
            name = _tool_name(tool)
            if name not in selected_names:
                selected_names.append(name)
                if len(selected_names) >= limit:
                    break

    kept = [tool_by_name[name] for name in selected_names if name in tool_by_name]
    # Preserve original order from tools list for cache stability
    order = {_tool_name(t): i for i, t in enumerate(tools)}
    kept.sort(key=lambda t: order.get(_tool_name(t), len(tools)))
    return kept, len(tools) - len(kept)


def head_could_be_escape(text: str) -> bool:
    """True if the opening characters could still turn out to be the token.

    The escape instruction puts the sentinel at the very start of the output, so
    once the head cannot be a prefix of it, a compliant model will not emit one
    later and sniffing can stop.
    """
    head = text[: len(ESCAPE_TOKEN)].lstrip()
    return ESCAPE_TOKEN.startswith(head) or head.startswith(ESCAPE_TOKEN)


def tail_looks_like_escape(text: str) -> bool:
    tail = text[-len(ESCAPE_TOKEN):]
    return ESCAPE_TOKEN.startswith(tail) or tail in ESCAPE_TOKEN


async def sniff_for_escape(
    chunks: AsyncIterator[bytes], limit: int, overflow: int, scan_chars: int
) -> Tuple[List[bytes], bool]:
    """Bounded look-ahead on a pruned route.

    Consumes from a live byte iterator and returns everything it buffered, plus
    whether the escape sentinel was seen. Reads at most `limit` (+ a small
    overflow so the token is never split across the boundary). It never buffers
    a whole stream -- the original implementation used `aread()`, which did
    exactly that.

    Takes an iterator rather than a Response on purpose: an httpx stream is
    single-pass, so the caller must hand the *same* iterator on to the relay
    afterwards. Starting a second `aiter_raw()` raises `StreamConsumed`.
    """
    buffered: List[bytes] = []
    total = 0
    text = ""

    async for chunk in chunks:
        buffered.append(chunk)
        total += len(chunk)
        text += chunk.decode("utf-8", errors="ignore")

        if ESCAPE_TOKEN in text:
            return buffered, True
        if total >= limit:
            if tail_looks_like_escape(text) and total < limit + overflow:
                continue
            break
        if len(text) >= scan_chars and not head_could_be_escape(text):
            break

    return buffered, False


# ------------------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------------------
def create_app(
    settings: Optional[Settings] = None,
    classifier: Optional[Classifier] = None,
    store: Optional[ArtifactStore] = None,
    memory: Optional[GraphMemory] = None,
    upstream_transport: Optional[httpx.AsyncBaseTransport] = None,
) -> FastAPI:
    settings = settings or load_settings()

    app = FastAPI(
        title="agent-context-gateway",
        version=__version__,
        description=(
            "Agent-agnostic, OpenAI-compatible gateway that prunes tool schemas, "
            "spills oversized context and maintains a self-pruning graph memory."
        ),
    )
    app.state.settings = settings
    app.state.classifier = classifier
    app.state.store = store
    app.state.memory = memory
    app.state.upstream_transport = upstream_transport

    # --- accessors (lazy so import never touches the filesystem) ------------
    def cfg() -> Settings:
        return app.state.settings

    def klass() -> Classifier:
        if app.state.classifier is None:
            app.state.classifier = get_classifier(cfg())
        return app.state.classifier

    def artifacts() -> ArtifactStore:
        if app.state.store is None:
            app.state.store = get_store(cfg())
        return app.state.store

    def graph() -> GraphMemory:
        if app.state.memory is None:
            app.state.memory = get_memory(cfg())
            # Route memory's retrieval audits into the same audit log as the
            # request telemetry, so `agent-gateway stats` sees one stream.
            if app.state.memory.audit_hook is None:
                app.state.memory.audit_hook = audit
        return app.state.memory

    def upstream_client() -> httpx.AsyncClient:
        timeout = httpx.Timeout(
            cfg().upstream_timeout, connect=cfg().upstream_connect_timeout
        )
        if app.state.upstream_transport is not None:
            return httpx.AsyncClient(timeout=timeout, transport=app.state.upstream_transport)
        return httpx.AsyncClient(timeout=timeout)

    # --- auth --------------------------------------------------------------
    def authorized(request: Request) -> bool:
        expected = cfg().gateway_api_key
        if not expected:
            return True
        import hmac

        presented = ""
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            presented = header[7:].strip()
        if not presented:
            # Claude Code authenticates with x-api-key.
            presented = request.headers.get("x-api-key", "").strip()
        return hmac.compare_digest(presented, expected)

    def openai_error(message: str, status: int, kind: str = "invalid_request_error"):
        return JSONResponse(
            {"error": {"message": message, "type": kind, "code": None}},
            status_code=status,
        )

    # --- telemetry ---------------------------------------------------------
    def audit(entry: Dict[str, Any]) -> None:
        try:
            path = cfg().audit_log_path
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {"ts": int(time.time()), "version": __version__, **entry}
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def telemetry(
        route: str, tool_action: str, elapsed_ms: float, spill: int, req_id: str, extra: int = 0,
        tools: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, str]:
        headers = {
            "X-Proxy-Route": route,
            "X-Proxy-Tool-Action": tool_action,
            "X-Proxy-Latency-MS": f"{elapsed_ms:.2f}",
            "X-Proxy-Spill-Count": str(spill),
            "X-Proxy-Request-Id": req_id,
            "X-Proxy-Intercepted": str(extra),
        }
        if tools is not None:
            before, after = tools
            headers["X-Proxy-Tools-Before"] = str(before)
            headers["X-Proxy-Tools-After"] = str(after)
        return headers

    def upstream_request(
        client: httpx.AsyncClient, body: Dict[str, Any], client_auth: str = ""
    ) -> httpx.Request:
        headers = {"Content-Type": "application/json"}
        auth = upstream_auth_header(cfg().upstream_api_key, client_auth)
        if auth:
            headers["Authorization"] = auth
        return client.build_request(
            "POST", f"{cfg().upstream_base_url}/chat/completions", json=body, headers=headers
        )

    async def stream_and_tap(
        upstream_response: httpx.Response,
        client: httpx.AsyncClient,
        iterator: AsyncIterator[bytes],
        bg,
        session_id: str,
        prompt: str,
        prefetched: Optional[List[bytes]] = None,
    ) -> AsyncIterator[bytes]:
        """Forward bytes untouched while silently accumulating text.

        `iterator` is the single live byte stream for `upstream_response`; when a
        sniff has already run it is the partially-consumed iterator from that
        sniff, never a fresh one.
        """
        tap = SSETap()
        try:
            for chunk in prefetched or []:
                yield chunk
                tap.feed(chunk)
            async for chunk in iterator:
                yield chunk
                tap.feed(chunk)
        finally:
            await _close_quietly(upstream_response, client)

        if tap.text:
            bg.add_task(graph().ingest_async, session_id, prompt, tap.text, klass())

    # ----------------------------------------------------------------------
    # Routes
    # ----------------------------------------------------------------------
    @app.get("/health")
    async def health():
        payload = {"status": "healthy", "version": __version__, **cfg().describe()}
        try:
            payload["artifacts"] = artifacts().stats()
        except Exception:
            payload["artifacts"] = None
        try:
            stats = graph().stats()
            payload["memory"] = {
                "relations": stats["relations"],
                "entities": stats["entities"],
                "journal_mode": stats["journal_mode"],
                "by_status": stats["by_status"],
            }
        except Exception:
            payload["memory"] = None
        if not os.path.isdir(cfg().shm_cache_dir):
            payload["warning"] = "shared-memory cache directory is not writable"
        payload["timestamp"] = time.time()
        return payload

    @app.get("/v1/models")
    async def list_models(request: Request):
        client = upstream_client()
        try:
            headers = {}
            auth = upstream_auth_header(
                cfg().upstream_api_key, client_upstream_auth(request, cfg().gateway_api_key)
            )
            if auth:
                headers["Authorization"] = auth
            response = await client.get(f"{cfg().upstream_base_url}/models", headers=headers)
            return JSONResponse(response.json(), status_code=response.status_code)
        except Exception as exc:
            return openai_error(f"Upstream /models unavailable: {exc}", 502, "upstream_error")
        finally:
            await _close_quietly(None, client)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, bg: BackgroundTasks):
        started = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"

        if not authorized(request):
            return openai_error("Invalid gateway API key.", 401, "authentication_error")

        try:
            body = await request.json()
        except Exception:
            return openai_error("Request body is not valid JSON.", 400)
        if not isinstance(body, dict):
            return openai_error("Request body must be a JSON object.", 400)

        bypass = request.headers.get("x-agent-gateway-bypass", "").lower() == "true"
        # The client's own credential, for forwarding a subscription session
        # upstream when we have no key of our own to inject.
        client_auth = client_upstream_auth(request, cfg().gateway_api_key)

        session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("x-agent-session-id")
            or "default"
        )
        messages = normalize_messages(body)
        prompt = last_user_text(messages)
        model = str(body.get("model", "") or "")

        route, tool_action = "Bypass", "Bypassed"
        spill_count = 0
        intercepted = 0
        tools_before = 0
        spilled_chars = 0

        if bypass:
            client = upstream_client()
            try:
                upstream = await client.send(
                    upstream_request(client, body, client_auth), stream=True
                )
            except Exception as exc:
                await _close_quietly(None, client)
                return openai_error(f"Upstream unreachable: {exc}", 502, "upstream_error")
            elapsed = (time.perf_counter() - started) * 1000
            return StreamingResponse(
                stream_and_tap(upstream, client, upstream.aiter_raw(), bg, "bypass", "", None),
                status_code=upstream.status_code,
                headers=telemetry(route, tool_action, elapsed, 0, req_id),
            )

        # 1. Answer outstanding synthetic tool calls locally.
        try:
            intercepted = artifacts().intercept_virtual_tool_results(messages)
        except Exception:
            intercepted = 0

        # 2. Ensure the synthetic retrieval tool is offered.
        tools = body.get("tools")
        has_tools = isinstance(tools, list) and bool(tools)
        if has_tools:
            names = [
                ((tool or {}).get("function") or {}).get("name")
                for tool in tools
                if isinstance(tool, dict)
            ]
            if "fetch_log" not in names:
                tools.append(FETCH_LOG_TOOL)
                body["tools"] = tools

        # What the upstream is about to be charged for if the gateway does
        # nothing: the caller's schema plus the synthetic retrieval tool.
        tools_before = len(tools) if has_tools else 0

        def tools_sent() -> int:
            """Schema size in the body as it will actually be dispatched."""
            current = body.get("tools")
            return len(current) if isinstance(current, list) else 0

        # 3. Route.
        last_role = messages[-1].get("role") if messages else ""
        is_tool_turn = last_role in ("tool", "function", "tool_call")
        decision = await decide_route(cfg(), klass(), model, prompt, has_tools, is_tool_turn)
        route, tool_action = decision.route, decision.tool_action

        if has_tools and not is_tool_turn:
            try:
                spill_count = artifacts().truncate_tool_outputs(messages, session_id)
            except Exception:
                spill_count = 0
            # Characters of tool output the upstream will not see -- what the
            # analytics engine books as the spillover saving.
            try:
                spilled_chars = artifacts().last_truncated_chars
            except Exception:
                spilled_chars = 0

        created_escape_message = False
        # Selective sub-tool pruning: with a large schema, keep only the tools
        # this prompt implies instead of choosing between everything and nothing.
        # Must precede the all-or-nothing strip so a stripped turn cannot rank.
        # Note the absence of `not is_tool_turn`: a mid-loop turn must not
        # silently expand back to the full catalog just because the model is
        # waiting on a result. The cap is unconditional: on a tool turn it
        # hard-caps strictly to <= 5 tools (active tool + CORE_TOOLS), ensuring a
        # 49-tool catalog can never leak through a multi-turn tool cycle.
        selective_count = 0
        if (
            has_tools
            and not decision.stripped
            and body.get("tool_choice") in (None, "auto")
        ):
            if is_tool_turn:
                pending: List[str] = []
                for name, _args in iter_tool_call_refs(messages).values():
                    if name and name not in pending:
                        pending.append(name)
                tools, selective_count = prune_for_tool_loop(
                    tools, pending, limit=min(5, cfg().selective_tool_limit or 5)
                )
            else:
                tools, selective_count = apply_selective_pruning(cfg(), prompt, tools)
            if selective_count:
                body["tools"] = tools
                has_tools = bool(tools)

        created_escape_message = False
        if decision.stripped:
            body.pop("tools", None)
            body.pop("tool_choice", None)
            created_escape_message = inject_escape_instruction(messages)

        # 4. Inject recalled facts.
        if (not is_tool_turn or cfg().memory_inject_tool_turns) and prompt and cfg().memory_injection:
            try:
                block = graph().retrieve(prompt)
                if block:
                    target = next(
                        (m for m in messages if m.get("role") == "system"),
                        None,
                    )
                    if target is not None:
                        target["content"] = text_of(target.get("content")) + "\n" + block
                    else:
                        messages.insert(0, {"role": "system", "content": block})
            except Exception:
                pass

        # 5. Dispatch.
        client = upstream_client()
        try:
            upstream = await client.send(
                upstream_request(client, body, client_auth), stream=True
            )
        except Exception as exc:
            await _close_quietly(None, client)
            elapsed = (time.perf_counter() - started) * 1000
            audit({
                "req_id": req_id, "surface": "openai", "route": route,
                "tool_action": tool_action, "spill_count": spill_count,
                "tools_before": tools_before, "tools_after": tools_sent(),
                "latency_ms": round(elapsed, 2), "error": str(exc),
            })
            return JSONResponse(
                {"error": {"message": f"Upstream unreachable: {exc}", "type": "upstream_error"}},
                status_code=502,
                headers=telemetry(route, tool_action, elapsed, spill_count, req_id, intercepted, (tools_before, tools_sent())),
            )

        if upstream.status_code >= 400:
            route = f"{route}->Upstream{upstream.status_code}"
            elapsed = (time.perf_counter() - started) * 1000
            audit({
                "req_id": req_id, "surface": "openai", "route": route,
                "tool_action": tool_action, "spill_count": spill_count,
                "tools_before": tools_before, "tools_after": tools_sent(),
                "latency_ms": round(elapsed, 2), "upstream_status": upstream.status_code,
            })
            return StreamingResponse(
                stream_and_tap(
                    upstream, client, upstream.aiter_raw(), bg, session_id, prompt, None
                ),
                status_code=upstream.status_code,
                headers=telemetry(route, tool_action, elapsed, spill_count, req_id, intercepted, (tools_before, tools_sent())),
            )

        # 6. Bounded escape look-ahead on the pruned route.
        # The stream is created exactly once here: httpx allows a single pass, so
        # the sniff must leave the iterator positioned for the relay below.
        iterator: AsyncIterator[bytes] = upstream.aiter_raw()
        prefetched: Optional[List[bytes]] = None
        if decision.stripped:
            try:
                prefetched, escaped = await sniff_for_escape(
                    iterator,
                    cfg().sniff_limit_bytes,
                    cfg().sniff_overflow_bytes,
                    cfg().escape_scan_chars,
                )
            except Exception:
                prefetched, escaped = [], False

            if escaped:
                await _close_quietly(upstream, None)
                route = f"{route}->EarlyEscapeAbort"
                tool_action = "Reverted-To-Full"

                body["tools"] = tools
                body["tool_choice"] = "auto"
                for message in messages:
                    if message.get("role") == "system":
                        message["content"] = strip_escape_instruction(
                            text_of(message.get("content"))
                        )
                if (
                    created_escape_message
                    and messages
                    and messages[0].get("role") == "system"
                    and not text_of(messages[0].get("content")).strip()
                ):
                    messages.pop(0)

                try:
                    upstream = await client.send(
                        upstream_request(client, body, client_auth), stream=True
                    )
                except Exception as exc:
                    await _close_quietly(None, client)
                    elapsed = (time.perf_counter() - started) * 1000
                    audit({
                        "req_id": req_id, "surface": "openai", "route": route,
                        "tool_action": tool_action, "latency_ms": round(elapsed, 2),
                        "error": str(exc),
                    })
                    return JSONResponse(
                        {"error": {"message": f"Upstream replay failed: {exc}", "type": "upstream_error"}},
                        status_code=502,
                        headers=telemetry(route, tool_action, elapsed, spill_count, req_id, intercepted),
                    )
                iterator = upstream.aiter_raw()
                prefetched = None

        elapsed = (time.perf_counter() - started) * 1000
        audit({
            "req_id": req_id, "surface": "openai", "session_id": session_id, "model": model,
            "route": route, "tool_action": tool_action, "reason": decision.reason,
            "spill_count": spill_count, "intercepted": intercepted,
            "spilled_chars": spilled_chars,
            "tools_before": tools_before, "tools_after": tools_sent(),
            "selective_dropped": selective_count,
            "latency_ms": round(elapsed, 2),
        })
        return StreamingResponse(
            stream_and_tap(
                upstream, client, iterator, bg, session_id, prompt, prefetched
            ),
            status_code=upstream.status_code,
            headers=telemetry(
                route, tool_action, elapsed, spill_count, req_id, intercepted,
                (tools_before, tools_sent()),
            ),
        )

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        """Anthropic token probe. Approximate, and intentionally inexpensive."""
        if not authorized(request):
            return JSONResponse(
                {"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key."}},
                status_code=401,
            )
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}

        try:
            translated = anthropic_to_openai_request(body)
            blob = json.dumps(translated.get("messages", []), ensure_ascii=False)
            blob += json.dumps(translated.get("tools", []), ensure_ascii=False)
            tokens = estimate_tokens(blob)
        except Exception:
            tokens = 0
        return JSONResponse({"input_tokens": tokens})

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request, bg: BackgroundTasks):
        started = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"

        if not authorized(request):
            return JSONResponse(
                {"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key."}},
                status_code=401,
            )

        try:
            body = await request.json()
        except Exception:
            body = None
        if not isinstance(body, dict):
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_request_error", "message": "Body must be a JSON object."}},
                status_code=400,
            )

        streaming = bool(body.get("stream"))
        session_id = (
            request.headers.get("x-session-id")
            or request.headers.get("x-agent-session-id")
            or "default"
        )

        try:
            translated = anthropic_to_openai_request(body)
        except Exception as exc:
            return JSONResponse(
                {"type": "error", "error": {"type": "invalid_request_error", "message": f"Could not translate request: {exc}"}},
                status_code=400,
            )

        # The translation layer does not carry `stream` across, so propagate it
        # explicitly: without this the upstream returns a single JSON body and the
        # client receives a stream with no content deltas in it.
        translated["stream"] = streaming

        # Claude Code asks for `claude-*` ids; a plain OpenAI upstream would 404.
        if cfg().anthropic_model_override:
            translated["model"] = cfg().anthropic_model_override

        model = str(translated.get("model", "") or "")
        forced_tools = translated.get("tool_choice") in ("required",) or isinstance(
            translated.get("tool_choice"), dict
        )
        norm_messages = normalize_messages(translated)
        last_role = norm_messages[-1].get("role") if norm_messages else ""
        is_tool_turn = last_role in ("tool", "function", "tool_call")
        prompt = last_user_text(norm_messages)
        tools = translated.get("tools")
        has_tools = isinstance(tools, list) and bool(tools)

        decision = await decide_route(
            cfg(), klass(), model, prompt, has_tools, is_tool_turn=is_tool_turn
        )
        # Never prune when the caller demanded a specific tool.
        if forced_tools and decision.stripped:
            decision.route = f"{decision.route}->ForcedToolChoice"
            decision.tool_action = "Retained-Forced"

        route, tool_action = decision.route, decision.tool_action

        # Same absolute cap on the Anthropic surface: a large schema is never
        # forwarded whole. Skipped for an explicit tool choice and for an
        # all-or-nothing strip, which removes the schema entirely.
        if has_tools and not forced_tools and not decision.stripped:
            if is_tool_turn:
                pending = []
                for name, _args in iter_tool_call_refs(norm_messages).values():
                    if name and name not in pending:
                        pending.append(name)
                pruned, dropped = prune_for_tool_loop(
                    tools, pending, limit=min(5, cfg().selective_tool_limit or 5)
                )
            else:
                pruned, dropped = apply_selective_pruning(cfg(), prompt, tools)
            if dropped:
                translated["tools"] = pruned
                tools = pruned
                has_tools = bool(pruned)

        if decision.stripped:
            translated.pop("tools", None)
            translated.pop("tool_choice", None)
            inject_escape_instruction(normalize_messages(translated))

        client = upstream_client()
        client_auth = client_upstream_auth(request, cfg().gateway_api_key)
        try:
            upstream = await client.send(
                upstream_request(client, translated, client_auth), stream=streaming
            )
        except Exception as exc:
            await _close_quietly(None, client)
            elapsed = (time.perf_counter() - started) * 1000
            audit({
                "req_id": req_id, "surface": "anthropic", "route": route,
                "tool_action": tool_action, "latency_ms": round(elapsed, 2), "error": str(exc),
            })
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": f"Upstream unreachable: {exc}"}},
                status_code=502,
            )

        elapsed = (time.perf_counter() - started) * 1000
        audit({
            "req_id": req_id, "surface": "anthropic", "session_id": session_id, "model": model,
            "route": route, "tool_action": tool_action, "reason": decision.reason,
            "streaming": streaming, "latency_ms": round(elapsed, 2),
        })

        if upstream.status_code >= 400:
            raw = await upstream.aread()
            await _close_quietly(upstream, client)
            return JSONResponse(
                {"type": "error", "error": {"type": "api_error", "message": raw.decode("utf-8", "replace")}},
                status_code=upstream.status_code,
            )

        if not streaming:
            payload = await upstream.aread()
            await _close_quietly(upstream, client)
            headers = telemetry(route, tool_action, elapsed, 0, req_id)
            try:
                parsed = json.loads(payload)
            except Exception:
                fallback = openai_to_anthropic_response(
                    {},
                    model,
                    cfg().anthropic_thinking_passthrough,
                )
                fallback["content"] = [
                    {"type": "text", "text": payload.decode("utf-8", "replace")}
                ]
                return JSONResponse(fallback, headers=headers)
            return JSONResponse(
                openai_to_anthropic_response(
                    parsed, model, cfg().anthropic_thinking_passthrough
                ),
                headers=headers,
            )

        input_tokens = estimate_tokens(json.dumps(translated.get("messages", [])))

        async def anthropic_body() -> AsyncIterator[bytes]:
            tap = SSETap()

            def observe(chunk: bytes) -> None:
                tap.feed(chunk)

            try:
                async for frame in translate_stream(
                    upstream.aiter_raw(),
                    model,
                    input_tokens,
                    observe,
                    cfg().anthropic_thinking_passthrough,
                ):
                    yield frame
            finally:
                await _close_quietly(upstream, client)
            if tap.text:
                bg.add_task(graph().ingest_async, session_id, prompt, tap.text, klass())

        return StreamingResponse(
            anthropic_body(),
            media_type="text/event-stream",
            headers={
                **telemetry(route, tool_action, elapsed, 0, req_id),
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ----------------------------------------------------------------------
    # Embedded web dashboard (GET /ui + its JSON endpoints)
    # ----------------------------------------------------------------------
    @app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard():
        return dashboard_response()

    @app.get("/ui/api/stats", include_in_schema=False)
    async def dashboard_stats():
        from .analytics import Analytics

        return JSONResponse(
            stats_payload(Analytics(cfg().audit_log_path), connection=cfg().connection)
        )

    @app.get("/ui/api/profile", include_in_schema=False)
    async def dashboard_profile():
        from .config import available_profiles

        return JSONResponse(
            profile_payload(
                available_profiles(),
                os.environ.get("AGENT_GATEWAY_PROFILE") or ".env",
            )
        )

    @app.post("/ui/api/profile", include_in_schema=False)
    async def dashboard_switch_profile(request: Request):
        from .config import available_profiles, find_profile

        try:
            body = await request.json()
        except Exception:
            body = None
        name = str((body or {}).get("profile") or "").strip()
        if not name:
            return JSONResponse({"error": "profile name required"}, status_code=400)
        if find_profile(name) is None:
            return JSONResponse(
                {"error": f"profile {name!r} does not exist"}, status_code=404
            )

        # Exported for the next process, and applied live via the app-state
        # singletons: they rebuild from the new settings on the next request.
        os.environ["AGENT_GATEWAY_PROFILE"] = name
        reloaded = load_settings(env_file=find_profile(name))
        app.state.settings = reloaded
        app.state.classifier = None
        app.state.memory = None
        app.state.store = None
        return JSONResponse(
            {
                "message": (
                    f"switched to {name}: upstream {reloaded.upstream_base_url}, "
                    f"classifier {reloaded.effective_classifier_mode}. "
                    f"Note: the listen port ({reloaded.port}) applies on restart."
                )
            }
        )

    @app.get("/ui/api/memory", include_in_schema=False)
    async def dashboard_memory():
        return JSONResponse({"relations": memory_relations(graph())})

    @app.delete("/ui/api/memory", include_in_schema=False)
    async def dashboard_forget(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = None
        relation_id = (body or {}).get("id")
        deleted = forget_relation(graph(), relation_id)
        if not deleted:
            return JSONResponse({"error": "no such relation"}, status_code=404)
        return JSONResponse({"deleted": relation_id})

    return app


def assert_no_loop(settings: Settings) -> None:
    """Refuse to start if the upstream points back into this gateway."""
    if settings.is_loop_upstream():
        sys.stderr.write(
            f"[FATAL] Upstream {settings.upstream_base_url!r} resolves to a local "
            f"gateway port. Refusing to start on {settings.label}: this would "
            f"forward requests to itself in an infinite loop.\n"
        )
        raise SystemExit(2)


app = create_app()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-context-gateway",
        description=(
            "Agent-agnostic OpenAI/Anthropic context gateway. With no arguments it "
            "reads .env; with --profile it reads .env.<name>."
        ),
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AGENT_GATEWAY_PROFILE", ""),
        metavar="NAME",
        help=(
            "load the .env.NAME profile (e.g. --profile antigravity). "
            "Falls back to .env when the flag is omitted."
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="print the discoverable profile names and exit",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_profiles:
        names = available_profiles()
        if not names:
            sys.stderr.write(
                "[agent-context-gateway] no profiles found. Add a .env.<name> "
                "file, or copy one of the shipped .env.* templates.\n"
            )
            return 1
        for name in names:
            print(name)
        return 0

    env_file = None
    if args.profile:
        env_file = find_profile(args.profile)
        if env_file is None:
            available = ", ".join(available_profiles()) or "none"
            sys.stderr.write(
                f"[FATAL] unknown profile {args.profile!r}. "
                f"Looked for .env.{args.profile} in the working directory, the "
                f"repository and ~/.agent-gateway. Available: {available}\n"
            )
            return 2
        # Also exported, so a gateway respawned by the sentinel -- or started as
        # `uvicorn src.gateway:app` -- resolves the same profile.
        os.environ["AGENT_GATEWAY_PROFILE"] = args.profile

    settings = load_settings(env_file=env_file)
    assert_no_loop(settings)
    settings.ensure_dirs()

    if settings.effective_classifier_mode == "local_jev":
        from .jev_lifecycle import ensure_local_jev_running

        ensure_local_jev_running(settings)

    # Built from the resolved settings, not the import-time module global, so
    # `--profile` actually takes effect.
    app = create_app(settings)

    import uvicorn

    sys.stderr.write(
        f"[agent-context-gateway] listening on http://{settings.label} -> {settings.upstream_base_url}\n"
    )
    if settings.env_file:
        sys.stderr.write(f"[agent-context-gateway] profile file: {settings.env_file}\n")
    if not settings.upstream_api_key:
        sys.stderr.write(
            "[agent-context-gateway] warning: no upstream API key set "
            "(UPSTREAM_API_KEY / OPENAI_API_KEY).\n"
        )
    if not settings.classifier_enabled:
        sys.stderr.write(
            f"[agent-context-gateway] notice: classifier mode "
            f"'{settings.effective_classifier_mode}' issues no network calls; "
            f"routing uses local heuristics and fails open.\n"
        )
    else:
        sys.stderr.write(
            f"[agent-context-gateway] classifier mode "
            f"'{settings.effective_classifier_mode}' -> "
            f"{settings.classifier_api_url} ({settings.classifier_model})\n"
        )
    # Pre-flight bind check. Uvicorn catches the bind OSError itself, logs a
    # terse line and exits with code 3 -- the user never gets a traceback, but
    # they also never get told what to do about it. Probing the port here lets
    # us say exactly that, with the fix.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((settings.host, settings.port))
    except OSError:
        sys.stderr.write(
            f"\n[ERROR] Port {settings.port} is in use by another application.\n"
            f"  Start elsewhere:  python -m src.gateway --port {settings.port + 1}\n"
            f"  See who owns it:  ss -ltnp | grep :{settings.port}\n"
            f"  Full diagnosis:   agent-gateway doctor\n"
        )
        return 1
    finally:
        probe.close()

    import signal
    import threading

    try:
        signal.signal(signal.SIGHUP, lambda s, f: os.kill(os.getpid(), signal.SIGTERM))
    except Exception:
        pass

    owner_pid_str = os.environ.get("AGENT_GATEWAY_OWNER_PID")
    if owner_pid_str and owner_pid_str.isdigit():
        owner_pid = int(owner_pid_str)

        def _watchdog_loop() -> None:
            while True:
                time.sleep(0.3)
                try:
                    os.kill(owner_pid, 0)
                except OSError:
                    # Owner process terminated (e.g. terminal window closed)
                    try:
                        if settings.effective_classifier_mode == "local_jev":
                            from .jev_lifecycle import stop_local_jev

                            stop_local_jev(settings)
                    finally:
                        os._exit(0)

        t = threading.Thread(target=_watchdog_loop, name="owner-watchdog", daemon=True)
        t.start()

    try:
        uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    finally:
        if settings.effective_classifier_mode == "local_jev":
            from .jev_lifecycle import stop_local_jev

            stop_local_jev(settings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
