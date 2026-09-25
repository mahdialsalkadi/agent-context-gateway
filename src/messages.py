"""
Message plumbing shared by the gateway, the artifact store and the memory engine.

OpenAI `content` fields are not always plain strings -- they may be lists of
typed parts (text, images). These helpers read and write them without silently
destroying the non-text parts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def text_of(content: Any) -> str:
    """Flatten a message `content` field into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def set_text(message: Dict[str, Any], text: str) -> None:
    """Write `text` back, preserving multimodal parts when present."""
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = text
                return
        content.append({"type": "text", "text": text})
    else:
        message["content"] = text


def last_user_text(messages: Iterable[Mapping[str, Any]]) -> str:
    """Text of the most recent user turn."""
    for message in reversed(list(messages)):
        if isinstance(message, Mapping) and message.get("role") == "user":
            return text_of(message.get("content"))
    return ""


def parse_arguments(raw: Any) -> Dict[str, Any]:
    """Tool-call arguments arrive as a JSON string; tolerate a bad one."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def iter_tool_call_refs(
    messages: Iterable[Mapping[str, Any]],
) -> Dict[str, Tuple[str, Dict[str, Any]]]:
    """Map `tool_call_id` -> (tool_name, parsed_arguments) across a conversation.

    Handles both the modern `tool_calls` list and the legacy `function_call`
    field, plus Anthropic-style `tool_use` blocks nested in content.
    """
    refs: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    for message in messages:
        if not isinstance(message, Mapping):
            continue

        for call in message.get("tool_calls") or []:
            if not isinstance(call, Mapping):
                continue
            call_id = call.get("id")
            function = call.get("function") or {}
            if call_id and isinstance(function, Mapping):
                refs[str(call_id)] = (
                    str(function.get("name") or ""),
                    parse_arguments(function.get("arguments")),
                )

        legacy = message.get("function_call")
        if isinstance(legacy, Mapping) and legacy.get("name"):
            # Legacy calls have no id; Anthropic/OpenAI tool results reference
            # them by name, so index under both spellings.
            args = parse_arguments(legacy.get("arguments"))
            refs[str(legacy["name"])] = (str(legacy["name"]), args)

        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                    continue
                block_id = block.get("id")
                if block_id:
                    refs[str(block_id)] = (
                        str(block.get("name") or ""),
                        parse_arguments(block.get("input")),
                    )

    return refs


def resolve_tool_result_ref(message: Mapping[str, Any]) -> Optional[str]:
    """Find the call id a tool-result message is answering."""
    for key in ("tool_call_id", "tool_use_id"):
        value = message.get(key)
        if value:
            return str(value)
    return None


def normalize_messages(body: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract a mutable messages list from a request body."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    return [message for message in messages if isinstance(message, dict)]
