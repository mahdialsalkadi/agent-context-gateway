"""
Artifact spillover: keep oversized tool output out of the prompt, but reachable.

When a conversation accumulates large tool results, replaying all of them on
every turn burns tokens and pushes the model toward context-window failures. The
store moves the bulk of that text into a shared-memory file and leaves a short
notice plus a stable handle behind. The model can retrieve any slice with the
synthetic `fetch_log` tool, and the gateway transparently answers that call
itself so the client needs no implementation of it.

Handles are derived from the content hash, so the same output always maps to the
same handle -- repeated reads do not multiply storage.
"""

from __future__ import annotations

import glob
import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .messages import iter_tool_call_refs, resolve_tool_result_ref, set_text

ERROR_SIGNATURES = (
    "Traceback (most recent call last):",
    "Error:",
    "Exception:",
    "FAILED",
    "fatal:",
    "error TS",
    "panicked at",
)

FETCH_LOG_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "fetch_log",
        "description": (
            "Read a slice of a large command output or log that was truncated to "
            "save context. Pass the handle id shown in the truncation notice."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Log handle, e.g. LOG_A94F12"},
                "offset": {
                    "type": "integer",
                    "description": "Character offset to start reading from",
                    "default": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of characters to return",
                    "default": 1000,
                },
            },
            "required": ["id"],
        },
    },
}

_HANDLE_RE = re.compile(r"^LOG_[0-9A-F]{6,}$")


def content_key(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()


def handle_for(content: str) -> str:
    """Deterministic, human-copyable handle derived from the content hash."""
    return f"LOG_{content_key(content)[:6].upper()}"


class ArtifactStore:
    """Filesystem-backed store for spilled context."""

    def __init__(self, cache_dir: Path | str, truncate_threshold_chars: int = 800) -> None:
        self.cache_dir = Path(cache_dir)
        self.truncate_threshold_chars = max(1, int(truncate_threshold_chars))
        # content hash -> (handle, path). Rebuilt from disk when cold.
        self._index: Dict[str, Tuple[str, str]] = {}
        # Characters replaced by handles during the most recent
        # `truncate_tool_outputs` call -- what the analytics engine books as
        # spillover savings. Recomputed per call, not accumulated.
        self.last_truncated_chars = 0

    # --- filesystem --------------------------------------------------------
    def ensure_dir(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def path_for_handle(self, handle_id: str) -> Optional[str]:
        safe = re.sub(r"[^A-Za-z0-9_]", "", handle_id or "")
        if not safe or not _HANDLE_RE.match(safe):
            return None

        for cached_handle, path in self._index.values():
            if cached_handle == safe and os.path.exists(path):
                return path

        matches = sorted(glob.glob(str(self.cache_dir / f"*_{safe}.log")))
        return matches[-1] if matches else None

    # --- write -------------------------------------------------------------
    def spill(self, content: str, session_id: str = "session") -> Tuple[str, str]:
        """Persist `content`, returning (handle, path).

        Identical content reuses the existing file: the handle is a content
        hash, so this is naturally idempotent.
        """
        key = content_key(content)
        handle = handle_for(content)

        cached = self._index.get(key)
        if cached and os.path.exists(cached[1]):
            return cached

        existing = self.path_for_handle(handle)
        if existing:
            self._index[key] = (handle, existing)
            return handle, existing

        self.ensure_dir()
        safe_session = re.sub(r"[^A-Za-z0-9_-]", "_", session_id or "session")[:40]
        filename = f"{safe_session}_{int(time.time())}_{handle}.log"
        path = self.cache_dir / filename
        try:
            path.write_text(content, encoding="utf-8")
        except OSError:
            return handle, ""
        self._index[key] = (handle, str(path))
        return handle, str(path)

    # --- read --------------------------------------------------------------
    def fetch(self, handle_id: str, offset: int = 0, limit: int = 1000) -> str:
        """Read a character window of a spilled artifact."""
        path = self.path_for_handle(handle_id)
        if not path:
            return (
                f"[fetch_log] Unknown or expired handle: {handle_id}. "
                f"The artifact may have been evicted by the maintenance daemon."
            )
        try:
            data = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"[fetch_log] Could not read {handle_id}: {exc}"

        offset = max(0, int(offset or 0))
        limit = max(1, min(int(limit or 1000), 50_000))
        window = data[offset:offset + limit]
        return (
            f"[fetch_log {handle_id}] chars {offset}-{offset + len(window)} of {len(data)}\n"
            f"{window}"
        )

    # --- pruning -----------------------------------------------------------
    def prune(
        self, max_bytes: int = 180 * 1024 * 1024, max_age_seconds: float = 7200.0
    ) -> Dict[str, int]:
        """FIFO eviction: oldest first, until under size, plus age expiry."""
        stats = {"files": 0, "removed": 0, "freed_bytes": 0}
        if not self.cache_dir.is_dir():
            return stats

        entries: List[Tuple[float, str, int]] = []
        total = 0
        for path in glob.glob(str(self.cache_dir / "*.log")):
            try:
                stat = os.stat(path)
            except OSError:
                continue
            entries.append((stat.st_mtime, path, stat.st_size))
            total += stat.st_size

        stats["files"] = len(entries)
        entries.sort(key=lambda item: item[0])  # oldest first

        now = time.time()
        for mtime, path, size in entries:
            if now - mtime <= max_age_seconds and total <= max_bytes:
                break
            try:
                os.remove(path)
                total -= size
                stats["removed"] += 1
                stats["freed_bytes"] += size
            except OSError:
                continue

        if stats["removed"]:
            self._index.clear()
        return stats

    def stats(self) -> Dict[str, Any]:
        files = glob.glob(str(self.cache_dir / "*.log"))
        total = 0
        for path in files:
            try:
                total += os.path.getsize(path)
            except OSError:
                continue
        return {"dir": str(self.cache_dir), "files": len(files), "bytes": total}

    # --- prompt compaction -------------------------------------------------
    def truncate_tool_outputs(
        self, messages: List[Dict[str, Any]], session_id: str = "session"
    ) -> int:
        """Replace bulky *historical* tool output with a handle.

        The last two messages are never touched: the model is actively reasoning
        about them, and mangling them mid-loop is how agents lose the plot.
        """
        spilled = 0
        total = len(messages)
        self.last_truncated_chars = 0

        for index, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            if index >= total - 2:
                continue

            raw = message.get("content")
            if not isinstance(raw, str) or len(raw) <= self.truncate_threshold_chars:
                continue

            handle, _ = self.spill(raw, session_id)
            spilled += 1
            self.last_truncated_chars += len(raw) - len(summarize(raw, handle))
            message["content"] = summarize(raw, handle)

        return spilled

    def intercept_virtual_tool_results(self, messages: List[Dict[str, Any]]) -> int:
        """Answer outstanding `fetch_log` calls from the local cache.

        Clients generally have no `fetch_log` implementation, so they will have
        reported an unknown-tool error for it. Substituting the real payload here
        keeps the retrieval loop working with an unmodified agent.
        """
        refs = iter_tool_call_refs(messages)
        if not refs:
            return 0

        intercepted = 0
        for message in messages:
            if message.get("role") != "tool":
                continue
            call_id = resolve_tool_result_ref(message)
            if not call_id or call_id not in refs:
                continue
            name, args = refs[call_id]
            if name != "fetch_log":
                continue
            set_text(
                message,
                self.fetch(
                    str(args.get("id", "")),
                    args.get("offset", 0) or 0,
                    args.get("limit", 1000) or 1000,
                ),
            )
            intercepted += 1
        return intercepted


def summarize(raw: str, handle: str) -> str:
    """Build the replacement notice, biased toward preserving error detail."""
    has_error = any(signature in raw for signature in ERROR_SIGNATURES)

    if has_error:
        error_lines = [
            line
            for line in raw.splitlines()
            if any(signature in line for signature in ERROR_SIGNATURES)
        ]
        head = "\n".join(error_lines[:10])
        return (
            f"[Output truncated | errors detected]\n{head}\n"
            f'[Full output ({len(raw)} chars) preserved as handle {handle}. '
            f'Call \'fetch_log(id="{handle}")\' to inspect the raw buffer.]'
        )

    head = raw[:250]
    tail = raw[-150:]
    return (
        f"{head}\n... [output truncated, {len(raw)} chars] ...\n{tail}\n"
        f'[Preserved as handle {handle}. '
        f'Call \'fetch_log(id="{handle}")\' to view the full text.]'
    )


# ------------------------------------------------------------------------------
# Process-wide store
# ------------------------------------------------------------------------------
_STORE: Optional[ArtifactStore] = None
_STORE_KEY: Optional[tuple] = None


def configure(store: ArtifactStore) -> ArtifactStore:
    global _STORE, _STORE_KEY
    _STORE = store
    _STORE_KEY = (str(store.cache_dir), store.truncate_threshold_chars)
    return store


def get_store(settings: Optional[Any] = None) -> ArtifactStore:
    """Return the process-wide store, rebuilding it if the config changed."""
    global _STORE, _STORE_KEY
    if settings is None:
        from .config import load_settings

        settings = load_settings()
    key = (str(settings.shm_cache_dir), settings.truncate_threshold_chars)
    if _STORE is None or _STORE_KEY != key:
        _STORE = ArtifactStore(settings.shm_cache_dir, settings.truncate_threshold_chars)
        _STORE_KEY = key
    return _STORE
