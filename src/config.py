"""
Central configuration.

Every path and credential is resolved from the environment, so nothing in this
project is tied to a particular agent, vendor or directory layout. A `.env` file
is loaded if present, but a real environment variable always wins over it.

Precedence: process environment -> .env file -> defaults.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8090
DEFAULT_DATA_DIR = "~/.agent-gateway/data"
DEFAULT_LOG_DIR = "~/.agent-gateway/logs"
DEFAULT_UPSTREAM = "https://api.openai.com/v1"

DEFAULT_CLASSIFIER_PROTOCOL = "openai"
DEFAULT_CLASSIFIER_TIMEOUT = 2.5
DEFAULT_NEEDS_TOOLS_THRESHOLD = 0.15
DEFAULT_SUPERSEDE_THRESHOLD = 0.75

DEFAULT_TRUNCATE_CHARS = 800
DEFAULT_SNIFF_LIMIT = 512
DEFAULT_SNIFF_OVERFLOW = 192
DEFAULT_ESCAPE_SCAN_CHARS = 64

ESCAPE_TOKEN = "[ESCAPE_NEED_TOOLS]"
ESCAPE_INSTRUCTION = (
    f"\n[SYSTEM INSTRUCTION: If you cannot fulfill this request without external "
    f"tools, start output with '{ESCAPE_TOKEN}']"
)

# Ports a gateway is most likely to be listening on. Used by the loop guard.
LEGACY_PROXY_PORTS = (8080, 8090)


# ------------------------------------------------------------------------------
# .env loading
# ------------------------------------------------------------------------------
def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def env_file_candidates() -> List[Path]:
    """Places a `.env` is looked for, in order."""
    candidates: List[Path] = []
    explicit = os.environ.get("AGENT_GATEWAY_ENV")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(Path.cwd() / ".env")
    candidates.append(_repo_root() / ".env")
    candidates.append(Path("~/.agent-gateway/.env").expanduser())
    return candidates


def load_env_file(path: Optional[Path] = None) -> Optional[Path]:
    """Populate os.environ from a `.env` file, never overriding real vars.

    Deliberately tiny and dependency-free: only simple `KEY=value` lines are
    understood, with optional surrounding quotes and `export` prefixes.
    """
    paths = [path] if path is not None else env_file_candidates()
    for candidate in paths:
        if candidate is None or not candidate.is_file():
            continue
        try:
            with open(candidate, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    if key.startswith("export "):
                        key = key[len("export "):].strip()
                    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                        continue
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
            return candidate
        except OSError:
            continue
    return None


# ------------------------------------------------------------------------------
# Small typed readers
# ------------------------------------------------------------------------------
def _first(env: Mapping[str, str], names: Sequence[str], default: str = "") -> str:
    for name in names:
        value = env.get(name)
        if value:
            return value
    return default


def _as_bool(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.strip().lower() not in ("0", "false", "no", "off")


def _as_float(value: str, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: str, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def sanitize_url(raw: str) -> str:
    """Normalise a URL pasted from documentation.

    Guards the classic copy/paste failure where a URL arrives wrapped in
    markdown link syntax: `[https://x/v1](https://x/v1)`.
    """
    raw = (raw or "").strip().strip("`").strip('"').strip("'")
    match = re.match(r"^\[[^\]]*\]\(\s*([^)]+?)\s*\)$", raw)
    if match:
        raw = match.group(1).strip()
    return raw


def resolve_indirect_key(
    explicit: str, indirect_env_name: str, env: Mapping[str, str]
) -> str:
    """Resolve a credential, optionally by naming another variable that holds it.

    The indirection exists so a key is never duplicated into a second file:
    `CLASSIFIER_API_KEY_ENV=UPSTREAM_API_KEY` reuses the key already present.
    """
    if explicit:
        return explicit
    if indirect_env_name:
        return env.get(indirect_env_name, "") or ""
    return ""


# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Settings:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    data_dir: Path = field(default_factory=lambda: Path(DEFAULT_DATA_DIR).expanduser())
    log_dir: Path = field(default_factory=lambda: Path(DEFAULT_LOG_DIR).expanduser())
    shm_cache_dir: Path = field(default_factory=lambda: Path("/tmp/agent_gateway"))

    upstream_base_url: str = DEFAULT_UPSTREAM
    upstream_api_key: str = ""
    upstream_timeout: float = 180.0
    upstream_connect_timeout: float = 15.0

    gateway_api_key: str = ""

    classifier_api_url: str = ""
    classifier_api_key: str = ""
    classifier_model: str = "gpt-4o-mini"
    classifier_protocol: str = DEFAULT_CLASSIFIER_PROTOCOL
    classifier_timeout: float = DEFAULT_CLASSIFIER_TIMEOUT
    classifier_needs_tools_threshold: float = DEFAULT_NEEDS_TOOLS_THRESHOLD
    classifier_supersede_threshold: float = DEFAULT_SUPERSEDE_THRESHOLD

    # Claude Code asks for `claude-*` model ids. If your upstream does not serve
    # those names, this replaces the model on the Anthropic surface only.
    anthropic_model_override: str = ""

    memory_injection: bool = True
    memory_inject_tool_turns: bool = False

    truncate_threshold_chars: int = DEFAULT_TRUNCATE_CHARS
    sniff_limit_bytes: int = DEFAULT_SNIFF_LIMIT
    sniff_overflow_bytes: int = DEFAULT_SNIFF_OVERFLOW
    escape_scan_chars: int = DEFAULT_ESCAPE_SCAN_CHARS

    extra_reasoning_patterns: tuple = ()
    env_file: Optional[Path] = None

    # --- derived paths -----------------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "graph.db"

    @property
    def audit_log_path(self) -> Path:
        return self.log_dir / "audit.log"

    @property
    def gateway_log_path(self) -> Path:
        return self.log_dir / "gateway.log"

    @property
    def label(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def classifier_enabled(self) -> bool:
        return bool(self.classifier_api_key and self.classifier_api_url)

    # --- safety ------------------------------------------------------------
    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.log_dir, self.shm_cache_dir):
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    def forbidden_upstreams(self) -> set:
        """(host, port) pairs the upstream may never resolve to."""
        hosts = {self.host.lower(), "127.0.0.1", "localhost", "0.0.0.0", "::1"}
        ports = {self.port, *LEGACY_PROXY_PORTS}
        return {(host, port) for host in hosts for port in ports}

    def is_loop_upstream(self, url: Optional[str] = None) -> bool:
        """True when the upstream points back at this gateway (a fatal loop)."""
        from urllib.parse import urlparse

        target = sanitize_url(url if url is not None else self.upstream_base_url)
        try:
            parsed = urlparse(target)
        except ValueError:
            return False
        if parsed.scheme not in ("http", "https"):
            return False
        host = (parsed.hostname or "").lower()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return (host, port) in self.forbidden_upstreams()

    def describe(self) -> Dict[str, object]:
        """Redacted snapshot, safe to log or return from /health."""
        return {
            "listen": self.label,
            "upstream": self.upstream_base_url,
            "upstream_key": bool(self.upstream_api_key),
            "gateway_auth": bool(self.gateway_api_key),
            "classifier_enabled": self.classifier_enabled,
            "classifier_url": self.classifier_api_url or None,
            "classifier_model": self.classifier_model if self.classifier_enabled else None,
            "classifier_protocol": self.classifier_protocol if self.classifier_enabled else None,
            "anthropic_model_override": self.anthropic_model_override or None,
            "data_dir": str(self.data_dir),
            "log_dir": str(self.log_dir),
            "shm_cache_dir": str(self.shm_cache_dir),
            "memory_injection": self.memory_injection,
        }


def pick_shm_dir(explicit: str, env: Mapping[str, str]) -> Path:
    """Prefer a RAM disk, but never fail because one is missing."""
    if explicit:
        return Path(explicit).expanduser()
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        return Path("/dev/shm/agent_gateway")
    return Path("/tmp/agent_gateway")


def load_settings(
    env: Optional[Mapping[str, str]] = None, env_file: Optional[Path] = None
) -> Settings:
    """Build `Settings` from the environment.

    `HERMES_*` / `REAL_UPSTREAM_BASE_URL` / `JEV_*` names are accepted as aliases
    so an existing deployment can migrate without rewriting its config.
    """
    if env is None:
        discovered = load_env_file(env_file)
        env = os.environ
    else:
        discovered = env_file

    upstream_base = sanitize_url(
        _first(
            env,
            ("UPSTREAM_BASE_URL", "REAL_UPSTREAM_BASE_URL", "OPENAI_BASE_URL"),
            DEFAULT_UPSTREAM,
        )
    ).rstrip("/")

    upstream_key = _first(
        env,
        ("UPSTREAM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"),
    )

    classifier_url = sanitize_url(
        _first(env, ("CLASSIFIER_API_URL", "JEV_API_URL"), "")
    )
    classifier_key = resolve_indirect_key(
        _first(env, ("CLASSIFIER_API_KEY", "JEV_API_KEY"), ""),
        _first(env, ("CLASSIFIER_API_KEY_ENV", "JEV_API_KEY_ENV"), ""),
        env,
    )

    extra_regex = _first(env, ("REASONING_MODEL_REGEX",), "")

    return Settings(
        host=_first(env, ("GATEWAY_HOST", "HERMES_PROXY_HOST"), DEFAULT_HOST),
        port=_as_int(_first(env, ("GATEWAY_PORT", "HERMES_PROXY_PORT"), str(DEFAULT_PORT)), DEFAULT_PORT),
        data_dir=Path(_first(env, ("DATA_DIR",), DEFAULT_DATA_DIR)).expanduser(),
        log_dir=Path(_first(env, ("LOG_DIR",), DEFAULT_LOG_DIR)).expanduser(),
        shm_cache_dir=pick_shm_dir(_first(env, ("SHM_CACHE_DIR",), ""), env),
        upstream_base_url=upstream_base,
        upstream_api_key=upstream_key,
        upstream_timeout=_as_float(_first(env, ("UPSTREAM_TIMEOUT_SECONDS",), ""), 180.0),
        upstream_connect_timeout=_as_float(
            _first(env, ("UPSTREAM_CONNECT_TIMEOUT_SECONDS",), ""), 15.0
        ),
        gateway_api_key=_first(env, ("GATEWAY_API_KEY",), ""),
        classifier_api_url=classifier_url,
        classifier_api_key=classifier_key,
        classifier_model=_first(
            env, ("CLASSIFIER_MODEL", "JEV_MODEL"), "gpt-4o-mini"
        ),
        classifier_protocol=_first(
            env, ("CLASSIFIER_PROTOCOL", "JEV_PROTOCOL"), DEFAULT_CLASSIFIER_PROTOCOL
        ).lower(),
        classifier_timeout=_as_float(
            _first(env, ("CLASSIFIER_TIMEOUT_SECONDS", "JEV_TIMEOUT"), ""),
            DEFAULT_CLASSIFIER_TIMEOUT,
        ),
        classifier_needs_tools_threshold=_as_float(
            _first(env, ("CLASSIFIER_NEEDS_TOOLS_THRESHOLD", "JEV_NEEDS_TOOLS_THRESHOLD"), ""),
            DEFAULT_NEEDS_TOOLS_THRESHOLD,
        ),
        classifier_supersede_threshold=_as_float(
            _first(env, ("CLASSIFIER_SUPERSEDE_THRESHOLD", "JEV_SUPERSEDE_THRESHOLD"), ""),
            DEFAULT_SUPERSEDE_THRESHOLD,
        ),
        anthropic_model_override=_first(
            env, ("ANTHROPIC_MODEL_OVERRIDE", "BRIDGE_MODEL_OVERRIDE"), ""
        ),
        memory_injection=_as_bool(_first(env, ("MEMORY_INJECTION", "HERMES_MEMORY_INJECTION"), "1"), True),
        memory_inject_tool_turns=_as_bool(
            _first(env, ("MEMORY_INJECT_TOOL_TURNS", "HERMES_MEMORY_INJECT_TOOL_TURNS"), "0"), False
        ),
        truncate_threshold_chars=_as_int(
            _first(env, ("TRUNCATE_THRESHOLD_CHARS",), ""), DEFAULT_TRUNCATE_CHARS
        ),
        sniff_limit_bytes=_as_int(_first(env, ("SNIFF_LIMIT_BYTES",), ""), DEFAULT_SNIFF_LIMIT),
        sniff_overflow_bytes=_as_int(
            _first(env, ("SNIFF_OVERFLOW_BYTES",), ""), DEFAULT_SNIFF_OVERFLOW
        ),
        escape_scan_chars=_as_int(
            _first(env, ("ESCAPE_SCAN_CHARS",), ""), DEFAULT_ESCAPE_SCAN_CHARS
        ),
        extra_reasoning_patterns=tuple(
            part.strip() for part in extra_regex.split(",") if part.strip()
        ),
        env_file=discovered,
    )
