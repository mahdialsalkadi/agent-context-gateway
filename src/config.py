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
DEFAULT_CLASSIFIER_MODEL = "gpt-4o-mini"

# ------------------------------------------------------------------------------
# Classifier routing modes
# ------------------------------------------------------------------------------
# `CLASSIFIER_MODE` selects where a routing verdict comes from. All four modes
# are first-class and none is required: the local heuristics always run first and
# an unreachable classifier always fails open.
CLASSIFIER_MODE_HEURISTICS = "heuristics"
CLASSIFIER_MODE_UPSTREAM_REUSED = "upstream_reused"
CLASSIFIER_MODE_LOCAL_OLLAMA = "local_ollama"
CLASSIFIER_MODE_EXTERNAL_JEV = "external_jev"
CLASSIFIER_MODES = (
    CLASSIFIER_MODE_HEURISTICS,
    CLASSIFIER_MODE_UPSTREAM_REUSED,
    CLASSIFIER_MODE_LOCAL_OLLAMA,
    CLASSIFIER_MODE_EXTERNAL_JEV,
)
# Unset (or `auto`) preserves the original behaviour exactly: an explicitly
# configured endpoint and key mean the external classifier, otherwise heuristics.
CLASSIFIER_MODE_AUTO = "auto"

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
DEFAULT_OLLAMA_CLASSIFIER_MODEL = "qwen2.5:0.5b"

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


PROFILE_PREFIX = ".env."


def profile_candidates(name: str) -> List[Path]:
    """Places a named profile (`<PROFILE_PREFIX><name>`) is looked for, in order."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "", name or "").lstrip(".")
    if not safe:
        return []
    return [
        Path.cwd() / f"{PROFILE_PREFIX}{safe}",
        _repo_root() / f"{PROFILE_PREFIX}{safe}",
        Path(f"~/.agent-gateway/{PROFILE_PREFIX}{safe}").expanduser(),
    ]


def find_profile(name: str) -> Optional[Path]:
    """Locate a profile file, or None when it does not exist anywhere."""
    for candidate in profile_candidates(name):
        if candidate.is_file():
            return candidate
    return None


def available_profiles() -> List[str]:
    """Names of discoverable profiles, for `--list-profiles`."""
    names = set()
    for root in (Path.cwd(), _repo_root(), Path("~/.agent-gateway").expanduser()):
        try:
            found = list(root.glob(f"{PROFILE_PREFIX}*"))
        except OSError:
            continue
        for path in found:
            name = path.name[len(PROFILE_PREFIX):]
            if not name or name == "example":
                continue
            if path.is_file():
                names.add(name)
    return sorted(names)


def env_file_candidates() -> List[Path]:
    """Places a `.env` is looked for, in order.

    An explicitly requested profile (`--profile x`, or `AGENT_GATEWAY_PROFILE=x`)
    outranks the plain `.env`: naming a profile is a deliberate act, so it should
    not be quietly overridden by a leftover default file.
    """
    candidates: List[Path] = []
    explicit = os.environ.get("AGENT_GATEWAY_ENV")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    profile = os.environ.get("AGENT_GATEWAY_PROFILE")
    if profile:
        candidates.extend(profile_candidates(profile))
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
    # `auto` derives the mode from what is configured, so an existing deployment
    # keeps working without setting CLASSIFIER_MODE at all.
    classifier_mode: str = CLASSIFIER_MODE_AUTO

    # Allow UPSTREAM_BASE_URL to target a port in LEGACY_PROXY_PORTS. Required for
    # local subscription bridges that live on 8080 (Antigravity, Copilot).
    allow_legacy_upstream: bool = False

    # Forward upstream reasoning text as Anthropic `thinking` blocks. Off by
    # default: unverified signatures can make strict Anthropic clients reject the
    # stream, and dropping reasoning is the safe, long-standing behaviour.
    anthropic_thinking_passthrough: bool = False

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
    def effective_classifier_mode(self) -> str:
        """The mode actually in force, resolving `auto` from the environment.

        An unrecognised value resolves like `auto` rather than crashing, so a
        typo degrades to a working default; `/health` reports the effective mode,
        which is where the typo becomes visible.
        """
        mode = (self.classifier_mode or "").strip().lower()
        if mode in CLASSIFIER_MODES:
            return mode
        return (
            CLASSIFIER_MODE_EXTERNAL_JEV
            if (self.classifier_api_url and self.classifier_api_key)
            else CLASSIFIER_MODE_HEURISTICS
        )

    @property
    def classifier_enabled(self) -> bool:
        """True when ambiguous turns should reach a model at all."""
        mode = self.effective_classifier_mode
        if mode == CLASSIFIER_MODE_HEURISTICS or not self.classifier_api_url:
            return False
        if mode in (CLASSIFIER_MODE_UPSTREAM_REUSED, CLASSIFIER_MODE_LOCAL_OLLAMA):
            # Local bridges and subscription gateways commonly ignore the bearer
            # token, so a missing key must not silently disable classification.
            return True
        return bool(self.classifier_api_key)

    # --- safety ------------------------------------------------------------
    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.log_dir, self.shm_cache_dir):
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

    def forbidden_upstreams(self) -> set:
        """(host, port) pairs the upstream may never resolve to.

        The gateway's own listen address is always forbidden -- that is the
        infinite loop. The legacy proxy ports are forbidden too, but only as a
        heuristic guess that something proxy-shaped lives there. Local
        subscription bridges genuinely occupy 8080, so `ALLOW_LEGACY_UPSTREAM_PORT`
        relaxes the guess without relaxing the real loop guard.
        """
        hosts = {self.host.lower(), "127.0.0.1", "localhost", "0.0.0.0", "::1"}
        ports = {self.port}
        if not self.allow_legacy_upstream:
            ports.update(LEGACY_PROXY_PORTS)
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
            "classifier_mode": self.effective_classifier_mode,
            "classifier_url": self.classifier_api_url or None,
            "classifier_model": self.classifier_model if self.classifier_enabled else None,
            "classifier_protocol": self.classifier_protocol if self.classifier_enabled else None,
            "anthropic_model_override": self.anthropic_model_override or None,
            "data_dir": str(self.data_dir),
            "log_dir": str(self.log_dir),
            "shm_cache_dir": str(self.shm_cache_dir),
            "memory_injection": self.memory_injection,
            "anthropic_thinking_passthrough": self.anthropic_thinking_passthrough,
            "allow_legacy_upstream": self.allow_legacy_upstream,
            "profile": os.environ.get("AGENT_GATEWAY_PROFILE") or None,
            "env_file": str(self.env_file) if self.env_file else None,
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

    # --- classifier mode ---------------------------------------------------
    # Each mode decides where the endpoint, credential and model come from. The
    # legacy variables keep working untouched, so `auto` is a pure no-op.
    raw_mode = _first(env, ("CLASSIFIER_MODE",), "").strip().lower()
    resolved_mode = raw_mode if raw_mode in CLASSIFIER_MODES else CLASSIFIER_MODE_AUTO
    classifier_model = _first(
        env, ("CLASSIFIER_MODEL", "JEV_MODEL"), DEFAULT_CLASSIFIER_MODEL
    )
    classifier_protocol = _first(
        env, ("CLASSIFIER_PROTOCOL", "JEV_PROTOCOL"), DEFAULT_CLASSIFIER_PROTOCOL
    ).lower()

    if resolved_mode == CLASSIFIER_MODE_HEURISTICS:
        # Zero network, unconditionally. Blank the endpoint rather than merely
        # ignoring it, so no code path -- and no future refactor -- can reach out.
        classifier_url = ""
        classifier_key = ""
    elif resolved_mode == CLASSIFIER_MODE_UPSTREAM_REUSED:
        # Reuse the subscription already being paid for, on the host already
        # being talked to: same credential, same quota, no second bill.
        classifier_url = classifier_url or f"{upstream_base}/chat/completions"
        classifier_key = classifier_key or upstream_key
        classifier_protocol = DEFAULT_CLASSIFIER_PROTOCOL
    elif resolved_mode == CLASSIFIER_MODE_LOCAL_OLLAMA:
        ollama_base = sanitize_url(
            _first(env, ("OLLAMA_BASE_URL",), DEFAULT_OLLAMA_BASE_URL)
        ).rstrip("/")
        classifier_url = classifier_url or f"{ollama_base}/chat/completions"
        classifier_key = classifier_key or "ollama"
        classifier_model = (
            _first(env, ("CLASSIFIER_MODEL", "JEV_MODEL"), "")
            or DEFAULT_OLLAMA_CLASSIFIER_MODEL
        )
        classifier_protocol = DEFAULT_CLASSIFIER_PROTOCOL
    elif resolved_mode == CLASSIFIER_MODE_EXTERNAL_JEV:
        # Unchanged: a dedicated endpoint, and `blueprint` remains available.
        pass

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
        classifier_model=classifier_model,
        classifier_protocol=classifier_protocol,
        classifier_mode=raw_mode or CLASSIFIER_MODE_AUTO,
        allow_legacy_upstream=_as_bool(
            _first(env, ("ALLOW_LEGACY_UPSTREAM_PORT",), ""), False
        ),
        anthropic_thinking_passthrough=_as_bool(
            _first(env, ("ANTHROPIC_THINKING_PASSTHROUGH",), ""), False
        ),
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
