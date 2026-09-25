"""
Single entrypoint for everything the gateway can do.

    agent-gateway start   [--profile NAME] [--port N] [--daemon]
    agent-gateway stop
    agent-gateway status
    agent-gateway stats   [--json] [--live]
    agent-gateway test
    agent-gateway service install [--user]

`start`, `stop` and `status` coordinate through a pidfile in LOG_DIR, so they
work no matter how the gateway was started by this CLI (foreground or daemon).
A gateway started some other way -- uvicorn, systemd directly -- is still
reported by `status` via its HTTP health endpoint; `stop` only ever signals a
PID recorded by this CLI, never a PID guessed from a port scan, because the
port may legitimately belong to something else.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from .ux import CliError


# ------------------------------------------------------------------------------
# pidfile helpers
# ------------------------------------------------------------------------------
def _pid_path() -> Path:
    from .config import load_settings

    return load_settings().log_dir / "gateway.pid"


def _read_pid() -> Optional[int]:
    try:
        return int(_pid_path().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


def _write_pid(pid: int) -> None:
    path = _pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid), encoding="utf-8")


def _clear_pid() -> None:
    try:
        _pid_path().unlink()
    except OSError:
        pass


# ------------------------------------------------------------------------------
# health
# ------------------------------------------------------------------------------
def _probe_health(timeout: float = 2.0) -> Optional[dict]:
    """The gateway's /health payload, or None when nothing answers."""
    from .config import load_settings

    settings = load_settings()
    try:
        import httpx

        response = httpx.get(
            f"http://{settings.host}:{settings.port}/health", timeout=timeout
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


# ------------------------------------------------------------------------------
# subcommands
# ------------------------------------------------------------------------------
def cmd_start(args: argparse.Namespace) -> int:
    if _pid_alive(_read_pid()):
        sys.stderr.write(
            f"[cli] a gateway is already running (pid {_read_pid()}). "
            f"Use `agent-gateway stop` first.\n"
        )
        return 1

    # Overrides must be in the environment before settings are resolved.
    if args.port:
        os.environ["GATEWAY_PORT"] = str(args.port)
    if args.profile:
        from .config import find_profile

        if find_profile(args.profile) is None:
            sys.stderr.write(f"[cli] unknown profile {args.profile!r}\n")
            return 2
        os.environ["AGENT_GATEWAY_PROFILE"] = args.profile

    from .config import load_settings

    settings = load_settings()
    settings.ensure_dirs()

    if args.daemon:
        return _start_daemon(args, settings)

    # Foreground: this process *is* the gateway. Ctrl-C stops it cleanly.
    _write_pid(os.getpid())
    try:
        from .gateway import main as gateway_main

        return gateway_main([])
    finally:
        _clear_pid()


def _start_daemon(args: argparse.Namespace, settings) -> int:
    """Detach: spawn the gateway into its own session and return immediately."""
    log_path = settings.gateway_log_path
    handle = open(log_path, "ab")

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        [sys.executable, "-m", "src.gateway"],
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    _write_pid(process.pid)

    # Wait briefly so `start --daemon` can fail loudly on a bad config --
    # a process that died instantly should not report success.
    deadline = time.time() + 8.0
    while time.time() < deadline:
        if not _pid_alive(process.pid):
            _clear_pid()
            sys.stderr.write(
                f"[cli] gateway exited immediately; last log lines:\n"
                f"{_tail(log_path, 10)}\n"
            )
            return 1
        if _probe_health(timeout=1.0) is not None:
            sys.stderr.write(
                f"[cli] gateway running (pid {process.pid}) on "
                f"http://{settings.label} -- log: {log_path}\n"
            )
            return 0
        time.sleep(0.25)

    sys.stderr.write(
        f"[cli] gateway spawned (pid {process.pid}) but not healthy yet; "
        f"check {log_path}\n"
    )
    return 0


def _tail(path: Path, lines: int) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(log unavailable)"


def cmd_stop(args: argparse.Namespace) -> int:
    pid = _read_pid()
    if not _pid_alive(pid):
        _clear_pid()
        if _probe_health(timeout=1.0) is not None:
            sys.stderr.write(
                "[cli] something answers on the port, but it was not started by "
                "this CLI; refusing to signal a PID we did not record.\n"
            )
            return 1
        sys.stderr.write("[cli] gateway is not running.\n")
        return 0

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        sys.stderr.write(f"[cli] could not signal pid {pid}: {exc}\n")
        return 1

    deadline = time.time() + 8.0
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(0.2)
    if _pid_alive(pid):
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.3)

    _clear_pid()
    sys.stderr.write(f"[cli] gateway stopped (pid {pid}).\n")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from .config import load_settings

    settings = load_settings()
    pid = _read_pid()
    alive = _pid_alive(pid)
    health = _probe_health()

    payload = {
        "pid": pid,
        "pid_alive": alive,
        "pid_recorded_by_cli": pid is not None,
        "listen": f"{settings.host}:{settings.port}",
        "healthy": bool(health),
        "version": (health or {}).get("version"),
        "upstream": (health or {}).get("upstream"),
        "classifier_mode": (health or {}).get("classifier_mode"),
        "profile": (health or {}).get("profile"),
        "env_file": (health or {}).get("env_file"),
        "foreign_service_on_port": bool(health) and not (health or {}).get("data_dir"),
    }
    print(json.dumps(payload, indent=2))
    return 0 if health else 1


def cmd_stats(args: argparse.Namespace) -> int:
    from .analytics import main as analytics_main

    argv: List[str] = []
    if args.json:
        argv.append("--json")
    if args.live:
        argv.append("--live")
    if args.audit_log:
        argv.extend(["--audit-log", args.audit_log])
    return analytics_main(argv)


def cmd_test(args: argparse.Namespace) -> int:
    """Run the offline test suite. No network, no credentials, no spend."""
    root = Path(__file__).resolve().parent.parent
    sys.stderr.write(f"[cli] running the offline suite in {root}\n")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=str(root),
    )
    return completed.returncode


SERVICE_UNIT = """\
[Unit]
Description=agent-context-gateway (context-pruning LLM proxy)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={workdir}
# '-' prefix: the unit still starts when the file does not exist.
EnvironmentFile=-{env_file}
Environment=PYTHONUNBUFFERED=1
ExecStart={python} -m src.cli start
ExecStop={python} -m src.cli stop
Restart=on-failure
RestartSec=3

[Install]
WantedBy=default.target
"""


def cmd_service(args: argparse.Namespace) -> int:
    """Generate and enable a systemd *user* unit for the gateway.

    User scope is deliberate: a system unit needs root and a full install,
    while `systemctl --user` is exactly the right tool for a single-user local
    proxy. `--user` is accepted to make the intent explicit.
    """
    if args.action != "install":
        sys.stderr.write(f"[cli] unknown service action {args.action!r}\n")
        return 2

    root = Path(__file__).resolve().parent.parent
    from .config import find_profile, load_settings

    settings = load_settings()
    env_file = None
    if args.profile:
        env_file = find_profile(args.profile)
        if env_file is None:
            sys.stderr.write(f"[cli] unknown profile {args.profile!r}\n")
            return 2

    unit_dir = Path("~/.config/systemd/user").expanduser()
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / "agent-gateway.service"
    unit_path.write_text(
        SERVICE_UNIT.format(
            workdir=root,
            env_file=env_file or (Path.cwd() / ".env"),
            python=sys.executable,
        ),
        encoding="utf-8",
    )
    sys.stderr.write(f"[cli] wrote {unit_path}\n")

    systemctl = None
    for candidate in ("/usr/bin/systemctl", "/bin/systemctl"):
        if Path(candidate).exists():
            systemctl = candidate
            break
    if systemctl is None:
        sys.stderr.write(
            "[cli] systemctl not found. Start the unit manually with:\n"
            f"  systemd-run --user --unit=agent-gateway "
            f"{sys.executable} -m src.cli start\n"
        )
        return 0

    commands = [
        [systemctl, "--user", "daemon-reload"],
        [systemctl, "--user", "enable", "--now", "agent-gateway.service"],
    ]
    for command in commands:
        result = subprocess.run(command)
        if result.returncode != 0:
            sys.stderr.write(
                f"[cli] {' '.join(command[2:])} failed. The unit file is written; "
                f"enable it manually with:\n"
                f"  systemctl --user enable --now agent-gateway.service\n"
            )
            return 1

    sys.stderr.write(
        "[cli] service enabled. Useful commands:\n"
        "  systemctl --user status agent-gateway\n"
        "  journalctl --user -u agent-gateway -f\n"
        "  agent-gateway stop && systemctl --user stop agent-gateway\n"
    )
    return 0


# ------------------------------------------------------------------------------
# shared low-level helpers used by the UX commands
# ------------------------------------------------------------------------------
def _gateway_healthy() -> bool:
    """True when OUR gateway answers on the configured port."""
    health = _probe_health(timeout=1.5)
    if not health:
        return False
    from .config import load_settings

    # Identity, not just a 200: the payload must carry our data dir.
    return str(health.get("data_dir") or "") == str(load_settings().data_dir)


def _ensure_gateway_running() -> int:
    """Start the daemon if needed; return 0 when a healthy gateway is up."""
    if _gateway_healthy():
        return 0
    if _probe_health(timeout=1.0) is not None:
        raise CliError(
            "the configured port answers, but it is not this gateway",
            ["Run `agent-gateway doctor` to see who owns the port.",
             "Start on another port: agent-gateway start --port 8091"],
        )
    namespace = argparse.Namespace(daemon=True, profile="", port=None)
    if cmd_start(namespace) != 0:
        raise CliError(
            "could not start the gateway in the background",
            ["Run `agent-gateway start` in the foreground to see the error.",
             "Run `agent-gateway doctor` for a full diagnosis."],
        )
    return 0


# ------------------------------------------------------------------------------
# `agent-gateway init` -- zero-config interactive setup
# ------------------------------------------------------------------------------
INIT_AGENTS = (
    ("1", "claude", "Claude Code"),
    ("2", "hermes", "Hermes Agent"),
    ("3", "aider", "Aider / Cursor"),
    ("4", "custom", "Custom / Other"),
)
INIT_BACKENDS = (
    ("1", "antigravity", "Google Antigravity Bridge (Google One Pro / free)"),
    ("2", "ollama", "Local Ollama"),
    ("3", "openrouter", "OpenRouter / commercial API"),
    ("4", "custom", "Direct custom URL"),
)


def detect_local_services(timeout: float = 0.8) -> dict:
    """Probe the standard local model-server ports.

    Used by `init` to preselect answers and by `doctor` to explain what is on
    8080/11434. Never raises: an unreachable port is simply not listed.
    """
    import httpx

    found = {}
    for port, label in ((8080, "antigravity"), (11434, "ollama")):
        try:
            response = httpx.get(
                f"http://127.0.0.1:{port}/v1/models", timeout=timeout
            )
            if response.status_code < 500:
                found[label] = {"port": port, "status": response.status_code}
        except Exception:
            continue
    return found


def build_init_preset(
    agent: str, backend: str, port: int, api_key: str = "", upstream_url: str = ""
) -> dict:
    """The .env contents for a chosen agent/backend pair.

    Pure so the wizard's decisions are testable without a terminal.
    """
    preset: dict = {"GATEWAY_HOST": "127.0.0.1", "GATEWAY_PORT": str(port)}

    if backend == "antigravity":
        preset["UPSTREAM_BASE_URL"] = upstream_url or "http://127.0.0.1:8080/v1"
        preset["ALLOW_LEGACY_UPSTREAM_PORT"] = "1"
        preset["CLASSIFIER_MODE"] = "upstream_reused"
        preset["CLASSIFIER_MODEL"] = "gemini-2.5-flash"
    elif backend == "ollama":
        preset["UPSTREAM_BASE_URL"] = upstream_url or "http://127.0.0.1:11434/v1"
        preset["CLASSIFIER_MODE"] = "local_ollama"
    elif backend == "openrouter":
        preset["UPSTREAM_BASE_URL"] = upstream_url or "https://openrouter.ai/api/v1"
        preset["UPSTREAM_API_KEY"] = api_key or "REPLACE_ME"
        preset["CLASSIFIER_MODE"] = "upstream_reused"
        preset["CLASSIFIER_MODEL"] = "google/gemini-2.5-flash-lite"
    else:  # custom
        preset["UPSTREAM_BASE_URL"] = upstream_url or "https://api.openai.com/v1"
        preset["UPSTREAM_API_KEY"] = api_key or "REPLACE_ME"
        preset["CLASSIFIER_MODE"] = "heuristics"

    if agent == "claude":
        preset["ANTHROPIC_SURFACE"] = "1"
    return preset


def render_env_file(preset: dict, header: str = "") -> str:
    """Serialize a preset the way .env files are expected to look."""
    lines = []
    if header:
        for line in header.strip().splitlines():
            lines.append(f"# {line}" if not line.startswith("#") else line)
        lines.append("")
    for key, value in preset.items():
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def choose(prompt: str, options, input_fn, default: str = "1") -> str:
    """Ask until one of the numbered options is picked."""
    while True:
        sys.stderr.write(prompt)
        for number, _key, label in options:
            sys.stderr.write(f"  [{number}] {label}\n")
        raw = input_fn(f"Select 1-{len(options)} [{default}]: ").strip() or default
        for number, key, _label in options:
            if raw == number:
                return key
        sys.stderr.write(f"  please enter 1-{len(options)}\n")


def cmd_init(args: argparse.Namespace) -> int:
    """Interactive first-run setup: ask, detect, write .env, print next step."""
    input_fn = args.input_fn or input

    from . import ux

    sys.stderr.write(ux.banner("agent-gateway setup", width=62) + "\n")

    services = detect_local_services()
    if services:
        for label, info in services.items():
            sys.stderr.write(
                ux.kv(
                    f"detected {label}",
                    ux.green(f"listening on :{info['port']}", stream=sys.stderr)
                    + ux.dim(" (will preselect)", stream=sys.stderr),
                )
                + "\n"
            )

    agent = choose("\nWhich agent will use the gateway?\n", INIT_AGENTS, input_fn)

    default_backend = "1"
    if "ollama" in services and "antigravity" not in services:
        default_backend = "2"
    if "antigravity" in services:
        default_backend = "1"
    backend = choose(
        "\nWhat is the upstream backend?\n", INIT_BACKENDS, input_fn, default_backend
    )

    port = 8091 if backend == "antigravity" else 8090
    api_key = ""
    upstream_url = ""
    if backend in ("openrouter", "custom"):
        api_key = input_fn("API key for the upstream (input hidden? no -- plain): ").strip()
        if not api_key:
            sys.stderr.write(
                ux.yellow(
                    "no key entered -- placeholder written; edit .env before starting\n",
                    stream=sys.stderr,
                )
            )
            api_key = ""
    if backend == "custom":
        upstream_url = input_fn("Upstream base URL (e.g. https://host/v1): ").strip()

    preset = build_init_preset(agent, backend, port, api_key, upstream_url)

    env_path = Path.cwd() / ".env"
    if env_path.exists() and not args.force:
        raise CliError(
            f"{env_path} already exists",
            [
                f"Keep it and re-run with --force to overwrite: agent-gateway init --force",
                "Or point a different file: copy the printed preset into .env.mine",
            ],
        )
    env_path.write_text(
        render_env_file(
            preset,
            header=(
                f"agent-context-gateway -- written by `agent-gateway init`\n"
                f"agent: {agent} | backend: {backend}"
            ),
        ),
        encoding="utf-8",
    )

    run_hint = {
        "claude": "agent-gateway run claude",
        "hermes": "agent-gateway run hermes",
        "aider": "agent-gateway run aider",
    }.get(agent, f"agent-gateway start   # then point your client at :{port}/v1")

    lines = [
        ux.green("setup complete", stream=sys.stderr),
        f"wrote  {env_path}",
        f"agent  {agent}   backend  {backend}   port  {port}",
        "",
        "next step:",
        ux.bold(f"  {run_hint}", stream=sys.stderr),
        "",
        ux.dim("change anything later by editing .env, then restart", stream=sys.stderr),
    ]
    sys.stderr.write("\n".join(ux.box(lines, stream=sys.stderr)) + "\n")
    return 0


# ------------------------------------------------------------------------------
# `agent-gateway doctor` -- 2-second diagnosis
# ------------------------------------------------------------------------------
def _check_gateway() -> tuple:
    from .config import load_settings

    settings = load_settings()
    health = _probe_health(timeout=1.5)
    if health is None:
        if _read_pid() and _pid_alive(_read_pid()):
            return (
                "fail",
                f"recorded pid {_read_pid()} is alive but the port is silent",
                "Wait a moment and re-run; if it persists: agent-gateway stop && agent-gateway start",
            )
        return (
            "warn",
            "gateway is not running",
            "Start it: agent-gateway start   (or: agent-gateway run claude)",
        )
    if str(health.get("data_dir") or "") != str(settings.data_dir):
        return (
            "fail",
            f"something else answers on :{settings.port} (foreign service)",
            f"Start on a free port: agent-gateway start --port 8091",
        )
    version = health.get("version") or "?"
    return ("ok", f"healthy on :{settings.port} (v{version})", "")


def _check_upstream() -> tuple:
    import httpx

    from .config import load_settings

    settings = load_settings()
    started = time.perf_counter()
    try:
        headers = {}
        if settings.upstream_api_key:
            headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
        response = httpx.get(
            f"{settings.upstream_base_url.rstrip('/')}/models", headers=headers, timeout=4.0
        )
    except Exception as exc:
        return (
            "fail",
            f"cannot reach {settings.upstream_base_url} ({type(exc).__name__})",
            "Check the upstream is running and UPSTREAM_BASE_URL is correct; run `agent-gateway init` to reconfigure.",
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if response.status_code >= 500:
        return (
            "fail",
            f"upstream answered {response.status_code} in {elapsed_ms:.0f}ms",
            "The upstream is up but unhealthy; check its own logs.",
        )
    if response.status_code >= 400:
        return (
            "warn",
            f"upstream answered {response.status_code} in {elapsed_ms:.0f}ms (auth?)",
            "Check UPSTREAM_API_KEY is valid for this provider.",
        )
    return ("ok", f"reachable in {elapsed_ms:.0f}ms", "")


def _check_classifier() -> tuple:
    from .config import load_settings

    settings = load_settings()
    mode = settings.effective_classifier_mode
    if mode == "heuristics":
        return ("ok", "heuristics -- no network call, fails open", "")
    if mode == "upstream_reused":
        ok, detail, hint = _check_upstream()
        return (ok, f"upstream_reused via {settings.classifier_api_url} -- {detail}", hint)
    if mode == "local_ollama":
        import httpx

        try:
            response = httpx.get(
                settings.classifier_api_url.rsplit("/", 1)[0].rsplit("/", 1)[0] + "/models",
                timeout=3.0,
            )
            if response.status_code < 500:
                return ("ok", f"ollama reachable ({settings.classifier_model})", "")
            return ("warn", f"ollama answered {response.status_code}", "Check the model is pulled: ollama pull " + settings.classifier_model)
        except Exception:
            return (
                "fail",
                f"ollama not reachable at {settings.classifier_api_url}",
                "Start it: ollama serve   and pull the model: ollama pull " + settings.classifier_model,
            )
    # external_jev
    import httpx

    try:
        response = httpx.get(settings.classifier_api_url, timeout=4.0)
        if response.status_code < 500:
            return ("ok", f"external endpoint reachable ({response.status_code})", "")
        return ("warn", f"external endpoint answered {response.status_code}", "Check the key and quota on the provider.")
    except Exception:
        return (
            "warn",
            "external classifier unreachable (routing still works, fails open)",
            "Check network/CLASSIFIER_API_URL; heuristics keep working meanwhile.",
        )


def _check_database() -> tuple:
    import sqlite3

    from .config import load_settings

    settings = load_settings()
    try:
        settings.ensure_dirs()
        connection = sqlite3.connect(str(settings.db_path), timeout=3.0)
        try:
            mode = connection.execute("PRAGMA journal_mode=WAL;").fetchone()[0]
            tables = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        finally:
            connection.close()
    except Exception as exc:
        return (
            "fail",
            f"cannot open {settings.db_path}: {type(exc).__name__}",
            f"Check permissions on {settings.data_dir}.",
        )
    if str(mode).lower() != "wal":
        return ("warn", f"journal mode is {mode}, expected wal", "Delete the db to recreate it in WAL mode.")
    return ("ok", f"WAL ok, {len(tables)} tables", "")


def _check_shared_memory() -> tuple:
    import shutil

    from .config import load_settings

    settings = load_settings()
    path = Path(settings.shm_cache_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".doctor-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return (
            "fail",
            f"{path} is not writable ({type(exc).__name__})",
            "Set SHM_CACHE_DIR to a writable path in .env.",
        )
    usage = shutil.disk_usage(path)
    free_mb = usage.free / 1_000_000
    if free_mb < 50:
        return (
            "warn",
            f"only {free_mb:.0f}MB free on {path}",
            "Free space, or point SHM_CACHE_DIR elsewhere; spills evict automatically at 180MB.",
        )
    return ("ok", f"{path} writable, {free_mb:.0f}MB free", "")


def _check_binaries() -> tuple:
    import shutil

    found = {name: shutil.which(name) for name in ("claude", "hermes", "aider", "docker")}
    installed = [name for name, path in found.items() if path]
    if not installed:
        return (
            "warn",
            "no known agent binaries on PATH (claude/hermes/aider/docker)",
            "Install one, or point any OpenAI SDK client at this gateway directly.",
        )
    return ("ok", "installed: " + ", ".join(sorted(installed)), "")


def run_checks() -> list:
    """The doctor checklist. Returns (name, status, detail, hint) tuples."""
    checks = [
        ("gateway", *_check_gateway()),
        ("upstream", *_check_upstream()),
        ("classifier", *_check_classifier()),
        ("database", *_check_database()),
        ("shared memory", *_check_shared_memory()),
        ("agent binaries", *_check_binaries()),
    ]
    return checks


def render_doctor(checks) -> str:
    """Plain text (colour applied by the caller's stream-aware helpers)."""
    lines = ["agent-gateway doctor", "=" * 46]
    for name, status, detail, hint in checks:
        marker = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}[status]
        lines.append(f"{marker} {name:<15} {detail}")
        if hint and status != "ok":
            lines.append(f"       fix: {hint}")
    failures = sum(1 for c in checks if c[1] == "fail")
    warnings = sum(1 for c in checks if c[1] == "warn")
    lines.append("=" * 46)
    lines.append(f"{failures} failed, {warnings} warnings, {len(checks)} checks")
    return "\n".join(lines)


def cmd_doctor(args: argparse.Namespace) -> int:
    from . import ux

    checks = run_checks()
    if args.json:
        print(
            json.dumps(
                [
                    {"check": name, "status": status, "detail": detail, "fix": hint}
                    for name, status, detail, hint in checks
                ],
                indent=2,
            )
        )
    else:
        for line in render_doctor(checks).splitlines():
            if "[FAIL]" in line:
                sys.stderr.write(ux.red(line, stream=sys.stderr) + "\n")
            elif "[WARN]" in line:
                sys.stderr.write(ux.yellow(line, stream=sys.stderr) + "\n")
            elif "[ OK ]" in line:
                sys.stderr.write(ux.green(line, stream=sys.stderr) + "\n")
            else:
                sys.stderr.write(line + "\n")
    return 1 if any(status == "fail" for _n, status, _d, _h in checks) else 0


# ------------------------------------------------------------------------------
# `agent-gateway run <agent>` -- one command, proxy + agent
# ------------------------------------------------------------------------------
RUN_AGENTS = {
    "claude": {
        "binary": "claude",
        "env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:{port}"},
        "args": [],
    },
    "hermes": {
        "binary": "hermes",
        "env": {"OPENAI_BASE_URL": "http://127.0.0.1:{port}/v1"},
        "args": [],
    },
    "aider": {
        "binary": "aider",
        "env": {"OPENAI_API_KEY": "dummy"},
        "args": ["--openai-api-base", "http://127.0.0.1:{port}/v1"],
    },
    "cursor": {
        "binary": "cursor-agent",
        "env": {"OPENAI_BASE_URL": "http://127.0.0.1:{port}/v1"},
        "args": [],
    },
}


def build_agent_env(agent: str, port: int) -> dict:
    """Environment variables to inject for `run`. Pure and testable."""
    spec = RUN_AGENTS.get(agent)
    if not spec:
        return {}
    return {
        key: value.format(port=port) for key, value in spec["env"].items()
    }


def build_agent_command(agent: str, extra_args: List[str], port: int) -> List[str]:
    """Full argv for the child agent, with gateway flags inserted. Pure."""
    spec = RUN_AGENTS.get(agent)
    if not spec:
        return []
    command = [spec["binary"]]
    command.extend(arg.format(port=port) for arg in spec["args"])
    command.extend(extra_args)
    return command


def cmd_run(args: argparse.Namespace) -> int:
    import shutil

    from . import ux

    agent = args.agent
    spec = RUN_AGENTS.get(agent)
    if spec is None:
        raise CliError(
            f"unknown agent {agent!r}",
            ["Known agents: " + ", ".join(sorted(RUN_AGENTS)),
             "For anything else, point the client at http://127.0.0.1:<port>/v1 yourself."],
        )
    if shutil.which(spec["binary"]) is None:
        raise CliError(
            f"{spec['binary']!r} is not installed or not on PATH",
            [
                f"Install {spec['binary']} first, then re-run this command.",
                "Or start only the gateway: agent-gateway start",
            ],
        )

    from .config import load_settings

    settings = load_settings()

    _ensure_gateway_running()
    sys.stderr.write(
        ux.dim(
            f"[run] gateway ready on :{settings.port} -- launching {spec['binary']}\n",
            stream=sys.stderr,
        )
    )

    environment = os.environ.copy()
    environment.update(build_agent_env(agent, settings.port))

    command = build_agent_command(agent, args.agent_args, settings.port)
    sys.stderr.write(
        ux.dim("[run] env: " + ", ".join(sorted(build_agent_env(agent, settings.port))) + "\n", stream=sys.stderr)
    )
    try:
        completed = subprocess.run(command, env=environment)
    except KeyboardInterrupt:
        return 130
    finally:
        # Leave the gateway running so the next `run` is instant; `stop` is opt-in.
        pass
    return completed.returncode


# ------------------------------------------------------------------------------
# `agent-gateway ui` -- open the dashboard
# ------------------------------------------------------------------------------
def cmd_ui(args: argparse.Namespace) -> int:
    import webbrowser

    from . import ux
    from .config import load_settings

    settings = load_settings()
    _ensure_gateway_running()
    url = f"http://{settings.host}:{settings.port}/ui"
    sys.stderr.write(f"[ui] opening {url}\n")
    try:
        webbrowser.open(url)
    except Exception:
        sys.stderr.write(f"[ui] could not launch a browser -- open {url} manually\n")
    if args.no_open:
        sys.stderr.write(f"[ui] dashboard: {url}\n")
    return 0


# ------------------------------------------------------------------------------
# parser
# ------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-gateway",
        description=(
            "Context-pruning LLM gateway. Run with no arguments for help; "
            "`agent-gateway start` serves, `agent-gateway stats` shows savings."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    start = subparsers.add_parser("start", help="run the gateway (foreground by default)")
    start.add_argument("--profile", help="load .env.NAME instead of .env")
    start.add_argument("--port", type=int, help="override GATEWAY_PORT")
    start.add_argument(
        "--daemon", action="store_true", help="detach after the health check passes"
    )
    start.set_defaults(func=cmd_start)

    stop = subparsers.add_parser("stop", help="stop a gateway started by this CLI")
    stop.set_defaults(func=cmd_stop)

    status = subparsers.add_parser("status", help="pid, health and configuration summary")
    status.set_defaults(func=cmd_status)

    stats = subparsers.add_parser("stats", help="token and cost savings dashboard")
    stats.add_argument("--json", action="store_true", help="machine-readable output")
    stats.add_argument("--live", action="store_true", help="refresh in place until Ctrl-C")
    stats.add_argument("--audit-log", help="override the audit log path")
    stats.set_defaults(func=cmd_stats)

    test = subparsers.add_parser("test", help="run the offline test suite")
    test.set_defaults(func=cmd_test)

    service = subparsers.add_parser("service", help="manage the systemd user service")
    service_sub = service.add_subparsers(dest="action")
    install = service_sub.add_parser("install", help="write and enable the user unit")
    install.add_argument(
        "--user", action="store_true", help="user scope (default; accepted for explicitness)"
    )
    install.add_argument("--profile", help="point EnvironmentFile at .env.NAME")
    install.set_defaults(func=cmd_service)

    init = subparsers.add_parser(
        "init", help="interactive setup: pick an agent and backend, write .env"
    )
    init.add_argument(
        "--force", action="store_true", help="overwrite an existing .env"
    )
    init.add_argument(
        "--input-fn", dest="input_fn", default=None, help=argparse.SUPPRESS
    )
    init.set_defaults(func=cmd_init)

    doctor = subparsers.add_parser(
        "doctor", help="diagnose gateway, upstream, classifier, database and paths"
    )
    doctor.add_argument("--json", action="store_true", help="machine-readable output")
    doctor.set_defaults(func=cmd_doctor)

    run = subparsers.add_parser(
        "run", help="start the gateway (if needed) and launch an agent through it"
    )
    run.add_argument("agent", help="claude | hermes | aider | cursor")
    run.add_argument(
        "agent_args", nargs="*", help="extra arguments passed to the agent"
    )
    run.set_defaults(func=cmd_run)

    ui = subparsers.add_parser(
        "ui", help="start the gateway (if needed) and open the web dashboard"
    )
    ui.add_argument(
        "--no-open", action="store_true", help="print the URL instead of opening a browser"
    )
    ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except CliError as error:
        from . import ux

        ux.render_error(error)
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\n[interrupted]\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
