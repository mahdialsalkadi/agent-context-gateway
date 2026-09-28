"""
Single entrypoint for everything the gateway can do.

    agent-gateway                       (no args) interactive launcher
    agent-gateway start   [--profile NAME] [--port N] [--daemon]
    agent-gateway stop
    agent-gateway status
    agent-gateway stats   [--json] [--live]
    agent-gateway test
    agent-gateway install-shim          global wrapper in ~/.local/bin
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
    if getattr(args, "upstream_url", None):
        os.environ["UPSTREAM_BASE_URL"] = args.upstream_url
    if getattr(args, "upstream_key", None):
        os.environ["UPSTREAM_API_KEY"] = args.upstream_key
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

    if settings.effective_classifier_mode == "local_jev":
        from .jev_lifecycle import ensure_local_jev_running

        ensure_local_jev_running(settings)

    if args.daemon:
        return _start_daemon(args, settings)

    # Foreground: this process *is* the gateway. Ctrl-C stops it cleanly.
    _write_pid(os.getpid())
    try:
        from .gateway import main as gateway_main

        return gateway_main([])
    finally:
        _clear_pid()
        if settings.effective_classifier_mode == "local_jev":
            from .jev_lifecycle import stop_local_jev

            stop_local_jev(settings)


def cmd_install_skill(args: argparse.Namespace) -> int:
    """Generate and install native Jev tool router skill for subscriptions."""
    from .skill_generator import install_skill, ping_jev

    target = getattr(args, "target", "claude") or "claude"
    dest = Path(args.dest) if getattr(args, "dest", None) else None
    script_path, doc_path = install_skill(target=target, dest_dir=dest)
    sys.stdout.write(f"[Skill Installed] Native Jev tool router skill created at: {script_path}\n")
    sys.stdout.write(f"[Skill Configured] Metadata written to: {doc_path}\n")
    sys.stdout.write(
        "[Skill Ready] Directly calls local Vulkan Jev (http://127.0.0.1:11435) with $0 subscription usage.\n"
    )
    if getattr(args, "test", False):
        alive = ping_jev()
        if alive:
            sys.stdout.write("[Skill Test] Local Jev server is ACTIVE.\n")
        else:
            sys.stderr.write("[Skill Test] Local Jev server is UNREACHABLE at http://127.0.0.1:11435\n")
    return 0


def _start_daemon(args: argparse.Namespace, settings) -> int:
    """Detach: spawn the gateway into its own session and return immediately."""
    log_path = settings.gateway_log_path
    handle = open(log_path, "ab")

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")

    python_bin = sys.executable
    venv_py = Path.cwd() / ".venv" / "bin" / "python"
    if venv_py.is_file():
        python_bin = str(venv_py)

    process = subprocess.Popen(
        [python_bin, "-m", "src.gateway"],
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
                f"[Proxy Ready] Listening on http://{settings.host}:{settings.port} -> "
                f"Forwarding to {settings.upstream_base_url} (pid {process.pid})\n"
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


def _safe_stderr(text: str) -> None:
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass


def cmd_stop(args: argparse.Namespace) -> int:
    from .config import load_settings

    settings = load_settings()
    keep_jev = getattr(args, "keep_jev", False) or os.environ.get("AGENT_GATEWAY_PRESERVE_JEV") == "1"
    if not keep_jev:
        from .jev_lifecycle import stop_local_jev

        stop_local_jev(settings)

    pid = _read_pid()
    if not _pid_alive(pid):
        _clear_pid()
        if _probe_health(timeout=1.0) is not None:
            _safe_stderr(
                "[cli] something answers on the port, but it was not started by "
                "this CLI; refusing to signal a PID we did not record.\n"
            )
            return 1
        if not keep_jev:
            _safe_stderr("[Stopped] Gateway is not running. Local Jev stopped (Ports & VRAM released).\n")
        else:
            _safe_stderr("[cli] gateway is not running.\n")
        return 0

    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as exc:
        _safe_stderr(f"[cli] could not signal pid {pid}: {exc}\n")
        return 1

    deadline = time.time() + 8.0
    while _pid_alive(pid) and time.time() < deadline:
        time.sleep(0.2)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        time.sleep(0.3)

    _clear_pid()
    if not keep_jev:
        _safe_stderr("[Stopped] Gateway and local Jev stopped. Ports & VRAM released.\n")
    else:
        _safe_stderr(f"[cli] gateway stopped (pid {pid}).\n")
    return 0


def _probe_jev_diagnostic(url: str = "http://127.0.0.1:11435") -> dict:
    import urllib.request
    from urllib.parse import urlparse

    online = False
    model_name = None
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme or 'http'}://{parsed.netloc or '127.0.0.1:11435'}"
        req = urllib.request.Request(
            f"{base}/v1/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                online = True
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("models") or data.get("data") or []
                if models and isinstance(models, list):
                    first = models[0]
                    model_name = first.get("name") or first.get("id") or "qwen3.5-2b"
                    if isinstance(model_name, str) and "/" in model_name:
                        model_name = Path(model_name).name
    except Exception:
        pass

    jev_pid = None
    vulkan_resident = False
    try:
        out = subprocess.run(
            ["pgrep", "-f", "llama-server.*11435"],
            capture_output=True,
            text=True,
            timeout=1.0,
        )
        pids = [int(p) for p in out.stdout.split() if p.isdigit()]
        if not pids:
            out2 = subprocess.run(
                ["pgrep", "llama-server"],
                capture_output=True,
                text=True,
                timeout=1.0,
            )
            pids = [int(p) for p in out2.stdout.split() if p.isdigit()]
        if pids:
            jev_pid = pids[0]
            maps_path = Path(f"/proc/{jev_pid}/maps")
            if maps_path.is_file():
                maps_content = maps_path.read_text(errors="ignore").lower()
                if any(k in maps_content for k in ("vulkan", "radv", "nvidia", "amdgpu")):
                    vulkan_resident = True
            cmd_path = Path(f"/proc/{jev_pid}/cmdline")
            if cmd_path.is_file():
                cmd_content = cmd_path.read_text(errors="ignore")
                if "-ngl" in cmd_content:
                    vulkan_resident = True
    except Exception:
        pass

    return {
        "port": 11435,
        "online": online,
        "model": model_name,
        "pid": jev_pid,
        "vulkan_gpu_resident": vulkan_resident,
    }


def _detect_installed_skills() -> list:
    candidates = [
        ("antigravity_cli", Path.home() / ".gemini" / "antigravity-cli" / "skills" / "jev-router"),
        ("antigravity_config", Path.home() / ".gemini" / "config" / "skills" / "jev-router"),
        ("claude", Path.home() / ".claude" / "skills"),
        ("codex", Path.home() / ".codex" / "skills"),
        ("workspace_gemini", Path.cwd() / ".gemini" / "skills" / "jev-router"),
        ("workspace_agents", Path.cwd() / ".agents" / "skills" / "jev-router"),
    ]
    detected = []
    seen = set()
    for name, p in candidates:
        if p.exists() and str(p) not in seen:
            seen.add(str(p))
            py_file = p / "jev-router.py" if p.is_dir() else p
            md_file = p / "SKILL.md" if p.is_dir() else None
            detected.append({
                "target": name,
                "path": str(p),
                "script_exists": py_file.is_file(),
                "script_executable": os.access(py_file, os.X_OK) if py_file.is_file() else False,
                "manifest_exists": md_file.is_file() if md_file else False,
            })
    return detected


def cmd_status(args: argparse.Namespace) -> int:
    from .config import load_settings

    settings = load_settings()
    pid = _read_pid()
    alive = _pid_alive(pid)
    health = _probe_health()
    jev_diag = _probe_jev_diagnostic(settings.local_jev_url)
    skills = _detect_installed_skills()

    payload = {
        "gateway": {
            "pid": pid,
            "pid_alive": alive,
            "listen": f"{settings.host}:{settings.port}",
            "healthy": bool(health),
            "version": (health or {}).get("version"),
            "upstream": (health or {}).get("upstream"),
            "classifier_mode": (health or {}).get("classifier_mode"),
        },
        "jev_server": jev_diag,
        "installed_skills": skills,
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


def _ensure_gateway_running(owner_pid: Optional[int] = None) -> int:
    """Start the daemon if needed; return 0 when a healthy gateway is up."""
    from .config import load_settings

    settings = load_settings()
    if settings.effective_classifier_mode == "local_jev":
        from .jev_lifecycle import ensure_local_jev_running

        ensure_local_jev_running(settings)

    if _gateway_healthy():
        return 0
    if _probe_health(timeout=1.0) is not None:
        raise CliError(
            "the configured port answers, but it is not this gateway",
            ["Run `agent-gateway doctor` to see who owns the port.",
             "Start on another port: agent-gateway start --port 8091"],
        )
    # If a previously recorded gateway process is alive but not answering healthily
    # on our configured port/data_dir, stop the stale/misconfigured process first.
    if _pid_alive(_read_pid()):
        cmd_stop(argparse.Namespace(keep_jev=True))
    if owner_pid is not None:
        os.environ["AGENT_GATEWAY_OWNER_PID"] = str(owner_pid)
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
    8080/11434/11435. Never raises: an unreachable port is simply not listed.
    """
    import httpx

    found = {}
    for port, label in ((11434, "ollama"), (11435, "local_jev")):
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

    # Best-effort: install the universal wrapper so the very next command works
    # from any shell and any directory without activating the virtualenv.
    try:
        shim = install_global_wrapper()
        sys.stderr.write(f"[cli] global wrapper ready: {shim}\n")
        if not local_bin_on_path(shim.parent):
            sys.stderr.write(path_guidance(shim.parent) + "\n")
    except Exception:
        pass
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
    if mode == "local_jev":
        import httpx

        try:
            base = settings.classifier_api_url.rsplit("/chat/completions", 1)[0].rsplit("/completions", 1)[0]
            response = httpx.get(f"{base}/models", timeout=3.0)
            if response.status_code < 500:
                return ("ok", f"local_jev reachable on Vulkan GPU ({settings.classifier_model})", "")
            return ("warn", f"local_jev answered {response.status_code}", "Check llama-server status.")
        except Exception:
            return (
                "fail",
                f"local_jev not reachable at {settings.classifier_api_url}",
                "Start llama-server on port 11435 with Vulkan GPU offload.",
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
        "env": {
            "OPENAI_BASE_URL": "http://127.0.0.1:{port}/v1",
            "OPENAI_API_KEY": "dummy",
        },
        "args": [],
    },
    "aider": {
        "binary": "aider",
        "env": {"OPENAI_API_KEY": "dummy"},
        "args": ["--openai-api-base", "http://127.0.0.1:{port}/v1"],
    },
    "cursor": {
        "binary": "cursor-agent",
        "env": {
            "OPENAI_BASE_URL": "http://127.0.0.1:{port}/v1",
            "OPENAI_API_KEY": "dummy",
        },
        "args": [],
    },
    "codex": {
        "binary": "codex",
        "env": {
            "OPENAI_BASE_URL": "http://127.0.0.1:{port}/v1",
            "OPENAI_API_KEY": "dummy",
        },
        "args": [],
    },
    "antigravity": {
        "binary": "agy",
        "env": {
            "HTTP_PROXY": "http://127.0.0.1:{port}",
            "HTTPS_PROXY": "http://127.0.0.1:{port}",
            "ALL_PROXY": "http://127.0.0.1:{port}",
            "http_proxy": "http://127.0.0.1:{port}",
            "https_proxy": "http://127.0.0.1:{port}",
            "all_proxy": "http://127.0.0.1:{port}",
            "GOOGLE_API_ENDPOINT": "http://127.0.0.1:{port}",
            "CLOUDCODE_BASE_URL": "http://127.0.0.1:{port}",
            "GEMINI_BASE_URL": "http://127.0.0.1:{port}/v1",
            "ANTIGRAVITY_ENDPOINT": "http://127.0.0.1:{port}",
            "ANTIGRAVITY_PROXY": "http://127.0.0.1:{port}",
            "DAILY_CLOUDCODE_ENDPOINT": "http://127.0.0.1:{port}",
            "OPENAI_BASE_URL": "http://127.0.0.1:{port}/v1",
            "OPENAI_API_KEY": "dummy",
        },
        "args": [],
    },
}


def build_agent_env(
    agent: str, port: int, api_key: str = "dummy", model: Optional[str] = None
) -> dict:
    """Environment variables to inject for `run`. Pure and testable."""
    spec = RUN_AGENTS.get(agent)
    if not spec:
        return {}
    res = {}
    for key, value in spec["env"].items():
        if key == "OPENAI_API_KEY" and api_key:
            res[key] = api_key
        else:
            res[key] = value.format(port=port)
    if agent == "hermes":
        m = model or os.environ.get("HERMES_MODEL") or os.environ.get("MODEL_NAME")
        if m:
            res["MODEL_NAME"] = m
            res["HERMES_MODEL"] = m
            res["OPENAI_MODEL"] = m
    return res


def resolve_agent_path(agent: str) -> Optional[str]:
    """Find the agent executable binary, searching PATH and known local/npm/nvm locations."""
    import shutil

    spec = RUN_AGENTS.get(agent)
    binary = spec["binary"] if spec else agent

    # 1. Standard PATH
    found = shutil.which(binary)
    if found:
        return found

    # 2. Known local / user / package manager paths
    home = Path.home()
    candidates: List[Path] = [
        home / ".local" / "bin" / binary,
        home / ".npm-global" / "bin" / binary,
        home / "bin" / binary,
        Path("/usr/local/bin") / binary,
        Path("/opt/homebrew/bin") / binary,
    ]

    # Node / nvm / fnm versions
    nvm_node_dir = home / ".nvm" / "versions" / "node"
    if nvm_node_dir.is_dir():
        try:
            for p in sorted(nvm_node_dir.glob("*/bin/" + binary), reverse=True):
                candidates.append(p)
        except Exception:
            pass

    candidates.append(home / ".fnm" / "current" / "bin" / binary)
    try:
        candidates.append(home / ".local" / "share" / "fnm" / "current" / "bin" / binary)
    except Exception:
        pass

    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)

    # If claude specifically, try `npm prefix -g` or checking global npm prefix
    if binary == "claude" or agent == "claude":
        try:
            npm = shutil.which("npm")
            if npm:
                res = subprocess.run([npm, "prefix", "-g"], capture_output=True, text=True, timeout=2.0)
                if res.returncode == 0 and res.stdout.strip():
                    npm_bin = Path(res.stdout.strip()) / "bin" / "claude"
                    if npm_bin.is_file() and os.access(npm_bin, os.X_OK):
                        return str(npm_bin)
        except Exception:
            pass

    if binary in ("agy", "antigravity") or agent == "antigravity":
        for alt in ("agy", "antigravity"):
            found = shutil.which(alt)
            if found:
                return found
            for cand in (home / ".local" / "bin" / alt, Path("/usr/local/bin") / alt):
                if cand.is_file() and os.access(cand, os.X_OK):
                    return str(cand)

    return None


def build_agent_command(
    agent: str,
    extra_args: List[str],
    port: int,
    binary_path: Optional[str] = None,
    model: Optional[str] = None,
) -> List[str]:
    """Full argv for the child agent, with gateway flags inserted. Pure."""
    spec = RUN_AGENTS.get(agent)
    if not spec:
        return []
    bin_name = binary_path or spec["binary"]
    command = [bin_name]
    command.extend(arg.format(port=port) for arg in spec["args"])
    if agent == "hermes":
        m = model or os.environ.get("HERMES_MODEL") or os.environ.get("MODEL_NAME")
        if m and not any(arg in ("-m", "--model") for arg in extra_args):
            command.extend(["-m", m])
    command.extend(extra_args)
    return command


def cmd_run(args: argparse.Namespace) -> int:
    from . import ux

    agent = args.agent
    spec = RUN_AGENTS.get(agent)
    if spec is None:
        raise CliError(
            f"unknown agent {agent!r}",
            ["Known agents: " + ", ".join(sorted(RUN_AGENTS)),
             "For anything else, point the client at http://127.0.0.1:<port>/v1 yourself."],
        )

    binary_path = resolve_agent_path(agent)
    if not binary_path:
        hints = [
            f"Install {spec['binary']} first, then re-run this command.",
            "Or start only the gateway: agent-gateway start",
        ]
        if agent == "claude":
            error_msg = (
                "[Error] 'claude' CLI was not found or is not installed on PATH. "
                "Install it via 'npm install -g @anthropic-ai/claude-code' "
                "or use the native skill mode: 'agent-gateway install-skill claude'."
            )
            hints = [
                "Install it via 'npm install -g @anthropic-ai/claude-code'",
                "Or use the native skill mode: 'agent-gateway install-skill claude'",
                "Or start in standalone gateway mode: 'agent-gateway start --daemon'",
            ]
        elif agent == "antigravity":
            error_msg = (
                "[Error] 'agy' / 'antigravity' CLI was not found or is not installed on PATH. "
                "Use the native skill mode: 'agent-gateway install-skill antigravity'."
            )
            hints = [
                "Use native skill mode: 'agent-gateway install-skill antigravity'",
                "Or start in standalone gateway mode: 'agent-gateway start --daemon'",
            ]
        else:
            error_msg = f"[Error] '{spec['binary']}' is not installed or not on PATH"

        input_fn = getattr(args, "input_fn", None)
        if input_fn is None and getattr(args, "interactive", False) and sys.stdin.isatty():
            input_fn = input

        if input_fn is not None:
            sys.stderr.write(f"\n{error_msg}\n\n")
            sys.stderr.write("What would you like to do?\n")
            sys.stderr.write("  [1] Start gateway in standalone background mode (--daemon)\n")
            sys.stderr.write("  [2] Exit\n")
            choice = input_fn("Select 1-2 [1]: ").strip() or "1"
            if choice == "1":
                _ensure_gateway_running()
                from .config import load_settings

                settings = load_settings()
                sys.stderr.write(
                    f"[launcher] gateway ready on :{settings.port} -- point your client at "
                    f"http://127.0.0.1:{settings.port}/v1\n"
                )
                return 0
            return 1

        raise CliError(error_msg, hints)

    from .config import load_settings

    settings = load_settings()

    if settings.effective_classifier_mode == "local_jev":
        from .jev_lifecycle import ensure_local_jev_running

        ensure_local_jev_running(settings)

    _ensure_gateway_running(owner_pid=os.getpid())
    sys.stderr.write(
        ux.dim(
            f"[run] gateway ready on :{settings.port} -- launching {spec['binary']}\n",
            stream=sys.stderr,
        )
    )

    model = (
        getattr(args, "model", None)
        or os.environ.get("HERMES_MODEL")
        or getattr(settings, "hermes_model", "")
        or None
    )
    agent_key = settings.upstream_api_key or "dummy"
    agent_env = build_agent_env(agent, settings.port, agent_key, model=model)
    environment = os.environ.copy()
    environment.update(agent_env)

    command = build_agent_command(
        agent, args.agent_args, settings.port, binary_path=binary_path, model=model
    )
    sys.stderr.write(
        ux.dim("[run] env: " + ", ".join(sorted(agent_env)) + "\n", stream=sys.stderr)
    )
    import atexit

    proc: Optional[subprocess.Popen] = None

    def _cleanup():
        nonlocal proc
        # Closed terminal PTY or broken pipe must never abort cleanup:
        try:
            devnull = open(os.devnull, "w")
            sys.stdout = devnull
            sys.stderr = devnull
        except Exception:
            pass

        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        try:
            cmd_stop(argparse.Namespace(keep_jev=True))
        except Exception:
            pass

    def _signal_handler(signum, frame):
        _cleanup()
        sys.exit(128 + signum)

    old_hup = None
    old_term = None
    try:
        old_hup = signal.signal(signal.SIGHUP, _signal_handler)
    except Exception:
        pass
    try:
        old_term = signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

    atexit.register(_cleanup)
    try:
        proc = subprocess.Popen(
            command,
            env=environment,
            stdin=sys.stdin,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        completed_code = proc.wait()
    except KeyboardInterrupt:
        _cleanup()
        return 130
    finally:
        _cleanup()
        try:
            atexit.unregister(_cleanup)
        except Exception:
            pass
        if old_hup is not None:
            try:
                signal.signal(signal.SIGHUP, old_hup)
            except Exception:
                pass
        if old_term is not None:
            try:
                signal.signal(signal.SIGTERM, old_term)
            except Exception:
                pass
    return completed_code


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
# Global wrapper: one `agent-gateway` command from any directory and any shell
# ------------------------------------------------------------------------------
WRAPPER_TEMPLATE = """\
#!/usr/bin/env bash
# Auto-generated by agent-context-gateway -- do not edit by hand.
# Runs the CLI with this project's virtualenv, from any directory, in any
# shell (bash / zsh / fish). `export PYTHONPATH` keeps `-m src.cli` importable
# without `cd`.
REPO_DIR="{repo_dir}"
VENV_PY="{venv_python}"
if [ ! -x "$VENV_PY" ]; then
    echo "[ERROR] Virtualenv python not found at $VENV_PY" >&2
    echo "  cd $REPO_DIR && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
    exit 1
fi
export PYTHONPATH="$REPO_DIR${{PYTHONPATH:+:$PYTHONPATH}}"
exec "$VENV_PY" -m src.cli "$@"
"""


def install_global_wrapper(
    repo_dir: Optional[Path] = None,
    bin_dir: Optional[Path] = None,
    venv_python: Optional[Path] = None,
) -> Path:
    """Write an executable `agent-gateway` shim into `~/.local/bin`.

    The wrapper pins the project's virtualenv and exports PYTHONPATH, so the
    command works from any cwd without `source .venv/bin/activate` first. Returns
    the path written.
    """
    repo = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent.parent
    target_bin = (
        Path(bin_dir) if bin_dir is not None else Path("~/.local/bin").expanduser()
    )
    python = Path(venv_python) if venv_python is not None else repo / ".venv" / "bin" / "python"
    if venv_python is None and not python.exists():
        # No virtualenv: still produce a working wrapper around the interpreter
        # running this CLI, so a plain `pip install` setup keeps working.
        python = Path(sys.executable)

    target_bin.mkdir(parents=True, exist_ok=True)
    path = target_bin / "agent-gateway"
    path.write_text(
        WRAPPER_TEMPLATE.format(repo_dir=repo, venv_python=python),
        encoding="utf-8",
    )
    try:
        path.chmod(0o755)
    except OSError:
        pass
    return path


def local_bin_on_path(bin_dir: Optional[Path] = None) -> bool:
    target = str(Path(bin_dir) if bin_dir is not None else Path("~/.local/bin").expanduser())
    return target in os.environ.get("PATH", "").split(os.pathsep)


def path_guidance(bin_dir: Optional[Path] = None) -> str:
    """A one-liner for Fish/Bash/Zsh when ~/.local/bin is not on PATH."""
    target = str(Path(bin_dir) if bin_dir is not None else Path("~/.local/bin").expanduser())
    return (
        f"{target} is not on PATH. Add it once:\n"
        f'  bash/zsh:  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.profile && . ~/.profile\n'
        f"  fish:      fish_add_path $HOME/.local/bin"
    )


def cmd_install_shim(args: argparse.Namespace) -> int:
    """`agent-gateway install-shim` -- install the universal wrapper."""
    bin_dir = getattr(args, "bin_dir", None)
    path = install_global_wrapper(bin_dir=Path(bin_dir) if bin_dir else None)
    sys.stderr.write(f"[cli] wrote {path}\n")
    if not local_bin_on_path(path.parent):
        sys.stderr.write(path_guidance(path.parent) + "\n")
    else:
        sys.stderr.write(f"[cli] {path.parent} is already on PATH -- run `agent-gateway` from anywhere.\n")
    return 0


# ------------------------------------------------------------------------------
# Interactive launcher (`agent-gateway` with no arguments)
# ------------------------------------------------------------------------------
WIZARD_AGENTS = (
    ("1", "hermes", "Hermes Agent"),
    ("2", "claude", "Claude Code"),
    ("3", "codex", "Codex"),
    ("4", "antigravity", "Google Antigravity"),
    ("5", "standalone", "Standalone Gateway (run in background only)"),
)

WIZARD_HERMES_UPSTREAMS = (
    (
        "1",
        "local",
        "Local Offline Engine (Ollama / llama-server at http://127.0.0.1:11434/v1 - $0)",
    ),
    (
        "2",
        "commercial",
        "External Commercial API (OpenRouter, Groq, OpenAI, custom)",
    ),
)

WIZARD_SUBSCRIPTION_AGENT_UPSTREAMS = (
    (
        "1",
        "native_skill",
        "Native Skill Mode (Official Subscription - Zero-Proxy / No local ports needed)",
    ),
    (
        "2",
        "local",
        "Local Offline Engine (via Gateway Proxy - $0)",
    ),
    (
        "3",
        "commercial",
        "External Commercial API (via Gateway Proxy)",
    ),
)

WIZARD_STANDALONE_UPSTREAMS = (
    ("1", "local", "Local Offline Engine"),
    ("2", "commercial", "External Commercial API"),
)

WIZARD_UPSTREAMS = WIZARD_HERMES_UPSTREAMS

# Step 3: consolidated 3 clear, practical choices (advanced available via --advanced flag)
WIZARD_ENGINES = (
    (
        "1",
        "local_jev",
        "Local Jev-2B Decision (Vulkan GPU accelerated, offline $0) [Default]",
    ),
    (
        "2",
        "external_jev",
        "Cloud / API Jev Decision (via OpenRouter / dedicated Jev endpoint)",
    ),
    (
        "3",
        "heuristics",
        "Fast Regex Heuristics (<1ms, rule-based)",
    ),
)
WIZARD_ADVANCED_ENGINES = (
    (
        "4",
        "upstream_reused",
        "Upstream Reused        (asks the provider you already chose to classify)",
    ),
    (
        "5",
        "local_ollama",
        "Local Ollama classifier (qwen2.5:0.5b / llama3.2)",
    ),
)
WIZARD_STRATEGIES = WIZARD_ENGINES + WIZARD_ADVANCED_ENGINES

_DEFAULT_UPSTREAM_BY_STRATEGY = {
    "local_ollama": "http://127.0.0.1:11434/v1",
    "upstream_reused": "https://openrouter.ai/api/v1",
    "external_jev": "https://openrouter.ai/api/v1",
    "local_jev": "https://api.openai.com/v1",
    "heuristics": "https://api.openai.com/v1",
}

# The commercial sub-menu, reached from option 2 or 3.
WIZARD_COMMERCIAL = (
    ("1", "openrouter", "OpenRouter                 https://openrouter.ai/api/v1"),
    ("2", "openai", "OpenAI                     https://api.openai.com/v1"),
    ("3", "groq", "Groq                       https://api.groq.com/openai/v1"),
    ("4", "custom", "Other OpenAI-compatible endpoint (enter URL + key)"),
)
WIZARD_SUBSCRIPTION_PROVIDERS = frozenset({"local"})
WIZARD_KEY_PROVIDERS = frozenset({"openrouter", "openai", "groq", "custom"})

WIZARD_UPSTREAM_URLS = {
    "local": "http://127.0.0.1:11434/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}

# A placeholder satisfies agents that insist on a non-empty key, while the real
# subscription session travels upstream in the client's Authorization header.
WIZARD_PLACEHOLDER_KEY = "dummy"

WIZARD_TIER_NOTES = {
    "local": "Fully local and offline. No API key needed.",
}


def wizard_banner(status: str, width: int = 64) -> str:
    """The boxed header the launcher opens with.

    Pure so tests can pin it: a plain title line, a status line that reports the
    free tier, and a rule. ASCII-safe inside the box padding; colour is applied
    by the caller through `ux` so non-TTY streams stay clean.
    """
    top = f"╭{'─' * (width - 2)}╮"
    bottom = f"╰{'─' * (width - 2)}╯"

    def row(text: str) -> str:
        padding = max(0, width - 4 - len(text))
        return f"│ {text}{' ' * padding} │"

    return "\n".join(
        [top, row("agent-gateway"), row(status), bottom]
    )


def choose_upstream(input_fn, agent: str = "hermes") -> str:
    """The clean contextual upstream provider menu."""
    from . import ux

    if agent in ("claude", "codex"):
        options = WIZARD_SUBSCRIPTION_AGENT_UPSTREAMS
    elif agent == "standalone":
        options = WIZARD_STANDALONE_UPSTREAMS
    else:
        options = WIZARD_HERMES_UPSTREAMS

    while True:
        sys.stderr.write("\nWhere should requests go upstream?\n")
        for number, _key, label in options:
            sys.stderr.write(f"  [{number}] {label}\n")
        raw = input_fn(f"Select 1-{len(options)} [1]: ").strip() or "1"
        for number, key, _label in options:
            if raw == number:
                return key
        sys.stderr.write(f"  please enter 1-{len(options)}\n")


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True when a local socket already owns the port."""
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, int(port)))
        return False
    except OSError:
        return True
    finally:
        probe.close()


def suggest_port(preferred: int = 8090) -> int:
    """A free port, biased to 8091 when 8090/8080 are taken (Antigravity)."""
    if not port_in_use(preferred) and not port_in_use(8080):
        return preferred
    for candidate in (8091, 8092, 8093, 8094, 8095):
        if not port_in_use(candidate):
            return candidate
    return 8091


def read_env_file(path: Path) -> dict:
    """Parse simple KEY=value lines; comments and junk are ignored."""
    values: dict = {}
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return values
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def build_wizard_preset(
    agent: str,
    strategy: str,
    port: int,
    existing: Optional[dict] = None,
    upstream_url: Optional[str] = None,
    api_key: Optional[str] = None,
    classifier_url: Optional[str] = None,
    classifier_key: Optional[str] = None,
    anthropic_surface: Optional[bool] = None,
    hermes_model: Optional[str] = None,
    jev_api_base_url: Optional[str] = None,
    jev_api_key: Optional[str] = None,
) -> dict:
    """The `.env` values the launcher should apply. Pure and testable.

    `upstream_url`/`api_key` let the caller point at any OpenAI-compatible
    provider (the "choose endpoint + API" step); when they are omitted the
    existing `.env` value wins, then the strategy's default. `anthropic_surface`
    forces the Claude Code surface on independently of the chosen agent, which is
    how the subscription-session upstream is wired.
    """
    existing = existing or {}
    preset: dict = {
        "GATEWAY_HOST": "127.0.0.1",
        "GATEWAY_PORT": str(port),
        "CLASSIFIER_MODE": strategy,
    }
    if upstream_url:
        upstream = upstream_url
    elif strategy == "local_ollama":
        candidate = existing.get("UPSTREAM_BASE_URL", "")
        if any(h in candidate for h in ("127.0.0.1", "localhost", "0.0.0.0", "::1")):
            upstream = candidate
        else:
            upstream = "http://127.0.0.1:11434/v1"
    else:
        upstream = (
            existing.get("UPSTREAM_BASE_URL")
            or _DEFAULT_UPSTREAM_BY_STRATEGY.get(strategy, "https://api.openai.com/v1")
        )
    preset["UPSTREAM_BASE_URL"] = upstream
    key = api_key or existing.get("UPSTREAM_API_KEY")
    if key:
        preset["UPSTREAM_API_KEY"] = key
    if strategy == "local_jev":
        preset["LOCAL_JEV_URL"] = existing.get(
            "LOCAL_JEV_URL", "http://127.0.0.1:11435/v1/chat/completions"
        )
    if hermes_model:
        preset["HERMES_MODEL"] = hermes_model
    elif agent == "hermes" and existing.get("HERMES_MODEL"):
        preset["HERMES_MODEL"] = existing["HERMES_MODEL"]

    if jev_api_base_url:
        preset["JEV_API_BASE_URL"] = jev_api_base_url
        preset["CLASSIFIER_API_URL"] = jev_api_base_url
    if jev_api_key:
        preset["JEV_API_KEY"] = jev_api_key
        preset["CLASSIFIER_API_KEY"] = jev_api_key

    if classifier_url:
        if "CLASSIFIER_API_URL" not in preset:
            preset["CLASSIFIER_API_URL"] = classifier_url
        if strategy == "external_jev" and "JEV_API_BASE_URL" not in preset:
            preset["JEV_API_BASE_URL"] = classifier_url
        resolved = classifier_key or existing.get("CLASSIFIER_API_KEY")
        if resolved and "CLASSIFIER_API_KEY" not in preset:
            preset["CLASSIFIER_API_KEY"] = resolved
        if strategy == "external_jev" and resolved and "JEV_API_KEY" not in preset:
            preset["JEV_API_KEY"] = resolved
    if agent == "claude" or anthropic_surface:
        preset["ANTHROPIC_SURFACE"] = "1"
    return preset


def update_env_file(path: Path, updates: dict) -> Path:
    """Merge `updates` into a `.env`, replacing keys and preserving comments."""
    path = Path(path)
    try:
        original = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        original = []

    remaining = dict(updates)
    rendered: List[str] = []
    for line in original:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in remaining:
                rendered.append(f"{key}={remaining.pop(key)}")
                continue
        rendered.append(line)

    if remaining:
        if rendered and rendered[-1].strip():
            rendered.append("")
        rendered.append("# --- written by `agent-gateway` interactive launcher ---")
        for key, value in remaining.items():
            rendered.append(f"{key}={value}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rendered) + "\n", encoding="utf-8")
    return path


def cmd_interactive(args: argparse.Namespace) -> int:
    """The no-argument experience: agent, provider, engine -- then launch.

    Three numbered steps under a status banner, a silent port, and a gateway
    that starts (and an agent that attaches) without a fourth question.
    """
    from . import ux

    input_fn = getattr(args, "input_fn", None) or input
    launching = getattr(args, "launch", True)
    forced_port = getattr(args, "port", None)

    # Piped or non-TTY (scripts, pytest, CI): keep the long-standing contract
    # that bare `agent-gateway` prints help rather than blocking on a prompt.
    if not sys.stdin.isatty() and getattr(args, "input_fn", None) is None:
        build_parser().print_help()
        return 0

    # --- the box: what is running, where ------------------------------------
    services = detect_local_services()
    if "ollama" in services and "local_jev" in services:
        status = "Ollama (:11434) & Local Jev (:11435) detected"
    elif "local_jev" in services:
        status = "Local Jev classifier detected on :11435"
    elif "ollama" in services:
        status = "Ollama detected on :11434"
    else:
        status = "no local LLM detected -- commercial APIs available"
    sys.stderr.write(ux.bold(wizard_banner(status), stream=sys.stderr) + "\n")

    # --- Background Process Check -------------------------------------------
    running_pid = _read_pid()
    if _pid_alive(running_pid):
        prompt_msg = (
            f"\n[Gateway is currently running (PID: {running_pid})] -> "
            f"[S]top Gateway | [R]estart | [C]ontinue [C]: "
        )
        action = input_fn(prompt_msg).strip().lower() or "c"
        if action.startswith("s"):
            cmd_stop(argparse.Namespace())
            return 0
        elif action.startswith("r"):
            cmd_stop(argparse.Namespace(keep_jev=True))

    # --- step 1: the agent ---------------------------------------------------
    sys.stderr.write(
        ux.dim("\n── Step 1 · Choose Agent ─────────────────────────────", stream=sys.stderr)
        + "\n"
    )
    agent = choose("", WIZARD_AGENTS, input_fn)

    env_path = Path.cwd() / ".env"
    existing = read_env_file(env_path)

    # --- step 2: contextual upstream ----------------------------------------
    sys.stderr.write(
        ux.dim(
            "\n── Step 2 · Choose Upstream Provider ─────────────────",
            stream=sys.stderr,
        )
        + "\n"
    )
    if agent == "hermes":
        upstream_options = WIZARD_HERMES_UPSTREAMS
    elif agent in ("claude", "codex", "antigravity"):
        upstream_options = WIZARD_SUBSCRIPTION_AGENT_UPSTREAMS
    elif agent == "standalone":
        upstream_options = WIZARD_STANDALONE_UPSTREAMS
    else:
        upstream_options = WIZARD_HERMES_UPSTREAMS

    for number, _key, label in upstream_options:
        sys.stderr.write(f"  [{number}] {label}\n")

    provider = ""
    while provider not in {key for _n, key, _l in upstream_options}:
        raw = input_fn(f"Select 1-{len(upstream_options)} [1]: ").strip() or "1"
        provider = next(
            (key for number, key, _l in upstream_options if raw == number), ""
        )

    if provider == "native_skill":
        from .skill_generator import install_skill
        from .config import load_settings
        from .jev_lifecycle import ensure_local_jev_running

        # ── Step 3 for native skill: choose routing engine ──────────
        sys.stderr.write(
            ux.dim(
                "\n── Step 3 · Routing & Pruning Engine ─────────────────",
                stream=sys.stderr,
            )
            + "\n"
        )
        is_advanced = getattr(args, "advanced", False)
        engines_to_show = WIZARD_STRATEGIES if is_advanced else WIZARD_ENGINES
        for number, _key, label in engines_to_show:
            sys.stderr.write(f"  [{number}] {label}\n")
        max_choice = len(engines_to_show)
        skill_strategy = ""
        while not skill_strategy:
            raw = input_fn(f"Select 1-{max_choice} [1]: ").strip() or "1"
            for number, key, _l in WIZARD_STRATEGIES:
                if raw == number:
                    skill_strategy = key
                    break

        settings = load_settings()
        if skill_strategy == "local_jev":
            ensure_local_jev_running(settings)
        script_path, doc_path = install_skill(target=agent)
        sys.stderr.write(
            ux.green(f"\n✔ Native Jev tool router skill installed\n", stream=sys.stderr)
        )
        sys.stderr.write(ux.dim(f"  script → {script_path}\n  metadata → {doc_path}\n", stream=sys.stderr))
        sys.stderr.write(
            "[Skill Ready] Directly calls local Vulkan Jev (http://127.0.0.1:11435) with $0 subscription usage.\n"
        )
        bin_path = resolve_agent_path(agent)
        if bin_path and launching:
            sys.stderr.write(f"[Skill Ready] Launching {agent} natively (subscription mode)...\n")
            return subprocess.run([bin_path]).returncode
        sys.stderr.write(
            f"[Skill Ready] Native Jev router installed. Run '{agent}' in your project directory whenever you are ready.\n"
        )
        return 0

    if provider == "commercial":
        provider = choose("\nWhich commercial provider?", WIZARD_COMMERCIAL, input_fn)

    existing_key = existing.get("UPSTREAM_API_KEY", "")
    anthropic_surface = (agent == "claude")

    if provider == "custom":
        default_url = existing.get("UPSTREAM_BASE_URL") or "https://api.openai.com/v1"
        upstream_url = (
            input_fn(f"Upstream base URL [{default_url}]: ").strip() or default_url
        )
    elif provider == "local":
        candidate_existing = existing.get("UPSTREAM_BASE_URL", "")
        if any(h in candidate_existing for h in ("127.0.0.1", "localhost", "0.0.0.0", "::1")):
            default_url = candidate_existing
        else:
            default_url = "http://127.0.0.1:11434/v1"
        typed = input_fn(f"Local engine base URL [{default_url}]: ").strip()
        upstream_url = typed or default_url
    else:
        upstream_url = WIZARD_UPSTREAM_URLS[provider]

    api_key = ""
    if provider in WIZARD_KEY_PROVIDERS:
        hint = " (blank keeps the existing key)" if existing_key else ""
        api_key = input_fn(f"API key for {provider}{hint}: ").strip()
        if not api_key and not existing_key:
            sys.stderr.write(
                ux.yellow(
                    "no API key entered -- set UPSTREAM_API_KEY in .env before starting\n",
                    stream=sys.stderr,
                )
            )
    elif provider in WIZARD_SUBSCRIPTION_PROVIDERS:
        api_key = existing_key or WIZARD_PLACEHOLDER_KEY
        note = WIZARD_TIER_NOTES.get(provider)
        if note:
            sys.stderr.write("\n" + ux.cyan(note, stream=sys.stderr) + "\n")

    # --- step 3: the routing engine -----------------------------------------
    sys.stderr.write(
        ux.dim(
            "\n── Step 3 · Routing & Pruning Engine ─────────────────",
            stream=sys.stderr,
        )
        + "\n"
    )
    is_advanced = getattr(args, "advanced", False)
    engines_to_show = WIZARD_STRATEGIES if is_advanced else WIZARD_ENGINES
    for number, _key, label in engines_to_show:
        sys.stderr.write(f"  [{number}] {label}\n")

    default_strategy_num = "1"
    if existing.get("CLASSIFIER_MODE") == "external_jev":
        default_strategy_num = "2"
    elif existing.get("CLASSIFIER_MODE") == "heuristics":
        default_strategy_num = "3"
    elif existing.get("CLASSIFIER_MODE") == "upstream_reused":
        default_strategy_num = "4" if is_advanced else "1"
    elif existing.get("CLASSIFIER_MODE") == "local_ollama":
        default_strategy_num = "5" if is_advanced else "1"

    max_choice = len(engines_to_show)
    strategy = ""
    while not strategy:
        raw = (
            input_fn(f"Select 1-{max_choice} [{default_strategy_num}]: ").strip()
            or default_strategy_num
        )
        for number, key, _l in WIZARD_STRATEGIES:
            if raw == number:
                strategy = key
                break
        if not strategy:
            sys.stderr.write(f"Please select 1-{max_choice}\n")

    # `external_jev` is the one strategy that needs its own classifier endpoint.
    classifier_url = ""
    classifier_key = ""
    jev_api_base_url = ""
    jev_api_key = ""
    if strategy == "external_jev":
        default_jev_url = (
            existing.get("JEV_API_BASE_URL")
            or existing.get("CLASSIFIER_API_URL")
            or "https://openrouter.ai/api/v1"
        )
        jev_url_input = (
            input_fn(f"Jev API base URL [{default_jev_url}]: ").strip()
            or default_jev_url
        )
        jev_api_base_url = jev_url_input
        classifier_url = jev_url_input

        existing_jev_key = existing.get("JEV_API_KEY") or existing.get("CLASSIFIER_API_KEY", "")
        hint = " (blank keeps the existing key)" if existing_jev_key else ""
        jev_key_input = input_fn(f"Jev API key{hint}: ").strip()
        jev_api_key = jev_key_input or existing_jev_key
        classifier_key = jev_api_key

    # --- the port is not a question -----------------------------------------
    # Silent by default: 8090 when free, 8091 when 8080/8090 are busy, and only
    # `--port` (or an existing .env port) overrides without asking.
    if forced_port:
        port = int(forced_port)
    elif existing.get("GATEWAY_PORT", "").isdigit():
        port = int(existing["GATEWAY_PORT"])
    else:
        port = suggest_port()

    preset = build_wizard_preset(
        agent,
        strategy,
        port,
        existing,
        upstream_url=upstream_url,
        api_key=api_key,
        classifier_url=classifier_url,
        classifier_key=classifier_key,
        anthropic_surface=anthropic_surface,
        jev_api_base_url=jev_api_base_url,
        jev_api_key=jev_api_key,
    )
    update_env_file(env_path, preset)
    os.environ.update({key: str(value) for key, value in preset.items()})

    # --- shim self-heal -------------------------------------------------------
    # The global wrapper is what makes `agent-gateway` work from any directory;
    # a missing or unreadable one is rewritten here rather than complained about.
    try:
        shim = install_global_wrapper()
        hint = f"[{shim.parent}]"
        if not local_bin_on_path(shim.parent):
            hint = path_guidance(shim.parent)
    except Exception as exc:  # pragma: no cover - filesystem edge
        hint = f"(could not write the global wrapper: {exc})"

    sys.stderr.write(
        ux.green("\n✔ setup saved to " + str(env_path), stream=sys.stderr) + "\n"
    )
    sys.stderr.write(
        ux.dim(
            f"  agent → {agent} · upstream → {upstream_url} · "
            f"engine → {strategy} · port → {port}",
            stream=sys.stderr,
        )
        + "\n"
    )
    sys.stderr.write(ux.dim("agent-gateway wrapper: " + hint, stream=sys.stderr) + "\n")

    # When interactive setup writes a new configuration, stop any old
    # gateway process running with previous settings so the new setup takes effect.
    if _pid_alive(_read_pid()):
        cmd_stop(argparse.Namespace(keep_jev=True))

    from .config import load_settings

    settings = load_settings()
    if strategy == "local_jev":
        from .jev_lifecycle import ensure_local_jev_running

        ensure_local_jev_running(settings)

    if agent == "standalone":
        if launching:
            _ensure_gateway_running()
            sys.stderr.write(
                ux.dim(
                    f"[launcher] gateway ready on :{port} -- point your client at "
                    f"http://127.0.0.1:{port}/v1\n",
                    stream=sys.stderr,
                )
            )
        return 0

    if not launching:
        sys.stderr.write(
            ux.dim(f"[launcher] would now run: agent-gateway run {agent}\n", stream=sys.stderr)
        )
        return 0

    sys.stderr.write(
        ux.dim(f"[launcher] starting {agent} through :{settings.port}\n", stream=sys.stderr)
    )
    return cmd_run(
        argparse.Namespace(
            agent=agent,
            agent_args=[],
            input_fn=input_fn,
            interactive=True,
            model=existing.get("HERMES_MODEL"),
        )
    )


# ------------------------------------------------------------------------------
# parser
# ------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-gateway",
        description=(
            "Context-pruning LLM gateway. Run with no arguments for the guided "
            "launcher; `agent-gateway start` serves, `agent-gateway stats` shows "
            "savings."
        ),
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="show advanced routing and pruning engines",
    )
    subparsers = parser.add_subparsers(dest="command")

    start = subparsers.add_parser("start", help="run the transparent proxy (foreground by default)")
    start.add_argument("--profile", help="load .env.NAME instead of .env")
    start.add_argument("--port", type=int, help="override GATEWAY_PORT")
    start.add_argument("--upstream-url", "-u", help="override UPSTREAM_BASE_URL")
    start.add_argument("--upstream-key", "-k", help="override UPSTREAM_API_KEY")
    start.add_argument(
        "--daemon", action="store_true", help="detach after the health check passes"
    )
    start.set_defaults(func=cmd_start)

    skill = subparsers.add_parser(
        "install-skill",
        help="generate and install native Jev tool router skill for subscriptions (claude, codex, antigravity, generic)",
    )
    skill.add_argument(
        "target",
        nargs="?",
        default="claude",
        choices=["claude", "codex", "antigravity", "generic"],
        help="subscription target environment (claude | codex | antigravity | generic, default: claude)",
    )
    skill.add_argument("--dest", help="custom destination directory")
    skill.add_argument("--workspace", action="store_true", help="install to current workspace in addition to global")
    skill.add_argument("--test", action="store_true", help="verify local Jev connectivity")
    skill.set_defaults(func=cmd_install_skill)

    stop = subparsers.add_parser("stop", help="stop a gateway started by this CLI")
    stop.add_argument("--keep-jev", action="store_true", help="keep local Jev server running in VRAM")
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
    run.add_argument("agent", help="claude | hermes | aider | cursor | codex | antigravity")
    run.add_argument("-m", "--model", help="target model name for the agent (e.g. hermes)")
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

    interactive = subparsers.add_parser(
        "interactive", help="guided launcher (same as running with no arguments)"
    )
    interactive.add_argument(
        "--input-fn", dest="input_fn", default=None, help=argparse.SUPPRESS
    )
    interactive.add_argument(
        "--no-launch", dest="launch", action="store_false", help=argparse.SUPPRESS
    )
    interactive.add_argument(
        "--advanced", action="store_true", help="show advanced routing and pruning engines"
    )
    interactive.add_argument(
        "--port", type=int, default=None, help="override the gateway port (skipped silently otherwise)"
    )
    interactive.set_defaults(func=cmd_interactive)

    shim = subparsers.add_parser(
        "install-shim",
        help="install the universal ~/.local/bin/agent-gateway wrapper",
    )
    shim.add_argument(
        "--bin-dir", dest="bin_dir", default=None, help=argparse.SUPPRESS
    )
    shim.set_defaults(func=cmd_install_shim)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not getattr(args, "func", None):
            # No arguments: the interactive launcher. In a non-TTY it prints help,
            # so scripts and pipelines keep the historical behaviour.
            return cmd_interactive(args)
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
