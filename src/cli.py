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

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
