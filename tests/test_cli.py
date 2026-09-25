"""
The `agent-gateway` CLI dispatcher.

Parsing, the pidfile lifecycle (start/stop/status against a real spawned
gateway on a private port) and the service unit generator. Everything runs
offline; the only network touched is the loopback.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from src import cli
from src.config import load_settings


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Every test gets its own LOG_DIR (pidfile) and default env."""
    saved = dict(os.environ)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("GATEWAY_PORT", "8399")
    monkeypatch.setenv("UPSTREAM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("UPSTREAM_API_KEY", "k")
    yield
    os.environ.clear()
    os.environ.update(saved)


# ------------------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------------------
def test_no_arguments_prints_help_and_exits_cleanly(capsys):
    assert cli.main([]) == 0
    assert "start" in capsys.readouterr().out


def test_every_subcommand_is_wired():
    parser = cli.build_parser()
    for name in ("start", "stop", "status", "stats", "test"):
        args = parser.parse_args([name])
        assert getattr(args, "func", None), f"{name} has no handler"
    # `service` needs an action subcommand to be runnable.
    args = parser.parse_args(["service", "install"])
    assert getattr(args, "func", None), "service install has no handler"


def test_start_accepts_profile_port_and_daemon_flags():
    args = cli.build_parser().parse_args(
        ["start", "--profile", "antigravity", "--port", "8123", "--daemon"]
    )
    assert args.profile == "antigravity"
    assert args.port == 8123
    assert args.daemon is True


def test_start_rejects_an_unknown_profile():
    assert cli.cmd_start(
        cli.build_parser().parse_args(["start", "--profile", "nope"])
    ) == 2


def test_service_install_rejects_unknown_actions():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["service", "frobnicate"])


# ------------------------------------------------------------------------------
# pidfile lifecycle
# ------------------------------------------------------------------------------
def test_pid_helpers_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_pid_path", lambda: tmp_path / "gw.pid")

    assert cli._read_pid() is None
    assert not cli._pid_alive(None)

    cli._write_pid(12345)
    assert cli._read_pid() == 12345
    # A PID far beyond the process range must read as dead without raising.
    assert cli._pid_alive(2**31 - 1) is False
    assert cli._pid_alive(-5) is False

    cli._clear_pid()
    assert cli._read_pid() is None


def test_start_refuses_to_double_start(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(cli, "_read_pid", lambda: 4242)

    assert cli.cmd_start(cli.build_parser().parse_args(["start"])) == 1
    assert "already running" in capsys.readouterr().err


def test_stop_without_a_gateway_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_read_pid", lambda: None)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(cli, "_probe_health", lambda timeout=2.0: None)

    assert cli.cmd_stop(cli.build_parser().parse_args(["stop"])) == 0
    assert "not running" in capsys.readouterr().err


def test_stop_never_signals_a_foreign_process(monkeypatch, capsys):
    """A healthy port with no recorded PID must not be killed by a guess."""
    monkeypatch.setattr(cli, "_read_pid", lambda: None)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(cli, "_probe_health", lambda timeout=1.0: {"status": "healthy"})

    assert cli.cmd_stop(cli.build_parser().parse_args(["stop"])) == 1
    assert "refusing" in capsys.readouterr().err


def test_stop_terminates_the_recorded_pid(monkeypatch):
    killed = []
    monkeypatch.setattr(cli, "_read_pid", lambda: 4711)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: pid == 4711)
    monkeypatch.setattr(cli, "_probe_health", lambda timeout=1.0: None)
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(cli, "_clear_pid", lambda: None)

    assert cli.cmd_stop(cli.build_parser().parse_args(["stop"])) == 0
    assert (4711, 15) in killed  # SIGTERM


# ------------------------------------------------------------------------------
# status
# ------------------------------------------------------------------------------
def test_status_reports_an_unhealthy_gateway(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_probe_health", lambda timeout=2.0: None)
    monkeypatch.setattr(cli, "_read_pid", lambda: None)

    assert cli.cmd_status(cli.build_parser().parse_args(["status"])) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["healthy"] is False


def test_status_reports_a_healthy_gateway(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "_probe_health",
        lambda timeout=2.0: {
            "status": "healthy",
            "version": "0.1.0",
            "upstream": "http://127.0.0.1:8080/v1",
            "classifier_mode": "upstream_reused",
            "profile": "antigravity",
            "env_file": "/x/.env.antigravity",
            "data_dir": str(load_settings().data_dir),
        },
    )
    monkeypatch.setattr(cli, "_read_pid", lambda: 4321)
    monkeypatch.setattr(cli, "_pid_alive", lambda pid: True)

    assert cli.cmd_status(cli.build_parser().parse_args(["status"])) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["healthy"] is True
    assert payload["pid"] == 4321
    assert payload["classifier_mode"] == "upstream_reused"
    assert payload["profile"] == "antigravity"
    assert payload["foreign_service_on_port"] is False


# ------------------------------------------------------------------------------
# stats passthrough
# ------------------------------------------------------------------------------
def test_stats_flags_are_forwarded_to_the_analytics_cli(monkeypatch):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("src.analytics.main", fake_main)

    cli.cmd_stats(
        cli.build_parser().parse_args(["stats", "--json", "--live", "--audit-log", "/a.log"])
    )
    assert seen["argv"] == ["--json", "--live", "--audit-log", "/a.log"]


# ------------------------------------------------------------------------------
# service install
# ------------------------------------------------------------------------------
def test_service_install_writes_a_user_unit(tmp_path, monkeypatch, capsys):
    from pathlib import Path as _Path

    from src import cli as cli_module

    monkeypatch.setenv("HOME", str(tmp_path))

    # Capture the real method BEFORE patching, or the fake recurses into itself.
    original_expanduser = _Path.expanduser

    def fake_expanduser(self):
        text = str(self)
        if text.startswith("~"):
            return _Path(tmp_path) / text.replace("~", "").lstrip("/")
        return original_expanduser(self)

    monkeypatch.setattr(cli_module.Path, "expanduser", fake_expanduser)
    monkeypatch.setattr(
        cli_module.subprocess,
        "run",
        lambda *a, **k: type("R", (), {"returncode": 0})(),
    )

    args = cli.build_parser().parse_args(["service", "install", "--user"])
    assert cli.cmd_service(args) == 0

    unit = (tmp_path / ".config/systemd/user/agent-gateway.service").read_text()
    assert "ExecStart=" in unit
    assert "systemctl" not in unit, "enablement is a command, not unit content"
    assert "WantedBy=default.target" in unit


# ------------------------------------------------------------------------------
# `agent-gateway start` really serves
# ------------------------------------------------------------------------------
async def test_start_daemon_spawns_and_reports_healthy(tmp_path, monkeypatch):
    """The real thing: spawn the daemon against the mock upstream and watch /health."""
    from tests.mock_upstream import start_mock_upstream

    server, _state, base_url = start_mock_upstream()
    try:
        monkeypatch.setenv("UPSTREAM_BASE_URL", base_url)
        monkeypatch.setenv("GATEWAY_PORT", "8398")
        monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

        args = cli.build_parser().parse_args(["start", "--daemon"])
        assert cli.cmd_start(args) == 0

        # The pidfile now points at a live, healthy gateway.
        pid = cli._read_pid()
        assert cli._pid_alive(pid)
        health = cli._probe_health(timeout=2.0)
        assert health and health["status"] == "healthy"

        # And stop tears it down. The user-visible invariant is that the port
        # stops serving; a PID check would be unreliable here because a freshly
        # exited PID can be reused by the next process the machine forks.
        assert cli.cmd_stop(cli.build_parser().parse_args(["stop"])) == 0
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if cli._probe_health(timeout=1.0) is None:
                break
            time.sleep(0.25)
        assert cli._probe_health(timeout=1.0) is None, "the port must stop serving"
        assert cli._read_pid() is None
    finally:
        server.shutdown()
        server.server_close()
