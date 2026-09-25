"""
The UX layer of the CLI: `doctor`, `init` and `run`.

The pieces that make decisions are pure (`build_init_preset`,
`build_agent_env`, `run_checks`) and are tested directly; the interactive
wizard is driven through an injected input function; `run` is tested up to the
subprocess boundary (environment construction and argv, without launching a
real agent).
"""

from __future__ import annotations

import json
import os

import pytest

from src import cli
from src.cli import (
    build_agent_command,
    build_agent_env,
    build_init_preset,
    build_wizard_preset,
    cmd_doctor,
    cmd_init,
    cmd_interactive,
    install_global_wrapper,
    render_doctor,
    render_env_file,
    run_checks,
    update_env_file,
)
from src.ux import CliError


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    saved = dict(os.environ)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("GATEWAY_PORT", "8399")
    monkeypatch.setenv("UPSTREAM_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("UPSTREAM_API_KEY", "k")
    # A profile leaked from another test would outrank .env entirely and make
    # preset-loading tests pass for the wrong reason.
    monkeypatch.delenv("AGENT_GATEWAY_PROFILE", raising=False)
    monkeypatch.chdir(tmp_path)
    yield
    os.environ.clear()
    os.environ.update(saved)


# ------------------------------------------------------------------------------
# init: presets
# ------------------------------------------------------------------------------
def test_antigravity_preset_relaxes_the_loop_guard_and_reuses_gemini():
    preset = build_init_preset("custom", "antigravity", 8091)

    assert preset["UPSTREAM_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert preset["ALLOW_LEGACY_UPSTREAM_PORT"] == "1"
    assert preset["CLASSIFIER_MODE"] == "upstream_reused"
    assert preset["CLASSIFIER_MODEL"] == "gemini-2.5-flash"
    assert preset["GATEWAY_PORT"] == "8091", "must not collide with the bridge on 8080"


def test_ollama_preset_is_fully_local():
    preset = build_init_preset("aider", "ollama", 8090)

    assert preset["UPSTREAM_BASE_URL"] == "http://127.0.0.1:11434/v1"
    assert preset["CLASSIFIER_MODE"] == "local_ollama"
    assert "UPSTREAM_API_KEY" not in preset, "a local runner needs no key"


def test_openrouter_preset_requires_a_key():
    preset = build_init_preset("hermes", "openrouter", 8090, api_key="sk-or-x")

    assert preset["UPSTREAM_API_KEY"] == "sk-or-x"
    assert preset["CLASSIFIER_MODE"] == "upstream_reused"


def test_custom_preset_without_a_key_writes_a_placeholder():
    preset = build_init_preset("custom", "custom", 8090)

    assert preset["UPSTREAM_API_KEY"] == "REPLACE_ME"
    assert preset["CLASSIFIER_MODE"] == "heuristics"


def test_claude_preset_marks_the_anthropic_surface():
    assert build_init_preset("claude", "ollama", 8090)["ANTHROPIC_SURFACE"] == "1"
    assert "ANTHROPIC_SURFACE" not in build_init_preset("aider", "ollama", 8090)


def test_render_env_file_is_loadable_by_the_gateway(tmp_path, monkeypatch):
    # The fixture sets UPSTREAM_BASE_URL; a .env file must not override a real
    # environment variable, so clear it to prove the file supplies the value.
    monkeypatch.delenv("UPSTREAM_BASE_URL", raising=False)
    text = render_env_file(
        build_init_preset("custom", "ollama", 8090), header="generated"
    )
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")

    from src.config import load_settings

    settings = load_settings(env_file=path)

    assert settings.upstream_base_url == "http://127.0.0.1:11434/v1"
    assert settings.effective_classifier_mode == "local_ollama"


# ------------------------------------------------------------------------------
# init: the interactive flow
# ------------------------------------------------------------------------------
def test_init_writes_env_from_answers(tmp_path, monkeypatch, capsys):
    answers = iter(["1", "2"])  # claude, ollama

    args = cli.build_parser().parse_args(["init", "--force"])
    args.input_fn = lambda _prompt: next(answers)

    assert cmd_init(args) == 0
    text = (tmp_path / ".env").read_text(encoding="utf-8")

    assert "CLASSIFIER_MODE=local_ollama" in text
    assert "UPSTREAM_BASE_URL=http://127.0.0.1:11434/v1" in text
    assert "agent: claude" in text


def test_init_preselects_a_detected_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "detect_local_services", lambda: {"ollama": {"port": 11434, "status": 200}})
    seen = {}

    def fake_input(prompt):
        seen[prompt] = prompt
        return "2" if "backend" in str(seen) and False else ""

    # First question (agent) -> "1"; second (backend) -> "" (takes the default).
    answers = iter(["1", ""])
    args = cli.build_parser().parse_args(["init", "--force"])
    args.input_fn = lambda _p: next(answers)

    assert cmd_init(args) == 0
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "CLASSIFIER_MODE=local_ollama" in text


def test_init_refuses_to_clobber_without_force(tmp_path):
    (tmp_path / ".env").write_text("GATEWAY_PORT=8090\n", encoding="utf-8")

    args = cli.build_parser().parse_args(["init"])
    args.input_fn = lambda _p: "1"

    with pytest.raises(CliError) as excinfo:
        cmd_init(args)
    assert "already exists" in excinfo.value.message
    assert any("--force" in hint for hint in excinfo.value.hints)


def test_init_re_prompts_on_a_bad_selection(tmp_path):
    answers = iter(["9", "1", "2"])  # invalid, then claude, then ollama
    args = cli.build_parser().parse_args(["init", "--force"])
    args.input_fn = lambda _p: next(answers)

    assert cmd_init(args) == 0
    assert "CLASSIFIER_MODE=local_ollama" in (tmp_path / ".env").read_text()


def test_init_never_writes_the_key_into_the_completion_box(tmp_path, capsys):
    answers = iter(["1", "3", "sk-or-secret-value"])
    args = cli.build_parser().parse_args(["init", "--force"])
    args.input_fn = lambda _p: next(answers)

    assert cmd_init(args) == 0
    err = capsys.readouterr().err
    # The key belongs in the file, not echoed back in the celebration box.
    assert "sk-or-secret-value" not in err
    assert "sk-or-secret-value" in (tmp_path / ".env").read_text()


# ------------------------------------------------------------------------------
# doctor
# ------------------------------------------------------------------------------
def test_doctor_reports_each_layer(monkeypatch, tmp_path):
    checks = {c[0]: c for c in run_checks()}

    assert "gateway" in checks
    assert "upstream" in checks
    assert "classifier" in checks
    assert "database" in checks
    assert "shared memory" in checks
    assert "agent binaries" in checks
    for _name, status, detail, _hint in checks.values():
        assert status in ("ok", "warn", "fail")
        assert detail


def test_doctor_flags_a_dead_gateway_as_warn(monkeypatch):
    monkeypatch.setattr(cli, "_probe_health", lambda timeout=1.5: None)
    monkeypatch.setattr(cli, "_read_pid", lambda: None)

    checks = {c[0]: c for c in run_checks()}
    name, status, _detail, hint = checks["gateway"]
    assert status == "warn"
    assert "start" in hint.lower()


def test_doctor_flags_a_foreign_service_as_fail(monkeypatch):
    monkeypatch.setattr(
        cli, "_probe_health", lambda timeout=1.5: {"status": "healthy", "data_dir": "/somewhere/else"}
    )

    checks = {c[0]: c for c in run_checks()}
    _name, status, detail, hint = checks["gateway"]

    assert status == "fail"
    assert "foreign" in detail
    assert "--port" in hint


def test_doctor_identifies_our_own_gateway_as_ok(monkeypatch, tmp_path):
    from src.config import load_settings

    settings = load_settings()
    monkeypatch.setattr(
        cli,
        "_probe_health",
        lambda timeout=1.5: {"status": "healthy", "version": "0.1.0", "data_dir": str(settings.data_dir)},
    )

    checks = {c[0]: c for c in run_checks()}
    _name, status, detail, _hint = checks["gateway"]

    assert status == "ok"
    assert "healthy" in detail


def test_doctor_json_is_machine_readable(monkeypatch, capsys):
    args = cli.build_parser().parse_args(["doctor", "--json"])
    cmd_doctor(args)

    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload
    for row in payload:
        assert set(row) == {"check", "status", "detail", "fix"}


def test_render_doctor_includes_fix_hints_for_failures():
    checks = [("gateway", "fail", "down", "Run: agent-gateway start")]
    text = render_doctor(checks)

    assert "[FAIL]" in text
    assert "fix: Run: agent-gateway start" in text
    assert "1 failed" in text


def test_doctor_exit_code_reflects_failures(monkeypatch, capsys):
    monkeypatch.setattr(
        cli,
        "run_checks",
        lambda: [("gateway", "fail", "down", "hint"), ("database", "ok", "fine", "")],
    )
    assert cmd_doctor(cli.build_parser().parse_args(["doctor"])) == 1

    monkeypatch.setattr(
        cli,
        "run_checks",
        lambda: [("gateway", "warn", "meh", "hint"), ("database", "ok", "fine", "")],
    )
    assert cmd_doctor(cli.build_parser().parse_args(["doctor"])) == 0


# ------------------------------------------------------------------------------
# run: env injection and argv
# ------------------------------------------------------------------------------
def test_claude_run_injects_anthropic_base_url():
    env = build_agent_env("claude", 8091)
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8091"
    # An origin, not a path: Claude Code appends /v1/messages itself.
    assert not env["ANTHROPIC_BASE_URL"].endswith("/v1")


def test_aider_run_injects_the_api_base_flag():
    env = build_agent_env("aider", 8091)
    assert env["OPENAI_API_KEY"] == "dummy"

    argv = build_agent_command("aider", ["--model", "gpt-4o"], 8091)
    assert "--openai-api-base" in argv
    assert "http://127.0.0.1:8091/v1" in argv
    assert argv[-2:] == ["--model", "gpt-4o"], "extra args stay at the end"


def test_hermes_run_injects_openai_base_url():
    assert build_agent_env("hermes", 8090)["OPENAI_BASE_URL"] == "http://127.0.0.1:8090/v1"


def test_unknown_agent_has_no_spec():
    assert build_agent_env("emacs", 8090) == {}
    assert build_agent_command("emacs", [], 8090) == []


def test_run_rejects_unknown_agents_with_a_hint():
    with pytest.raises(CliError) as excinfo:
        cli.cmd_run(cli.build_parser().parse_args(["run", "emacs"]))
    assert "unknown agent" in excinfo.value.message
    assert any("claude" in hint for hint in excinfo.value.hints)


def test_run_rejects_a_missing_binary_without_a_traceback():
    import shutil

    with pytest.raises(CliError) as excinfo:
        cli.cmd_run(
            cli.build_parser().parse_args(
                ["run", "claude", "--no such binary--"] if shutil.which("claude") else ["run", "claude"]
            )
        )
    message = excinfo.value.message
    assert "not installed" in message or "not on PATH" in message


# ------------------------------------------------------------------------------
# graceful errors: no tracebacks on predicted failures
# ------------------------------------------------------------------------------
def test_cli_error_is_rendered_without_a_traceback(capsys):
    from src.ux import render_error

    render_error(CliError("port 8090 is in use", ["start --port 8091", "doctor"]))
    err = capsys.readouterr().err

    assert "port 8090 is in use" in err
    assert "start --port 8091" in err
    assert "Traceback" not in err


def test_main_renders_cli_errors_as_guidance(monkeypatch, capsys):
    def explode(_args):
        raise CliError("the upstream is unreachable", ["check UPSTREAM_BASE_URL"])

    monkeypatch.setattr(cli, "cmd_status", explode)
    code = cli.main(["status"])

    assert code == 1
    err = capsys.readouterr().err
    assert "the upstream is unreachable" in err
    assert "Traceback" not in err


def test_unexpected_errors_still_traceback(monkeypatch):
    """A real bug must stay loud -- fail-safe ergonomics must not hide it."""
    def explode(_args):
        raise RuntimeError("genuine bug")

    monkeypatch.setattr(cli, "cmd_status", explode)
    with pytest.raises(RuntimeError):
        cli.main(["status"])


# ------------------------------------------------------------------------------
# Global wrapper: one command from any directory, any shell
# ------------------------------------------------------------------------------
def test_install_global_wrapper_writes_an_executable_shim(tmp_path):
    path = install_global_wrapper(
        repo_dir=tmp_path,
        bin_dir=tmp_path / "bin",
        venv_python=tmp_path / "bin" / "python",
    )

    assert path == tmp_path / "bin" / "agent-gateway"
    assert os.access(path, os.X_OK), "the wrapper must be executable"
    text = path.read_text(encoding="utf-8")
    assert "-m src.cli" in text
    assert "exec" in text
    assert str(tmp_path) in text, "the wrapper pins this project's directory"
    assert "PYTHONPATH" in text, "must work without `cd`"


def test_install_shim_subcommand_is_wired():
    args = cli.build_parser().parse_args(["install-shim"])
    assert getattr(args, "func", None)


def test_path_guidance_mentions_fish_and_bash():
    guidance = cli.path_guidance("/x/bin")
    assert "fish" in guidance
    assert "bash" in guidance.lower()


# ------------------------------------------------------------------------------
# Interactive launcher
# ------------------------------------------------------------------------------
def test_wizard_preset_selects_local_jev_and_preserves_upstream():
    preset = build_wizard_preset(
        "claude",
        "local_jev",
        8091,
        {"UPSTREAM_BASE_URL": "https://api.example/v1", "UPSTREAM_API_KEY": "k"},
    )

    assert preset["CLASSIFIER_MODE"] == "local_jev"
    assert preset["GATEWAY_PORT"] == "8091"
    assert preset["UPSTREAM_BASE_URL"] == "https://api.example/v1"
    assert preset["UPSTREAM_API_KEY"] == "k"
    assert preset["ANTHROPIC_SURFACE"] == "1"
    assert preset["LOCAL_JEV_URL"].startswith("http://127.0.0.1:11435")


def test_wizard_preset_standalone_has_no_surface_flag():
    preset = build_wizard_preset("standalone", "heuristics", 8090)
    assert "ANTHROPIC_SURFACE" not in preset
    assert preset["CLASSIFIER_MODE"] == "heuristics"


def test_wizard_lists_other_providers_and_their_endpoints():
    kinds = {key for _n, key, _l in cli.WIZARD_UPSTREAMS}
    assert {"openrouter", "openai", "groq", "antigravity", "ollama", "custom"} <= kinds
    assert cli.WIZARD_UPSTREAM_URLS["groq"] == "https://api.groq.com/openai/v1"


def test_wizard_preset_honours_a_chosen_endpoint_and_key():
    preset = build_wizard_preset(
        "aider",
        "heuristics",
        8090,
        upstream_url="https://api.groq.com/openai/v1",
        api_key="gsk_example",
    )

    assert preset["UPSTREAM_BASE_URL"] == "https://api.groq.com/openai/v1"
    assert preset["UPSTREAM_API_KEY"] == "gsk_example"
    assert "ALLOW_LEGACY_UPSTREAM_PORT" not in preset


def test_wizard_preset_relaxes_the_loop_guard_for_a_bridge():
    preset = build_wizard_preset(
        "hermes", "upstream_reused", 8091,
        upstream_url="http://127.0.0.1:8080/v1",
    )
    assert preset["ALLOW_LEGACY_UPSTREAM_PORT"] == "1"


def test_wizard_preset_carries_the_external_classifier_endpoint():
    preset = build_wizard_preset(
        "hermes", "external_jev", 8090,
        classifier_url="https://jev.example/v1/chat/completions",
        classifier_key="jev-key",
    )
    assert preset["CLASSIFIER_API_URL"] == "https://jev.example/v1/chat/completions"
    assert preset["CLASSIFIER_API_KEY"] == "jev-key"


def test_update_env_file_merges_without_duplicating_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# keep me\nGATEWAY_PORT=8090\nUPSTREAM_BASE_URL=http://old/v1\n",
        encoding="utf-8",
    )

    update_env_file(env, {"GATEWAY_PORT": "8091", "CLASSIFIER_MODE": "local_jev"})
    text = env.read_text(encoding="utf-8")

    assert "# keep me" in text
    assert text.count("GATEWAY_PORT=") == 1
    assert "GATEWAY_PORT=8091" in text
    assert "UPSTREAM_BASE_URL=http://old/v1" in text
    assert "CLASSIFIER_MODE=local_jev" in text


def test_suggest_port_prefers_8091_when_8080_is_busy(monkeypatch):
    monkeypatch.setattr(
        cli, "port_in_use", lambda port, host="127.0.0.1": port in (8080, 8090)
    )
    assert cli.suggest_port() == 8091


def test_suggest_port_keeps_8090_when_everything_is_free(monkeypatch):
    monkeypatch.setattr(cli, "port_in_use", lambda port, host="127.0.0.1": False)
    assert cli.suggest_port() == 8090


def test_cmd_interactive_writes_env_from_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "port_in_use", lambda port, host="127.0.0.1": False)
    monkeypatch.setattr(
        cli, "install_global_wrapper", lambda *a, **k: tmp_path / "bin" / "agent-gateway"
    )

    args = cli.build_parser().parse_args(["interactive", "--no-launch"])
    # Claude Code, local_jev, local Ollama provider, default port
    answers = iter(["2", "2", "5", ""])
    args.input_fn = lambda _prompt: next(answers)

    assert cmd_interactive(args) == 0
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "CLASSIFIER_MODE=local_jev" in text
    assert "ANTHROPIC_SURFACE=1" in text
    assert "UPSTREAM_BASE_URL=http://127.0.0.1:11434/v1" in text


def test_cmd_interactive_accepts_a_custom_provider_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "port_in_use", lambda port, host="127.0.0.1": False)
    monkeypatch.setattr(
        cli, "install_global_wrapper", lambda *a, **k: tmp_path / "bin" / "agent-gateway"
    )

    args = cli.build_parser().parse_args(["interactive", "--no-launch"])
    # Aider, upstream_reused, custom provider, URL, key, port
    answers = iter(["3", "3", "6", "https://my.gateway/v1", "sk-custom", ""])
    args.input_fn = lambda _prompt: next(answers)

    assert cmd_interactive(args) == 0
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "UPSTREAM_BASE_URL=https://my.gateway/v1" in text
    assert "UPSTREAM_API_KEY=sk-custom" in text


def test_cmd_interactive_external_jev_prompts_for_the_classifier_endpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(cli, "port_in_use", lambda port, host="127.0.0.1": False)
    monkeypatch.setattr(
        cli, "install_global_wrapper", lambda *a, **k: tmp_path / "bin" / "agent-gateway"
    )

    args = cli.build_parser().parse_args(["interactive", "--no-launch"])
    # Hermes, external_jev, local Ollama upstream, classifier URL, key, port
    answers = iter(["1", "5", "5", "https://jev.example/v1", "jev-key", ""])
    args.input_fn = lambda _prompt: next(answers)

    assert cmd_interactive(args) == 0
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "CLASSIFIER_API_URL=https://jev.example/v1" in text
    assert "CLASSIFIER_API_KEY=jev-key" in text


def test_cmd_interactive_standalone_starts_the_gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "port_in_use", lambda port, host="127.0.0.1": False)
    monkeypatch.setattr(
        cli, "install_global_wrapper", lambda *a, **k: tmp_path / "bin" / "agent-gateway"
    )
    started = {}
    monkeypatch.setattr(
        cli, "_ensure_gateway_running", lambda: started.setdefault("up", True)
    )

    args = cli.build_parser().parse_args(["interactive"])
    # standalone, heuristics, local Ollama provider, default port
    answers = iter(["4", "1", "5", ""])
    args.input_fn = lambda _prompt: next(answers)

    assert cmd_interactive(args) == 0
    assert started.get("up") is True
    assert "CLASSIFIER_MODE=heuristics" in (tmp_path / ".env").read_text()
