"""Unit tests for the native Jev tool router skill generator."""

import os
import subprocess
import sys
import json
from pathlib import Path
import pytest

from src.cli import build_parser, cmd_install_skill
from src.skill_generator import get_default_skill_dir, install_skill, ping_jev


def test_install_skill_claude(tmp_path):
    dest = tmp_path / ".claude" / "skills"
    script_path, doc_path = install_skill(target="claude", dest_dir=dest)

    assert script_path.is_file()
    assert doc_path.is_file()
    assert os.access(script_path, os.X_OK), "Script must be executable"

    script_content = script_path.read_text(encoding="utf-8")
    assert "#!/usr/bin/env python3" in script_content
    assert "http://127.0.0.1:11435" in script_content
    assert "route_tools" in script_content

    doc_content = doc_path.read_text(encoding="utf-8")
    assert "name: jev-router" in doc_content


def test_install_skill_codex_and_generic(tmp_path):
    dest_codex = tmp_path / ".codex" / "skills"
    s_codex, d_codex = install_skill(target="codex", dest_dir=dest_codex)
    assert s_codex.is_file()
    assert d_codex.is_file()

    dest_generic = tmp_path / "generic"
    s_gen, d_gen = install_skill(target="generic", dest_dir=dest_generic)
    assert s_gen.is_file()
    assert d_gen.is_file()


def test_installed_skill_execution(tmp_path):
    dest = tmp_path / "skills"
    script_path, _ = install_skill(target="generic", dest_dir=dest)

    # Test running with --help
    res = subprocess.run([sys.executable, str(script_path), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Jev Semantic Tool Router" in res.stdout

    # Test running tool evaluation with mocked Jev or offline fallback
    # When Jev server is unreachable, it should safely return all tools or empty
    res2 = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--prompt", "read git status",
            "--tools", '[{"name": "git_status", "description": "check status"}]',
            "--jev-url", "http://127.0.0.1:99999/v1/chat/completions",
        ],
        capture_output=True,
        text=True,
    )
    assert res2.returncode == 0
    assert "git_status" in res2.stdout


def test_cli_install_skill_command(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "custom_skills"
    parser = build_parser()
    args = parser.parse_args(["install-skill", "claude", "--dest", str(dest)])

    ret = cmd_install_skill(args)
    assert ret == 0

    captured = capsys.readouterr()
    assert "[Skill Installed]" in captured.out
    assert "[Skill Ready]" in captured.out
    assert (dest / "jev-router.py").is_file()


def test_install_skill_antigravity(tmp_path, monkeypatch):
    dest = tmp_path / "antigravity_skills"
    script_path, doc_path = install_skill(target="antigravity", dest_dir=dest)

    assert script_path.is_file()
    assert doc_path.is_file()
    assert os.access(script_path, os.X_OK)

    script_content = script_path.read_text(encoding="utf-8")
    assert "--api-key" in script_content
    assert "JEV_API_BASE_URL" in script_content
    assert "Authorization" in script_content

    # Check default antigravity dir
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    gemini_dir = tmp_path / ".gemini" / "antigravity-cli"
    gemini_dir.mkdir(parents=True, exist_ok=True)
    default_dir = get_default_skill_dir("antigravity")
    assert str(default_dir).endswith(".gemini/antigravity-cli/skills")


def test_cli_install_skill_antigravity(tmp_path, capsys):
    dest = tmp_path / "custom_agy_skills"
    parser = build_parser()
    args = parser.parse_args(["install-skill", "antigravity", "--dest", str(dest)])

    ret = cmd_install_skill(args)
    assert ret == 0
    assert (dest / "jev-router.py").is_file()
    assert (dest / "jev-router" / "SKILL.md").is_file()
    assert (dest / "jev-router" / "jev-router.py").is_file()


def test_antigravity_skill_descriptor_and_telemetry(tmp_path, monkeypatch):
    dest = tmp_path / "antigravity_test"
    audit_file = tmp_path / "logs" / "audit.log"
    script_path, doc_path = install_skill(target="antigravity", dest_dir=dest)

    # 1. Descriptor check
    doc_text = doc_path.read_text(encoding="utf-8")
    assert "Before executing actions or selecting tools from the catalog" in doc_text
    assert "run the jev-router skill to determine the exact minimal tool subset" in doc_text
    assert "name: jev-router" in doc_text

    # 2. Execution and telemetry check
    import subprocess
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--prompt", "list git files",
            "--tools", json.dumps([{"name": "git_ls", "description": "list files"}, {"name": "browser", "description": "web"}]),
            "--audit-file", str(audit_file),
            "--agent", "antigravity",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert audit_file.is_file()
    lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record["agent"] == "antigravity"
    assert record["route"] == "Jev-Skill"
    assert record["tools_in"] == 2
    assert "tools_out" in record
    assert "selected_tools" in record
    assert "latency_ms" in record
    assert "timestamp" in record


def test_antigravity_dual_installation(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path / "workspace")
    (tmp_path / "workspace").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".gemini" / "antigravity-cli").mkdir(parents=True, exist_ok=True)

    script_path, doc_path = install_skill(target="antigravity")

    # Global CLI
    assert (tmp_path / ".gemini" / "antigravity-cli" / "skills" / "jev-router" / "SKILL.md").is_file()
    # Global Config
    assert (tmp_path / ".gemini" / "config" / "skills" / "jev-router" / "SKILL.md").is_file()
    # Workspace Gemini
    assert (tmp_path / "workspace" / ".gemini" / "skills" / "jev-router" / "SKILL.md").is_file()
    # Workspace Agents Rule
    rule = tmp_path / "workspace" / ".agents" / "rules" / "jev-router.md"
    assert rule.is_file()
    assert "run the jev-router skill" in rule.read_text(encoding="utf-8")
