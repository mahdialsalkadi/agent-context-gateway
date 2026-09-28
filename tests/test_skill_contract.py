"""Comprehensive skill contract verification tests for Claude Code and Codex subscriptions.

Guarantees standard input/output contracts, metadata validity, zero stdout log leakage,
and subprocess telemetry integration for external agent environments.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


def parse_frontmatter(content: str) -> dict[str, any]:
    """Parse YAML-like frontmatter from markdown file without third-party dependencies."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
    assert match is not None, "SKILL.md must contain frontmatter enclosed by '---'"
    frontmatter_text = match.group(1)

    parsed = {}
    current_list_key = None

    for line in frontmatter_text.splitlines():
        line_clean = line.strip()
        if not line_clean or line_clean.startswith("#"):
            continue

        if line.startswith("  - ") and current_list_key:
            parsed[current_list_key].append(line_clean[2:].strip().strip("\"'"))
            continue

        if ":" in line:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            current_list_key = None

            if not val:
                # Key starting a list or dict
                parsed[key] = []
                current_list_key = key
            elif val.startswith("[") and val.endswith("]"):
                items = [x.strip().strip("\"'") for x in val[1:-1].split(",") if x.strip()]
                parsed[key] = items
            else:
                parsed[key] = val.strip("\"'")

    return parsed


def test_claude_skill_contract(tmp_path):
    """Verify Claude Code native skill installation, contract, and subprocess invocation."""
    dest = tmp_path / "claude_skills"
    audit_file = tmp_path / "claude_logs" / "audit.log"

    # 1. Execute agent-gateway install-skill claude --dest <dest>
    res = subprocess.run(
        [sys.executable, "-m", "src.cli", "install-skill", "claude", "--dest", str(dest)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"install-skill claude failed with stderr: {res.stderr}"
    assert "[Skill Installed]" in res.stdout
    assert "[Skill Ready]" in res.stdout

    # Verify generated artifacts
    script_path = dest / "jev-router.py"
    doc_path = dest / "jev-router" / "SKILL.md"
    assert script_path.is_file(), "Primary executable jev-router.py missing"
    assert doc_path.is_file(), "Descriptor SKILL.md missing"
    assert os.access(script_path, os.X_OK), "jev-router.py must have executable permissions"

    # 2. Verify SKILL.md frontmatter (name, description, allowed tools)
    doc_content = doc_path.read_text(encoding="utf-8")
    frontmatter = parse_frontmatter(doc_content)

    assert frontmatter.get("name") == "jev-router"
    assert "description" in frontmatter and len(frontmatter["description"]) > 10
    assert "version" in frontmatter

    # Allowed tools verification
    allowed_tools = frontmatter.get("allowed-tools") or frontmatter.get("allowed_tools")
    assert allowed_tools is not None, "Frontmatter must declare allowed tools for Claude Code"
    assert isinstance(allowed_tools, list)
    assert any("bash" in t.lower() for t in allowed_tools), "Allowed tools must include Bash execution"

    # 3. Simulate Claude Code agent invoking the script via subprocess with --tools
    candidate_tools = [
        {"name": "view_file", "description": "View file content from filesystem"},
        {"name": "run_command", "description": "Run shell bash commands"},
        {"name": "send_email", "description": "Send email notifications"},
    ]
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--prompt", "Please read config.py and check git status",
            "--tools", json.dumps(candidate_tools),
            "--audit-file", str(audit_file),
            "--agent", "claude",
        ],
        capture_output=True,
        text=True,
    )

    # Assert exit code is strictly 0
    assert proc.returncode == 0, f"Subprocess exited with {proc.returncode}, stderr: {proc.stderr}"

    # Assert stdout is strictly parseable JSON (no debug logs leaking)
    stdout_trimmed = proc.stdout.strip()
    try:
        output_tools = json.loads(stdout_trimmed)
    except json.JSONDecodeError as err:
        pytest.fail(f"Stdout is not strictly valid JSON: {err}. Raw stdout: {proc.stdout}")

    assert isinstance(output_tools, list), "Output must be a JSON array of tool names"
    assert all(isinstance(name, str) for name in output_tools), "All items in array must be strings"
    assert stdout_trimmed == json.dumps(output_tools, indent=2).strip(), (
        "Stdout contains unexpected logs or data outside formatted JSON array"
    )

    # Assert telemetry audit log was recorded
    assert audit_file.is_file(), "Audit telemetry file must be created"
    records = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    assert len(records) >= 1
    last_record = records[-1]
    assert last_record["agent"] == "claude"
    assert last_record["route"] == "Jev-Skill"
    assert last_record["tools_in"] == 3
    assert "latency_ms" in last_record
    assert "timestamp" in last_record

    # 4. Simulate agent invoking script via stdin
    proc_stdin = subprocess.run(
        [sys.executable, str(script_path), "--audit-file", str(audit_file)],
        input=json.dumps({"prompt": "check git diff", "tools": [{"name": "run_command"}]}),
        capture_output=True,
        text=True,
    )
    assert proc_stdin.returncode == 0
    stdin_output = json.loads(proc_stdin.stdout.strip())
    assert isinstance(stdin_output, list)
    assert proc_stdin.stdout.strip() == json.dumps(stdin_output, indent=2).strip()


def test_codex_skill_contract(tmp_path):
    """Verify Codex native skill installation, execution schema, and input/output contracts."""
    dest = tmp_path / "codex_skills"
    audit_file = tmp_path / "codex_logs" / "audit.log"

    # 1. Execute agent-gateway install-skill codex --dest <dest>
    res = subprocess.run(
        [sys.executable, "-m", "src.cli", "install-skill", "codex", "--dest", str(dest)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"install-skill codex failed: {res.stderr}"

    script_path = dest / "jev-router.py"
    doc_path = dest / "jev-router" / "SKILL.md"
    assert script_path.is_file()
    assert doc_path.is_file()
    assert os.access(script_path, os.X_OK)

    # 2. Verify Codex metadata and command execution schema match Codex skill requirements
    doc_content = doc_path.read_text(encoding="utf-8")
    frontmatter = parse_frontmatter(doc_content)
    assert frontmatter.get("name") == "jev-router"
    assert "description" in frontmatter

    # Command Execution Schema and parameters check
    assert "## Execution Schema" in doc_content
    assert "python3 jev-router.py --prompt" in doc_content
    assert "## Input Parameters" in doc_content
    assert "--prompt" in doc_content
    assert "--tools" in doc_content
    assert "## Output Format" in doc_content

    # 3. Subprocess execution with Codex agent identity
    candidate_tools = [
        {"name": "fetch_file", "description": "Fetch source file"},
        {"name": "patch_file", "description": "Patch code file"},
    ]
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--prompt", "Patch the authentication handler",
            "--tools", json.dumps(candidate_tools),
            "--audit-file", str(audit_file),
            "--agent", "codex",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    codex_tools = json.loads(proc.stdout.strip())
    assert isinstance(codex_tools, list)
    assert proc.stdout.strip() == json.dumps(codex_tools, indent=2).strip()

    # Verify audit log reflects codex agent
    assert audit_file.is_file()
    records = [json.loads(l) for l in audit_file.read_text(encoding="utf-8").strip().splitlines() if l.strip()]
    assert len(records) >= 1
    assert records[-1]["agent"] == "codex"
    assert records[-1]["tools_in"] == 2


def test_skill_unreachable_jev_fallback_produces_clean_json(tmp_path):
    """Verify that when Jev is offline, warnings go to stderr and stdout is strictly parseable JSON."""
    dest = tmp_path / "offline_skill"
    audit_file = tmp_path / "offline_logs" / "audit.log"

    subprocess.run(
        [sys.executable, "-m", "src.cli", "install-skill", "claude", "--dest", str(dest)],
        capture_output=True,
        check=True,
    )
    script_path = dest / "jev-router.py"

    candidate_tools = ["tool_alpha", "tool_beta"]
    proc = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--prompt", "perform task",
            "--tools", json.dumps(candidate_tools),
            "--jev-url", "http://127.0.0.1:59999/v1/chat/completions",
            "--audit-file", str(audit_file),
        ],
        capture_output=True,
        text=True,
    )

    # Subprocess must succeed (fail-open)
    assert proc.returncode == 0
    # Warning should be emitted to stderr, NOT stdout
    assert "[jev-router] Warning:" in proc.stderr
    # Stdout must be strictly parseable JSON containing all tools retained safely
    stdout_parsed = json.loads(proc.stdout.strip())
    assert isinstance(stdout_parsed, list)
    assert set(stdout_parsed) == {"tool_alpha", "tool_beta"}
    assert proc.stdout.strip() == json.dumps(stdout_parsed, indent=2).strip()
