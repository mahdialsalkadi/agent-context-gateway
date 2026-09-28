"""Native Jev tool router skill generator for subscription environments.

Generates standalone, zero-dependency skill scripts that run inside
Claude Code, Codex, Antigravity, or generic environments, directly querying
the local Vulkan-accelerated Jev model at http://127.0.0.1:11435/v1/chat/completions.
Records routing telemetry directly to ~/.agent-gateway/logs/audit.log.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Optional

DEFAULT_JEV_URL = os.environ.get(
    "JEV_API_BASE_URL",
    os.environ.get("JEV_URL", "http://127.0.0.1:11435/v1/chat/completions"),
)

SKILL_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Jev Semantic Tool Router - Native Skill for Subscriptions.

Zero-proxy, zero-network-interception tool pruning.
Directly queries the local Vulkan-accelerated Jev model at http://127.0.0.1:11435
or configured cloud Jev endpoint.
Emits telemetry to ~/.agent-gateway/logs/audit.log.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set

DEFAULT_JEV_URL = os.environ.get(
    "JEV_API_BASE_URL",
    os.environ.get("JEV_URL", "http://127.0.0.1:11435/v1/chat/completions"),
)
DEFAULT_JEV_API_KEY = os.environ.get("JEV_API_KEY", "")
DEFAULT_AUDIT_LOG = os.environ.get(
    "AGENT_GATEWAY_AUDIT_LOG",
    str(Path.home() / ".agent-gateway" / "logs" / "audit.log"),
)
DEFAULT_AGENT = "{target_agent}"
ROUTER_SYSTEM_PROMPT = (
    "You are a semantic tool router. Given candidate tools and user request/context, "
    "select ONLY the tool names strictly required to fulfill this turn.\\n"
    "Rules:\\n"
    "1. Return ONLY a valid JSON array of tool name strings, e.g. [\\"tool1\\", \\"tool2\\"].\\n"
    "2. If no tools are needed (e.g. conversational questions, explanations, creative writing), output [].\\n"
    "3. Do not include markdown codeblocks, thoughts, or explanations. Only the raw JSON array."
)


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return tool.strip()
    if isinstance(tool, dict):
        if "name" in tool:
            return str(tool["name"])
        if "function" in tool and isinstance(tool["function"], dict):
            return str(tool["function"].get("name", ""))
    return ""


def _tool_description(tool: Any) -> str:
    if isinstance(tool, dict):
        desc = ""
        if "function" in tool and isinstance(tool["function"], dict):
            desc = str(tool["function"].get("description") or "")
        else:
            desc = str(tool.get("description") or "")
        first_line = desc.strip().splitlines()[0] if desc.strip() else ""
        return first_line[:100]
    return ""


def parse_jev_output(text: str, candidate_names: Set[str]) -> List[str]:
    if not text:
        return []
    cleaned = text.strip()
    match = re.search(r"\\[.*?\\]", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                result = []
                for item in parsed:
                    name = str(item).strip().strip("\'\\"`")
                    if name in candidate_names and name not in result:
                        result.append(name)
                return result
        except Exception:
            pass
    tokens = re.findall(r"[a-zA-Z0-9_\\-]+", cleaned)
    result = []
    for token in tokens:
        if token in candidate_names and token not in result:
            result.append(token)
    return result


def log_audit(
    agent: str,
    route: str,
    tools_in: int,
    tools_out: int,
    selected_tools: List[str],
    latency_ms: float,
    audit_file: Optional[str] = None,
) -> None:
    """Append structured telemetry record to gateway audit log."""
    try:
        target_path = Path(audit_file) if audit_file else Path(DEFAULT_AUDIT_LOG)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        now_utc = datetime.now(timezone.utc)
        record = {
            "timestamp": now_utc.isoformat(),
            "agent": agent,
            "route": route,
            "tools_in": tools_in,
            "tools_out": tools_out,
            "selected_tools": selected_tools,
            "latency_ms": latency_ms,
            "ts": now_utc.timestamp(),
            "tools_before": tools_in,
            "tools_after": tools_out,
        }
        with open(target_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\\n")
    except Exception:
        pass


def route_tools(
    prompt: str,
    tools: List[Any],
    jev_url: str = DEFAULT_JEV_URL,
    api_key: Optional[str] = None,
    context: Optional[str] = None,
    agent: str = DEFAULT_AGENT,
    audit_file: Optional[str] = None,
    timeout: float = 5.0,
) -> List[str]:
    """Prune candidate tools to only the minimal subset required for the prompt."""
    start_time = time.perf_counter()
    if not tools:
        if prompt:
            log_audit(
                agent=agent,
                route="Jev-Skill",
                tools_in=0,
                tools_out=0,
                selected_tools=[],
                latency_ms=0.0,
                audit_file=audit_file,
            )
        return []

    candidate_names: Set[str] = set()
    tool_lines: List[str] = []
    for t in tools:
        name = _tool_name(t)
        if not name:
            continue
        candidate_names.add(name)
        desc = _tool_description(t)
        tool_lines.append(f"- {name}: {desc}" if desc else f"- {name}")

    if not candidate_names:
        return []

    tools_summary = "\\n".join(tool_lines)
    prompt_snippet = prompt[:2500] if len(prompt) > 2500 else prompt

    user_content = f"Candidate Tools:\\n{tools_summary}\\n\\n"
    if context:
        user_content += f"Conversation Context:\\n{context[:1500]}\\n\\n"
    user_content += f"User Request: {prompt_snippet}\\n\\nSelected Tools (JSON array):"

    payload = {
        "model": "qwen3.5-2b",
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }

    target_url = jev_url.rstrip("/")
    if not target_url.endswith("/chat/completions"):
        target_url = f"{target_url}/chat/completions"

    headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
    resolved_key = api_key if api_key is not None else DEFAULT_JEV_API_KEY
    if resolved_key and resolved_key != "local":
        headers["Authorization"] = f"Bearer {resolved_key}"

    req = urllib.request.Request(
        target_url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    selected: List[str] = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    selected = parse_jev_output(content, candidate_names)
            else:
                selected = list(candidate_names)
    except Exception as exc:
        sys.stderr.write(f"[jev-router] Warning: routing via Jev failed ({exc}); retaining all tools.\\n")
        selected = list(candidate_names)

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    log_audit(
        agent=agent,
        route="Jev-Skill",
        tools_in=len(candidate_names),
        tools_out=len(selected),
        selected_tools=selected,
        latency_ms=latency_ms,
        audit_file=audit_file,
    )
    return selected


def ping_jev(
    jev_url: str = DEFAULT_JEV_URL,
    api_key: Optional[str] = None,
    timeout: float = 2.0,
) -> bool:
    """Check whether Jev llama-server or API endpoint is healthy."""
    target = jev_url.rstrip("/")
    base = target.rsplit("/v1/", 1)[0] + "/health" if "/v1/" in target else target
    headers = {}
    resolved_key = api_key if api_key is not None else DEFAULT_JEV_API_KEY
    if resolved_key and resolved_key != "local":
        headers["Authorization"] = f"Bearer {resolved_key}"
    req = urllib.request.Request(base, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 404)  # 404 means server is up but no /health
    except Exception:
        # Fallback to GET /v1/models
        try:
            models_url = target.rsplit("/chat/completions", 1)[0] + "/models" if "/chat/completions" in target else f"{target}/models"
            req2 = urllib.request.Request(models_url, headers=headers, method="GET")
            with urllib.request.urlopen(req2, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Jev Semantic Tool Router (Native Skill)")
    parser.add_argument("--prompt", "-p", help="User request or instruction")
    parser.add_argument("--tools", "-t", help="Candidate tools as JSON string or file path")
    parser.add_argument("--context", "-c", help="Optional conversation execution context")
    parser.add_argument("--agent", default=DEFAULT_AGENT, help="Agent identity for telemetry")
    parser.add_argument("--audit-file", default=None, help="Custom audit log path")
    parser.add_argument("--jev-url", default=DEFAULT_JEV_URL, help="Jev endpoint URL")
    parser.add_argument("--api-key", "-k", default=DEFAULT_JEV_API_KEY, help="Jev API key (optional)")
    parser.add_argument("--test", "--ping", action="store_true", help="Ping Jev server")

    args = parser.parse_args()

    if args.test:
        alive = ping_jev(args.jev_url, api_key=args.api_key)
        if alive:
            sys.stdout.write(f"[jev-router] Jev server is ACTIVE at {args.jev_url}\\n")
            return 0
        else:
            sys.stderr.write(f"[jev-router] Jev server is UNREACHABLE at {args.jev_url}\\n")
            return 1

    prompt = args.prompt or ""
    raw_tools = args.tools

    if not raw_tools and not sys.stdin.isatty():
        try:
            stdin_data = json.load(sys.stdin)
            if isinstance(stdin_data, dict):
                prompt = prompt or stdin_data.get("prompt", "")
                raw_tools = stdin_data.get("tools", [])
            elif isinstance(stdin_data, list):
                raw_tools = stdin_data
        except Exception:
            pass

    if isinstance(raw_tools, str):
        if os.path.isfile(raw_tools):
            with open(raw_tools, "r", encoding="utf-8") as f:
                raw_tools = json.load(f)
        else:
            try:
                raw_tools = json.loads(raw_tools)
            except Exception:
                raw_tools = [s.strip() for s in raw_tools.split(",") if s.strip()]

    if not isinstance(raw_tools, list):
        raw_tools = []

    selected = route_tools(
        prompt=prompt,
        tools=raw_tools,
        jev_url=args.jev_url,
        api_key=args.api_key,
        context=args.context,
        agent=args.agent,
        audit_file=args.audit_file,
    )
    sys.stdout.write(json.dumps(selected, indent=2) + "\\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

SKILL_MARKDOWN_TEMPLATE = """---
name: jev-router
description: Semantic tool router powered by local Vulkan-accelerated Jev-2B. Prunes tool candidates to minimal required subset for 0$ token spend. Before executing actions or selecting tools from the catalog, run the jev-router skill to determine the exact minimal tool subset required for the user's turn.
version: 1.0.0
allowed-tools:
  - Bash
metadata:
  icon: router
---

# Jev Semantic Tool Router

Before executing actions or selecting tools from the catalog, run the jev-router skill to determine the exact minimal tool subset required for the user's turn.

## Description
Evaluates the user's request against available candidate tools and returns ONLY the minimal required subset as a JSON array, saving 100% of pruned schema overhead with $0 token spend via local Vulkan-accelerated Jev.

## When to Use
- **Before selecting tools**: Always execute before making actions or picking tools from the catalog to keep token usage lean.
- **Zero API costs**: Evaluated 100% offline via local `llama-server` on Vulkan GPU (`http://127.0.0.1:11435`).
- **Telemetry**: Automatically records routing decisions to gateway telemetry (`~/.agent-gateway/logs/audit.log`).

## Input Parameters
- `--prompt` (`string`, required): The user's prompt or task instruction for the turn.
- `--tools` (`string` or `JSON array`, required): List of tool definitions or tool names available to the agent.
- `--context` (`string`, optional): Recent conversation context or task history.
- `--jev-url` (`string`, optional): Endpoint for Jev (default: `http://127.0.0.1:11435/v1/chat/completions`).
- `--agent` (`string`, optional): Agent name for telemetry (default: `{agent}`).

## Output Format
JSON array containing the filtered list of tool names to keep:
```json
["tool_a", "tool_b"]
```
If no tools are required (e.g. conversational questions, explanations), returns `[]`.

## Execution Schema
```bash
python3 jev-router.py --prompt "<user_prompt>" --tools '[{"name": "...", "description": "..."}]'
```
Or pipe via stdin:
```bash
echo '{"prompt": "check git diff", "tools": [{"name": "run_command"}]}' | python3 jev-router.py
```

## Self-Test
```bash
python3 jev-router.py --test
```
"""


def get_default_skill_dir(target: str = "claude") -> Path:
    target_clean = target.lower().strip()
    if target_clean in ("claude", "claude-code"):
        return Path.home() / ".claude" / "skills"
    elif target_clean in ("codex", "openai-codex"):
        return Path.home() / ".codex" / "skills"
    elif target_clean in ("antigravity", "google-antigravity", "agy"):
        gemini_dir = Path.home() / ".gemini" / "antigravity-cli" / "skills"
        if gemini_dir.parent.exists():
            return gemini_dir
        return Path.home() / ".antigravity" / "skills"
    else:
        return Path.home() / ".agent-gateway" / "skills"


def install_skill(
    target: str = "claude",
    dest_dir: Optional[Path] = None,
    workspace: bool = False,
) -> tuple[Path, Path]:
    """Generate and write the native Jev tool router skill.

    Supports dual installation for Antigravity (global + workspace).
    Returns (script_path, doc_path) of the primary installation.
    """
    target_clean = target.lower().strip()
    agent_id = "antigravity" if target_clean in ("antigravity", "google-antigravity", "agy") else target_clean
    script_content = SKILL_SCRIPT_TEMPLATE.replace("{target_agent}", agent_id)
    doc_content = SKILL_MARKDOWN_TEMPLATE.replace("{agent}", agent_id)

    if dest_dir is not None:
        base_dir = Path(dest_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        script_path = base_dir / "jev-router.py"
        script_path.write_text(script_content, encoding="utf-8")
        script_path.chmod(0o755)

        skill_sub_dir = base_dir / "jev-router"
        skill_sub_dir.mkdir(parents=True, exist_ok=True)
        doc_path = skill_sub_dir / "SKILL.md"
        doc_path.write_text(doc_content, encoding="utf-8")
        sub_script = skill_sub_dir / "jev-router.py"
        sub_script.write_text(script_content, encoding="utf-8")
        sub_script.chmod(0o755)
        return script_path, doc_path

    # Production installation path (dest_dir is None)
    base_dir = get_default_skill_dir(target)
    base_dir.mkdir(parents=True, exist_ok=True)

    script_path = base_dir / "jev-router.py"
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(0o755)

    skill_sub_dir = base_dir / "jev-router"
    skill_sub_dir.mkdir(parents=True, exist_ok=True)
    doc_path = skill_sub_dir / "SKILL.md"
    doc_path.write_text(doc_content, encoding="utf-8")
    sub_script = skill_sub_dir / "jev-router.py"
    sub_script.write_text(script_content, encoding="utf-8")
    sub_script.chmod(0o755)

    # For Antigravity: complete dual installation across global and workspace
    if target_clean in ("antigravity", "google-antigravity", "agy"):
        # 1. Additional global location: ~/.gemini/config/skills/jev-router/
        config_skill_dir = Path.home() / ".gemini" / "config" / "skills" / "jev-router"
        try:
            config_skill_dir.mkdir(parents=True, exist_ok=True)
            (config_skill_dir / "SKILL.md").write_text(doc_content, encoding="utf-8")
            s = config_skill_dir / "jev-router.py"
            s.write_text(script_content, encoding="utf-8")
            s.chmod(0o755)
        except OSError:
            pass

        # 2. Workspace locations (if in a workspace / repository or workspace=True)
        cwd = Path.cwd()
        ws_candidates = [
            cwd / ".gemini" / "skills" / "jev-router",
            cwd / ".agents" / "skills" / "jev-router",
        ]
        for ws_dir in ws_candidates:
            try:
                ws_dir.mkdir(parents=True, exist_ok=True)
                (ws_dir / "SKILL.md").write_text(doc_content, encoding="utf-8")
                ws_script = ws_dir / "jev-router.py"
                ws_script.write_text(script_content, encoding="utf-8")
                ws_script.chmod(0o755)
            except OSError:
                pass

        # 3. Workspace rule trigger: .agents/rules/jev-router.md
        rules_dir = cwd / ".agents" / "rules"
        try:
            rules_dir.mkdir(parents=True, exist_ok=True)
            rule_file = rules_dir / "jev-router.md"
            rule_text = (
                "---\n"
                "trigger: always_on\n"
                "---\n\n"
                "# Jev Semantic Tool Router Instruction\n\n"
                "Before executing actions or selecting tools from the catalog, run the jev-router "
                "skill to determine the exact minimal tool subset required for the user's turn.\n"
            )
            rule_file.write_text(rule_text, encoding="utf-8")
        except OSError:
            pass

    return script_path, doc_path


def ping_jev(
    jev_url: str = DEFAULT_JEV_URL,
    api_key: Optional[str] = None,
    timeout: float = 2.0,
) -> bool:
    """Check whether local Jev llama-server or API endpoint is healthy."""
    import urllib.request
    target = jev_url.rstrip("/")
    base = target.rsplit("/v1/", 1)[0] + "/health" if "/v1/" in target else target
    headers = {}
    if api_key and api_key != "local":
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(base, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 404)
    except Exception:
        try:
            models_url = target.rsplit("/chat/completions", 1)[0] + "/models" if "/chat/completions" in target else f"{target}/models"
            req2 = urllib.request.Request(models_url, headers=headers, method="GET")
            with urllib.request.urlopen(req2, timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False
