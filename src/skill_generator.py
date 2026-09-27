"""Native Jev tool router skill generator for subscription environments.

Generates standalone, zero-dependency skill scripts that run inside
Claude Code, Codex, or generic environments, directly querying the local
Vulkan-accelerated Jev model at http://127.0.0.1:11435/v1/chat/completions.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Optional

DEFAULT_JEV_URL = "http://127.0.0.1:11435/v1/chat/completions"

SKILL_SCRIPT_TEMPLATE = '''#!/usr/bin/env python3
"""Jev Semantic Tool Router - Native Skill for Subscriptions.

Zero-proxy, zero-network-interception tool pruning.
Directly queries the local Vulkan-accelerated Jev model at http://127.0.0.1:11435.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Set

DEFAULT_JEV_URL = os.environ.get("JEV_URL", "http://127.0.0.1:11435/v1/chat/completions")
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


def route_tools(
    prompt: str,
    tools: List[Any],
    jev_url: str = DEFAULT_JEV_URL,
    context: Optional[str] = None,
    timeout: float = 5.0,
) -> List[str]:
    """Prune candidate tools to only the minimal subset required for the prompt."""
    if not tools:
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

    req = urllib.request.Request(
        jev_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    return parse_jev_output(content, candidate_names)
    except Exception as exc:
        sys.stderr.write(f"[jev-router] Warning: routing via Jev failed ({exc}); retaining all tools.\\n")
        return list(candidate_names)

    return list(candidate_names)


def ping_jev(jev_url: str = DEFAULT_JEV_URL, timeout: float = 2.0) -> bool:
    """Check whether local Jev llama-server is healthy."""
    base = jev_url.rsplit("/v1/", 1)[0] + "/health" if "/v1/" in jev_url else jev_url
    req = urllib.request.Request(base, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 404)  # 404 means server is up but no /health
    except Exception:
        # Fallback to GET /v1/models
        try:
            models_url = jev_url.rsplit("/chat/completions", 1)[0] + "/models"
            with urllib.request.urlopen(urllib.request.Request(models_url), timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Jev Semantic Tool Router (Native Skill)")
    parser.add_argument("--prompt", "-p", help="User request or instruction")
    parser.add_argument("--tools", "-t", help="Candidate tools as JSON string or file path")
    parser.add_argument("--context", "-c", help="Optional conversation execution context")
    parser.add_argument("--jev-url", default=DEFAULT_JEV_URL, help="Jev endpoint URL")
    parser.add_argument("--test", "--ping", action="store_true", help="Ping local Jev server")

    args = parser.parse_args()

    if args.test:
        alive = ping_jev(args.jev_url)
        if alive:
            sys.stdout.write(f"[jev-router] Local Jev server is ACTIVE at {args.jev_url}\\n")
            return 0
        else:
            sys.stderr.write(f"[jev-router] Local Jev server is UNREACHABLE at {args.jev_url}\\n")
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

    selected = route_tools(prompt=prompt, tools=raw_tools, jev_url=args.jev_url, context=args.context)
    sys.stdout.write(json.dumps(selected, indent=2) + "\\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

SKILL_MARKDOWN_TEMPLATE = """---
name: jev-router
description: "Semantic tool router powered by local Vulkan-accelerated Jev-2B. Prunes tool candidates to minimal required subset for 0$ token spend."
version: 1.0.0
---

# Jev Semantic Tool Router

Prunes large tool schemas down to the exact subset needed for the current prompt using your local Vulkan-accelerated Jev model.

## Advantages for Subscriptions (Claude Code, Codex)
- **Zero API costs**: Evaluated 100% offline via local `llama-server` on Vulkan GPU (`http://127.0.0.1:11435`).
- **Zero Network Interception**: Does not proxy or break official subscription auth.
- **Context Preservation**: Keeps token usage lean and reduces hallucinated tool calls.

## Usage

### Direct CLI
```bash
python3 jev-router.py --prompt "check git diff of repo" --tools '[{"name": "git_diff", "description": "view changes"}, {"name": "browser", "description": "web browser"}]'
```

### Self-Test
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
    else:
        return Path.home() / ".agent-gateway" / "skills"


def install_skill(
    target: str = "claude",
    dest_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Generate and write the native Jev tool router skill.

    Returns (script_path, doc_path).
    """
    base_dir = Path(dest_dir) if dest_dir else get_default_skill_dir(target)
    base_dir.mkdir(parents=True, exist_ok=True)

    script_path = base_dir / "jev-router.py"
    script_path.write_text(SKILL_SCRIPT_TEMPLATE, encoding="utf-8")
    script_path.chmod(0o755)

    skill_sub_dir = base_dir / "jev-router"
    skill_sub_dir.mkdir(parents=True, exist_ok=True)
    doc_path = skill_sub_dir / "SKILL.md"
    doc_path.write_text(SKILL_MARKDOWN_TEMPLATE, encoding="utf-8")
    return script_path, doc_path


def ping_jev(jev_url: str = DEFAULT_JEV_URL, timeout: float = 2.0) -> bool:
    """Check whether local Jev llama-server is healthy."""
    import urllib.request
    base = jev_url.rsplit("/v1/", 1)[0] + "/health" if "/v1/" in jev_url else jev_url
    req = urllib.request.Request(base, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status in (200, 404)
    except Exception:
        try:
            models_url = jev_url.rsplit("/chat/completions", 1)[0] + "/models"
            with urllib.request.urlopen(urllib.request.Request(models_url), timeout=timeout) as resp:
                return resp.status == 200
        except Exception:
            return False

