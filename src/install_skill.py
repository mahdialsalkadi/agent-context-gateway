"""Install native Jev tool router skill.

Re-exports core functionality from skill_generator.
"""

from __future__ import annotations

from .skill_generator import (
    DEFAULT_JEV_URL,
    SKILL_MARKDOWN_TEMPLATE,
    SKILL_SCRIPT_TEMPLATE,
    get_default_skill_dir,
    install_skill,
    ping_jev,
)

__all__ = [
    "DEFAULT_JEV_URL",
    "SKILL_MARKDOWN_TEMPLATE",
    "SKILL_SCRIPT_TEMPLATE",
    "get_default_skill_dir",
    "install_skill",
    "ping_jev",
]
