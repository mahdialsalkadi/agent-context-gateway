"""
Terminal presentation helpers.

Everything user-facing renders through here so the CLI speaks one visual
language. Colours are plain ANSI escapes -- no dependency -- and degrade to no
colour at all when the stream is not a TTY or `NO_COLOR` is set (the
convention documented at https://no-color.org/).
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple


# ------------------------------------------------------------------------------
# Colour primitives
# ------------------------------------------------------------------------------
def _supports_colour(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _paint(code: str, text: str, stream) -> str:
    if not _supports_colour(stream):
        return text
    return f"\033[{code}m{text}\033[0m"


def _resolve(stream):
    """Late-bind the default stream.

    Defaulting a parameter to `sys.stderr` binds it at import time, which breaks
    anything that swaps sys.stderr afterwards (pytest's capsys, redirection).
    """
    return stream if stream is not None else sys.stderr


def bold(text: str, stream=None) -> str:
    return _paint("1", text, _resolve(stream))


def red(text: str, stream=None) -> str:
    return _paint("31", text, _resolve(stream))


def green(text: str, stream=None) -> str:
    return _paint("32", text, _resolve(stream))


def yellow(text: str, stream=None) -> str:
    return _paint("33", text, _resolve(stream))


def cyan(text: str, stream=None) -> str:
    return _paint("36", text, _resolve(stream))


def dim(text: str, stream=None) -> str:
    return _paint("2", text, _resolve(stream))


# ------------------------------------------------------------------------------
# Badges, dividers, boxes
# ------------------------------------------------------------------------------
def badge(text: str, kind: str = "ok", stream=None) -> str:
    """[OK] / [WARN] / [FAIL] with colour; the words survive without colour."""
    text = text.upper()
    if kind == "ok":
        return green(f"[{text}]", stream)
    if kind == "warn":
        return yellow(f"[{text}]", stream)
    return red(f"[{text}]", stream)


def divider(stream=None, width: int = 62) -> str:
    return dim("─" * width, stream)


def banner(title: str, stream=None, width: int = 62) -> str:
    return f"{bold(title, stream)}\n{divider(stream, width)}"


def box(lines: Sequence[str], stream=None, width: int = 66) -> List[str]:
    """A simple rounded box around already-coloured lines.

    Colour escapes are zero-width; padding is computed on the visible text by
    asking each line for its plain form, which is why `plain()` exists.
    """
    rendered: List[str] = []
    rendered.append(dim(f"╭{'─' * (width - 2)}╮", stream))
    for line in lines:
        visible = len(plain(line))
        padding = max(0, width - 4 - visible)
        rendered.append(
            f"{dim('│', stream)} {line}{' ' * padding} {dim('│', stream)}"
        )
    rendered.append(dim(f"╰{'─' * (width - 2)}╯", stream))
    return rendered


_ANSI_RE = None


def plain(text: str) -> str:
    """`text` with ANSI escapes removed, for width calculations."""
    global _ANSI_RE
    if _ANSI_RE is None:
        import re

        _ANSI_RE = re.compile(r"\033\[[0-9;]*m")
    return _ANSI_RE.sub("", text)


def kv(key: str, value: str, stream=None, key_width: int = 18) -> str:
    """Aligned `key  value` row."""
    return f"  {dim(key.ljust(key_width), stream)}{value}"


# ------------------------------------------------------------------------------
# Error presentation
# ------------------------------------------------------------------------------
class CliError(Exception):
    """An operational error with a human answer.

    Raised by commands that can predict failure (port in use, unknown profile,
    missing binary). `main()` renders `.message` and its `.hints` instead of a
    traceback; anything else still traceback, because an unexpected bug should
    be loud and reportable.
    """

    def __init__(self, message: str, hints: Optional[Sequence[str]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.hints = list(hints or [])


def render_error(error: CliError, stream=None) -> None:
    """Print a failure the user can act on, with no traceback."""
    stream = _resolve(stream)
    stream.write(f"\n{badge('error', 'fail', stream)} {error.message}\n")
    if error.hints:
        stream.write("\nTry one of these:\n")
        for hint in error.hints:
            stream.write(f"  {cyan('→', stream)} {hint}\n")
    stream.write(f"\n{dim('More diagnosis: agent-gateway doctor', stream)}\n")


def wrap_errors(func):
    """Decorator for `main()` entrypoints: render `CliError`, re-raise the rest."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except CliError as error:
            render_error(error)
            return 1
        except KeyboardInterrupt:
            sys.stderr.write("\n[interrupted]\n")
            return 130

    wrapper.__name__ = getattr(func, "__name__", "wrapped")
    return wrapper
