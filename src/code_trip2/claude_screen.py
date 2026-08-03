"""Classify a Claude Code pane capture as running vs waiting-for-input.

Pure text heuristics over ``tmux capture-pane`` output, kept in one
module so every assumption about Claude Code's TUI strings is pinned by
a constant here (and a fixture test) rather than scattered through the
producer. When a Claude Code release changes its chrome, this is the
only file to touch.

Precedence, applied to the last ``tail`` lines of the capture:

1. ``esc to interrupt`` anywhere → RUNNING. Claude Code renders the
   input box even while streaming, so the spinner marker must win over
   prompt-box markers.
2. A numbered option list (``1.`` and ``2.`` lines) or a "Do you want"
   question → WAITING_PERMISSION.
3. A ``❯`` / ``>`` prompt line → WAITING_INPUT.
4. Otherwise → RUNNING. Fail-quiet: an ambiguous screen (mid-redraw, a
   pager, Claude exited to a ``$`` shell) must not create queue tasks.
"""

from __future__ import annotations

import re

RUNNING = "running"
WAITING_INPUT = "waiting_input"
WAITING_PERMISSION = "waiting_permission"

# capture-pane -e emits SGR (color/style) sequences only, but strip the
# full CSI family so a stray cursor code can't hide a marker.
_ANSI_RE = re.compile(r"\x1b\[[0-9;:?]*[a-zA-Z]")

_BOX_CHARS = "│┃║╭╮╰╯├┤─━┌┐└┘"

# Survives variants like "(esc to interrupt · ctrl+t to hide todos)".
_RUNNING_MARKER = "esc to interrupt"
_PERMISSION_HINT = "Do you want"
# Permission/plan dialogs list numbered options, the first optionally
# cursor-prefixed: "❯ 1. Yes".
_OPTION_1_RE = re.compile(r"^\s*[❯>]?\s*1\.\s+\S")
_OPTION_2_RE = re.compile(r"^\s*2\.\s+\S")
# Idle input box ("│ ❯ ") or the legacy bare "> " prompt.
_PROMPT_RE = re.compile(r"^\s*[❯>](\s|$)")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _tail_lines(text: str, tail: int) -> list[str]:
    """Last ``tail`` non-blank lines, ANSI-stripped, box borders removed."""
    lines = []
    for raw in strip_ansi(text).splitlines():
        line = raw.strip().strip(_BOX_CHARS).strip()
        if line:
            lines.append(line)
    return lines[-tail:]


def detect_state(text: str, *, tail: int = 40) -> str:
    lines = _tail_lines(text, tail)
    if any(_RUNNING_MARKER in line for line in lines):
        return RUNNING
    has_options = any(_OPTION_1_RE.match(line) for line in lines) and any(
        _OPTION_2_RE.match(line) for line in lines
    )
    if has_options or any(_PERMISSION_HINT in line for line in lines):
        return WAITING_PERMISSION
    if any(_PROMPT_RE.match(line) for line in lines):
        return WAITING_INPUT
    return RUNNING


def permission_question(text: str, *, tail: int = 40) -> str | None:
    """The question a permission dialog is asking, for headlines.

    Prefers the "Do you want …" line; falls back to the nearest
    non-empty line above the ``1.`` option (plan-approval dialogs phrase
    the question differently).
    """
    lines = _tail_lines(text, tail)
    for line in lines:
        if _PERMISSION_HINT in line:
            return line
    for i, line in enumerate(lines):
        if _OPTION_1_RE.match(line) and i > 0:
            return lines[i - 1]
    return None
