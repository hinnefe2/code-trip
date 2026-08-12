"""Classify a Claude Code pane capture as running vs waiting-for-input.

Pure text heuristics over ``tmux capture-pane`` output, kept in one
module so every assumption about Claude Code's TUI strings is pinned by
a constant here (and a fixture test) rather than scattered through the
producer. When a Claude Code release changes its chrome, this is the
only file to touch.

Precedence, applied to the last ``tail`` lines of the capture:

1. Any live-work marker → RUNNING. Claude Code renders the input box
   even while streaming, so these must win over prompt-box markers.
2. An *active* numbered option dialog → WAITING_PERMISSION.
3. A ``❯`` / ``>`` prompt line → WAITING_INPUT.
4. Otherwise → RUNNING. Fail-quiet: an ambiguous screen (mid-redraw, a
   pager, Claude exited to a ``$`` shell) must not create queue tasks.

Two lessons are baked into the markers below, both learned from a
window that sat in the queue as "waiting" while its Claude was busy
running three subagents:

- **Don't key on the spinner glyph.** It animates (``✶ ✻ ✽ ·``), so a
  capture catches whichever frame was on screen.
- **Don't key on one string.** ``esc to interrupt`` vanished from the
  chrome entirely in the multi-agent releases, which is what caused
  that bug; the live spinner now reads ``Musing… (3m 38s · ↓ 11.4k
  tokens)``. Several independent markers mean one more disappearing
  doesn't silently invert the verdict.

Note that a *finished* turn also leaves a summary line — ``Worked for
1h 13m 21s`` — and a pane commonly shows both that and a live spinner
from the turn after it. The discriminator is the shape: an ellipsis
plus a parenthesized live counter means running; "for <duration>"
means done.
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

# Substring markers that mean work is in flight. "esc to interrupt"
# survives variants like "(esc to interrupt · ctrl+t to hide todos)"
# and is kept for older Claude Code builds; "to run in background" is
# the multi-agent chrome's equivalent hint.
_RUNNING_SUBSTRINGS = (
    "esc to interrupt",
    "to run in background",
)
# Live spinner: "✻ Quantumizing… (10m 0s · ↓ 30.0k tokens)". Matched by
# shape because the glyph animates and the gerund is randomized. The
# parenthesized counter is what separates it from the finished-turn
# summary ("✻ Sautéed for 4m 17s"), which carries no ellipsis.
_LIVE_SPINNER_RE = re.compile(r"…\s*\([^)]*·[^)]*tokens\)")
# Active tool call or subagent fan-out: "● Running 3 agents…",
# "● Running frontend TypeScript check in main checkout · 42s".
_RUNNING_LINE_RE = re.compile(r"^●\s+Running\b")

_PERMISSION_HINT = "Do you want"
# Permission/plan dialogs list numbered options, the first optionally
# cursor-prefixed: "❯ 1. Yes".
_OPTION_1_RE = re.compile(r"^\s*[❯>]?\s*1\.\s+\S")
_OPTION_2_RE = re.compile(r"^\s*2\.\s+\S")
# Idle input box ("│ ❯ ") or the legacy bare "> " prompt. The separator
# after the caret can be a non-breaking space when the user has typed.
_PROMPT_RE = re.compile(r"^\s*[❯>](\s|$)")

# A dialog is live UI: it sits at the bottom of the pane, in place of
# the input box. Numbered lists further up are prose — Claude writing
# "1. Land the PR. 2. Delete the legacy predicates." is not a question,
# and reading it as one put a permission-flavored task in the queue.
_DIALOG_TAIL = 15
# How far below option 1 option 2 may sit and still be the same list.
_OPTION_GAP = 3


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


def _is_running(lines: list[str]) -> bool:
    """True if any line says work is in flight right now."""
    for line in lines:
        if any(marker in line for marker in _RUNNING_SUBSTRINGS):
            return True
        if _LIVE_SPINNER_RE.search(line) or _RUNNING_LINE_RE.match(line):
            return True
    return False


def _dialog_option_index(lines: list[str]) -> int | None:
    """Index of option 1 of an *active* dialog, or ``None``.

    Three conditions separate a live dialog from numbered prose, and
    all three have to hold:

    - the options sit inside the bottom ``_DIALOG_TAIL`` lines, where
      the live UI lives;
    - option 2 follows option 1 closely, as a rendered list does;
    - no input prompt appears below them. Claude Code shows the dialog
      *instead of* the input box, so a live ``❯`` underneath means the
      user is free to type and nothing is being asked. (Option lines
      can themselves start with ``❯``, so those don't count.)
    """
    window_start = max(0, len(lines) - _DIALOG_TAIL)
    for i in range(len(lines) - 1, window_start - 1, -1):
        if not _OPTION_1_RE.match(lines[i]):
            continue
        if not any(
            _OPTION_2_RE.match(l) for l in lines[i + 1 : i + 1 + _OPTION_GAP]
        ):
            continue
        if any(
            _PROMPT_RE.match(l) and not _OPTION_1_RE.match(l)
            for l in lines[i + 1 :]
        ):
            continue
        return i
    return None


def _hint_index(lines: list[str]) -> int | None:
    """Index of a "Do you want …" line that is asking, not narrating.

    Same bottom-of-pane rule as :func:`_dialog_option_index`: the
    phrase turns up in ordinary prose often enough that finding it 30
    lines up is no evidence of a question.
    """
    window_start = max(0, len(lines) - _DIALOG_TAIL)
    for i in range(len(lines) - 1, window_start - 1, -1):
        if _PERMISSION_HINT not in lines[i]:
            continue
        if any(_PROMPT_RE.match(l) for l in lines[i + 1 :]):
            continue
        return i
    return None


def detect_state(text: str, *, tail: int = 40) -> str:
    lines = _tail_lines(text, tail)
    if _is_running(lines):
        return RUNNING
    if _dialog_option_index(lines) is not None or _hint_index(lines) is not None:
        return WAITING_PERMISSION
    if any(_PROMPT_RE.match(line) for line in lines):
        return WAITING_INPUT
    return RUNNING


def permission_question(text: str, *, tail: int = 40) -> str | None:
    """The question a permission dialog is asking, for headlines.

    Prefers the "Do you want …" line; falls back to the nearest
    non-empty line above the ``1.`` option (plan-approval dialogs phrase
    the question differently). Returns ``None`` when no *active* dialog
    is on screen, so a headline is never built from stray prose.
    """
    lines = _tail_lines(text, tail)
    hint = _hint_index(lines)
    if hint is not None:
        return lines[hint]
    option = _dialog_option_index(lines)
    if option is not None and option > 0:
        return lines[option - 1]
    return None
