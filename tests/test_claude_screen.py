"""Fixture tests for claude_screen state detection.

Each fixture pins an assumption about Claude Code's TUI chrome; if a
Claude Code release changes a marker string, the failing test names the
constant to update in ``claude_screen.py``.
"""

from __future__ import annotations

from code_trip2 import claude_screen
from code_trip2.claude_screen import (
    RUNNING,
    WAITING_INPUT,
    WAITING_PERMISSION,
    detect_state,
    permission_question,
)

RUNNING_SPINNER = """\
● I'll start by reading the config module.

✻ Cogitating… (esc to interrupt · 32s · ↓ 1.2k tokens)
"""

# Claude Code renders the input box even while streaming — the spinner
# marker must win over the prompt box.
RUNNING_WITH_INPUT_BOX = """\
● Working on it.

╭──────────────────────────────────────────╮
│ ❯                                        │
╰──────────────────────────────────────────╯
  ✻ Thinking… (esc to interrupt)
"""

IDLE_PROMPT_BOX = """\
● Done. Tests pass in both files.

╭──────────────────────────────────────────╮
│ ❯                                        │
╰──────────────────────────────────────────╯
  ? for shortcuts
"""

LEGACY_BARE_PROMPT = """\
Claude finished the refactor.

>
"""

PERMISSION_DIALOG = """\
╭─ Bash command ───────────────────────────╮
│ rm -rf /tmp/scratch                      │
│                                          │
│ Do you want to proceed?                  │
│ ❯ 1. Yes                                 │
│   2. Yes, and don't ask again            │
│   3. No, and tell Claude what to do      │
╰──────────────────────────────────────────╯
"""

PLAN_APPROVAL_DIALOG = """\
Would you like to proceed with this plan?

❯ 1. Yes, and auto-accept edits
  2. Yes, and manually approve edits
  3. No, keep planning
"""

SHELL_PROMPT = """\
$ pytest -x
2 passed in 0.41s
$
"""


def test_running_spinner():
    assert detect_state(RUNNING_SPINNER) == RUNNING


def test_running_wins_over_visible_input_box():
    assert detect_state(RUNNING_WITH_INPUT_BOX) == RUNNING


def test_idle_prompt_box_is_waiting_input():
    assert detect_state(IDLE_PROMPT_BOX) == WAITING_INPUT


def test_legacy_bare_prompt_is_waiting_input():
    assert detect_state(LEGACY_BARE_PROMPT) == WAITING_INPUT


def test_permission_dialog_is_waiting_permission():
    assert detect_state(PERMISSION_DIALOG) == WAITING_PERMISSION


def test_permission_question_extracted():
    assert permission_question(PERMISSION_DIALOG) == "Do you want to proceed?"


def test_numbered_options_without_do_you_want():
    assert detect_state(PLAN_APPROVAL_DIALOG) == WAITING_PERMISSION
    assert (
        permission_question(PLAN_APPROVAL_DIALOG)
        == "Would you like to proceed with this plan?"
    )


def test_shell_prompt_defaults_to_running():
    # Fail-quiet: a window where Claude exited to a plain shell must not
    # become a queue task.
    assert detect_state(SHELL_PROMPT) == RUNNING


def test_empty_and_garbage_default_to_running():
    assert detect_state("") == RUNNING
    assert detect_state("   \n\n  ") == RUNNING
    assert detect_state("lorem ipsum\ndolor sit amet") == RUNNING


def test_ansi_wrapped_markers_still_detected():
    running = "\x1b[38;5;10m✻ Thinking…\x1b[0m (esc to interrupt)"
    assert detect_state(running) == RUNNING
    idle = "output\n\x1b[2m│\x1b[0m \x1b[1m❯\x1b[0m \x1b[2m│\x1b[0m"
    assert detect_state(idle) == WAITING_INPUT


def test_running_marker_outside_tail_window_ignored():
    # An old "esc to interrupt" scrolled far up must not mask a current
    # idle prompt.
    old = "✻ Thinking… (esc to interrupt)\n" + "\n".join(
        f"output {i}" for i in range(60)
    )
    assert detect_state(old + "\n❯ ") == WAITING_INPUT


def test_strip_ansi():
    assert claude_screen.strip_ansi("\x1b[31mred\x1b[0m") == "red"
