"""Fixture tests for claude_screen state detection.

Each fixture pins an assumption about Claude Code's TUI chrome; if a
Claude Code release changes a marker string, the failing test names the
constant to update in ``claude_screen.py``.
"""

from __future__ import annotations

from pathlib import Path

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


# --- captured-pane fixtures ------------------------------------------------
#
# Real ``tmux capture-pane`` output from a remote Claude Code session,
# with the chrome preserved verbatim and the work prose replaced by
# filler. These pin the multi-agent-era TUI, which dropped the
# ``esc to interrupt`` marker the detector used to rely on — a window
# running three subagents was reported as waiting-for-input, and sat in
# the queue as a task the user hadn't been asked for.

_PANES = Path(__file__).parent / "fixtures" / "panes"


def _pane(name: str) -> str:
    return (_PANES / name).read_text(encoding="utf-8")


def test_capture_running_subagents_is_not_waiting():
    """The regression: live spinner + empty input box + no interrupt hint."""
    text = _pane("running_subagents.txt")
    assert "esc to interrupt" not in text     # the marker really is gone
    assert detect_state(text) == RUNNING


def test_capture_running_tool_call_is_not_waiting():
    assert detect_state(_pane("running_tool_call.txt")) == RUNNING


def test_capture_numbered_prose_is_not_a_permission_dialog():
    """Claude writing "1. … 2. …" is not asking the user to choose."""
    text = _pane("waiting_numbered_prose.txt")
    assert detect_state(text) == WAITING_INPUT
    assert permission_question(text) is None


def test_capture_finished_turn_is_waiting_input():
    assert detect_state(_pane("waiting_after_finished_turn.txt")) == WAITING_INPUT


def test_capture_done_marker_with_trailing_clause_is_waiting_input():
    assert detect_state(
        _pane("waiting_done_marker_with_suffix.txt")
    ) == WAITING_INPUT


def test_finished_and_live_markers_coexist_and_running_wins():
    """A pane routinely shows a finished turn's summary above the next
    turn's live spinner; the summary must not decide the verdict."""
    text = _pane("running_subagents.txt")
    assert "Worked for 1h 13m 21s" in text     # finished-turn summary
    assert "Musing… (3m 38s" in text           # live spinner below it
    assert detect_state(text) == RUNNING


# --- marker robustness -----------------------------------------------------


def test_spinner_glyph_animation_does_not_change_the_verdict():
    """The spinner cycles frames, so a capture catches an arbitrary one."""
    for glyph in ("✶", "✻", "✽", "·"):
        line = f"{glyph} Pondering… (12s · ↓ 3.1k tokens)\n\n❯\n"
        assert detect_state(line) == RUNNING, glyph


def test_finished_turn_summary_alone_is_not_running():
    for done in (
        "✻ Sautéed for 4m 17s",
        "✻ Worked for 1h 13m 21s",
        "✻ Brewed for 2m 58s · 1 monitor still running",
    ):
        assert detect_state(f"{done}\n\n❯\n") == WAITING_INPUT, done


def test_background_hint_and_running_line_are_running_markers():
    assert detect_state(
        "(ctrl+b ctrl+b (twice) to run in background)\n❯\n"
    ) == RUNNING
    assert detect_state("● Running 3 agents…\n❯\n") == RUNNING
    assert detect_state("● Running type check · 42s\n❯\n") == RUNNING


def test_typed_text_after_nbsp_caret_is_waiting_input():
    # Claude Code separates the caret from typed text with U+00A0.
    assert detect_state("✻ Crunched for 56s\n\n❯ go ahead\n") == WAITING_INPUT


# --- dialog vs prose -------------------------------------------------------


def test_options_far_above_a_live_prompt_are_prose():
    prose = (
        "1. First option-looking line\n"
        "2. Second option-looking line\n"
        + "\n".join(f"more output {i}" for i in range(12))
        + "\n❯\n"
    )
    assert detect_state(prose) == WAITING_INPUT


def test_prompt_below_options_means_the_box_is_live():
    """A real dialog replaces the input box; a caret under the options
    means the user can type, so nothing is being asked."""
    text = "Do you want to proceed?\n❯ 1. Yes\n  2. No\n❯\n"
    assert detect_state(text) == WAITING_INPUT


def test_do_you_want_in_prose_is_not_a_question():
    text = (
        "● I checked the config. Do you want the summary in the PR body too?\n"
        + "\n".join(f"line {i}" for i in range(14))
        + "\n❯ no thanks\n"
    )
    assert detect_state(text) == WAITING_INPUT
    assert permission_question(text) is None


def test_option_two_must_follow_option_one_closely():
    scattered = (
        "1. Yes\n"
        + "\n".join(f"filler {i}" for i in range(5))
        + "\n2. No\n"
    )
    assert detect_state(scattered) == RUNNING   # fail-quiet, no task
