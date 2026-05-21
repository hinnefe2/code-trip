"""Tests for the TUI render function.

We don't start a real ``Live`` display in tests — just verify ``render``
produces a non-empty Layout that prints cleanly for the relevant states
(idle queue, active task, populated queue, etc.).
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rich.console import Console
from rich.layout import Layout

from code_trip2 import chords, modes, tui
from code_trip2.producers import ProducerSupervisor
from code_trip2.tasks import Task
from code_trip2.window import Chord, KeyStroke


def _make_ctx(*, app_mode="queue", summarizer_enabled=False):
    tts = MagicMock()
    tts.is_playing.return_value = False
    cfg = SimpleNamespace(
        ssh_host="",
        ssh_options=(),
        tmux_session="main",
        work_window="work",
        linear_window="linear",
        terminal_apps=("kitty",),
    )
    summarizer = None
    if summarizer_enabled:
        summarizer = SimpleNamespace(enabled=True, _model="gpt-4o-mini")
    ctx = modes.Context(
        config=cfg,  # type: ignore[arg-type]
        tts=tts,
        log=MagicMock(),
        thinking=MagicMock(),
        summarizer=summarizer,
    )
    ctx.app_mode = app_mode
    ctx.active_window = "ticket-42"
    return ctx


def _render_to_string(layout: Layout) -> str:
    """Render the layout to a string so we can assert presence of pieces."""
    buf = io.StringIO()
    console = Console(file=buf, width=120, height=40, force_terminal=False)
    console.print(layout)
    return buf.getvalue()


# --- format helpers --------------------------------------------------------


def test_format_age_units():
    assert tui._format_age(5) == "5s"
    assert tui._format_age(120) == "2m"
    assert tui._format_age(3700) == "1h"
    assert tui._format_age(86_400 * 3) == "3d"


def test_truncate_handles_newlines_and_caps():
    assert tui._truncate("a\nb\nc", 80) == "a b c"
    assert tui._truncate("x" * 100, 10) == "x" * 9 + "…"


# --- render ----------------------------------------------------------------


def test_render_idle_queue_mode():
    ctx = _make_ctx(app_mode="queue")
    layout = tui.render(ctx, supervisor=None)
    out = _render_to_string(layout)
    assert "QUEUE" in out
    assert "ticket-42" in out
    assert "queue empty" in out.lower()
    assert "idle" in out.lower() or "say 'next'" in out.lower()


def test_render_focused_mode_shows_summarizer_state():
    ctx = _make_ctx(app_mode="focused", summarizer_enabled=True)
    layout = tui.render(ctx, supervisor=None)
    out = _render_to_string(layout)
    assert "FOCUSED" in out
    assert "gpt-4o-mini" in out


def test_render_summarizer_off_when_disabled():
    ctx = _make_ctx(summarizer_enabled=False)
    layout = tui.render(ctx, supervisor=None)
    out = _render_to_string(layout)
    assert "off" in out  # the "off" label for missing summarizer


def test_render_current_task():
    ctx = _make_ctx()
    t = Task(
        kind="claude_reply",
        topic="ticket-42",
        headline="replied to: run the tests",
        body="Tests passed in two files.",
    )
    ctx.current_task = t
    layout = tui.render(ctx, supervisor=None)
    out = _render_to_string(layout)
    assert "claude_reply" in out
    assert "ticket-42" in out
    assert "replied to: run the tests" in out


def test_render_populated_queue_shows_pending_count_and_top():
    ctx = _make_ctx()
    ctx.queue.add(Task(kind="claude_reply", topic="ticket-42", headline="top item",
                       created_at=1.0))
    ctx.queue.add(Task(kind="slack_msg", topic="general", headline="alice pinged",
                       created_at=100.0))
    layout = tui.render(ctx, supervisor=None)
    out = _render_to_string(layout)
    assert "2 pending" in out
    assert "top item" in out
    assert "alice pinged" in out


def test_render_recent_topics_shows_recent_first():
    ctx = _make_ctx()
    import time as _t
    ctx.recent_topics.touch("ticket-1", now=_t.time() - 60)
    ctx.recent_topics.touch("ticket-2", now=_t.time() - 5)
    layout = tui.render(ctx, supervisor=None)
    out = _render_to_string(layout)
    # Both topics rendered; ticket-2 (more recent) should appear before
    # ticket-1 in the output stream.
    assert out.find("ticket-2") < out.find("ticket-1")


def test_detect_tui_host_app_maps_known_terms(monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "kitty")
    assert tui.detect_tui_host_app() == "kitty"
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    assert tui.detect_tui_host_app() == "Terminal"
    monkeypatch.setenv("TERM_PROGRAM", "iTerm.app")
    assert tui.detect_tui_host_app() == "iTerm2"


def test_detect_tui_host_app_unknown_falls_through(monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "MysteryTerm")
    assert tui.detect_tui_host_app() == "MysteryTerm"


def test_detect_tui_host_app_missing_returns_none(monkeypatch):
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    assert tui.detect_tui_host_app() is None


# --- TUI-host keystroke suppression ---------------------------------------


_FAKE_STROKE = KeyStroke(chords=(Chord(key="x"),))


def _stroke_ctx(*, tui_host_app=None):
    ctx = _make_ctx()
    ctx.tui_host_app = tui_host_app
    return ctx


def test_send_stroke_fires_when_no_tui_host():
    ctx = _stroke_ctx(tui_host_app=None)
    with patch("code_trip2.chords.window.send_keystroke") as send:
        chords._send_stroke(ctx, _FAKE_STROKE)
    send.assert_called_once_with(_FAKE_STROKE)


def test_send_stroke_suppressed_when_active_app_is_tui_host(monkeypatch):
    ctx = _stroke_ctx(tui_host_app="kitty")
    monkeypatch.setattr("code_trip2.chords.window.active_app", lambda: "kitty")
    with patch("code_trip2.chords.window.send_keystroke") as send:
        chords._send_stroke(ctx, _FAKE_STROKE)
    send.assert_not_called()


def test_send_stroke_fires_when_active_app_differs(monkeypatch):
    ctx = _stroke_ctx(tui_host_app="kitty")
    monkeypatch.setattr("code_trip2.chords.window.active_app", lambda: "Google Chrome")
    with patch("code_trip2.chords.window.send_keystroke") as send:
        chords._send_stroke(ctx, _FAKE_STROKE)
    send.assert_called_once_with(_FAKE_STROKE)


def test_send_stroke_fires_when_active_app_lookup_errors(monkeypatch):
    """If we can't determine focus, default to firing — failing open is
    less surprising than silent suppression."""
    from code_trip2 import window

    def boom():
        raise window.WindowError("nope")

    ctx = _stroke_ctx(tui_host_app="kitty")
    monkeypatch.setattr("code_trip2.chords.window.active_app", boom)
    with patch("code_trip2.chords.window.send_keystroke") as send:
        chords._send_stroke(ctx, _FAKE_STROKE)
    send.assert_called_once_with(_FAKE_STROKE)


def test_yes_tap_suppressed_in_focused_mode_when_tui_host_focused(monkeypatch):
    """End-to-end: a YES tap in focused mode should not synthesize Enter
    when the user is looking at the TUI host terminal."""
    ctx = _stroke_ctx(tui_host_app="kitty")
    ctx.app_mode = "focused"
    monkeypatch.setattr("code_trip2.chords.window.active_app", lambda: "kitty")
    with patch("code_trip2.chords.window.send_keystroke") as send:
        chords.handle_tap(ctx, "yes")
    send.assert_not_called()


def test_render_keymap_in_queue_mode_shows_only_queue_relevant_keys():
    ctx = _make_ctx(app_mode="queue")
    layout = tui.render(ctx, supervisor=None)
    out = _render_to_string(layout)
    assert "Macropad" in out
    # Queue-flavored solo taps.
    assert "accept" in out or "expand" in out
    assert "skip task" in out
    # NAV solo flips mode; ACT solo interrupts audio.
    assert "→ focused" in out
    assert "stop audio" in out
    # NAV-modifier chords still useful (user often glances at the screen).
    assert "NAV+" in out
    # ACT+NO is a shell-input affordance — irrelevant when away from screen.
    assert "ACT+" not in out
    assert "Ctrl+U" not in out
    assert "clear line" not in out


def test_render_keymap_in_focused_mode_shows_full_chord_set():
    ctx = _make_ctx(app_mode="focused")
    layout = tui.render(ctx, supervisor=None)
    out = _render_to_string(layout)
    assert "Macropad" in out
    # Keyboard-style solo taps.
    assert "Enter" in out
    assert "Esc" in out
    # NAV solo flips mode; ACT solo does per-app.
    assert "→ queue" in out
    assert "per-app" in out
    # All chord rows shown.
    assert "NAV+" in out
    assert "ACT+" in out
    assert "Ctrl+U" in out


def test_keymap_panel_height_changes_with_mode():
    """Queue mode keymap renders fewer rows, so the panel reserves less
    vertical space; focused mode reserves more."""
    queue_ctx = _make_ctx(app_mode="queue")
    focused_ctx = _make_ctx(app_mode="focused")
    assert tui._keymap_panel_size(queue_ctx) < tui._keymap_panel_size(focused_ctx)


def test_render_producers_status_uses_supervisor():
    ctx = _make_ctx()
    sup = ProducerSupervisor()
    # Fake two producers: one idle, one running.
    sup.add(SimpleNamespace(name="claude", _thread=SimpleNamespace(is_alive=lambda: True),
                            start=lambda: None, stop=lambda: None))
    sup.add(SimpleNamespace(name="slack", _thread=None,
                            start=lambda: None, stop=lambda: None))
    layout = tui.render(ctx, supervisor=sup)
    out = _render_to_string(layout)
    assert "claude" in out
    assert "running" in out
    assert "slack" in out
    assert "idle" in out
