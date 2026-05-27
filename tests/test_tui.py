"""Tests for the Textual TUI panel-builders and host detection.

The Textual ``CodeTripApp`` itself isn't exercised here — that's a Pilot
test (``test_codetrip_app.py``-style harness, future work). These tests
verify the pure Rich panel-builders that the app's Static widgets render,
plus the macOS host-app detection used to suppress synthesized keystrokes
when the user is looking at the TUI host terminal.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rich.console import Console
from rich.panel import Panel

from code_trip2 import chords, modes, tui
from code_trip2.producers import ProducerSupervisor
from code_trip2.tasks import Task
from code_trip2.window import Chord, KeyStroke
from conftest import make_mock_tts


def _make_ctx(*, app_mode="queue", summarizer_enabled=False):
    tts = make_mock_tts()
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


def _render(renderable) -> str:
    buf = io.StringIO()
    console = Console(file=buf, width=120, height=40, force_terminal=False)
    console.print(renderable)
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


# --- panel builders -------------------------------------------------------


def test_header_renders_mode_and_window():
    ctx = _make_ctx(app_mode="queue")
    out = _render(tui._header(ctx))
    assert "QUEUE" in out
    assert "ticket-42" in out


def test_header_focused_mode_shows_summarizer_model():
    ctx = _make_ctx(app_mode="focused", summarizer_enabled=True)
    out = _render(tui._header(ctx))
    assert "FOCUSED" in out
    assert "gpt-4o-mini" in out


def test_header_summarizer_off_when_disabled():
    ctx = _make_ctx(summarizer_enabled=False)
    out = _render(tui._header(ctx))
    assert "off" in out


def test_current_task_idle_panel():
    ctx = _make_ctx()
    out = _render(tui._current_task_panel(ctx))
    assert "idle" in out.lower() or "say 'next'" in out.lower()


def test_current_task_panel_with_active_task():
    ctx = _make_ctx()
    ctx.current_task = Task(
        kind="claude_reply",
        topic="ticket-42",
        headline="replied to: run the tests",
        body="Tests passed in two files.",
    )
    out = _render(tui._current_task_panel(ctx))
    assert "claude_reply" in out
    assert "ticket-42" in out
    assert "replied to: run the tests" in out


def test_queue_table_empty():
    ctx = _make_ctx()
    out = _render(tui._queue_table(ctx))
    assert "queue empty" in out.lower()


def test_queue_table_populated_shows_pending_count_and_top():
    ctx = _make_ctx()
    ctx.queue.add(Task(kind="claude_reply", topic="ticket-42", headline="top item",
                       created_at=1.0))
    ctx.queue.add(Task(kind="slack_msg", topic="general", headline="alice pinged",
                       created_at=100.0))
    out = _render(tui._queue_table(ctx))
    assert "2 pending" in out
    assert "top item" in out
    assert "alice pinged" in out


def test_topics_panel_orders_most_recent_first():
    ctx = _make_ctx()
    import time as _t
    ctx.recent_topics.touch("ticket-1", now=_t.time() - 60)
    ctx.recent_topics.touch("ticket-2", now=_t.time() - 5)
    out = _render(tui._topics_panel(ctx))
    assert out.find("ticket-2") < out.find("ticket-1")


def test_keymap_queue_mode_shows_queue_relevant_keys():
    ctx = _make_ctx(app_mode="queue")
    out = _render(tui._keymap_panel(ctx))
    assert "Macropad" in out
    assert "accept" in out or "expand" in out
    assert "skip task" in out
    assert "→ focused" in out
    assert "stop audio" in out
    assert "NAV+" in out
    assert "ACT+" in out
    assert "dismiss" in out
    assert "Ctrl+U" not in out
    assert "clear line" not in out


def test_keymap_focused_mode_shows_full_chord_set():
    ctx = _make_ctx(app_mode="focused")
    out = _render(tui._keymap_panel(ctx))
    assert "Macropad" in out
    assert "Enter" in out
    assert "Esc" in out
    assert "→ queue" in out
    assert "per-app" in out
    assert "NAV+" in out
    assert "ACT+" in out
    assert "Ctrl+U" in out


def test_keymap_panel_height_same_in_both_modes():
    queue_ctx = _make_ctx(app_mode="queue")
    focused_ctx = _make_ctx(app_mode="focused")
    assert tui._keymap_panel_size(queue_ctx) == tui._keymap_panel_size(focused_ctx)


def test_producers_panel_uses_supervisor_status():
    sup = ProducerSupervisor()
    sup.add(SimpleNamespace(name="claude", request_stop=lambda: None,
                            run=lambda: None))
    sup.add(SimpleNamespace(name="slack", request_stop=lambda: None,
                            run=lambda: None))
    # Simulate "claude task was created and is alive"; slack never started.
    sup._tasks["claude"] = SimpleNamespace(done=lambda: False)
    out = _render(tui._producers_panel(sup))
    assert "claude" in out
    assert "running" in out
    assert "slack" in out
    assert "idle" in out


def test_producers_panel_no_supervisor():
    out = _render(tui._producers_panel(None))
    assert "no supervisor" in out


# --- host-terminal detection ----------------------------------------------


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


@pytest.mark.asyncio
async def test_send_stroke_fires_when_no_tui_host():
    ctx = _stroke_ctx(tui_host_app=None)
    with patch("code_trip2.chords.window.send_keystroke", new_callable=AsyncMock) as send:
        await chords._send_stroke(ctx, _FAKE_STROKE)
    send.assert_awaited_once_with(_FAKE_STROKE)


@pytest.mark.asyncio
async def test_send_stroke_suppressed_when_active_app_is_tui_host(monkeypatch):
    ctx = _stroke_ctx(tui_host_app="kitty")
    monkeypatch.setattr("code_trip2.chords.window.active_app", AsyncMock(return_value="kitty"))
    with patch("code_trip2.chords.window.send_keystroke", new_callable=AsyncMock) as send:
        await chords._send_stroke(ctx, _FAKE_STROKE)
    send.assert_not_called()


@pytest.mark.asyncio
async def test_send_stroke_fires_when_active_app_differs(monkeypatch):
    ctx = _stroke_ctx(tui_host_app="kitty")
    monkeypatch.setattr(
        "code_trip2.chords.window.active_app", AsyncMock(return_value="Google Chrome"),
    )
    with patch("code_trip2.chords.window.send_keystroke", new_callable=AsyncMock) as send:
        await chords._send_stroke(ctx, _FAKE_STROKE)
    send.assert_awaited_once_with(_FAKE_STROKE)


@pytest.mark.asyncio
async def test_send_stroke_fires_when_active_app_lookup_errors(monkeypatch):
    """If we can't determine focus, default to firing — failing open is
    less surprising than silent suppression."""
    from code_trip2 import window

    async def boom():
        raise window.WindowError("nope")

    ctx = _stroke_ctx(tui_host_app="kitty")
    monkeypatch.setattr("code_trip2.chords.window.active_app", boom)
    with patch("code_trip2.chords.window.send_keystroke", new_callable=AsyncMock) as send:
        await chords._send_stroke(ctx, _FAKE_STROKE)
    send.assert_awaited_once_with(_FAKE_STROKE)


@pytest.mark.asyncio
async def test_yes_tap_suppressed_in_focused_mode_when_tui_host_focused(monkeypatch):
    """End-to-end: a YES tap in focused mode should not synthesize Enter
    when the user is looking at the TUI host terminal."""
    ctx = _stroke_ctx(tui_host_app="kitty")
    ctx.app_mode = "focused"
    monkeypatch.setattr("code_trip2.chords.window.active_app", AsyncMock(return_value="kitty"))
    with patch("code_trip2.chords.window.send_keystroke", new_callable=AsyncMock) as send:
        await chords.handle_tap(ctx, "yes")
    send.assert_not_called()


# --- CodeTripApp via Pilot ------------------------------------------------


@pytest.mark.asyncio
async def test_app_input_submit_dispatches_handle_voice(monkeypatch):
    """Submitting the Input widget calls handle_voice when no PTT release
    primed skill mode."""
    ctx = _make_ctx()
    called: list[str] = []

    async def fake_handle_voice(c, t):
        called.append(("voice", t))

    async def fake_handle_skill(c, t):
        called.append(("skill", t))

    monkeypatch.setattr("code_trip2.dispatch.handle_voice", fake_handle_voice)
    monkeypatch.setattr("code_trip2.dispatch.handle_skill", fake_handle_skill)

    app = tui.CodeTripApp(ctx, supervisor=None, local_stt=True)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#voice_input", tui.Input)
        input_widget.value = "what's next"
        await input_widget.action_submit()
        await pilot.pause()
        # Yield once more so the create_task() coroutine actually runs.
        await pilot.pause()

    assert called == [("voice", "what's next")]


@pytest.mark.asyncio
async def test_app_ptt_release_routes_to_skill(monkeypatch):
    """PttReleased(skill_mode=True) primes skill_mode; the next Input submit
    dispatches handle_skill."""
    ctx = _make_ctx()
    called: list[str] = []

    async def fake_handle_voice(c, t):
        called.append(("voice", t))

    async def fake_handle_skill(c, t):
        called.append(("skill", t))

    monkeypatch.setattr("code_trip2.dispatch.handle_voice", fake_handle_voice)
    monkeypatch.setattr("code_trip2.dispatch.handle_skill", fake_handle_skill)

    app = tui.CodeTripApp(ctx, supervisor=None, local_stt=True)
    async with app.run_test() as pilot:
        app.post_message(tui.PttReleased(skill_mode=True))
        await pilot.pause()
        input_widget = app.query_one("#voice_input", tui.Input)
        input_widget.value = "archive this email"
        await input_widget.action_submit()
        await pilot.pause()
        await pilot.pause()

    assert called == [("skill", "archive this email")]


@pytest.mark.asyncio
async def test_app_input_hidden_in_openai_mode():
    """In openai-STT mode the Input is composed but marked hidden so it
    doesn't take up screen real estate."""
    ctx = _make_ctx()
    app = tui.CodeTripApp(ctx, supervisor=None, local_stt=False)
    async with app.run_test():
        input_widget = app.query_one("#voice_input", tui.Input)
        assert "hidden" in input_widget.classes


@pytest.mark.asyncio
async def test_app_ptt_release_ignored_when_not_local_stt(monkeypatch):
    """PTT release in openai-STT mode is a no-op — the audio path handles
    skill_mode via on_audio's skill_mode kwarg, not via the Input widget."""
    ctx = _make_ctx()
    called: list[str] = []

    async def fake_handle_voice(c, t):
        called.append(("voice", t))

    monkeypatch.setattr("code_trip2.dispatch.handle_voice", fake_handle_voice)

    app = tui.CodeTripApp(ctx, supervisor=None, local_stt=False)
    async with app.run_test() as pilot:
        app.post_message(tui.PttReleased(skill_mode=True))
        await pilot.pause()
        # No flag should have been set; verify the input widget wasn't
        # focused/cleared and pending state stays false.
        assert app._pending_skill_mode is False
