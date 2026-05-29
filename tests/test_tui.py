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


def _make_ctx(
    *,
    app_mode="queue",
    summarizer_enabled=False,
    autohandle_enabled=False,
    autohandle_kinds=(),
):
    tts = make_mock_tts()
    cfg = SimpleNamespace(
        ssh_host="",
        ssh_options=(),
        tmux_session="main",
        work_window="work",
        linear_window="linear",
        terminal_apps=("kitty",),
        autohandle_enabled=autohandle_enabled,
        autohandle_kinds=autohandle_kinds,
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


def test_current_task_panel_preserves_multiline_body():
    """Body lines that fit under the cap render with their newlines
    intact — markdown structure stays readable instead of collapsing
    to one line like the old single-line preview did."""
    ctx = _make_ctx()
    ctx.current_task = Task(
        kind="linear_issue",
        topic="ai-7",
        headline="AI-7: a thing",
        body="Line one\nLine two\nLine three",
    )
    out = _render(tui._current_task_panel(ctx))
    assert "Line one" in out
    assert "Line two" in out
    assert "Line three" in out
    assert "truncated" not in out.lower()


def test_current_task_panel_caps_body_and_marks_truncation():
    ctx = _make_ctx()
    long_body = "\n".join(f"line {i}" for i in range(40))
    ctx.current_task = Task(
        kind="linear_issue",
        topic="ai-7",
        headline="AI-7: long",
        body=long_body,
    )
    out = _render(tui._current_task_panel(ctx))
    # First N lines included; later ones are not.
    assert "line 0" in out
    assert f"line {tui._CURRENT_TASK_BODY_MAX_LINES - 1}" in out
    assert f"line {tui._CURRENT_TASK_BODY_MAX_LINES}" not in out
    assert "truncated" in out.lower()


def test_clip_body_short_body_passes_through():
    clipped, truncated = tui._clip_body("a\nb\nc", max_lines=10)
    assert clipped == "a\nb\nc"
    assert truncated is False


def test_clip_body_caps_at_max_lines():
    body = "\n".join(str(i) for i in range(20))
    clipped, truncated = tui._clip_body(body, max_lines=5)
    assert clipped.splitlines() == ["0", "1", "2", "3", "4"]
    assert truncated is True


def test_clip_body_empty_returns_empty():
    assert tui._clip_body("", max_lines=10) == ("", False)


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


def test_queue_table_marker_follows_cursor():
    """The ▶ marker tracks ctx.current_task so arrow-key navigation moves
    visibly without reshuffling the queue."""
    ctx = _make_ctx()
    top = ctx.queue.add(
        Task(kind="claude_reply", topic="t1", headline="top item", created_at=1.0)
    )
    second = ctx.queue.add(
        Task(kind="slack_msg", topic="t2", headline="second item", created_at=100.0)
    )
    # No cursor → marker on top row (matches auto-announce target).
    out = _render(tui._queue_table(ctx))
    top_line = next(line for line in out.splitlines() if "top item" in line)
    second_line = next(line for line in out.splitlines() if "second item" in line)
    assert "▶" in top_line
    assert "▶" not in second_line
    # Cursor on the second task → marker moves to it; top stays in the list.
    ctx.current_task = second
    out = _render(tui._queue_table(ctx))
    top_line = next(line for line in out.splitlines() if "top item" in line)
    second_line = next(line for line in out.splitlines() if "second item" in line)
    assert "▶" in second_line
    assert "▶" not in top_line


def test_keymap_queue_mode_shows_queue_relevant_keys():
    ctx = _make_ctx(app_mode="queue")
    out = _render(tui._keymap_panel(ctx))
    assert "Macropad" in out
    assert "submit" in out                 # YES = Enter / submit Input
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


def test_producers_panel_shows_polling_state():
    """Producer with ``is_polling=True`` shows as ``polling`` in cyan."""
    sup = ProducerSupervisor()
    sup.add(SimpleNamespace(
        name="email",
        is_polling=True,
        request_stop=lambda: None,
        run=lambda: None,
    ))
    sup._tasks["email"] = SimpleNamespace(done=lambda: False)
    out = _render(tui._producers_panel(sup))
    assert "email" in out
    assert "polling" in out


# --- auto-handle log panel ------------------------------------------------


def _log_entry(
    *,
    action: str = "handled",
    skill: str = "accept-invite",
    headline: str = "John Doe: Standup invite",
    summary: str | None = "Accepted and archived.",
    error: str | None = None,
    dry_run_nominated: bool = False,
    age_seconds: float = 5.0,
):
    """Build one ``AutohandleLogEntry`` for panel tests."""
    import time as _t
    from code_trip2.screener import AutohandleLogEntry, ScreeningOutcome
    task = Task(kind="email_msg", topic="john-doe", headline=headline)
    outcome = ScreeningOutcome(
        action=action,
        task=task,
        skill=skill,
        summary=summary,
        error=error,
        dry_run_nominated=dry_run_nominated,
    )
    return AutohandleLogEntry(ts=_t.time() - age_seconds, outcome=outcome)


def test_autohandle_panel_empty_when_disabled():
    ctx = _make_ctx(autohandle_enabled=False, autohandle_kinds=())
    out = _render(tui._autohandle_panel(ctx))
    assert "auto-handle disabled" in out.lower()


def test_autohandle_panel_empty_when_enabled_with_no_entries():
    ctx = _make_ctx(autohandle_enabled=True, autohandle_kinds=("email_msg",))
    out = _render(tui._autohandle_panel(ctx))
    assert "no recent actions" in out.lower()


def test_autohandle_panel_renders_handled_entry():
    ctx = _make_ctx(autohandle_enabled=True, autohandle_kinds=("email_msg",))
    ctx.autohandle_log.append(_log_entry())
    out = _render(tui._autohandle_panel(ctx))
    assert "HANDLED" in out
    assert "accept-invite" in out
    assert "John Doe" in out


def test_autohandle_panel_renders_failed_entry_with_error_suffix():
    ctx = _make_ctx(autohandle_enabled=True, autohandle_kinds=("email_msg",))
    ctx.autohandle_log.append(_log_entry(
        action="failed",
        summary=None,
        error="MCP timed out after 60s",
    ))
    out = _render(tui._autohandle_panel(ctx))
    assert "FAILED" in out
    assert "MCP timed out" in out


def test_autohandle_panel_renders_dry_run_nominated_entry():
    ctx = _make_ctx(autohandle_enabled=True, autohandle_kinds=("email_msg",))
    ctx.autohandle_log.append(_log_entry(
        action="forward",
        dry_run_nominated=True,
        summary=None,
    ))
    out = _render(tui._autohandle_panel(ctx))
    assert "DRY-RUN" in out
    assert "accept-invite" in out


def test_autohandle_panel_renders_dismissed_entry():
    ctx = _make_ctx(autohandle_enabled=True, autohandle_kinds=("slack_msg",))
    ctx.autohandle_log.append(_log_entry(
        action="dismissed",
        skill="drop-standups",
        headline="Alice: standup update for thursday",
        summary=None,
    ))
    out = _render(tui._autohandle_panel(ctx))
    assert "DISMISSED" in out
    assert "drop-standups" in out
    assert "Alice" in out


def test_autohandle_panel_newest_entry_appears_first():
    ctx = _make_ctx(autohandle_enabled=True, autohandle_kinds=("email_msg",))
    ctx.autohandle_log.append(_log_entry(
        headline="OLDER ITEM", age_seconds=600,
    ))
    ctx.autohandle_log.append(_log_entry(
        headline="NEWER ITEM", age_seconds=10,
    ))
    out = _render(tui._autohandle_panel(ctx))
    assert out.find("NEWER ITEM") < out.find("OLDER ITEM")


def test_autohandle_panel_caps_visible_rows_with_overflow_indicator():
    """More entries than the visible cap → title says how many are hidden."""
    ctx = _make_ctx(autohandle_enabled=True, autohandle_kinds=("email_msg",))
    for i in range(15):
        ctx.autohandle_log.append(_log_entry(
            headline=f"item-{i:02d}", age_seconds=float(i),
        ))
    out = _render(tui._autohandle_panel(ctx))
    assert "(15)" in out          # total count
    assert "+5 older" in out      # 15 entries, 10 visible


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


# --- submit_input (macropad YES tap path) ---------------------------------


@pytest.mark.asyncio
async def test_submit_input_dispatches_current_value(monkeypatch):
    """submit_input() submits whatever's in the Input as if Enter was hit."""
    ctx = _make_ctx()
    called: list[tuple[str, str]] = []

    async def fake_handle_voice(c, t):
        called.append(("voice", t))

    monkeypatch.setattr("code_trip2.dispatch.handle_voice", fake_handle_voice)

    app = tui.CodeTripApp(ctx, supervisor=None, local_stt=True)
    async with app.run_test() as pilot:
        input_widget = app.query_one("#voice_input", tui.Input)
        input_widget.value = "what's next"
        assert app.submit_input() is True
        await pilot.pause()
        await pilot.pause()
        assert input_widget.value == ""
    assert called == [("voice", "what's next")]


@pytest.mark.asyncio
async def test_submit_input_empty_is_noop(monkeypatch):
    """No text in Input → submit_input returns False, no dispatch."""
    ctx = _make_ctx()
    called: list[tuple[str, str]] = []

    async def fake_handle_voice(c, t):
        called.append(("voice", t))

    monkeypatch.setattr("code_trip2.dispatch.handle_voice", fake_handle_voice)

    app = tui.CodeTripApp(ctx, supervisor=None, local_stt=True)
    async with app.run_test() as pilot:
        assert app.submit_input() is False
        await pilot.pause()
    assert called == []


@pytest.mark.asyncio
async def test_submit_input_after_ptt_uses_skill_mode(monkeypatch):
    """A PTT release primes skill mode; submit_input dispatches via skill."""
    ctx = _make_ctx()
    called: list[tuple[str, str]] = []

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
        assert app.submit_input() is True
        await pilot.pause()
        await pilot.pause()
    assert called == [("skill", "archive this email")]


@pytest.mark.asyncio
async def test_ptt_release_does_not_arm_autosubmit():
    """After PTT release the Input should be focused but text should sit
    there indefinitely — no timer auto-submits it on a quiet pause."""
    ctx = _make_ctx()
    app = tui.CodeTripApp(ctx, supervisor=None, local_stt=True)
    async with app.run_test() as pilot:
        app.post_message(tui.PttReleased(skill_mode=False))
        await pilot.pause()
        input_widget = app.query_one("#voice_input", tui.Input)
        input_widget.value = "would have auto-submitted before"
        # Let several event-loop turns elapse — well past the old
        # 0.4 s quiet-pause threshold.
        for _ in range(20):
            await pilot.pause()
        assert input_widget.value == "would have auto-submitted before"
