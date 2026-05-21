"""Unit tests for dispatch.py: mode flip + queue-mode voice handlers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from code_trip2 import dispatch, modes
from code_trip2.tasks import RecentTopics, Task, TaskQueue


def _make_ctx(app_mode: str = "focused") -> modes.Context:
    """Build a Context wired up enough for dispatch tests."""
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
    log = MagicMock()
    thinking = MagicMock()
    ctx = modes.Context(
        config=cfg,  # type: ignore[arg-type]
        tts=tts,
        log=log,
        thinking=thinking,
    )
    ctx.app_mode = app_mode
    return ctx


# --- mode flip --------------------------------------------------------------


def test_flip_mode_toggles_and_plays_earcon_only():
    """The two mode chimes are distinct enough to identify the mode by
    ear; no spoken label is played on flip."""
    ctx = _make_ctx(app_mode="focused")
    with patch.object(dispatch, "earcon") as earcon_mock:
        dispatch.flip_mode(ctx)
    assert ctx.app_mode == "queue"
    earcon_mock.mode_chime.assert_called_once_with("queue")
    ctx.tts.speak.assert_not_called()

    with patch.object(dispatch, "earcon") as earcon_mock:
        dispatch.flip_mode(ctx)
    assert ctx.app_mode == "focused"
    earcon_mock.mode_chime.assert_called_once_with("focused")
    ctx.tts.speak.assert_not_called()


# --- queue-mode voice handling ----------------------------------------------


def test_handle_voice_focused_falls_through():
    ctx = _make_ctx(app_mode="focused")
    with patch.object(modes, "handle_voice") as mocked:
        dispatch.handle_voice(ctx, "some transcript")
    mocked.assert_called_once_with(ctx, "some transcript")


def test_queue_voice_status_speaks_mode():
    ctx = _make_ctx(app_mode="queue")
    ctx.active_window = "ticket-42"
    dispatch.handle_voice(ctx, "status")
    ctx.tts.speak.assert_called_with("Queue mode. Window ticket-42.")


def test_queue_voice_next_announces_top_task():
    ctx = _make_ctx(app_mode="queue")
    t = Task(kind="claude_reply", topic="ticket-42", headline="ready")
    ctx.queue.add(t)
    with patch.object(modes, "speak_chunked") as mocked:
        dispatch.handle_voice(ctx, "next")
    assert ctx.current_task is t
    mocked.assert_called()


def test_queue_voice_next_empty_speaks_empty():
    ctx = _make_ctx(app_mode="queue")
    dispatch.handle_voice(ctx, "next")
    ctx.tts.speak.assert_called_with("Queue is empty.")


def test_queue_voice_skip_defers_current():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    ctx.queue.mark_active(t.id)
    dispatch.handle_voice(ctx, "skip")
    assert ctx.current_task is None
    assert ctx.queue.get(t.id).state == "pending"
    assert ctx.queue.get(t.id).ready_at > 0


def test_queue_voice_dismiss_marks_done():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    dispatch.handle_voice(ctx, "dismiss")
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None


def test_queue_voice_snooze_parses_seconds():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    dispatch.handle_voice(ctx, "snooze 30 seconds")
    assert ctx.queue.get(t.id).ready_at > 0
    # 30s is much smaller than the 600s default; sanity-check the parser.
    elapsed = ctx.queue.get(t.id).ready_at
    import time as _t
    assert _t.time() + 60 > elapsed  # well under a minute


def test_queue_voice_add_manual_creates_note_task():
    ctx = _make_ctx(app_mode="queue")
    dispatch.handle_voice(ctx, "add a task call the doctor")
    pending = ctx.queue.pending()
    assert len(pending) == 1
    assert pending[0].kind == "note"
    assert "call the doctor" in pending[0].body


# --- queue tap delegates ---------------------------------------------------


def test_queue_yes_tap_with_no_task_pulls_next():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(kind="note", headline="hi"))
    with patch.object(modes, "speak_chunked"):
        dispatch.queue_yes_tap(ctx)
    assert ctx.current_task is t


def test_queue_yes_tap_with_active_task_expands_body():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(kind="note", headline="hi", body="more details here"))
    ctx.current_task = t
    with patch.object(modes, "speak_chunked") as mocked:
        dispatch.queue_yes_tap(ctx)
    mocked.assert_called_with(ctx, "more details here")


def test_queue_no_tap_skips_current():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    dispatch.queue_no_tap(ctx)
    assert ctx.current_task is None
    assert ctx.queue.get(t.id).state == "pending"


def test_queue_no_tap_with_nothing_active_speaks():
    ctx = _make_ctx(app_mode="queue")
    dispatch.queue_no_tap(ctx)
    ctx.tts.speak.assert_called_with("Nothing active.")


def test_dismiss_current_task_marks_done_and_stops_playback():
    """ACT+NO in queue mode: permanently drop the active task and
    interrupt any in-flight announcement."""
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="annoying msg"))
    ctx.current_task = t
    # Simulate active playback.
    ctx.playback_queue = ["chunk1", "chunk2"]
    dispatch.dismiss_current_task(ctx)
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None
    ctx.tts.stop.assert_called()  # playback interrupted
    assert ctx.playback_queue == []  # stop_playback clears the queue


def test_dismiss_current_task_with_nothing_active_speaks():
    ctx = _make_ctx(app_mode="queue")
    dispatch.dismiss_current_task(ctx)
    ctx.tts.speak.assert_called_with("Nothing active.")
