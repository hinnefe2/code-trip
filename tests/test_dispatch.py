"""Unit tests for dispatch.py: mode flip + queue-mode voice handlers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_trip2 import dispatch, modes
from code_trip2.tasks import RecentTopics, Task, TaskQueue


def _make_ctx(app_mode: str = "focused") -> modes.Context:
    """Build a Context wired up enough for dispatch tests."""
    tts = MagicMock()
    tts.is_playing.return_value = False
    tts.speak = AsyncMock(return_value=None)
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


@pytest.mark.asyncio
async def test_handle_voice_focused_falls_through():
    ctx = _make_ctx(app_mode="focused")
    with patch.object(modes, "handle_focused_voice", new_callable=AsyncMock) as mocked:
        await dispatch.handle_voice(ctx, "some transcript")
    mocked.assert_awaited_once_with(ctx, "some transcript")


@pytest.mark.asyncio
async def test_queue_voice_status_speaks_mode():
    ctx = _make_ctx(app_mode="queue")
    ctx.active_window = "ticket-42"
    await dispatch.handle_voice(ctx, "status")
    ctx.tts.speak.assert_called_with("Queue mode. Window ticket-42.")


@pytest.mark.asyncio
async def test_queue_voice_next_announces_top_task():
    ctx = _make_ctx(app_mode="queue")
    t = Task(kind="claude_reply", topic="ticket-42", headline="ready")
    ctx.queue.add(t)
    with patch.object(modes, "speak_chunked") as mocked:
        await dispatch.handle_voice(ctx, "next")
    assert ctx.current_task is t
    mocked.assert_called()


@pytest.mark.asyncio
async def test_queue_voice_next_empty_speaks_empty():
    ctx = _make_ctx(app_mode="queue")
    await dispatch.handle_voice(ctx, "next")
    ctx.tts.speak.assert_called_with("Queue is empty.")


@pytest.mark.asyncio
async def test_queue_voice_skip_defers_current():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    ctx.queue.mark_active(t.id)
    await dispatch.handle_voice(ctx, "skip")
    assert ctx.current_task is None
    assert ctx.queue.get(t.id).state == "pending"
    assert ctx.queue.get(t.id).ready_at > 0


@pytest.mark.asyncio
async def test_queue_voice_dismiss_marks_done():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    await dispatch.handle_voice(ctx, "dismiss")
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None


@pytest.mark.asyncio
async def test_queue_voice_snooze_parses_seconds():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    await dispatch.handle_voice(ctx, "snooze 30 seconds")
    assert ctx.queue.get(t.id).ready_at > 0
    # 30s is much smaller than the 600s default; sanity-check the parser.
    elapsed = ctx.queue.get(t.id).ready_at
    import time as _t
    assert _t.time() + 60 > elapsed  # well under a minute


@pytest.mark.asyncio
async def test_queue_voice_add_manual_creates_note_task():
    ctx = _make_ctx(app_mode="queue")
    await dispatch.handle_voice(ctx, "add a task call the doctor")
    pending = ctx.queue.pending()
    assert len(pending) == 1
    assert pending[0].kind == "note"
    assert "call the doctor" in pending[0].body


# --- queue tap delegates ---------------------------------------------------


@pytest.mark.asyncio
async def test_queue_yes_tap_with_no_task_pulls_next():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(kind="note", headline="hi"))
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_yes_tap(ctx)
    assert ctx.current_task is t


@pytest.mark.asyncio
async def test_queue_yes_tap_with_active_task_expands_body():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(kind="note", headline="hi", body="more details here"))
    ctx.current_task = t
    with patch.object(modes, "speak_chunked") as mocked:
        await dispatch.queue_yes_tap(ctx)
    mocked.assert_called_with(ctx, "more details here")


@pytest.mark.asyncio
async def test_queue_no_tap_skips_current():
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    await dispatch.queue_no_tap(ctx)
    assert ctx.current_task is None
    assert ctx.queue.get(t.id).state == "pending"


@pytest.mark.asyncio
async def test_queue_no_tap_with_nothing_active_speaks():
    ctx = _make_ctx(app_mode="queue")
    await dispatch.queue_no_tap(ctx)
    ctx.tts.speak.assert_called_with("Nothing active.")


@pytest.mark.asyncio
async def test_dismiss_current_task_marks_done_and_stops_playback():
    """ACT+NO in queue mode: permanently drop the active task and
    interrupt any in-flight announcement."""
    ctx = _make_ctx(app_mode="queue")
    t = ctx.queue.add(Task(headline="annoying msg"))
    ctx.current_task = t
    # Simulate active playback.
    ctx.playback_queue = ["chunk1", "chunk2"]
    await dispatch.dismiss_current_task(ctx)
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None
    ctx.tts.stop.assert_called()  # playback interrupted
    assert ctx.playback_queue == []  # stop_playback clears the queue


@pytest.mark.asyncio
async def test_dismiss_current_task_with_nothing_active_speaks():
    ctx = _make_ctx(app_mode="queue")
    await dispatch.dismiss_current_task(ctx)
    ctx.tts.speak.assert_called_with("Nothing active.")


# --- handle_skill (ACT+PTT) ------------------------------------------------


@pytest.mark.asyncio
async def test_handle_skill_invokes_agent_and_marks_task_done():
    ctx = _make_ctx(app_mode="queue")
    mcp = MagicMock()
    mcp.enabled = True
    mcp.run_agent = AsyncMock(return_value="Accepted 'Lunch' and archived the email.")
    ctx.agent_mcp = mcp
    ctx.agent_allowed_tools = ("mcp__svc__tool_a", "mcp__svc__tool_b")
    t = ctx.queue.add(Task(
        kind="email_msg",
        topic="alice",
        headline="Calendar invite",
        body="Please join...",
        source={"thread_id": "T1", "message_id": "M1", "subject": "Lunch"},
    ))
    ctx.current_task = t
    await dispatch.handle_skill(ctx, "accept and archive")
    mcp.run_agent.assert_called_once()
    call = mcp.run_agent.call_args
    prompt = call.kwargs["prompt"]
    assert "accept and archive" in prompt
    assert "T1" in prompt  # thread_id baked into the prompt
    assert "email_msg" in prompt
    # allowed_tools propagates from Context to the agent call.
    assert call.kwargs["allowed_tools"] == ("mcp__svc__tool_a", "mcp__svc__tool_b")
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None
    ctx.tts.speak.assert_called_with("Accepted 'Lunch' and archived the email.")


@pytest.mark.asyncio
async def test_handle_skill_with_no_active_task_speaks():
    ctx = _make_ctx(app_mode="queue")
    ctx.agent_mcp = MagicMock(enabled=True)
    await dispatch.handle_skill(ctx, "do something")
    ctx.tts.speak.assert_called_with("Nothing active to act on.")
    ctx.agent_mcp.run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_handle_skill_without_agent_mcp_speaks_error():
    ctx = _make_ctx(app_mode="queue")
    ctx.agent_mcp = None
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    await dispatch.handle_skill(ctx, "do something")
    ctx.tts.speak.assert_called_with("Agent MCP is not configured.")
    assert ctx.queue.get(t.id).state != "done"


@pytest.mark.asyncio
async def test_handle_skill_agent_error_keeps_task_active():
    ctx = _make_ctx(app_mode="queue")
    mcp = MagicMock()
    mcp.enabled = True
    mcp.run_agent = AsyncMock(side_effect=RuntimeError("budget exceeded"))
    ctx.agent_mcp = mcp
    t = ctx.queue.add(Task(kind="email_msg", source={"thread_id": "T1"}))
    ctx.current_task = t
    await dispatch.handle_skill(ctx, "accept invite")
    assert ctx.queue.get(t.id).state != "done"
    assert ctx.current_task is t
