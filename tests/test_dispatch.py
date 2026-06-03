"""Unit tests for dispatch.py: mode flip + queue-mode voice handlers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_trip2 import dispatch, modes
from code_trip2.tasks import RecentTopics, Task, TaskQueue
from conftest import make_mock_tts


def _make_ctx(app_mode: str = "focused") -> modes.Context:
    """Build a Context wired up enough for dispatch tests."""
    tts = make_mock_tts()
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
async def test_queue_yes_tap_calls_submit_input():
    """YES in queue mode invokes ctx.submit_input."""
    ctx = _make_ctx(app_mode="queue")
    submit = MagicMock(return_value=True)
    ctx.submit_input = submit
    await dispatch.queue_yes_tap(ctx)
    submit.assert_called_once_with()


@pytest.mark.asyncio
async def test_queue_yes_tap_without_submit_input_is_noop():
    """No TUI (e.g. headless OpenAI-STT mode) — YES tap is a no-op."""
    ctx = _make_ctx(app_mode="queue")
    ctx.submit_input = None
    await dispatch.queue_yes_tap(ctx)  # no raise


@pytest.mark.asyncio
async def test_queue_yes_tap_swallows_submit_exceptions():
    """A submit_input that raises shouldn't crash the chord handler."""
    ctx = _make_ctx(app_mode="queue")
    submit = MagicMock(side_effect=RuntimeError("widget gone"))
    ctx.submit_input = submit
    await dispatch.queue_yes_tap(ctx)  # no raise


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


# --- queue_navigate (TUI arrow keys) --------------------------------------


def _seed_three(ctx) -> tuple[Task, Task, Task]:
    """Add three pending tasks with descending priority so the ranked
    order is deterministic: oldest first wins on the age tiebreaker."""
    import time as _time
    base = _time.time() - 100
    a = ctx.queue.add(Task(kind="note", topic="a", headline="A", created_at=base))
    b = ctx.queue.add(Task(kind="note", topic="b", headline="B", created_at=base + 10))
    c = ctx.queue.add(Task(kind="note", topic="c", headline="C", created_at=base + 20))
    return a, b, c


@pytest.mark.asyncio
async def test_queue_navigate_down_from_idle_points_cursor_at_top():
    ctx = _make_ctx(app_mode="queue")
    a, _b, _c = _seed_three(ctx)
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=+1)
    assert ctx.current_task is a
    # Cursor doesn't change task state — the queue stays static.
    assert ctx.queue.get(a.id).state == "pending"
    assert len(ctx.queue.pending()) == 3


@pytest.mark.asyncio
async def test_queue_navigate_up_from_idle_points_cursor_at_bottom():
    ctx = _make_ctx(app_mode="queue")
    _a, _b, c = _seed_three(ctx)
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=-1)
    assert ctx.current_task is c


@pytest.mark.asyncio
async def test_queue_navigate_down_moves_cursor_without_disturbing_queue():
    ctx = _make_ctx(app_mode="queue")
    a, b, _c = _seed_three(ctx)
    ctx.current_task = a
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=+1)
    assert ctx.current_task is b
    # Previous cursor task stays pending in the queue (no defer, no
    # state change) so the user can arrow back to it.
    assert ctx.queue.get(a.id).state == "pending"
    assert ctx.queue.get(a.id).ready_at == 0
    assert {t.id for t in ctx.queue.pending()} == {a.id, b.id, _c.id}


@pytest.mark.asyncio
async def test_queue_navigate_up_walks_backward():
    ctx = _make_ctx(app_mode="queue")
    a, b, _c = _seed_three(ctx)
    ctx.current_task = b
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=-1)
    assert ctx.current_task is a


@pytest.mark.asyncio
async def test_queue_navigate_clamps_at_end():
    ctx = _make_ctx(app_mode="queue")
    _a, _b, c = _seed_three(ctx)
    ctx.current_task = c
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=+1)
    assert ctx.current_task is c  # clamped, not wrapped


@pytest.mark.asyncio
async def test_queue_navigate_clamps_at_start():
    ctx = _make_ctx(app_mode="queue")
    a, _b, _c = _seed_three(ctx)
    ctx.current_task = a
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=-1)
    assert ctx.current_task is a


@pytest.mark.asyncio
async def test_queue_navigate_on_empty_queue_speaks_empty():
    ctx = _make_ctx(app_mode="queue")
    await dispatch.queue_navigate(ctx, direction=+1)
    ctx.tts.speak.assert_called_with("Queue is empty.")
    assert ctx.current_task is None


@pytest.mark.asyncio
async def test_queue_navigate_does_not_touch_recent_topics():
    """Arrow browsing is exploratory — it shouldn't bias the scheduler
    the way an explicit voice ``next`` or skill engagement does."""
    ctx = _make_ctx(app_mode="queue")
    _a, _b, _c = _seed_three(ctx)
    before = ctx.recent_topics.as_list()
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=+1)
        await dispatch.queue_navigate(ctx, direction=+1)
    assert ctx.recent_topics.as_list() == before


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


def test_kind_label_meeting_followup_uses_meeting_when_available():
    t = Task(
        kind="meeting_followup",
        topic="planning-sync",
        headline="Draft retention doc",
        source={"meeting": "Planning sync"},
    )
    assert "Planning sync" in dispatch._kind_label(t)


def test_kind_label_meeting_followup_falls_back_to_topic():
    t = Task(
        kind="meeting_followup",
        topic="planning-sync",
        headline="Draft retention doc",
        source={},
    )
    assert "planning-sync" in dispatch._kind_label(t)


def test_kind_label_meeting_followup_with_no_context():
    t = Task(kind="meeting_followup", topic="", headline="x", source={})
    assert dispatch._kind_label(t) == "Meeting follow-up"
