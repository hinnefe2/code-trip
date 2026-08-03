"""Unit tests for dispatch.py: queue voice handlers + tap delegates."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest

from code_trip2 import dispatch, modes
from code_trip2.earcon import Thinking
from code_trip2.producers.claude_mcp import ClaudeMCPClient
from code_trip2.session_log import SessionLogger
from code_trip2.tasks import RecentTopics, Task, TaskQueue
from conftest import make_mock_tts


def _make_ctx() -> modes.Context:
    """Build a Context wired up enough for dispatch tests.

    ``log`` and ``thinking`` are spec'd against the real classes so a
    typo in a method name or a kwarg/positional collision (the exact
    bug shape that hid the ``kind=task.kind`` regression for months)
    raises ``TypeError`` at test time instead of being silently
    swallowed by a bare :class:`MagicMock`.
    """
    tts = make_mock_tts()
    cfg = SimpleNamespace(
        ssh_host="",
        ssh_options=(),
        tmux_session="main",
        work_window="work",
        linear_window="linear",
        terminal_apps=("kitty",),
    )
    log = create_autospec(SessionLogger, instance=True)
    thinking = create_autospec(Thinking, instance=True)
    return modes.Context(
        config=cfg,  # type: ignore[arg-type]
        tts=tts,
        log=log,
        thinking=thinking,
    )


# --- queue voice handling ---------------------------------------------------


@pytest.mark.asyncio
async def test_handle_voice_empty_transcript_is_noop():
    ctx = _make_ctx()
    await dispatch.handle_voice(ctx, "   ")
    ctx.tts.speak.assert_not_called()


@pytest.mark.asyncio
async def test_queue_voice_status_announces_count():
    ctx = _make_ctx()
    ctx.queue.add(Task(kind="remote_window", topic="ticket-42", headline="ready"))
    await dispatch.handle_voice(ctx, "status")
    ctx.tts.speak.assert_called_with("1 pending. 1 remote window.")


@pytest.mark.asyncio
async def test_queue_voice_unmatched_with_no_task_nudges():
    ctx = _make_ctx()
    await dispatch.handle_voice(ctx, "what is the meaning of life")
    ctx.tts.speak.assert_called_with("Nothing active. Say next to pull a task.")


@pytest.mark.asyncio
async def test_queue_voice_next_announces_top_task():
    ctx = _make_ctx()
    t = Task(kind="remote_window", topic="ticket-42", headline="ready")
    ctx.queue.add(t)
    with patch.object(modes, "speak_chunked") as mocked:
        await dispatch.handle_voice(ctx, "next")
    assert ctx.current_task is t
    mocked.assert_called()


@pytest.mark.asyncio
async def test_queue_voice_next_empty_speaks_empty():
    ctx = _make_ctx()
    await dispatch.handle_voice(ctx, "next")
    ctx.tts.speak.assert_called_with("Queue is empty.")


@pytest.mark.asyncio
async def test_queue_voice_skip_defers_current():
    ctx = _make_ctx()
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    ctx.queue.mark_active(t.id)
    await dispatch.handle_voice(ctx, "skip")
    assert ctx.current_task is None
    assert ctx.queue.get(t.id).state == "pending"
    assert ctx.queue.get(t.id).ready_at > 0


@pytest.mark.asyncio
async def test_queue_voice_dismiss_marks_done():
    ctx = _make_ctx()
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    await dispatch.handle_voice(ctx, "dismiss")
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None


@pytest.mark.asyncio
async def test_queue_voice_snooze_parses_seconds():
    ctx = _make_ctx()
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
    ctx = _make_ctx()
    await dispatch.handle_voice(ctx, "add a task call the doctor")
    pending = ctx.queue.pending()
    assert len(pending) == 1
    assert pending[0].kind == "note"
    assert "call the doctor" in pending[0].body


# --- queue tap delegates ---------------------------------------------------


@pytest.mark.asyncio
async def test_queue_yes_tap_calls_submit_input():
    """YES in queue mode invokes ctx.submit_input; a consumed submit
    reports True so the caller knows not to fall back."""
    ctx = _make_ctx()
    submit = MagicMock(return_value=True)
    ctx.submit_input = submit
    assert await dispatch.queue_yes_tap(ctx) is True
    submit.assert_called_once_with()


@pytest.mark.asyncio
async def test_queue_yes_tap_empty_input_reports_unconsumed():
    """submit_input finding an empty Input returns False — the tap
    wasn't consumed, so the caller may fall back to a focused-app
    Enter."""
    ctx = _make_ctx()
    ctx.submit_input = MagicMock(return_value=False)
    assert await dispatch.queue_yes_tap(ctx) is False


@pytest.mark.asyncio
async def test_queue_yes_tap_without_submit_input_reports_unconsumed():
    """No TUI (e.g. headless OpenAI-STT mode) — nothing consumed the
    tap, so the fallback path is available."""
    ctx = _make_ctx()
    ctx.submit_input = None
    assert await dispatch.queue_yes_tap(ctx) is False


@pytest.mark.asyncio
async def test_queue_yes_tap_swallows_submit_exceptions():
    """A submit_input that raises shouldn't crash the chord handler —
    and must report consumed, so the caller doesn't stack a synthetic
    Enter on top of an unknown widget state."""
    ctx = _make_ctx()
    submit = MagicMock(side_effect=RuntimeError("widget gone"))
    ctx.submit_input = submit
    assert await dispatch.queue_yes_tap(ctx) is True  # no raise


@pytest.mark.asyncio
async def test_queue_no_tap_skips_current():
    ctx = _make_ctx()
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    await dispatch.queue_no_tap(ctx)
    assert ctx.current_task is None
    assert ctx.queue.get(t.id).state == "pending"


@pytest.mark.asyncio
async def test_queue_no_tap_with_nothing_active_speaks():
    ctx = _make_ctx()
    await dispatch.queue_no_tap(ctx)
    ctx.tts.speak.assert_called_with("Nothing active.")


@pytest.mark.asyncio
async def test_dismiss_current_task_marks_done_and_stops_playback():
    """ACT+NO in queue mode: permanently drop the active task and
    interrupt any in-flight announcement."""
    ctx = _make_ctx()
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
    ctx = _make_ctx()
    await dispatch.dismiss_current_task(ctx)
    ctx.tts.speak.assert_called_with("Nothing active.")


@pytest.mark.asyncio
async def test_dismiss_advances_cursor_to_next_ranked_task():
    """ACT+NO mid-list: the cursor lands on the task below the
    dismissed one, not back at the top of the queue."""
    ctx = _make_ctx()
    _a, b, c = _seed_three(ctx)
    ctx.current_task = b
    with patch.object(modes, "speak_chunked") as spoke:
        await dispatch.dismiss_current_task(ctx)
    assert ctx.queue.get(b.id).state == "done"
    assert ctx.current_task is c
    spoke.assert_called()  # successor headline announced


@pytest.mark.asyncio
async def test_dismiss_bottom_task_wraps_cursor_to_top():
    ctx = _make_ctx()
    a, _b, c = _seed_three(ctx)
    ctx.current_task = c
    with patch.object(modes, "speak_chunked"):
        await dispatch.dismiss_current_task(ctx)
    assert ctx.queue.get(c.id).state == "done"
    assert ctx.current_task is a


@pytest.mark.asyncio
async def test_dismiss_only_task_leaves_cursor_empty():
    """No successor: cursor stays empty and the consumer's
    auto-announce owns the next move."""
    ctx = _make_ctx()
    t = ctx.queue.add(Task(headline="only one"))
    ctx.current_task = t
    await dispatch.dismiss_current_task(ctx)
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None


@pytest.mark.asyncio
async def test_dismiss_advance_touches_recent_topics_like_announce_next():
    """The advance replaces an auto-announce, so it carries the same
    scheduler bookkeeping — unlike exploratory arrow navigation."""
    ctx = _make_ctx()
    _a, b, c = _seed_three(ctx)
    ctx.current_task = b
    with patch.object(modes, "speak_chunked"):
        await dispatch.dismiss_current_task(ctx)
    assert c.topic in [topic for topic, _ts in ctx.recent_topics.as_list()]


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
    ctx = _make_ctx()
    a, _b, _c = _seed_three(ctx)
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=+1)
    assert ctx.current_task is a
    # Cursor doesn't change task state — the queue stays static.
    assert ctx.queue.get(a.id).state == "pending"
    assert len(ctx.queue.pending()) == 3


@pytest.mark.asyncio
async def test_queue_navigate_up_from_idle_points_cursor_at_bottom():
    ctx = _make_ctx()
    _a, _b, c = _seed_three(ctx)
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=-1)
    assert ctx.current_task is c


@pytest.mark.asyncio
async def test_queue_navigate_down_moves_cursor_without_disturbing_queue():
    ctx = _make_ctx()
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
    ctx = _make_ctx()
    a, b, _c = _seed_three(ctx)
    ctx.current_task = b
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=-1)
    assert ctx.current_task is a


@pytest.mark.asyncio
async def test_queue_navigate_wraps_at_end():
    ctx = _make_ctx()
    a, _b, c = _seed_three(ctx)
    ctx.current_task = c
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=+1)
    assert ctx.current_task is a  # down from the bottom wraps to top


@pytest.mark.asyncio
async def test_queue_navigate_wraps_at_start():
    ctx = _make_ctx()
    a, _b, c = _seed_three(ctx)
    ctx.current_task = a
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=-1)
    assert ctx.current_task is c  # up from the top wraps to bottom


@pytest.mark.asyncio
async def test_queue_navigate_on_empty_queue_speaks_empty():
    ctx = _make_ctx()
    await dispatch.queue_navigate(ctx, direction=+1)
    ctx.tts.speak.assert_called_with("Queue is empty.")
    assert ctx.current_task is None


@pytest.mark.asyncio
async def test_queue_navigate_does_not_touch_recent_topics():
    """Arrow browsing is exploratory — it shouldn't bias the scheduler
    the way an explicit voice ``next`` or skill engagement does."""
    ctx = _make_ctx()
    _a, _b, _c = _seed_three(ctx)
    before = ctx.recent_topics.as_list()
    with patch.object(modes, "speak_chunked"):
        await dispatch.queue_navigate(ctx, direction=+1)
        await dispatch.queue_navigate(ctx, direction=+1)
    assert ctx.recent_topics.as_list() == before


# --- handle_skill (ACT+PTT) ------------------------------------------------


@pytest.mark.asyncio
async def test_handle_skill_invokes_agent_and_marks_task_done():
    ctx = _make_ctx()
    mcp = create_autospec(ClaudeMCPClient, instance=True)
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
    ctx = _make_ctx()
    ctx.agent_mcp = MagicMock(enabled=True)
    await dispatch.handle_skill(ctx, "do something")
    ctx.tts.speak.assert_called_with("Nothing active to act on.")
    ctx.agent_mcp.run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_handle_skill_without_agent_mcp_speaks_error():
    ctx = _make_ctx()
    ctx.agent_mcp = None
    t = ctx.queue.add(Task(headline="x"))
    ctx.current_task = t
    await dispatch.handle_skill(ctx, "do something")
    ctx.tts.speak.assert_called_with("Agent MCP is not configured.")
    assert ctx.queue.get(t.id).state != "done"


@pytest.mark.asyncio
async def test_handle_skill_agent_error_keeps_task_active():
    ctx = _make_ctx()
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.enabled = True
    mcp.run_agent = AsyncMock(side_effect=RuntimeError("budget exceeded"))
    ctx.agent_mcp = mcp
    t = ctx.queue.add(Task(kind="email_msg", source={"thread_id": "T1"}))
    ctx.current_task = t
    await dispatch.handle_skill(ctx, "accept invite")
    assert ctx.queue.get(t.id).state != "done"
    assert ctx.current_task is t


@pytest.mark.asyncio
async def test_handle_skill_writes_queue_turn_to_real_session_log(tmp_path):
    """Regression: ``ctx.log.event("queue_turn", kind=task.kind, …)`` used
    to raise ``TypeError: got multiple values for argument 'kind'`` because
    ``SessionLogger.event(kind, **fields)`` takes ``kind`` positionally.

    The error fired *after* ``mark_done`` and *after* the agent had
    already run, so the task vanished while the side effect (Gmail
    archive) may not have completed — and the user heard nothing because
    the failure path bypassed the spoken summary. Wire a real
    SessionLogger here so the same kwarg-collision can't sneak back in.
    """
    from code_trip2.session_log import SessionLogger
    log = SessionLogger(tmp_path / "session.jsonl")
    ctx = _make_ctx()
    ctx.log = log
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.enabled = True
    mcp.run_agent = AsyncMock(return_value="Archived.")
    ctx.agent_mcp = mcp
    t = ctx.queue.add(Task(
        kind="email_msg",
        topic="cobalt",
        headline="x",
        source={"thread_id": "T1", "subject": "x"},
    ))
    ctx.current_task = t
    await dispatch.handle_skill(ctx, "archive that email")
    log.close()
    written = (tmp_path / "session.jsonl").read_text()
    assert '"kind": "queue_turn"' in written
    assert '"task_kind": "email_msg"' in written
    # And the spoken summary path completed (no crash before tts.speak).
    ctx.tts.speak.assert_called_with("Archived.")


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


# --- ACT+YES: create_linear_ticket_from_followup ---------------------------


def _followup_ctx(*, agent_reply: str | Exception = "Filed in AI: Draft doc."):
    """Context wired for the agent-driven follow-up filing path."""
    ctx = _make_ctx()
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.enabled = True
    if isinstance(agent_reply, Exception):
        mcp.run_agent = AsyncMock(side_effect=agent_reply)
    else:
        mcp.run_agent = AsyncMock(return_value=agent_reply)
    ctx.agent_mcp = mcp
    ctx.agent_allowed_tools = (
        "mcp__claude_ai_Linear__list_teams",
        "mcp__claude_ai_Linear__list_projects",
        "mcp__claude_ai_Linear__save_issue",
    )
    return ctx, mcp


async def _flush_bg(ctx) -> None:
    """Let the chord's background side effects run to completion."""
    while ctx.background_jobs:
        await asyncio.gather(*list(ctx.background_jobs))


@pytest.mark.asyncio
async def test_create_linear_ticket_invokes_file_meeting_followup_skill():
    ctx, mcp = _followup_ctx(
        agent_reply="Filed in AI / Retention: Draft retention metrics doc.",
    )
    t = ctx.queue.add(Task(
        kind="meeting_followup",
        topic="planning-sync",
        headline="Draft retention metrics doc",
        body="From planning sync: draft retention metrics doc for review.",
        source={"meeting": "Planning Sync"},
    ))
    ctx.current_task = t

    await dispatch.create_linear_ticket_from_followup(ctx, t)

    # Local effect is immediate: done + cursor cleared before the
    # agent reply lands.
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None
    ctx.tts.speak.assert_called_with("Filing.")

    await _flush_bg(ctx)

    mcp.run_agent.assert_awaited_once()
    call = mcp.run_agent.await_args
    prompt = call.kwargs["prompt"]
    # The prompt must name the skill, surface the task context the
    # skill body references, and forbid asking for confirmation.
    assert "file-meeting-followup" in prompt
    assert "Draft retention metrics doc" in prompt
    assert "planning-sync" in prompt
    assert "Planning Sync" in prompt  # meeting from source
    # Allowed tools propagate from Context so the agent can actually
    # call list_teams / save_issue.
    assert call.kwargs["allowed_tools"] == ctx.agent_allowed_tools

    assert ctx.queue.get(t.id).state == "done"
    ctx.tts.speak.assert_called_with(
        "Filed in AI / Retention: Draft retention metrics doc."
    )


@pytest.mark.asyncio
async def test_create_linear_ticket_uses_default_when_agent_returns_empty():
    """If the agent forgets to reply, the user still hears something."""
    ctx, _mcp = _followup_ctx(agent_reply="")
    t = ctx.queue.add(Task(kind="meeting_followup", headline="x"))
    ctx.current_task = t
    await dispatch.create_linear_ticket_from_followup(ctx, t)
    await _flush_bg(ctx)
    ctx.tts.speak.assert_called_with("Filed.")
    assert ctx.queue.get(t.id).state == "done"


@pytest.mark.asyncio
async def test_create_linear_ticket_without_agent_mcp_speaks_error_and_keeps_task():
    ctx, _mcp = _followup_ctx()
    ctx.agent_mcp = None
    t = ctx.queue.add(Task(kind="meeting_followup", headline="x"))
    ctx.current_task = t
    await dispatch.create_linear_ticket_from_followup(ctx, t)
    ctx.tts.speak.assert_called_with("Agent MCP is not configured.")
    assert ctx.queue.get(t.id).state != "done"


@pytest.mark.asyncio
async def test_create_linear_ticket_with_disabled_mcp_speaks_error():
    """``run_agent`` requires an enabled MCP — short-circuit before
    calling so a disabled client doesn't crash silently."""
    ctx, mcp = _followup_ctx()
    mcp.enabled = False
    t = ctx.queue.add(Task(kind="meeting_followup", headline="x"))
    ctx.current_task = t
    await dispatch.create_linear_ticket_from_followup(ctx, t)
    ctx.tts.speak.assert_called_with("Agent MCP is not configured.")
    mcp.run_agent.assert_not_called()
    assert ctx.queue.get(t.id).state != "done"


@pytest.mark.asyncio
async def test_create_linear_ticket_agent_failure_returns_task_to_pending():
    """Optimistic filing marks the task done at chord time; a failed
    agent run must resurrect it so the work isn't silently lost."""
    ctx, _mcp = _followup_ctx(agent_reply=RuntimeError("subprocess died"))
    t = ctx.queue.add(Task(kind="meeting_followup", headline="x"))
    ctx.current_task = t
    await dispatch.create_linear_ticket_from_followup(ctx, t)
    assert ctx.queue.get(t.id).state == "done"  # optimistic
    await _flush_bg(ctx)
    assert ctx.queue.get(t.id).state == "pending"  # resurrected
    spoken = ctx.tts.speak.call_args.args[0]
    assert "subprocess died" in spoken


@pytest.mark.asyncio
async def test_create_linear_ticket_runs_thinking_earcon():
    """Agent calls are slow (5–15s) — the thinking earcon must bookend
    the run so the user doesn't think the chord did nothing."""
    ctx, _mcp = _followup_ctx()
    t = ctx.queue.add(Task(kind="meeting_followup", headline="x"))
    ctx.current_task = t
    await dispatch.create_linear_ticket_from_followup(ctx, t)
    await _flush_bg(ctx)
    ctx.thinking.start.assert_called_once()
    ctx.thinking.stop.assert_called_once()


@pytest.mark.asyncio
async def test_create_linear_ticket_advances_cursor_to_successor():
    """ACT+YES leaves the queue like a dismissal does — the cursor
    moves to the next ranked task immediately."""
    ctx, _mcp = _followup_ctx()
    a, b, _c = _seed_three(ctx)
    a.kind = "meeting_followup"
    ctx.current_task = a
    with patch.object(modes, "speak_chunked"):
        await dispatch.create_linear_ticket_from_followup(ctx, a)
    assert ctx.current_task is b  # before the agent reply lands
    await _flush_bg(ctx)


@pytest.mark.asyncio
async def test_create_linear_ticket_local_effect_precedes_agent_run():
    """Responsiveness regression: while the agent is still running, the
    task is already done and the chord handler has returned."""
    ctx, mcp = _followup_ctx()
    gate = asyncio.Event()

    async def slow_agent(**kwargs):
        await gate.wait()
        return "Filed in AI: x."

    mcp.run_agent = AsyncMock(side_effect=slow_agent)
    t = ctx.queue.add(Task(kind="meeting_followup", headline="x"))
    ctx.current_task = t
    await dispatch.create_linear_ticket_from_followup(ctx, t)
    # Agent is parked on the gate; local state already settled.
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None
    gate.set()
    await _flush_bg(ctx)
    ctx.tts.speak.assert_called_with("Filed in AI: x.")


# --- ACT+YES: start_linear_ticket (remote /do-ticket) ----------------------


def _linear_ctx():
    """Context wired to reach the remote (non-empty ssh_host)."""
    ctx = _make_ctx()
    ctx.config.ssh_host = "coder.dec-8"
    return ctx


def _linear_task(ctx, identifier: str = "AI-1389"):
    t = ctx.queue.add(Task(
        kind="linear_issue",
        topic=identifier.lower(),
        headline=f"{identifier}: Do the thing",
        source={"identifier": identifier, "status": "Triage"},
    ))
    ctx.current_task = t
    return t


@pytest.mark.asyncio
async def test_start_linear_ticket_sends_do_ticket_and_marks_done():
    ctx = _linear_ctx()
    t = _linear_task(ctx)
    with patch.object(dispatch.remote, "send", new=AsyncMock()) as send:
        await dispatch.start_linear_ticket(ctx, t)
    # One send-keys into the standing Claude window, exact command.
    send.assert_awaited_once_with(
        "coder.dec-8", (), "main", "work", "/do-ticket AI-1389",
    )
    assert ctx.queue.get(t.id).state == "done"
    assert ctx.current_task is None
    ctx.tts.speak.assert_called_with("Starting AI-1389.")


@pytest.mark.asyncio
async def test_start_linear_ticket_does_not_touch_linear_status():
    """We never set the ticket to Todo — that would re-arm the auto-handle
    and double-fire /do-ticket on the next poll. The remote owns status."""
    ctx = _linear_ctx()
    ctx.linear_mcp = create_autospec(ClaudeMCPClient, instance=True)
    t = _linear_task(ctx)
    with patch.object(dispatch.remote, "send", new=AsyncMock()):
        await dispatch.start_linear_ticket(ctx, t)
    ctx.linear_mcp.call_tool.assert_not_called()
    # Source status left untouched — no local "Todo" write.
    assert ctx.queue.get(t.id).source["status"] == "Triage"


@pytest.mark.asyncio
async def test_start_linear_ticket_remote_error_keeps_task():
    ctx = _linear_ctx()
    t = _linear_task(ctx)
    with patch.object(
        dispatch.remote, "send",
        new=AsyncMock(side_effect=dispatch.remote.RemoteError("boom")),
    ):
        await dispatch.start_linear_ticket(ctx, t)
    # Send failed → task stays pending so the user can retry.
    assert ctx.queue.get(t.id).state == "pending"
    assert ctx.current_task is t
    ctx.tts.speak.assert_called_with("Could not reach Claude: boom")


@pytest.mark.asyncio
async def test_start_linear_ticket_without_ssh_speaks_error():
    ctx = _make_ctx()  # ssh_host == ""
    t = _linear_task(ctx)
    with patch.object(dispatch.remote, "send", new=AsyncMock()) as send:
        await dispatch.start_linear_ticket(ctx, t)
    send.assert_not_awaited()
    assert ctx.queue.get(t.id).state == "pending"
    ctx.tts.speak.assert_called_with("SSH is not configured.")


@pytest.mark.asyncio
async def test_start_linear_ticket_missing_identifier_keeps_task():
    ctx = _linear_ctx()
    t = ctx.queue.add(Task(kind="linear_issue", topic="x", headline="x", source={}))
    ctx.current_task = t
    with patch.object(dispatch.remote, "send", new=AsyncMock()) as send:
        await dispatch.start_linear_ticket(ctx, t)
    send.assert_not_awaited()
    assert ctx.queue.get(t.id).state == "pending"
    ctx.tts.speak.assert_called_with("Missing issue id for this task.")


@pytest.mark.asyncio
async def test_start_linear_ticket_advances_cursor_to_successor():
    ctx = _linear_ctx()
    a, b, _c = _seed_three(ctx)
    a.kind = "linear_issue"
    a.source = {"identifier": "AI-1", "status": "Triage"}
    ctx.current_task = a
    with patch.object(dispatch.remote, "send", new=AsyncMock()), \
            patch.object(modes, "speak_chunked"):
        await dispatch.start_linear_ticket(ctx, a)
    assert ctx.current_task is b
