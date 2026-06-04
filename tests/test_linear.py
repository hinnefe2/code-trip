"""Tests for LinearState, LinearProducer, and the Linear reply path."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from code_trip2 import chords, dispatch, modes
from code_trip2.linear_state import LinearState
from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.producers.linear import LinearProducer
from code_trip2.tasks import Task, TaskQueue
from conftest import make_mock_tts


# --- LinearState ----------------------------------------------------------


def test_linear_state_roundtrip(tmp_path: Path):
    p = tmp_path / "linear-state.json"
    s = LinearState(path=p)
    s.set_last_updated_at("2026-05-28T17:00:53.328Z")
    assert LinearState(path=p).last_updated_at() == "2026-05-28T17:00:53.328Z"


def test_linear_state_missing_file_is_empty(tmp_path: Path):
    assert LinearState(path=tmp_path / "nope.json").last_updated_at() is None


def test_linear_state_refuses_to_regress(tmp_path: Path):
    s = LinearState(path=tmp_path / "s.json")
    s.set_last_updated_at("2026-05-28T17:00:53.328Z")
    s.set_last_updated_at("2026-05-28T10:00:00.000Z")
    assert s.last_updated_at() == "2026-05-28T17:00:53.328Z"


def test_linear_state_handles_corrupt_file(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{ not json")
    assert LinearState(path=p).last_updated_at() is None


# --- LinearProducer ------------------------------------------------------


def _producer(
    tmp_path: Path,
    *,
    poll_interval=180.0,
    state_types=("triage", "unstarted", "started"),
    max_results=50,
):
    cfg = SimpleNamespace(
        linear_poll_interval=poll_interval,
        linear_state_types=tuple(state_types),
        linear_max_results=max_results,
    )
    state = LinearState(path=tmp_path / "linear-state.json")
    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.enabled = True
    q = TaskQueue()
    p = LinearProducer(config=cfg, queue=q, mcp=mcp, state=state)
    return p, q, mcp, state


@pytest.mark.asyncio
async def test_producer_skips_when_mcp_unavailable(tmp_path: Path):
    cfg = SimpleNamespace(
        linear_poll_interval=180.0,
        linear_state_types=("started",),
        linear_max_results=50,
    )
    p = LinearProducer(config=cfg, queue=TaskQueue(), mcp=None)
    await asyncio.wait_for(p.run(), timeout=1.0)


@pytest.mark.asyncio
async def test_producer_wide_first_poll_calls_per_state_type(tmp_path: Path):
    """Wide poll pushes the state filter server-side — one call per allowed
    state, no ``updatedAt`` floor. Avoids the response bloat that would
    otherwise come from unfiltered ``orderBy: updatedAt`` queries."""
    p, _q, mcp, state = _producer(
        tmp_path, state_types=("triage", "unstarted", "started"),
    )
    state.set_last_updated_at("2026-05-01T00:00:00.000Z")  # would normally constrain
    mcp.call_tool.return_value = {"issues": []}
    await p._poll_once()
    assert mcp.call_tool.call_count == 3
    states_called = sorted(c.args[1]["state"] for c in mcp.call_tool.call_args_list)
    assert states_called == ["started", "triage", "unstarted"]
    for c in mcp.call_tool.call_args_list:
        assert c.args[0] == "list_issues"
        args = c.args[1]
        assert "updatedAt" not in args
        assert args.get("assignee") == "me"
        assert args.get("includeArchived") is False


@pytest.mark.asyncio
async def test_producer_incremental_poll_single_call_with_updated_at(tmp_path: Path):
    """Incremental poll is a single unfiltered call with the ``updatedAt``
    cursor — the response window is small (only what changed since the
    cursor) so we can afford to filter the few state mismatches client-side."""
    p, _q, mcp, state = _producer(tmp_path)
    state.set_last_updated_at("2026-05-28T10:00:00.000Z")
    p._first_poll = False  # simulate not-the-first-poll-of-the-session
    mcp.call_tool.return_value = {"issues": []}
    await p._poll_once()
    assert mcp.call_tool.call_count == 1
    args = mcp.call_tool.call_args.args[1]
    assert args["updatedAt"] == "2026-05-28T10:00:00.000Z"
    assert "state" not in args


@pytest.mark.asyncio
async def test_producer_wide_pull_keeps_running_when_one_state_fails(tmp_path: Path):
    """A transient failure on one state doesn't void the whole wide poll;
    the surviving states still populate."""
    p, q, mcp, _state = _producer(
        tmp_path, state_types=("unstarted", "started"),
    )

    async def fake_call(_tool, args):
        if args["state"] == "unstarted":
            raise ClaudeMCPError("network")
        return {
            "issues": [{
                "id": "AI-7",
                "title": "Active",
                "statusType": "started",
                "url": "u",
                "updatedAt": "2026-05-28T10:00:00.000Z",
            }]
        }

    mcp.call_tool.side_effect = fake_call
    await p._poll_once()
    [task] = q.all()
    assert task.source["identifier"] == "AI-7"


@pytest.mark.asyncio
async def test_producer_first_poll_flag_flips_after_one_call(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"issues": []}
    assert p._first_poll is True
    await p._poll_once()
    assert p._first_poll is False


@pytest.mark.asyncio
async def test_producer_emits_task_from_structured_issues(tmp_path: Path):
    p, q, mcp, state = _producer(tmp_path, state_types=("started",))
    mcp.call_tool.return_value = {
        "issues": [
            {
                "id": "AI-1389",
                "title": "Hide irrelevant skills from claude code",
                "description": "Long description here",
                "status": "In Review",
                "statusType": "started",
                "url": "https://linear.app/picnichealth/issue/AI-1389/hide",
                "updatedAt": "2026-05-28T17:00:53.328Z",
                "priority": {"value": 2, "name": "High"},
            }
        ],
        "hasNextPage": False,
    }
    await p._poll_once()
    [task] = q.all()
    assert task.kind == "linear_issue"
    assert task.source["identifier"] == "AI-1389"
    assert task.source["url"].endswith("/AI-1389/hide")
    assert task.source["status"] == "In Review"
    assert task.source["priority"] == "High"
    assert "AI-1389" in task.headline
    assert "Hide irrelevant skills" in task.headline
    assert task.topic == "ai-1389"
    # Linear-API tasks always carry the namespaced subject key so the
    # queue can cluster them with cross-producer references (e.g. a
    # Gmail notification about the same issue).
    assert task.subject_key == "linear:AI-1389"
    assert state.last_updated_at() == "2026-05-28T17:00:53.328Z"


@pytest.mark.asyncio
async def test_producer_filters_out_disallowed_state_types(tmp_path: Path):
    """statusType outside the allow-list never becomes a task. Belt-and-
    suspenders for the incremental path, which doesn't push the state
    filter server-side."""
    p, q, mcp, _state = _producer(tmp_path, state_types=("started",))
    p._first_poll = False  # incremental path: no server-side state filter
    mcp.call_tool.return_value = {
        "issues": [
            {
                "id": "AI-1",
                "title": "Active",
                "statusType": "started",
                "url": "https://linear.app/x/issue/AI-1",
                "updatedAt": "2026-05-28T17:00:00.000Z",
            },
            {
                "id": "AI-2",
                "title": "Done",
                "statusType": "completed",
                "url": "https://linear.app/x/issue/AI-2",
                "updatedAt": "2026-05-28T18:00:00.000Z",
            },
            {
                "id": "AI-3",
                "title": "Todo",
                "statusType": "unstarted",
                "url": "https://linear.app/x/issue/AI-3",
                "updatedAt": "2026-05-28T16:00:00.000Z",
            },
        ]
    }
    await p._poll_once()
    tasks = q.all()
    assert len(tasks) == 1
    assert tasks[0].source["identifier"] == "AI-1"


@pytest.mark.asyncio
async def test_producer_cursor_advances_past_filtered_issues(tmp_path: Path):
    """Filtered-out issues still bump the cursor so we don't keep re-pulling them."""
    p, _q, mcp, state = _producer(tmp_path, state_types=("started",))
    p._first_poll = False  # incremental path: cursor advances even past filtered
    mcp.call_tool.return_value = {
        "issues": [
            {
                "id": "AI-X",
                "title": "Done",
                "statusType": "completed",
                "url": "u",
                "updatedAt": "2026-05-28T18:00:00.000Z",
            },
        ]
    }
    await p._poll_once()
    assert state.last_updated_at() == "2026-05-28T18:00:00.000Z"


@pytest.mark.asyncio
async def test_producer_retires_task_when_ticket_leaves_active_set(tmp_path: Path):
    """If a ticket the producer previously surfaced now returns with a
    non-allowed statusType (user closed/canceled it in Linear), the
    pending task is marked done. Linear has no push notification for
    status changes, so this incremental-poll sweep is what keeps the
    queue in sync without restart."""
    p, q, mcp, _state = _producer(tmp_path, state_types=("started",))
    p._first_poll = False  # incremental path is where the sweep runs

    # First poll surfaces an active ticket.
    mcp.call_tool.return_value = {
        "issues": [{
            "id": "AI-7",
            "title": "active",
            "statusType": "started",
            "url": "u",
            "updatedAt": "2026-05-28T10:00:00.000Z",
        }]
    }
    await p._poll_once()
    [task] = q.pending()
    assert task.source["identifier"] == "AI-7"

    # User closes AI-7 in Linear → next incremental poll returns it
    # with statusType="completed". The producer should mark the
    # existing task done rather than just skipping it.
    mcp.call_tool.return_value = {
        "issues": [{
            "id": "AI-7",
            "title": "active",
            "statusType": "completed",
            "url": "u",
            "updatedAt": "2026-05-28T11:00:00.000Z",
        }]
    }
    await p._poll_once()
    assert q.pending() == []
    assert q.get(task.id).state == "done"


@pytest.mark.asyncio
async def test_producer_filtered_issue_with_no_pending_task_is_silent(tmp_path: Path):
    """A filtered-out issue we never had a task for is a plain skip —
    we don't try to mark anything done and don't error."""
    p, q, mcp, _state = _producer(tmp_path, state_types=("started",))
    p._first_poll = False
    mcp.call_tool.return_value = {
        "issues": [{
            "id": "AI-NEVERSEEN",
            "title": "done before we noticed",
            "statusType": "completed",
            "url": "u",
            "updatedAt": "2026-05-28T11:00:00.000Z",
        }]
    }
    await p._poll_once()
    assert q.all() == []


@pytest.mark.asyncio
async def test_producer_collapses_repeat_sightings_into_single_task(tmp_path: Path):
    """Seeing the same identifier on the next poll updates the existing task."""
    p, q, mcp, _state = _producer(tmp_path, state_types=("started",))
    mcp.call_tool.return_value = {
        "issues": [
            {
                "id": "AI-7",
                "title": "Original title",
                "description": "first body",
                "statusType": "started",
                "url": "https://linear.app/x/issue/AI-7",
                "updatedAt": "2026-05-28T10:00:00.000Z",
            }
        ]
    }
    await p._poll_once()
    [first] = q.all()
    mcp.call_tool.return_value = {
        "issues": [
            {
                "id": "AI-7",
                "title": "Updated title",
                "description": "second body",
                "statusType": "started",
                "url": "https://linear.app/x/issue/AI-7",
                "updatedAt": "2026-05-28T11:00:00.000Z",
            }
        ]
    }
    await p._poll_once()
    pending = q.pending()
    assert len(pending) == 1
    assert pending[0].id == first.id
    assert "Updated title" in pending[0].headline
    assert "second body" in pending[0].body


@pytest.mark.asyncio
async def test_producer_collapses_update_into_active_issue_task(tmp_path: Path):
    """A Linear update for a ticket the user is currently viewing
    (task is ACTIVE) refreshes the open task in place, not a sibling
    in the queue."""
    p, q, mcp, _state = _producer(tmp_path, state_types=("started",))
    mcp.call_tool.return_value = {
        "issues": [
            {
                "id": "AI-7",
                "title": "Original",
                "description": "first body",
                "statusType": "started",
                "url": "https://linear.app/x/issue/AI-7",
                "updatedAt": "2026-05-28T10:00:00.000Z",
            }
        ]
    }
    await p._poll_once()
    [first] = q.all()
    q.mark_active(first.id)  # user opened it in the Current panel

    mcp.call_tool.return_value = {
        "issues": [
            {
                "id": "AI-7",
                "title": "Title updated mid-view",
                "description": "fresh body",
                "statusType": "started",
                "url": "https://linear.app/x/issue/AI-7",
                "updatedAt": "2026-05-28T11:00:00.000Z",
            }
        ]
    }
    await p._poll_once()

    all_tasks = q.all()
    assert len(all_tasks) == 1, (
        "active task should absorb the update, not spawn a sibling"
    )
    assert all_tasks[0].id == first.id
    assert "Title updated mid-view" in all_tasks[0].headline


@pytest.mark.asyncio
async def test_producer_handles_empty_results(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path, state_types=("started",))
    mcp.call_tool.return_value = {"issues": []}
    await p._poll_once()
    assert q.all() == []


@pytest.mark.asyncio
async def test_producer_resilient_to_mcp_errors(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path, state_types=("started",))
    mcp.call_tool.side_effect = ClaudeMCPError("network")
    await p._poll_once()  # must not raise
    assert q.all() == []
    # is_polling must clear even on error — otherwise the TUI would
    # be stuck showing "polling" forever after a transient failure.
    assert p.is_polling is False


@pytest.mark.asyncio
async def test_producer_is_polling_true_while_call_in_flight(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path, state_types=("started",))
    observed: list[bool] = []

    async def slow_call(*_a, **_kw):
        observed.append(p.is_polling)
        return {"issues": []}

    mcp.call_tool.side_effect = slow_call
    assert p.is_polling is False
    await p._poll_once()
    assert observed == [True]
    assert p.is_polling is False


@pytest.mark.asyncio
async def test_producer_routes_through_intake(tmp_path: Path):
    """Producer emits via the injected intake (so the screener gets first look)."""
    cfg = SimpleNamespace(
        linear_poll_interval=180.0,
        linear_state_types=("started",),
        linear_max_results=50,
    )
    state = LinearState(path=tmp_path / "linear-state.json")
    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.enabled = True
    mcp.call_tool = AsyncMock(return_value={
        "issues": [{
            "id": "AI-99",
            "title": "Intake test",
            "statusType": "started",
            "url": "u",
            "updatedAt": "2026-05-28T10:00:00.000Z",
        }]
    })
    q = TaskQueue()
    intake_called: list[Task] = []
    p = LinearProducer(
        config=cfg, queue=q, mcp=mcp, state=state,
        intake=intake_called.append,
    )
    await p._poll_once()
    assert len(intake_called) == 1
    assert intake_called[0].source["identifier"] == "AI-99"
    # And the intake bypasses queue.add (the test intake just appends).
    assert q.all() == []


# --- reply path ----------------------------------------------------------


def _ctx_with_linear_mcp(mcp):
    cfg = SimpleNamespace(
        ssh_host="", ssh_options=(), tmux_session="main",
        work_window="work", linear_window="linear", terminal_apps=("kitty",),
    )
    return modes.Context(
        config=cfg,  # type: ignore[arg-type]
        tts=make_mock_tts(),
        log=MagicMock(),
        thinking=MagicMock(),
        linear_mcp=mcp,
    )


@pytest.mark.asyncio
async def test_respond_linear_posts_comment_via_save_comment():
    mcp = MagicMock()
    mcp.call_tool = AsyncMock()
    ctx = _ctx_with_linear_mcp(mcp)
    task = Task(
        kind="linear_issue",
        topic="ai-7",
        source={
            "identifier": "AI-7",
            "url": "https://linear.app/x/issue/AI-7",
            "title": "Help",
        },
    )
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_linear(ctx, task, "Looks good, shipping today")
    mcp.call_tool.assert_called_once()
    call = mcp.call_tool.call_args
    assert call.args[0] == "save_comment"
    args = call.args[1]
    assert args == {"issueId": "AI-7", "body": "Looks good, shipping today"}
    assert ctx.queue.get(task.id).state == "done"
    assert ctx.current_task is None


@pytest.mark.asyncio
async def test_respond_linear_without_mcp_speaks_error():
    ctx = _ctx_with_linear_mcp(mcp=None)
    task = Task(kind="linear_issue", source={"identifier": "AI-7"})
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_linear(ctx, task, "reply")
    ctx.tts.speak.assert_called_with("Linear MCP is not configured.")
    assert ctx.queue.get(task.id).state != "done"


@pytest.mark.asyncio
async def test_respond_linear_without_identifier_speaks_error():
    mcp = MagicMock()
    mcp.call_tool = AsyncMock()
    ctx = _ctx_with_linear_mcp(mcp)
    task = Task(kind="linear_issue", source={"url": "https://x"})  # no identifier
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_linear(ctx, task, "reply")
    mcp.call_tool.assert_not_called()
    assert ctx.queue.get(task.id).state != "done"


@pytest.mark.asyncio
async def test_respond_linear_api_error_keeps_task_active():
    mcp = MagicMock()
    mcp.call_tool = AsyncMock(side_effect=ClaudeMCPError("network down"))
    ctx = _ctx_with_linear_mcp(mcp)
    task = Task(kind="linear_issue", source={"identifier": "AI-7"})
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_linear(ctx, task, "reply")
    assert ctx.queue.get(task.id).state != "done"
    assert ctx.current_task is task


@pytest.mark.asyncio
async def test_dispatch_task_response_routes_linear_kind():
    mcp = MagicMock()
    mcp.call_tool = AsyncMock()
    ctx = _ctx_with_linear_mcp(mcp)
    task = Task(kind="linear_issue", source={"identifier": "AI-7"})
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._dispatch_task_response(ctx, task, "ship it")
    mcp.call_tool.assert_called_once()
    assert mcp.call_tool.call_args.args[0] == "save_comment"


# --- spoken labels --------------------------------------------------------


def test_kind_label_linear_issue_includes_identifier():
    task = Task(kind="linear_issue", source={"identifier": "AI-7"})
    assert dispatch._kind_label(task) == "Linear AI-7"


def test_kind_label_linear_issue_without_identifier_falls_back():
    task = Task(kind="linear_issue", source={})
    assert dispatch._kind_label(task) == "Linear issue"


# --- ACT+YES open-in-browser URL resolution -------------------------------


def test_task_browser_url_linear_uses_source_url():
    task = Task(
        kind="linear_issue",
        source={"url": "https://linear.app/picnichealth/issue/AI-7/help"},
    )
    assert chords._task_browser_url(task) == "https://linear.app/picnichealth/issue/AI-7/help"


def test_task_browser_url_linear_without_url_returns_none():
    task = Task(kind="linear_issue", source={"identifier": "AI-7"})
    assert chords._task_browser_url(task) is None


def test_task_browser_url_email_still_builds_gmail_url():
    task = Task(kind="email_msg", source={"thread_id": "T1"})
    url = chords._task_browser_url(task)
    assert url is not None
    assert "T1" in url
    assert "mail.google.com" in url


def test_task_browser_url_slack_uses_permalink():
    task = Task(
        kind="slack_msg",
        source={"url": "https://picnichealth.slack.com/archives/C09/p1779911327920139"},
    )
    assert chords._task_browser_url(task) == (
        "https://picnichealth.slack.com/archives/C09/p1779911327920139"
    )


def test_task_browser_url_slack_without_url_returns_none():
    task = Task(kind="slack_msg", source={"channel_id": "C09"})
    assert chords._task_browser_url(task) is None


def test_task_browser_url_unknown_kind_returns_none():
    task = Task(kind="claude_reply", source={})
    assert chords._task_browser_url(task) is None
