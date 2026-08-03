"""Unit tests for WindowProducer with mocked remote calls."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from code_trip2 import remote
from code_trip2.producers.windows import WindowProducer
from code_trip2.tasks import STATE_DONE, STATE_PENDING, Task, TaskQueue

WAITING_CAP = """\
● Done. Tests pass.

╭──────────────╮
│ ❯            │
╰──────────────╯
"""

RUNNING_CAP = "✻ Thinking… (esc to interrupt)"

PERMISSION_CAP = """\
│ Do you want to proceed?  │
│ ❯ 1. Yes                 │
│   2. No                  │
"""


def _make_producer(**kwargs):
    cfg = SimpleNamespace(
        ssh_host="remote",
        ssh_options=(),
        tmux_session="dev",
        window_poll_interval=1.5,
        window_capture_lines=100,
    )
    q = TaskQueue()
    p = WindowProducer(config=cfg, queue=q, **kwargs)  # type: ignore[arg-type]
    return p, q


def _remote_mocks(windows, captures):
    """Patch list_windows/capture. ``captures`` maps window name to pane
    text or an exception to raise."""
    async def fake_capture(host, opts, session, window, **kw):
        result = captures[window]
        if isinstance(result, Exception):
            raise result
        return result

    return (
        patch(
            "code_trip2.remote.list_windows",
            AsyncMock(return_value=[(i, w, "/w") for i, w in enumerate(windows)]),
        ),
        patch("code_trip2.remote.capture", side_effect=fake_capture),
    )


@pytest.mark.asyncio
async def test_waiting_ticket_window_becomes_task():
    p, q = _make_producer()
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": WAITING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())

    tasks = q.all()
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == "remote_window"
    assert t.topic == "AI-1"
    assert t.headline == "waiting for input"
    assert t.origin_key == "claude:AI-1"
    assert t.subject_key == "linear:AI-1"
    assert t.source["claude_state"] == "waiting_input"
    assert p._mirrors["AI-1"] == WAITING_CAP


@pytest.mark.asyncio
async def test_permission_dialog_headline_includes_question():
    p, q = _make_producer()
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": PERMISSION_CAP})
    with lw, cap:
        await p._poll_once("remote", ())

    t = q.all()[0]
    assert t.headline == "needs permission: Do you want to proceed?"
    assert t.source["claude_state"] == "waiting_permission"
    assert t.body == "Do you want to proceed?"


@pytest.mark.asyncio
async def test_non_ticket_windows_ignored():
    p, q = _make_producer()
    lw, cap = _remote_mocks(
        ["work", "linear", "ai-1", "AI-1-followup"],
        {},  # capture must never be called
    )
    with lw, cap:
        await p._poll_once("remote", ())
    assert q.all() == []
    assert p._mirrors == {}


@pytest.mark.asyncio
async def test_running_window_has_no_task_but_mirror_updates():
    p, q = _make_producer()
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": RUNNING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())
    assert q.all() == []
    assert p._mirrors["AI-1"] == RUNNING_CAP


@pytest.mark.asyncio
async def test_waiting_then_running_retires_task():
    p, q = _make_producer()
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": WAITING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())
    assert q.pending()

    lw, cap = _remote_mocks(["AI-1"], {"AI-1": RUNNING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())
    assert q.pending() == []
    assert q.all()[0].state == STATE_DONE


@pytest.mark.asyncio
async def test_vanished_window_retires_task_and_drops_entries():
    p, q = _make_producer()
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": WAITING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())

    lw, cap = _remote_mocks([], {})
    with lw, cap:
        await p._poll_once("remote", ())
    assert q.pending() == []
    assert "AI-1" not in p._mirrors
    assert "AI-1" not in p._last_state


@pytest.mark.asyncio
async def test_steady_waiting_fires_no_queue_events():
    p, q = _make_producer()
    events: list[str] = []
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": WAITING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())
    q.add_listener(lambda event, task: events.append(event))
    with lw, cap:
        await p._poll_once("remote", ())
        await p._poll_once("remote", ())
    assert events == []


@pytest.mark.asyncio
async def test_substate_flap_does_not_resurrect_answered_task():
    p, q = _make_producer()
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": WAITING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())
    q.mark_done(q.all()[0].id)

    # Still waiting, but the sub-state changed (e.g. the user's answer
    # opened a permission dialog before any run started).
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": PERMISSION_CAP})
    with lw, cap:
        await p._poll_once("remote", ())
    assert q.pending() == []
    assert len(q.all()) == 1


@pytest.mark.asyncio
async def test_new_turn_after_done_mints_fresh_task():
    p, q = _make_producer()
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": WAITING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())
    first_id = q.all()[0].id
    q.mark_done(first_id)

    lw, cap = _remote_mocks(["AI-1"], {"AI-1": RUNNING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())

    lw, cap = _remote_mocks(["AI-1"], {"AI-1": WAITING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())

    pending = q.pending()
    assert len(pending) == 1
    assert pending[0].id != first_id
    assert pending[0].state == STATE_PENDING


@pytest.mark.asyncio
async def test_capture_error_for_one_window_keeps_others_working():
    p, q = _make_producer()
    lw, cap = _remote_mocks(
        ["AI-1", "AI-2"],
        {"AI-1": remote.RemoteError("boom"), "AI-2": WAITING_CAP},
    )
    with lw, cap:
        await p._poll_once("remote", ())

    tasks = q.all()
    assert len(tasks) == 1
    assert tasks[0].topic == "AI-2"
    assert "AI-1" not in p._mirrors  # no stale/fake mirror invented


@pytest.mark.asyncio
async def test_capture_error_does_not_flap_existing_task():
    p, q = _make_producer()
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": WAITING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())
    assert len(q.pending()) == 1

    lw, cap = _remote_mocks(["AI-1"], {"AI-1": remote.RemoteError("boom")})
    with lw, cap:
        await p._poll_once("remote", ())
    # Task and mirror survive a transient capture failure.
    assert len(q.pending()) == 1
    assert p._mirrors["AI-1"] == WAITING_CAP


@pytest.mark.asyncio
async def test_external_submit_receives_if_terminal():
    """The screening-gate submit in main.py takes (task, *, if_terminal)."""
    calls: list[tuple[Task, str]] = []

    def submit(task: Task, *, if_terminal: str = "new") -> Task:
        calls.append((task, if_terminal))
        return task

    p, _q = _make_producer(submit=submit)
    lw, cap = _remote_mocks(["AI-1"], {"AI-1": WAITING_CAP})
    with lw, cap:
        await p._poll_once("remote", ())
    assert len(calls) == 1
    assert calls[0][1] == "new"
