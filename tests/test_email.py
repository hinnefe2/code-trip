"""Tests for EmailState, EmailProducer, and the email reply path."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from code_trip2 import dispatch, modes
from code_trip2.email_state import EmailState
from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.producers.email import EmailProducer, _parse_ts, _split_name_addr
from code_trip2.tasks import Task, TaskQueue


# --- EmailState ----------------------------------------------------------


def test_email_state_roundtrip(tmp_path: Path):
    p = tmp_path / "email-state.json"
    s = EmailState(path=p)
    s.set_last_message_ts(1716000000)
    assert EmailState(path=p).last_message_ts() == 1716000000


def test_email_state_missing_file_is_empty(tmp_path: Path):
    assert EmailState(path=tmp_path / "nope.json").last_message_ts() is None


def test_email_state_refuses_to_regress(tmp_path: Path):
    s = EmailState(path=tmp_path / "s.json")
    s.set_last_message_ts(1716000100)
    s.set_last_message_ts(1716000050)
    assert s.last_message_ts() == 1716000100


def test_email_state_handles_corrupt_file(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{ not json")
    assert EmailState(path=p).last_message_ts() is None


# --- helpers -------------------------------------------------------------


def test_parse_ts_handles_epoch_seconds():
    assert _parse_ts(1716000000) == 1716000000
    assert _parse_ts("1716000000") == 1716000000


def test_parse_ts_handles_epoch_millis():
    """Gmail's internalDate is in milliseconds; we collapse to seconds."""
    assert _parse_ts("1716000000000") == 1716000000
    assert _parse_ts(1716000000_000) == 1716000000


def test_parse_ts_handles_iso_strings():
    """ISO-ish formats should yield a unix timestamp."""
    out = _parse_ts("2026-05-20 13:43:54")
    assert out > 0


def test_parse_ts_empty_inputs_return_zero():
    assert _parse_ts(None) == 0
    assert _parse_ts("") == 0
    assert _parse_ts("garbage") == 0


def test_split_name_addr_name_and_email():
    assert _split_name_addr("Alice Smith <alice@example.com>") == ("Alice Smith", "alice@example.com")


def test_split_name_addr_quoted_name():
    assert _split_name_addr('"Alice Smith" <alice@example.com>') == ("Alice Smith", "alice@example.com")


def test_split_name_addr_bare_email():
    assert _split_name_addr("alice@example.com") == ("", "alice@example.com")


def test_split_name_addr_empty():
    assert _split_name_addr("") == ("", "")


# --- EmailProducer -------------------------------------------------------


def _producer(tmp_path: Path, *, poll_interval=120.0, search_query="in:inbox -from:me is:unread", max_results=20):
    cfg = SimpleNamespace(
        email_poll_interval=poll_interval,
        email_search_query=search_query,
        email_max_results=max_results,
    )
    state = EmailState(path=tmp_path / "email-state.json")
    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.enabled = True
    q = TaskQueue()
    p = EmailProducer(config=cfg, queue=q, mcp=mcp, state=state)
    return p, q, mcp, state


@pytest.mark.asyncio
async def test_producer_skips_when_mcp_unavailable(tmp_path: Path):
    cfg = SimpleNamespace(
        email_poll_interval=120.0,
        email_search_query="in:inbox",
        email_max_results=20,
    )
    p = EmailProducer(config=cfg, queue=TaskQueue(), mcp=None)
    await asyncio.wait_for(p.run(), timeout=1.0)


@pytest.mark.asyncio
async def test_producer_passes_after_param_in_query(tmp_path: Path):
    """Each poll should append ``after:<unix_ts>`` to the configured query."""
    p, _q, mcp, state = _producer(tmp_path)
    state.set_last_message_ts(1716000000)
    mcp.call_tool.return_value = {"threads": []}
    await p._poll_once()
    call = mcp.call_tool.call_args
    assert call.args[0] == "search_threads"
    assert "after:1716000000" in call.args[1]["query"]
    assert "in:inbox" in call.args[1]["query"]


@pytest.mark.asyncio
async def test_producer_emits_task_from_structured_threads(tmp_path: Path):
    p, q, mcp, state = _producer(tmp_path)
    mcp.call_tool.return_value = {
        "threads": [
            {
                "id": "T1",
                "messages": [
                    {
                        "id": "M1",
                        "subject": "Project update",
                        "from": "Alice <alice@example.com>",
                        "snippet": "Here is the latest",
                        "internalDate": "1716000100000",  # ms
                    }
                ],
            }
        ]
    }
    await p._poll_once()
    [task] = q.all()
    assert task.kind == "email_msg"
    assert task.source["thread_id"] == "T1"
    assert task.source["message_id"] == "M1"
    assert task.source["sender_email"] == "alice@example.com"
    assert task.source["sender_name"] == "Alice"
    assert "Project update" in task.headline
    assert "Alice" in task.headline
    assert state.last_message_ts() == 1716000100


@pytest.mark.asyncio
async def test_producer_skips_threads_at_or_before_last_ts(tmp_path: Path):
    p, q, mcp, state = _producer(tmp_path)
    state.set_last_message_ts(1716000100)
    mcp.call_tool.return_value = {
        "threads": [
            {
                "id": "T1",
                "messages": [
                    {
                        "id": "M1",
                        "subject": "old",
                        "from": "x@y.com",
                        "internalDate": "1716000050000",
                    }
                ],
            },
            {
                "id": "T2",
                "messages": [
                    {
                        "id": "M2",
                        "subject": "new",
                        "from": "y@z.com",
                        "internalDate": "1716000200000",
                    }
                ],
            },
        ]
    }
    await p._poll_once()
    [task] = q.all()
    assert task.source["thread_id"] == "T2"
    assert state.last_message_ts() == 1716000200


@pytest.mark.asyncio
async def test_producer_collapses_thread_replies_into_single_task(tmp_path: Path):
    """A new message in the same thread updates the existing task — no
    pile-up of N tasks for one conversation."""
    p, q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {
        "threads": [
            {
                "id": "TX",
                "messages": [
                    {
                        "id": "M1",
                        "subject": "Help",
                        "from": "Alice <alice@example.com>",
                        "snippet": "first email",
                        "internalDate": "1716000100000",
                    }
                ],
            }
        ]
    }
    await p._poll_once()
    [first] = q.all()
    # Force the next poll to consider the new ts even though the producer's
    # dedup key is per (thread, message) — we want to test thread collapse.
    mcp.call_tool.return_value = {
        "threads": [
            {
                "id": "TX",
                "messages": [
                    {
                        "id": "M2",
                        "subject": "Help",
                        "from": "Alice <alice@example.com>",
                        "snippet": "follow-up email",
                        "internalDate": "1716000200000",
                    }
                ],
            }
        ]
    }
    await p._poll_once()
    pending = q.pending()
    assert len(pending) == 1
    assert pending[0].id == first.id
    assert "follow-up email" in pending[0].body
    assert pending[0].source["message_id"] == "M2"


@pytest.mark.asyncio
async def test_producer_handles_empty_results(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"threads": []}
    await p._poll_once()
    assert q.all() == []


@pytest.mark.asyncio
async def test_producer_resilient_to_mcp_errors(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.side_effect = ClaudeMCPError("network")
    await p._poll_once()  # must not raise
    assert q.all() == []


@pytest.mark.asyncio
async def test_producer_parses_markdown_fallback(tmp_path: Path):
    """If the MCP ever returns a Slack-style detailed markdown blob, we
    still extract the thread fields."""
    p, q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {
        "result": (
            "# Email search results\n\n"
            "### Thread 1 of 1\n"
            "Subject: Project update\n"
            "From: Alice <alice@example.com>\n"
            "Thread ID: TMD\n"
            "Message ID: MMD\n"
            "Date: 2026-05-20 13:43:54\n"
            "Snippet: here is the latest\n"
            "---\n"
        )
    }
    await p._poll_once()
    [task] = q.all()
    assert task.source["thread_id"] == "TMD"
    assert task.source["message_id"] == "MMD"
    assert "Project update" in task.headline


# --- reply path ----------------------------------------------------------


def _ctx_with_email_mcp(mcp):
    cfg = SimpleNamespace(
        ssh_host="", ssh_options=(), tmux_session="main",
        work_window="work", linear_window="linear", terminal_apps=("kitty",),
    )
    tts = MagicMock()
    tts.speak = AsyncMock(return_value=None)
    return modes.Context(
        config=cfg,  # type: ignore[arg-type]
        tts=tts,
        log=MagicMock(),
        thinking=MagicMock(),
        email_mcp=mcp,
    )


@pytest.mark.asyncio
async def test_respond_email_creates_draft_via_mcp():
    mcp = MagicMock()
    mcp.call_tool = AsyncMock()
    ctx = _ctx_with_email_mcp(mcp)
    task = Task(
        kind="email_msg",
        topic="alice",
        source={
            "thread_id": "T1",
            "message_id": "M1",
            "sender_name": "Alice",
            "sender_email": "alice@example.com",
            "subject": "Help",
        },
    )
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_email(ctx, task, "Sure, ill take a look")
    mcp.call_tool.assert_called_once()
    call = mcp.call_tool.call_args
    assert call.args[0] == "create_draft"
    args = call.args[1]
    assert args["to"] == ["alice@example.com"]
    assert args["replyToMessageId"] == "M1"
    assert args["subject"] == "Re: Help"
    assert "Sure" in args["body"]
    assert ctx.queue.get(task.id).state == "done"
    assert ctx.current_task is None


@pytest.mark.asyncio
async def test_respond_email_preserves_existing_re_prefix():
    """Don't double-prefix the subject when it already starts with Re:."""
    mcp = MagicMock()
    mcp.call_tool = AsyncMock()
    ctx = _ctx_with_email_mcp(mcp)
    task = Task(
        kind="email_msg",
        source={
            "thread_id": "T1",
            "message_id": "M1",
            "sender_email": "x@y.com",
            "subject": "Re: ongoing thread",
        },
    )
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_email(ctx, task, "ok")
    assert mcp.call_tool.call_args.args[1]["subject"] == "Re: ongoing thread"


@pytest.mark.asyncio
async def test_respond_email_without_mcp_speaks_error():
    ctx = _ctx_with_email_mcp(mcp=None)
    task = Task(kind="email_msg", source={"message_id": "M1", "sender_email": "x@y.com"})
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_email(ctx, task, "reply")
    ctx.tts.speak.assert_called_with("Gmail MCP is not configured.")
    assert ctx.queue.get(task.id).state != "done"


@pytest.mark.asyncio
async def test_respond_email_without_sender_speaks_error():
    mcp = MagicMock()
    mcp.call_tool = AsyncMock()
    ctx = _ctx_with_email_mcp(mcp)
    task = Task(kind="email_msg", source={"message_id": "M1", "subject": "x"})
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_email(ctx, task, "reply")
    mcp.call_tool.assert_not_called()
    assert ctx.queue.get(task.id).state != "done"


@pytest.mark.asyncio
async def test_respond_email_api_error_keeps_task_active():
    mcp = MagicMock()
    mcp.call_tool = AsyncMock(side_effect=ClaudeMCPError("network down"))
    ctx = _ctx_with_email_mcp(mcp)
    task = Task(
        kind="email_msg",
        source={"message_id": "M1", "sender_email": "x@y.com", "subject": "h"},
    )
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_email(ctx, task, "reply")
    assert ctx.queue.get(task.id).state != "done"
    assert ctx.current_task is task


@pytest.mark.asyncio
async def test_dispatch_task_response_routes_email_kind():
    mcp = MagicMock()
    mcp.call_tool = AsyncMock()
    ctx = _ctx_with_email_mcp(mcp)
    task = Task(
        kind="email_msg",
        source={"message_id": "M1", "sender_email": "x@y.com", "subject": "h"},
    )
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._dispatch_task_response(ctx, task, "responding")
    mcp.call_tool.assert_called_once()
