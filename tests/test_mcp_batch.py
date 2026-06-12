"""Tests for the MCP call batcher (mcp_batch.py) and poll alignment helpers."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from code_trip2._async_utils import next_tick_delay
from code_trip2.config import Config, polling_active
from code_trip2.producers import claude_mcp as cm
from code_trip2.producers import mcp_batch as mb
from code_trip2.producers.claude_mcp import ClaudeMCPError
from code_trip2.producers.mcp_batch import BatchedMCPClient, MCPBatcher


# --- stream-json fabrication ------------------------------------------------


def _tool_use_event(use_id: str, name: str, args: dict) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "id": use_id, "name": name, "input": args},
        ]},
    })


def _tool_result_event(use_id: str, payload) -> str:
    return json.dumps({
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": use_id,
             "content": [{"type": "text", "text": json.dumps(payload)}]},
        ]},
    })


def _stream(*lines: str) -> str:
    return "\n".join(lines) + "\n"


def _batcher(monkeypatch, *, stdout: str = "", window: float = 0.01) -> tuple[MCPBatcher, list]:
    """Batcher with a fake subprocess; returns (batcher, recorded calls)."""
    b = MCPBatcher(window_s=window)
    b._client._available = True
    calls: list[dict] = []

    async def fake_run(cmd, *, input_, timeout, what):
        calls.append({"cmd": list(cmd), "input": input_, "timeout": timeout})
        return (stdout, "", 0)

    monkeypatch.setattr(mb, "_run_subprocess", fake_run)
    monkeypatch.setattr(cm, "_run_subprocess", fake_run)
    return b, calls


# --- coalescing + structured matching ---------------------------------------


@pytest.mark.asyncio
async def test_three_calls_coalesce_into_one_session(monkeypatch):
    """Concurrent calls from different servers run in ONE subprocess, and
    each future gets the result matched to its (name, args) pair."""
    stdout = _stream(
        _tool_use_event("u1", "search_threads", {"query": "in:inbox"}),
        _tool_use_event("u2", "slack_search_public", {"query": "@henry"}),
        _tool_use_event("u3", "list_issues", {"assignee": "me"}),
        _tool_result_event("u1", {"threads": [1]}),
        _tool_result_event("u2", {"matches": [2]}),
        _tool_result_event("u3", {"issues": [3]}),
    )
    b, calls = _batcher(monkeypatch, stdout=stdout)
    email = BatchedMCPClient(server_id="claude_ai_Gmail", batcher=b)
    slack = BatchedMCPClient(server_id="claude_ai_Slack", batcher=b)
    linear = BatchedMCPClient(server_id="claude_ai_Linear", batcher=b)

    r_email, r_slack, r_linear = await asyncio.gather(
        email.call_tool("search_threads", {"query": "in:inbox"}),
        slack.call_tool("slack_search_public", {"query": "@henry"}),
        linear.call_tool("list_issues", {"assignee": "me"}),
    )

    assert len(calls) == 1  # one claude session for all three
    cmd = calls[0]["cmd"]
    tools = cmd[cmd.index("--allowedTools") + 1:]
    assert sorted(tools) == [
        "mcp__claude_ai_Gmail__search_threads",
        "mcp__claude_ai_Linear__list_issues",
        "mcp__claude_ai_Slack__slack_search_public",
    ]
    assert r_email == {"threads": [1]}
    assert r_slack == {"matches": [2]}
    assert r_linear == {"issues": [3]}


@pytest.mark.asyncio
async def test_same_tool_twice_matched_by_exact_args(monkeypatch):
    """Two calls to the same tool with different args each get the result
    whose tool_use input matches their args exactly."""
    args_a = {"channel_id": "C1", "message_ts": "1.0"}
    args_b = {"channel_id": "C2", "message_ts": "2.0"}
    stdout = _stream(
        # Model made the calls in REVERSE order — matching must follow
        # args, not arrival order.
        _tool_use_event("u2", "slack_read_thread", args_b),
        _tool_use_event("u1", "slack_read_thread", args_a),
        _tool_result_event("u2", {"thread": "B"}),
        _tool_result_event("u1", {"thread": "A"}),
    )
    b, calls = _batcher(monkeypatch, stdout=stdout)
    slack = BatchedMCPClient(server_id="claude_ai_Slack", batcher=b)
    ra, rb = await asyncio.gather(
        slack.call_tool("slack_read_thread", args_a),
        slack.call_tool("slack_read_thread", args_b),
    )
    assert len(calls) == 1
    assert ra == {"thread": "A"}
    assert rb == {"thread": "B"}


@pytest.mark.asyncio
async def test_single_call_uses_single_tool_path(monkeypatch):
    """A lone request (nothing else in the window) runs the proven
    single-tool flow — same prompt contract as before batching."""
    stdout = _stream(
        _tool_use_event("u1", "search_threads", {"query": "q"}),
        _tool_result_event("u1", {"threads": []}),
    )
    b, calls = _batcher(monkeypatch, stdout=stdout)
    email = BatchedMCPClient(server_id="claude_ai_Gmail", batcher=b)
    result = await email.call_tool("search_threads", {"query": "q"})
    assert result == {"threads": []}
    assert len(calls) == 1
    cmd = calls[0]["cmd"]
    tools = cmd[cmd.index("--allowedTools") + 1:]
    assert tools == ["mcp__claude_ai_Gmail__search_threads"]


@pytest.mark.asyncio
async def test_skipped_call_falls_back_to_individual_session(monkeypatch):
    """The model skipped one of the requested calls — the unmatched
    request re-runs as its own single-tool session."""
    stdout_batch = _stream(
        _tool_use_event("u1", "search_threads", {"query": "q"}),
        _tool_result_event("u1", {"threads": ["ok"]}),
        # list_issues never called by the model
    )
    stdout_single = _stream(
        _tool_use_event("u9", "list_issues", {"assignee": "me"}),
        _tool_result_event("u9", {"issues": ["recovered"]}),
    )
    b = MCPBatcher(window_s=0.01)
    b._client._available = True
    calls: list[list[str]] = []

    async def fake_run(cmd, *, input_, timeout, what):
        calls.append(list(cmd))
        return (stdout_batch if len(calls) == 1 else stdout_single, "", 0)

    monkeypatch.setattr(mb, "_run_subprocess", fake_run)
    monkeypatch.setattr(cm, "_run_subprocess", fake_run)
    email = BatchedMCPClient(server_id="claude_ai_Gmail", batcher=b)
    linear = BatchedMCPClient(server_id="claude_ai_Linear", batcher=b)
    r_email, r_linear = await asyncio.gather(
        email.call_tool("search_threads", {"query": "q"}),
        linear.call_tool("list_issues", {"assignee": "me"}),
    )
    assert r_email == {"threads": ["ok"]}
    assert r_linear == {"issues": ["recovered"]}
    assert len(calls) == 2  # batch + one fallback
    fallback_tools = calls[1][calls[1].index("--allowedTools") + 1:]
    assert fallback_tools == ["mcp__claude_ai_Linear__list_issues"]


@pytest.mark.asyncio
async def test_mutated_args_do_not_match(monkeypatch):
    """If the model altered the arguments, the structured match fails —
    we never return a result for a query we didn't ask."""
    stdout_batch = _stream(
        _tool_use_event("u1", "search_threads", {"query": "DIFFERENT"}),
        _tool_result_event("u1", {"threads": ["wrong"]}),
        _tool_use_event("u2", "list_issues", {"assignee": "me"}),
        _tool_result_event("u2", {"issues": []}),
    )
    stdout_single = _stream(
        _tool_use_event("u9", "search_threads", {"query": "q"}),
        _tool_result_event("u9", {"threads": ["right"]}),
    )
    b = MCPBatcher(window_s=0.01)
    b._client._available = True
    n = {"count": 0}

    async def fake_run(cmd, *, input_, timeout, what):
        n["count"] += 1
        return (stdout_batch if n["count"] == 1 else stdout_single, "", 0)

    monkeypatch.setattr(mb, "_run_subprocess", fake_run)
    monkeypatch.setattr(cm, "_run_subprocess", fake_run)
    email = BatchedMCPClient(server_id="claude_ai_Gmail", batcher=b)
    linear = BatchedMCPClient(server_id="claude_ai_Linear", batcher=b)
    r_email, _ = await asyncio.gather(
        email.call_tool("search_threads", {"query": "q"}),
        linear.call_tool("list_issues", {"assignee": "me"}),
    )
    assert r_email == {"threads": ["right"]}  # from the fallback, not the mutant


@pytest.mark.asyncio
async def test_batch_subprocess_failure_falls_back_for_all(monkeypatch):
    """A failed batch session degrades to individual calls for every
    request — the pre-batching cost profile, not missing data."""
    stdout_single = _stream(
        _tool_use_event("u1", "search_threads", {"query": "q"}),
        _tool_result_event("u1", {"threads": []}),
    )
    b = MCPBatcher(window_s=0.01)
    b._client._available = True
    n = {"count": 0}

    async def fake_run(cmd, *, input_, timeout, what):
        n["count"] += 1
        if n["count"] == 1:
            raise ClaudeMCPError("claude timed out")
        return (stdout_single, "", 0)

    monkeypatch.setattr(mb, "_run_subprocess", fake_run)
    monkeypatch.setattr(cm, "_run_subprocess", fake_run)
    email = BatchedMCPClient(server_id="claude_ai_Gmail", batcher=b)
    linear = BatchedMCPClient(server_id="claude_ai_Linear", batcher=b)
    r1, r2 = await asyncio.gather(
        email.call_tool("search_threads", {"query": "q"}),
        linear.call_tool("search_threads", {"query": "q"}),
        return_exceptions=True,
    )
    # Both resolved via fallback (the fake single stream answers both).
    assert not isinstance(r1, Exception)
    assert not isinstance(r2, Exception)
    assert n["count"] == 3  # 1 failed batch + 2 fallbacks


@pytest.mark.asyncio
async def test_batch_prompt_lists_every_call_with_exact_args(monkeypatch):
    stdout = _stream(
        _tool_use_event("u1", "a_tool", {"x": 1}),
        _tool_use_event("u2", "b_tool", {"y": 2}),
        _tool_result_event("u1", {}),
        _tool_result_event("u2", {}),
    )
    b, calls = _batcher(monkeypatch, stdout=stdout)
    c1 = BatchedMCPClient(server_id="s1", batcher=b)
    c2 = BatchedMCPClient(server_id="s2", batcher=b)
    await asyncio.gather(
        c1.call_tool("a_tool", {"x": 1}),
        c2.call_tool("b_tool", {"y": 2}),
    )
    prompt = calls[0]["input"]
    assert "mcp__s1__a_tool" in prompt
    assert "mcp__s2__b_tool" in prompt
    assert '"x": 1' in prompt
    assert '"y": 2' in prompt
    assert "EXACT" in prompt


@pytest.mark.asyncio
async def test_disabled_batcher_raises(monkeypatch):
    b = MCPBatcher()
    b._client._available = False
    client = BatchedMCPClient(server_id="s", batcher=b)
    assert client.enabled is False
    with pytest.raises(ClaudeMCPError):
        await client.call_tool("t", {})


# --- next_tick_delay ---------------------------------------------------------


def test_next_tick_delay_aligns_to_wall_clock_multiples():
    assert next_tick_delay(120.0, now=1000 * 120.0 + 30.0) == pytest.approx(90.0)
    assert next_tick_delay(60.0, now=59.0) == pytest.approx(1.0)


def test_next_tick_delay_on_boundary_waits_full_interval():
    assert next_tick_delay(120.0, now=240.0) == pytest.approx(120.0)


def test_next_tick_delay_coincidence():
    """Producers on 120s and 180s ticks share an instant every 360s —
    the alignment property the batcher relies on."""
    now = 360.0 * 7 - 50.0
    d120 = next_tick_delay(120.0, now=now)
    d180 = next_tick_delay(180.0, now=now)
    assert now + d120 == now + d180 == 360.0 * 7


# --- polling_active ----------------------------------------------------------


def test_polling_active_inside_window():
    cfg = Config()
    assert polling_active(cfg, datetime(2026, 6, 12, 12, 0)) is True
    assert polling_active(cfg, datetime(2026, 6, 12, 7, 0)) is True


def test_polling_paused_overnight():
    cfg = Config()
    assert polling_active(cfg, datetime(2026, 6, 12, 19, 0)) is False
    assert polling_active(cfg, datetime(2026, 6, 12, 23, 30)) is False
    assert polling_active(cfg, datetime(2026, 6, 12, 3, 0)) is False
    assert polling_active(cfg, datetime(2026, 6, 12, 6, 59)) is False


def test_polling_window_crossing_midnight():
    cfg = Config(poll_start_hour=22, poll_end_hour=6)
    assert polling_active(cfg, datetime(2026, 6, 12, 23, 0)) is True
    assert polling_active(cfg, datetime(2026, 6, 12, 3, 0)) is True
    assert polling_active(cfg, datetime(2026, 6, 12, 12, 0)) is False


def test_polling_equal_hours_disables_window():
    cfg = Config(poll_start_hour=0, poll_end_hour=0)
    assert polling_active(cfg, datetime(2026, 6, 12, 3, 0)) is True


def test_polling_active_tolerates_config_doubles_without_fields():
    """SimpleNamespace configs in older tests must read as always-active,
    never time-dependent."""
    assert polling_active(SimpleNamespace()) is True


# --- producer loop honors the window ----------------------------------------


@pytest.mark.asyncio
async def test_email_producer_skips_polls_outside_active_hours(monkeypatch):
    """Outside the window the loop keeps ticking but never touches claude."""
    from unittest.mock import AsyncMock, create_autospec

    from code_trip2.producers import email as email_module
    from code_trip2.producers.claude_mcp import ClaudeMCPClient
    from code_trip2.producers.email import EmailProducer
    from code_trip2.email_state import EmailState
    from code_trip2.tasks import TaskQueue

    monkeypatch.setattr(email_module.config_mod, "polling_active", lambda cfg: False)
    monkeypatch.setattr(EmailProducer, "_STARTUP_DELAY_S", 0.0, raising=False)
    cfg = SimpleNamespace(
        email_poll_interval=0.02,
        email_search_query="in:inbox",
        email_max_results=5,
    )
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.enabled = True
    mcp.call_tool = AsyncMock(return_value={"threads": []})
    p = EmailProducer(config=cfg, queue=TaskQueue(), mcp=mcp)
    task = asyncio.create_task(p.run())
    await asyncio.sleep(0.15)
    p.request_stop()
    await asyncio.wait_for(task, timeout=1.0)
    mcp.call_tool.assert_not_called()


@pytest.mark.asyncio
async def test_email_producer_polls_inside_active_hours(monkeypatch):
    """Control case: same setup, window open → polls happen."""
    from unittest.mock import AsyncMock, create_autospec

    from code_trip2.producers import email as email_module
    from code_trip2.producers.claude_mcp import ClaudeMCPClient
    from code_trip2.producers.email import EmailProducer
    from code_trip2.tasks import TaskQueue

    monkeypatch.setattr(email_module.config_mod, "polling_active", lambda cfg: True)
    monkeypatch.setattr(EmailProducer, "_STARTUP_DELAY_S", 0.0, raising=False)
    cfg = SimpleNamespace(
        email_poll_interval=0.02,
        email_search_query="in:inbox",
        email_max_results=5,
    )
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.enabled = True
    mcp.call_tool = AsyncMock(return_value={"threads": []})
    p = EmailProducer(config=cfg, queue=TaskQueue(), mcp=mcp)
    task = asyncio.create_task(p.run())
    await asyncio.sleep(0.15)
    p.request_stop()
    await asyncio.wait_for(task, timeout=1.0)
    assert mcp.call_tool.call_count >= 1
