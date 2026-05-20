"""Tests for ClaudeMCPClient, SlackState, SlackProducer, and the reply path."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from code_trip2 import dispatch, modes
from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.producers.slack import SlackProducer
from code_trip2.slack_state import SlackState
from code_trip2.tasks import Task, TaskQueue


# --- ClaudeMCPClient stream parsing --------------------------------------


def _stream_event(tool_result_text: str) -> str:
    """One stream-json line that includes a tool_result content block."""
    return json.dumps({
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_call_1",
                    "content": [{"type": "text", "text": tool_result_text}],
                }
            ],
        },
    })


def _mk_client(**kwargs) -> ClaudeMCPClient:
    """Build a client that's marked enabled regardless of shutil.which()."""
    c = ClaudeMCPClient(**kwargs)
    c._available = True
    return c


def test_claude_mcp_client_disabled_when_binary_missing():
    with patch("code_trip2.producers.claude_mcp.shutil.which", return_value=None):
        c = ClaudeMCPClient()
    assert c.enabled is False
    with pytest.raises(ClaudeMCPError):
        c.call_tool("anything", {})


def test_claude_mcp_call_tool_parses_json_object_from_tool_result():
    c = _mk_client()
    stdout = "\n".join([
        '{"type": "system", "subtype": "init"}',
        _stream_event('{"messages": [{"ts": "1.0", "text": "hi"}]}'),
        '{"type": "result", "is_error": false}',
    ])
    with patch("code_trip2.producers.claude_mcp.subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout=stdout, stderr="", returncode=0)
        result = c.call_tool("slack_search_public_and_private", {"query": "x"})
    assert result == {"messages": [{"ts": "1.0", "text": "hi"}]}


def test_claude_mcp_call_tool_handles_string_tool_result_content():
    c = _mk_client()
    stdout = json.dumps({
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": '{"id": "U123"}'},
            ],
        },
    })
    with patch("code_trip2.producers.claude_mcp.subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout=stdout, stderr="", returncode=0)
        assert c.call_tool("slack_read_user_profile", {}) == {"id": "U123"}


def test_claude_mcp_call_tool_raises_when_no_tool_result():
    c = _mk_client()
    stdout = '{"type": "system"}\n{"type": "result"}\n'
    with patch("code_trip2.producers.claude_mcp.subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout=stdout, stderr="", returncode=0)
        with pytest.raises(ClaudeMCPError) as exc:
            c.call_tool("slack_send_message", {"channel_id": "C1", "message": "hi"})
    assert "No tool_result" in str(exc.value)


def test_claude_mcp_call_tool_raises_on_nonzero_exit():
    c = _mk_client()
    with patch("code_trip2.producers.claude_mcp.subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout="", stderr="boom", returncode=1)
        with pytest.raises(ClaudeMCPError) as exc:
            c.call_tool("slack_read_user_profile", {})
    assert "exited 1" in str(exc.value)


def test_claude_mcp_call_tool_raises_on_timeout():
    c = _mk_client()
    with patch("code_trip2.producers.claude_mcp.subprocess.run") as run:
        run.side_effect = subprocess.TimeoutExpired(cmd=["claude"], timeout=60)
        with pytest.raises(ClaudeMCPError) as exc:
            c.call_tool("slack_read_user_profile", {})
    assert "timed out" in str(exc.value)


def test_claude_mcp_command_passes_expected_args():
    c = _mk_client(model="haiku", server_id="claude_ai_Slack", max_budget_usd=0.05)
    stdout = _stream_event('{"id": "U1"}')
    with patch("code_trip2.producers.claude_mcp.subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout=stdout, stderr="", returncode=0)
        c.call_tool("slack_read_user_profile", {})
    cmd = run.call_args.args[0]
    assert "--allowedTools" in cmd
    allowed_idx = cmd.index("--allowedTools")
    assert cmd[allowed_idx + 1] == "mcp__claude_ai_Slack__slack_read_user_profile"
    assert "--max-budget-usd" in cmd
    budget_idx = cmd.index("--max-budget-usd")
    assert cmd[budget_idx + 1] == "0.05"
    assert "--model" in cmd
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == "haiku"


def test_claude_mcp_prompt_goes_via_stdin_not_as_arg():
    """Regression: ``--allowedTools`` is variadic, so a trailing positional
    prompt gets eaten as a second tool name. The prompt must go through
    stdin instead."""
    c = _mk_client()
    stdout = _stream_event('{"id": "U1"}')
    with patch("code_trip2.producers.claude_mcp.subprocess.run") as run:
        run.return_value = SimpleNamespace(stdout=stdout, stderr="", returncode=0)
        c.call_tool("slack_read_user_profile", {"foo": "bar"})
    kwargs = run.call_args.kwargs
    cmd = run.call_args.args[0]
    # Prompt must be passed via stdin.
    assert "input" in kwargs and "slack_read_user_profile" in kwargs["input"]
    assert "foo" in kwargs["input"]
    # Prompt must NOT appear in cmd args (which would let --allowedTools
    # absorb it).
    assert kwargs["input"] not in cmd
    # The last token of cmd should be the allowed tool id, with no
    # trailing positional argument.
    assert cmd[-1] == "mcp__claude_ai_Slack__slack_read_user_profile"


# --- SlackState ----------------------------------------------------------


def test_slack_state_persists_roundtrip(tmp_path: Path):
    p = tmp_path / "slack-state.json"
    s = SlackState(path=p)
    s.set_last_search_ts("1716000000.000100")
    s2 = SlackState(path=p)
    assert s2.last_search_ts() == "1716000000.000100"


def test_slack_state_missing_file_is_empty(tmp_path: Path):
    s = SlackState(path=tmp_path / "nope.json")
    assert s.last_search_ts() is None


def test_slack_state_does_not_regress_ts(tmp_path: Path):
    s = SlackState(path=tmp_path / "s.json")
    s.set_last_search_ts("1716000100.000000")
    s.set_last_search_ts("1716000050.000000")  # older — ignored
    assert s.last_search_ts() == "1716000100.000000"


def test_slack_state_handles_corrupt_file(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{ not json")
    s = SlackState(path=p)
    assert s.last_search_ts() is None


# --- SlackProducer -------------------------------------------------------


def _producer(tmp_path: Path, *, poll_interval=30.0):
    cfg = SimpleNamespace(slack_poll_interval=poll_interval)
    state = SlackState(path=tmp_path / "slack-state.json")
    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.enabled = True
    q = TaskQueue()
    p = SlackProducer(config=cfg, queue=q, mcp=mcp, state=state)
    return p, q, mcp, state


def test_producer_skips_when_mcp_unavailable(tmp_path: Path):
    cfg = SimpleNamespace(slack_poll_interval=30.0)
    p = SlackProducer(config=cfg, queue=TaskQueue(), mcp=None)
    p.start()
    assert p._thread is None


def test_producer_skips_when_user_id_fails(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.side_effect = ClaudeMCPError("auth broken")
    p.start()
    assert p._thread is None


def test_producer_skips_when_user_id_empty(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"id": ""}
    p.start()
    assert p._thread is None


def test_producer_fetch_user_id_reads_id_from_user_block(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"user": {"id": "UABC", "name": "henry"}}
    assert p._fetch_user_id() == "UABC"


def test_producer_fetch_user_id_reads_id_from_top_level(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"id": "UDEF"}
    assert p._fetch_user_id() == "UDEF"


def test_producer_poll_emits_tasks_and_advances_state(tmp_path: Path):
    p, q, mcp, state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {
        "messages": [
            {
                "ts": "1716000001.000000",
                "text": "hey <@UME> can you check the deploy",
                "user": "UALICE",
                "username": "Alice",
                "channel": {"id": "CRAND", "name": "random"},
                "thread_ts": "1716000001.000000",
            },
            {
                "ts": "1716000002.000000",
                "text": "follow-up question",
                "user": "UBOB",
                "username": "Bob",
                "channel": {"id": "CENG", "name": "engineering"},
            },
        ]
    }
    p._poll_once()

    tasks = q.all()
    assert len(tasks) == 2
    by_ts = {t.source["ts"]: t for t in tasks}
    t1 = by_ts["1716000001.000000"]
    assert t1.topic == "slack-random"
    assert "Alice" in t1.headline
    assert t1.source["channel_id"] == "CRAND"
    assert t1.source["sender_id"] == "UALICE"
    assert t1.source["thread_ts"] == "1716000001.000000"
    t2 = by_ts["1716000002.000000"]
    assert t2.source["thread_ts"] == "1716000002.000000"  # falls back to ts
    assert state.last_search_ts() == "1716000002.000000"


def test_producer_poll_skips_messages_at_or_before_last_ts(tmp_path: Path):
    p, q, mcp, state = _producer(tmp_path)
    p._user_id = "UME"
    state.set_last_search_ts("1716000005.000000")
    mcp.call_tool.return_value = {
        "messages": [
            {"ts": "1716000005.000000", "text": "old", "user": "U1",
             "channel": {"id": "C1", "name": "c"}},
            {"ts": "1716000006.000000", "text": "new", "user": "U1",
             "channel": {"id": "C1", "name": "c"}},
        ]
    }
    p._poll_once()
    assert len(q.all()) == 1
    assert q.all()[0].body == "new"


def test_producer_poll_dedupes_within_session(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    msg = {"ts": "1716000010.000000", "text": "ping", "user": "U1",
           "channel": {"id": "C1", "name": "c"}}
    mcp.call_tool.return_value = {"messages": [msg]}
    p._poll_once()
    p._poll_once()  # same poll result returned; should not emit twice
    assert len(q.all()) == 1


def test_producer_poll_handles_empty_results(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"messages": []}
    p._poll_once()
    assert q.all() == []


def test_producer_poll_handles_alt_response_shapes(tmp_path: Path):
    """Slack MCP variants put matches under several keys; handle them."""
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {
        "messages": {
            "matches": [
                {"ts": "1.0", "text": "via matches", "user": "U1",
                 "channel": {"id": "C1", "name": "c"}}
            ]
        }
    }
    p._poll_once()
    assert len(q.all()) == 1


def test_producer_poll_resilient_to_mcp_errors(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.side_effect = ClaudeMCPError("network")
    p._poll_once()  # should not raise
    assert q.all() == []


def test_producer_after_param_uses_state_ts(tmp_path: Path):
    p, _q, _mcp, state = _producer(tmp_path)
    state.set_last_search_ts("1716000100.000000")
    assert p._after_param(state.last_search_ts()) == "1716000100"


def test_producer_after_param_falls_back_to_lookback(tmp_path: Path):
    p, _q, _mcp, _state = _producer(tmp_path)
    out = p._after_param(None)
    # Should be a Unix timestamp roughly _INITIAL_LOOKBACK_S in the past.
    import time
    now = time.time()
    assert (now - 8 * 24 * 3600) < int(out) < now


# --- reply path ---------------------------------------------------------


def _ctx_with_slack_mcp(mcp):
    cfg = SimpleNamespace(
        ssh_host="", ssh_options=(), tmux_session="main",
        work_window="work", linear_window="linear", terminal_apps=("kitty",),
    )
    return modes.Context(
        config=cfg,  # type: ignore[arg-type]
        tts=MagicMock(),
        log=MagicMock(),
        thinking=MagicMock(),
        slack_mcp=mcp,
    )


def test_respond_slack_calls_send_message_via_mcp():
    mcp = MagicMock()
    ctx = _ctx_with_slack_mcp(mcp)
    task = Task(
        kind="slack_msg",
        topic="slack-general",
        source={"channel_id": "C123", "ts": "1.0", "thread_ts": "1.0"},
    )
    ctx.queue.add(task)
    ctx.current_task = task
    dispatch._respond_slack(ctx, task, "Yes, ship it.")
    mcp.call_tool.assert_called_once_with(
        "slack_send_message",
        {"channel_id": "C123", "message": "Yes, ship it.", "thread_ts": "1.0"},
    )
    assert ctx.queue.get(task.id).state == "done"
    assert ctx.current_task is None


def test_respond_slack_without_mcp_speaks_error():
    ctx = _ctx_with_slack_mcp(mcp=None)
    task = Task(kind="slack_msg",
                source={"channel_id": "C1", "ts": "1.0", "thread_ts": "1.0"})
    ctx.queue.add(task)
    ctx.current_task = task
    dispatch._respond_slack(ctx, task, "reply")
    ctx.tts.speak.assert_called_with("Slack MCP is not configured.")
    assert ctx.queue.get(task.id).state != "done"


def test_respond_slack_api_error_keeps_task_active():
    mcp = MagicMock()
    mcp.call_tool.side_effect = ClaudeMCPError("network down")
    ctx = _ctx_with_slack_mcp(mcp)
    task = Task(kind="slack_msg",
                source={"channel_id": "C1", "ts": "1.0", "thread_ts": "1.0"})
    ctx.queue.add(task)
    ctx.current_task = task
    dispatch._respond_slack(ctx, task, "reply")
    assert ctx.queue.get(task.id).state != "done"
    assert ctx.current_task is task


def test_dispatch_task_response_routes_slack_kind():
    mcp = MagicMock()
    ctx = _ctx_with_slack_mcp(mcp)
    task = Task(kind="slack_msg",
                source={"channel_id": "C1", "ts": "1.0", "thread_ts": "1.0"})
    ctx.queue.add(task)
    ctx.current_task = task
    dispatch._dispatch_task_response(ctx, task, "responding now")
    mcp.call_tool.assert_called_once()
