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


def test_producer_setup_returns_false_when_user_id_fails(tmp_path: Path):
    """user_id resolution now happens on the worker thread; if it raises,
    setup returns False and the poll loop is skipped."""
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.side_effect = ClaudeMCPError("auth broken")
    assert p._setup_in_thread() is False


def test_producer_setup_returns_false_when_user_id_empty(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"id": ""}
    assert p._setup_in_thread() is False


def test_producer_setup_returns_true_and_caches_user_id(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"id": "UABC"}
    assert p._setup_in_thread() is True
    assert p._user_id == "UABC"


def test_producer_start_spawns_thread_without_blocking(tmp_path: Path):
    """Critical: start() must not block on the user_id fetch, even when
    that fetch takes a long time. The fetch happens inside the worker
    thread instead."""
    p, _q, mcp, _state = _producer(tmp_path)

    def slow_call(*_args, **_kwargs):
        import time as _t
        _t.sleep(10)
        return {"id": "UABC"}

    mcp.call_tool.side_effect = slow_call

    import time as _t
    started_at = _t.monotonic()
    p.start()
    elapsed = _t.monotonic() - started_at
    assert elapsed < 0.5, f"start() blocked for {elapsed:.2f}s — should be near-instant"
    assert p._thread is not None and p._thread.is_alive()
    p.stop()  # don't leave the slow-sleeping thread running across tests


def test_producer_fetch_user_id_reads_id_from_user_block(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"user": {"id": "UABC", "name": "henry"}}
    assert p._fetch_user_id() == "UABC"


def test_producer_fetch_user_id_reads_id_from_top_level(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"id": "UDEF"}
    assert p._fetch_user_id() == "UDEF"


def test_producer_fetch_user_id_parses_plain_text_result(tmp_path: Path):
    """The Slack MCP actually returns human-readable text in a ``result``
    key — regex out the User ID line."""
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {
        "result": (
            "User ID: U02L5V8H9RS\n"
            "Username: henry.hinnefeld\n"
            "Display Name: Henry Hinnefeld\n"
            "Real Name: Henry Hinnefeld\n"
        )
    }
    assert p._fetch_user_id() == "U02L5V8H9RS"


def test_producer_fetch_user_id_returns_empty_when_unparseable(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"result": "no recognizable user ID here"}
    assert p._fetch_user_id() == ""


_DETAILED_FIXTURE = (
    "# Search Results for: <@UME>\n\n"
    "## Messages (2 results)\n"
    "### Result 1 of 2\n"
    "Channel: #random (ID: CRAND)\n"
    "From: Alice (ID: UALICE) \n"
    "Time: 2026-05-20 13:43:54 CDT\n"
    "Message_ts: 1716000001.000000\n"
    "Permalink: [link](https://x.slack.com/archives/CRAND/p1716000001000000?thread_ts=1716000001.000000&cid=CRAND)\n"
    "Text: \n"
    "hey <@UME> can you check the deploy\n"
    "\n"
    "---\n\n"
    "### Result 2 of 2\n"
    "Channel: #engineering (ID: CENG)\n"
    "From: Bob (ID: UBOB) \n"
    "Time: 2026-05-20 13:44:54 CDT\n"
    "Message_ts: 1716000002.000000\n"
    "Reply count: 3\n"
    "Permalink: [link](https://x.slack.com/archives/CENG/p1716000002000000?thread_ts=1716000000.500000&cid=CENG)\n"
    "Text: \n"
    "follow-up question\n"
    "\n"
    "---\n\n"
)


def test_producer_poll_emits_tasks_from_detailed_markdown(tmp_path: Path):
    p, q, mcp, state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"results": _DETAILED_FIXTURE}
    p._poll_once()

    tasks = q.all()
    assert len(tasks) == 2
    by_ts = {t.source["ts"]: t for t in tasks}
    t1 = by_ts["1716000001.000000"]
    assert t1.topic == "slack-random"
    assert "Alice" in t1.headline
    assert "deploy" in t1.body
    assert t1.source["channel_id"] == "CRAND"
    assert t1.source["sender_id"] == "UALICE"
    assert t1.source["thread_ts"] == "1716000001.000000"
    t2 = by_ts["1716000002.000000"]
    assert t2.source["thread_ts"] == "1716000000.500000"  # extracted from permalink
    assert state.last_search_ts() == "1716000002.000000"


def test_producer_poll_skips_bot_messages_by_default(tmp_path: Path):
    bot_fixture = (
        "# Search Results for: <@UME>\n\n"
        "### Result 1 of 1\n"
        "Channel: #alerts (ID: CALERT)\n"
        "From: Linear (ID: UBOTID)  [BOT]\n"
        "Message_ts: 1716000099.000000\n"
        "Permalink: [link](https://x.slack.com/archives/CALERT/p1716000099000000?thread_ts=1716000099.000000)\n"
        "Text: \n"
        "Created issue ABC-123\n"
        "\n"
        "---\n\n"
    )
    p, q, mcp, state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"results": bot_fixture}
    p._poll_once()
    assert q.all() == []
    # State should still advance so we don't keep re-parsing the same bot
    # message on every poll.
    assert state.last_search_ts() == "1716000099.000000"


def test_producer_poll_skips_messages_at_or_before_last_ts(tmp_path: Path):
    fixture = (
        "### Result 1 of 2\n"
        "Channel: #c (ID: C1)\n"
        "From: A (ID: U1) \n"
        "Message_ts: 1716000005.000000\n"
        "Permalink: [link](https://x.slack.com/archives/C1/p1716000005000000?thread_ts=1716000005.000000)\n"
        "Text: \n"
        "old\n\n"
        "---\n\n"
        "### Result 2 of 2\n"
        "Channel: #c (ID: C1)\n"
        "From: A (ID: U1) \n"
        "Message_ts: 1716000006.000000\n"
        "Permalink: [link](https://x.slack.com/archives/C1/p1716000006000000?thread_ts=1716000006.000000)\n"
        "Text: \n"
        "new\n\n"
        "---\n\n"
    )
    p, q, mcp, state = _producer(tmp_path)
    p._user_id = "UME"
    state.set_last_search_ts("1716000005.000000")
    mcp.call_tool.return_value = {"results": fixture}
    p._poll_once()
    assert len(q.all()) == 1
    assert "new" in q.all()[0].body


def test_producer_poll_dedupes_within_session(tmp_path: Path):
    fixture = (
        "### Result 1 of 1\n"
        "Channel: #c (ID: C1)\n"
        "From: A (ID: U1) \n"
        "Message_ts: 1716000010.000000\n"
        "Permalink: [link](https://x.slack.com/archives/C1/p1716000010000000?thread_ts=1716000010.000000)\n"
        "Text: \n"
        "ping\n\n"
        "---\n\n"
    )
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"results": fixture}
    p._poll_once()
    p._poll_once()  # same poll result; second pass should be a no-op
    assert len(q.all()) == 1


def test_producer_poll_handles_empty_results(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"results": "# Search Results for: <@UME>\n\n## Messages (0 results)\n"}
    p._poll_once()
    assert q.all() == []


def test_producer_poll_handles_structured_messages_too(tmp_path: Path):
    """Future-proof: if the MCP ever returns a structured ``messages``
    list instead of markdown, we still cope."""
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {
        "messages": [
            {
                "ts": "1.0",
                "text": "ping",
                "channel_id": "C1",
                "channel_name": "c",
                "sender_id": "U1",
                "sender_name": "Alice",
                "thread_ts": "1.0",
            }
        ]
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


# --- slack_to_plain_text -------------------------------------------------


def test_slack_to_plain_text_user_mention_with_display_name():
    from code_trip2.producers.slack import slack_to_plain_text
    out = slack_to_plain_text("<@U02L5V8H9RS|Henry Hinnefeld> can you check this?")
    assert out == "Henry Hinnefeld can you check this?"


def test_slack_to_plain_text_user_mention_without_display_name():
    """Bare <@USERID> has no readable form — drop entirely."""
    from code_trip2.producers.slack import slack_to_plain_text
    out = slack_to_plain_text("hey <@U02L5V8H9RS> here is the question")
    assert out == "hey here is the question"


def test_slack_to_plain_text_channel_mention_with_name():
    from code_trip2.producers.slack import slack_to_plain_text
    assert slack_to_plain_text("in <#C123|general> earlier") == "in #general earlier"


def test_slack_to_plain_text_channel_mention_without_name():
    from code_trip2.producers.slack import slack_to_plain_text
    assert slack_to_plain_text("see <#C123> for context") == "see a channel for context"


def test_slack_to_plain_text_link_with_label():
    from code_trip2.producers.slack import slack_to_plain_text
    out = slack_to_plain_text("see <https://example.com/x|the docs> please")
    assert out == "see the docs please"


def test_slack_to_plain_text_bare_link_is_dropped():
    from code_trip2.producers.slack import slack_to_plain_text
    out = slack_to_plain_text("ref <https://example.com/x> and read")
    assert out == "ref and read"


def test_slack_to_plain_text_broadcasts():
    from code_trip2.producers.slack import slack_to_plain_text
    assert slack_to_plain_text("<!channel> heads up") == "channel heads up"
    assert slack_to_plain_text("<!here> deploying") == "here deploying"


def test_slack_to_plain_text_subteam_mention():
    from code_trip2.producers.slack import slack_to_plain_text
    assert (
        slack_to_plain_text("<!subteam^S0123|@platform-eng> please review")
        == "platform-eng please review"
    )


def test_slack_to_plain_text_decodes_slack_html_entities():
    from code_trip2.producers.slack import slack_to_plain_text
    assert slack_to_plain_text("foo &amp; bar &lt;= baz") == "foo & bar <= baz"


def test_slack_to_plain_text_realistic_message():
    """The actual @-mention text from a recent live poll."""
    from code_trip2.producers.slack import slack_to_plain_text
    raw = (
        "<@U02L5V8H9RS|Henry Hinnefeld> we talked about this briefly in the "
        "past, but any chance we could get a weekly round up / digest? "
        "<https://x.slack.com/archives/C074G552R8A/p1779300838350159|Example 1> "
        "and <https://x.slack.com/archives/C074G552R8A/p1779302538982649|example 2> "
        "in <#C074G552R8A|xteam-delivery>"
    )
    out = slack_to_plain_text(raw)
    assert "U02L5V8H9RS" not in out
    assert "slack.com" not in out
    assert "Henry Hinnefeld" in out
    assert "Example 1" in out
    assert "example 2" in out
    assert "#xteam-delivery" in out


def test_producer_emit_task_cleans_slack_markup(tmp_path: Path):
    """End-to-end: a task's body should already have the markup stripped."""
    fixture = (
        "### Result 1 of 1\n"
        "Channel: #ai-tools (ID: C123)\n"
        "From: Katie Fox (ID: U999) \n"
        "Message_ts: 1716000020.000000\n"
        "Permalink: [link](https://x.slack.com/archives/C123/p1716000020000000?thread_ts=1716000020.000000)\n"
        "Text: \n"
        "<@U02L5V8H9RS|Henry Hinnefeld> can you look at <https://example.com|the PR>?\n"
        "\n"
        "---\n\n"
    )
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "U02L5V8H9RS"
    mcp.call_tool.return_value = {"results": fixture}
    p._poll_once()
    [t] = q.all()
    assert "U02L5V8H9RS" not in t.body
    assert "example.com" not in t.body
    assert "Henry Hinnefeld" in t.body
    assert "the PR" in t.body
    # Headline is derived from the cleaned body.
    assert "U02L5V8H9RS" not in t.headline


def test_producer_after_param_falls_back_to_previous_workday_5pm(tmp_path: Path):
    p, _q, _mcp, _state = _producer(tmp_path)
    out = p._after_param(None)
    # Should be the unix timestamp for 5pm of the most recent weekday.
    from code_trip2.producers.slack import _previous_workday_5pm_unix
    assert int(out) == _previous_workday_5pm_unix()


# --- _previous_workday_5pm_unix ------------------------------------------


def _expected_unix(year, month, day):
    from datetime import datetime
    return int(datetime(year, month, day, 17, 0, 0).timestamp())


def test_previous_workday_5pm_unix_from_wednesday():
    """Wed → previous workday is Tue."""
    from datetime import datetime
    from code_trip2.producers.slack import _previous_workday_5pm_unix
    wed_noon = datetime(2026, 5, 20, 12, 0, 0)  # 2026-05-20 is a Wednesday
    assert _previous_workday_5pm_unix(wed_noon) == _expected_unix(2026, 5, 19)


def test_previous_workday_5pm_unix_from_friday():
    """Fri → previous workday is Thu."""
    from datetime import datetime
    from code_trip2.producers.slack import _previous_workday_5pm_unix
    fri = datetime(2026, 5, 22, 9, 0, 0)
    assert _previous_workday_5pm_unix(fri) == _expected_unix(2026, 5, 21)


def test_previous_workday_5pm_unix_from_monday_skips_weekend():
    """Mon → previous workday is Fri (skip Sat/Sun)."""
    from datetime import datetime
    from code_trip2.producers.slack import _previous_workday_5pm_unix
    mon = datetime(2026, 5, 25, 8, 30, 0)
    assert _previous_workday_5pm_unix(mon) == _expected_unix(2026, 5, 22)


def test_previous_workday_5pm_unix_from_saturday():
    """Sat → previous workday is Fri."""
    from datetime import datetime
    from code_trip2.producers.slack import _previous_workday_5pm_unix
    sat = datetime(2026, 5, 23, 14, 0, 0)
    assert _previous_workday_5pm_unix(sat) == _expected_unix(2026, 5, 22)


def test_previous_workday_5pm_unix_from_sunday():
    """Sun → previous workday is Fri (skip Sat)."""
    from datetime import datetime
    from code_trip2.producers.slack import _previous_workday_5pm_unix
    sun = datetime(2026, 5, 24, 23, 59, 59)
    assert _previous_workday_5pm_unix(sun) == _expected_unix(2026, 5, 22)


def test_previous_workday_5pm_unix_from_friday_after_5pm():
    """Fri evening still rolls back to Thu — we want yesterday's
    after-hours window, not 'today minus a few hours'."""
    from datetime import datetime
    from code_trip2.producers.slack import _previous_workday_5pm_unix
    fri_late = datetime(2026, 5, 22, 22, 0, 0)
    assert _previous_workday_5pm_unix(fri_late) == _expected_unix(2026, 5, 21)


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
