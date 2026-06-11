"""Tests for ClaudeMCPClient, SlackState, SlackProducer, and the reply path."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest

from code_trip2 import dispatch, modes
from code_trip2.earcon import Thinking
from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.producers.slack import SlackProducer
from code_trip2.session_log import SessionLogger
from code_trip2.slack_state import SlackState
from code_trip2.tasks import Task, TaskQueue
from conftest import make_mock_tts


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


def _patch_exec(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    hang: bool = False,
) -> dict:
    """Fake asyncio.create_subprocess_exec for claude_mcp tests.

    Returns a dict that captures ``argv`` and ``input`` from the last
    call. With ``hang=True`` the fake proc's communicate() never
    returns — pair with a tiny client.timeout to exercise the
    asyncio.TimeoutError path.
    """
    captured: dict = {"argv": None, "input": None}

    async def fake_exec(*argv, **kwargs):
        captured["argv"] = list(argv)
        proc = MagicMock()

        async def communicate(input=None):
            captured["input"] = input.decode("utf-8") if input else None
            if hang:
                await asyncio.sleep(60)
            return (stdout.encode("utf-8"), stderr.encode("utf-8"))

        async def wait():
            return returncode

        proc.communicate = communicate
        proc.wait = wait
        proc.kill = MagicMock()
        proc.returncode = returncode
        return proc

    monkeypatch.setattr(
        "code_trip2.producers.claude_mcp.asyncio.create_subprocess_exec",
        fake_exec,
    )
    return captured


@pytest.mark.asyncio
async def test_claude_mcp_client_disabled_when_binary_missing():
    with patch("code_trip2.producers.claude_mcp.shutil.which", return_value=None):
        c = ClaudeMCPClient()
    assert c.enabled is False
    with pytest.raises(ClaudeMCPError):
        await c.call_tool("anything", {})


@pytest.mark.asyncio
async def test_claude_mcp_call_tool_parses_json_object_from_tool_result(monkeypatch):
    c = _mk_client()
    stdout = "\n".join([
        '{"type": "system", "subtype": "init"}',
        _stream_event('{"messages": [{"ts": "1.0", "text": "hi"}]}'),
        '{"type": "result", "is_error": false}',
    ])
    _patch_exec(monkeypatch, stdout=stdout)
    result = await c.call_tool("slack_search_public_and_private", {"query": "x"})
    assert result == {"messages": [{"ts": "1.0", "text": "hi"}]}


@pytest.mark.asyncio
async def test_claude_mcp_call_tool_handles_string_tool_result_content(monkeypatch):
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
    _patch_exec(monkeypatch, stdout=stdout)
    assert await c.call_tool("slack_read_user_profile", {}) == {"id": "U123"}


@pytest.mark.asyncio
async def test_claude_mcp_call_tool_recovers_from_dumped_file(monkeypatch, tmp_path):
    """Large tool results get diverted to a file by claude --print; the
    parser should follow the sentinel and read the real payload."""
    # Write the on-disk file the way claude --print writes it: an outer
    # JSON array of content blocks whose ``text`` field carries the
    # actual tool result as a JSON string.
    dump_path = tmp_path / "mcp-dump.txt"
    payload = {"issues": [{"id": "AI-1", "title": "Big"}]}
    dump_path.write_text(json.dumps([{"type": "text", "text": json.dumps(payload)}]))
    notice = (
        f"Error: result (90000 characters) exceeds maximum allowed tokens. "
        f"Output has been saved to {dump_path}.\n"
        f"Format: JSON array with schema: [{{type: string, text: string}}]\n"
    )
    c = _mk_client()
    _patch_exec(monkeypatch, stdout=_stream_event(notice))
    result = await c.call_tool("mcp__claude_ai_Linear__list_issues", {"assignee": "me"})
    assert result == payload


@pytest.mark.asyncio
async def test_claude_mcp_call_tool_raises_when_no_tool_result(monkeypatch):
    c = _mk_client()
    _patch_exec(monkeypatch, stdout='{"type": "system"}\n{"type": "result"}\n')
    with pytest.raises(ClaudeMCPError) as exc:
        await c.call_tool("slack_send_message", {"channel_id": "C1", "message": "hi"})
    assert "No tool_result" in str(exc.value)


@pytest.mark.asyncio
async def test_claude_mcp_call_tool_raises_on_nonzero_exit(monkeypatch):
    c = _mk_client()
    _patch_exec(monkeypatch, stdout="", stderr="boom", returncode=1)
    with pytest.raises(ClaudeMCPError) as exc:
        await c.call_tool("slack_read_user_profile", {})
    assert "exited 1" in str(exc.value)


@pytest.mark.asyncio
async def test_claude_mcp_call_tool_raises_on_timeout(monkeypatch):
    c = _mk_client(timeout=0.01)
    _patch_exec(monkeypatch, hang=True)
    with pytest.raises(ClaudeMCPError) as exc:
        await c.call_tool("slack_read_user_profile", {})
    assert "timed out" in str(exc.value)


@pytest.mark.asyncio
async def test_claude_mcp_command_passes_expected_args(monkeypatch):
    c = _mk_client(model="haiku", server_id="claude_ai_Slack", max_budget_usd=0.05)
    captured = _patch_exec(monkeypatch, stdout=_stream_event('{"id": "U1"}'))
    await c.call_tool("slack_read_user_profile", {})
    cmd = captured["argv"]
    assert "--allowedTools" in cmd
    allowed_idx = cmd.index("--allowedTools")
    assert cmd[allowed_idx + 1] == "mcp__claude_ai_Slack__slack_read_user_profile"
    assert "--max-budget-usd" in cmd
    budget_idx = cmd.index("--max-budget-usd")
    assert cmd[budget_idx + 1] == "0.05"
    assert "--model" in cmd
    model_idx = cmd.index("--model")
    assert cmd[model_idx + 1] == "haiku"


@pytest.mark.asyncio
async def test_claude_mcp_prompt_goes_via_stdin_not_as_arg(monkeypatch):
    """Regression: ``--allowedTools`` is variadic, so a trailing positional
    prompt gets eaten as a second tool name. The prompt must go through
    stdin instead."""
    c = _mk_client()
    captured = _patch_exec(monkeypatch, stdout=_stream_event('{"id": "U1"}'))
    await c.call_tool("slack_read_user_profile", {"foo": "bar"})
    cmd = captured["argv"]
    stdin_text = captured["input"]
    # Prompt must be passed via stdin.
    assert stdin_text is not None
    assert "slack_read_user_profile" in stdin_text
    assert "foo" in stdin_text
    # Prompt must NOT appear in cmd args (which would let --allowedTools
    # absorb it).
    assert stdin_text not in cmd
    # The last token of cmd should be the allowed tool id, with no
    # trailing positional argument.
    assert cmd[-1] == "mcp__claude_ai_Slack__slack_read_user_profile"


# --- ClaudeMCPClient.run_agent ------------------------------------------


def _assistant_text_event(text: str) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    })


@pytest.mark.asyncio
async def test_run_agent_passes_allowed_tools_after_other_flags(monkeypatch):
    c = _mk_client()
    captured = _patch_exec(monkeypatch, stdout=_assistant_text_event("Did the thing."))
    result = await c.run_agent(
        prompt="please do X",
        allowed_tools=["mcp__svc__a", "mcp__svc__b"],
    )
    assert result == "Did the thing."
    cmd = captured["argv"]
    # --allowedTools is variadic and must come LAST so it doesn't swallow
    # a subsequent flag as a tool name.
    assert "--allowedTools" in cmd
    idx = cmd.index("--allowedTools")
    assert cmd[idx + 1 :] == ["mcp__svc__a", "mcp__svc__b"]
    # Prompt is on stdin, not in argv.
    assert captured["input"] == "please do X"


@pytest.mark.asyncio
async def test_run_agent_omits_allowed_tools_when_list_empty(monkeypatch):
    c = _mk_client()
    captured = _patch_exec(monkeypatch, stdout=_assistant_text_event("ok"))
    await c.run_agent(prompt="hi")
    assert "--allowedTools" not in captured["argv"]


@pytest.mark.asyncio
async def test_run_agent_extracts_last_assistant_text(monkeypatch):
    """Skill flows interleave tool_use / tool_result / text blocks; we
    want the final natural-language summary, not an intermediate one."""
    c = _mk_client()
    stdout = "\n".join([
        _assistant_text_event("Searching calendar..."),
        json.dumps({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "[{...}]"},
        ]}}),
        _assistant_text_event("Accepted 'Lunch' and archived the email."),
    ])
    _patch_exec(monkeypatch, stdout=stdout)
    out = await c.run_agent(prompt="accept invite", allowed_tools=["t1"])
    assert out == "Accepted 'Lunch' and archived the email."


@pytest.mark.asyncio
async def test_run_agent_raises_on_nonzero_exit(monkeypatch):
    c = _mk_client()
    _patch_exec(monkeypatch, stdout="", stderr="oh no", returncode=2)
    with pytest.raises(ClaudeMCPError) as exc:
        await c.run_agent(prompt="x")
    assert "exited 2" in str(exc.value)


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


def _producer(tmp_path: Path, *, poll_interval=30.0, watch_channels=()):
    cfg = SimpleNamespace(
        slack_poll_interval=poll_interval,
        slack_watch_channels=tuple(watch_channels),
    )
    state = SlackState(path=tmp_path / "slack-state.json")
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.enabled = True
    q = TaskQueue()
    p = SlackProducer(config=cfg, queue=q, mcp=mcp, state=state)
    return p, q, mcp, state


@pytest.mark.asyncio
async def test_producer_skips_when_mcp_unavailable(tmp_path: Path):
    cfg = SimpleNamespace(slack_poll_interval=30.0, slack_watch_channels=())
    p = SlackProducer(config=cfg, queue=TaskQueue(), mcp=None)
    # run() should return immediately when no MCP client is available.
    await asyncio.wait_for(p.run(), timeout=1.0)


@pytest.mark.asyncio
async def test_producer_setup_returns_false_when_user_id_fails(tmp_path: Path):
    """If user_id resolution raises, setup returns False and the poll
    loop is skipped."""
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.side_effect = ClaudeMCPError("auth broken")
    assert await p._setup_in_thread() is False


@pytest.mark.asyncio
async def test_producer_setup_returns_false_when_user_id_empty(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"id": ""}
    assert await p._setup_in_thread() is False


@pytest.mark.asyncio
async def test_producer_setup_returns_true_and_caches_user_id(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"id": "UABC"}
    assert await p._setup_in_thread() is True
    assert p._user_id == "UABC"


@pytest.mark.asyncio
async def test_producer_setup_resolves_watch_channels(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path, watch_channels=("team-ai",))
    # First call: user profile. Second: channel search.
    mcp.call_tool.side_effect = [
        {"id": "UABC"},
        {"results": (
            "# Search Results for: team-ai\n"
            "### Channel 1 of 2\n"
            "Name: team-ai\n"
            "ID: C0TEAMAI\n"
            "Members: 42\n"
            "---\n"
            "### Channel 2 of 2\n"
            "Name: team-ai-deprecated\n"
            "ID: C0OLD\n"
            "---\n"
        )},
    ]
    assert await p._setup_in_thread() is True
    assert p._watched == {"team-ai": "C0TEAMAI"}


@pytest.mark.asyncio
async def test_producer_setup_tolerates_failed_channel_lookup(tmp_path: Path):
    """A bad channel name shouldn't keep the producer from running the
    mention search."""
    p, _q, mcp, _state = _producer(tmp_path, watch_channels=("nope",))
    mcp.call_tool.side_effect = [
        {"id": "UABC"},
        ClaudeMCPError("network glitch"),
    ]
    assert await p._setup_in_thread() is True
    assert p._watched == {}


@pytest.mark.asyncio
async def test_producer_run_retries_setup_after_transient_failure(tmp_path: Path):
    """A one-time setup failure shouldn't permanently kill the producer.
    The task should retry on the next poll-interval tick."""
    p, _q, mcp, _state = _producer(tmp_path, poll_interval=0.01)
    p._STARTUP_DELAY_S = 0.01  # skip the 2s stagger for the test
    calls = []

    def call_tool(name, args):
        calls.append(name)
        if name == "slack_read_user_profile" and len(calls) == 1:
            raise ClaudeMCPError("transient")
        if name == "slack_read_user_profile":
            return {"id": "UABC"}
        return {"results": ""}

    mcp.call_tool.side_effect = call_tool

    task = asyncio.create_task(p.run())
    # Wait for two setup attempts to land.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while loop.time() < deadline:
        if calls.count("slack_read_user_profile") >= 2:
            break
        await asyncio.sleep(0.02)
    p.request_stop()
    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    assert calls.count("slack_read_user_profile") >= 2, (
        f"expected at least two setup attempts, got calls={calls}"
    )
    assert p._user_id == "UABC"


@pytest.mark.asyncio
async def test_producer_fetch_user_id_reads_id_from_user_block(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"user": {"id": "UABC", "name": "henry"}}
    assert await p._fetch_user_id() == "UABC"


@pytest.mark.asyncio
async def test_producer_fetch_user_id_reads_id_from_top_level(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"id": "UDEF"}
    assert await p._fetch_user_id() == "UDEF"


@pytest.mark.asyncio
async def test_producer_fetch_user_id_parses_plain_text_result(tmp_path: Path):
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
    assert await p._fetch_user_id() == "U02L5V8H9RS"


@pytest.mark.asyncio
async def test_producer_fetch_user_id_returns_empty_when_unparseable(tmp_path: Path):
    p, _q, mcp, _state = _producer(tmp_path)
    mcp.call_tool.return_value = {"result": "no recognizable user ID here"}
    assert await p._fetch_user_id() == ""


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
    "<@UME> follow-up question\n"
    "\n"
    "---\n\n"
)


@pytest.mark.asyncio
async def test_producer_poll_emits_tasks_from_detailed_markdown(tmp_path: Path):
    p, q, mcp, state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"results": _DETAILED_FIXTURE}
    await p._poll_once()

    tasks = q.all()
    assert len(tasks) == 2
    by_ts = {t.source["ts"]: t for t in tasks}
    t1 = by_ts["1716000001.000000"]
    assert t1.topic == "random"
    assert "Alice" in t1.headline
    assert "deploy" in t1.body
    assert t1.source["channel_id"] == "CRAND"
    assert t1.source["sender_id"] == "UALICE"
    assert t1.source["thread_ts"] == "1716000001.000000"
    t2 = by_ts["1716000002.000000"]
    assert t2.source["thread_ts"] == "1716000000.500000"  # extracted from permalink
    # Full permalink captured so ACT+YES can hand it to the Slack desktop app.
    assert t1.source["url"] == (
        "https://x.slack.com/archives/CRAND/p1716000001000000"
        "?thread_ts=1716000001.000000&cid=CRAND"
    )
    assert t2.source["url"].startswith("https://x.slack.com/archives/CENG/")
    assert state.last_search_ts() == "1716000002.000000"


@pytest.mark.asyncio
async def test_producer_poll_skips_bot_messages_by_default(tmp_path: Path):
    bot_fixture = (
        "# Search Results for: <@UME>\n\n"
        "### Result 1 of 1\n"
        "Channel: #alerts (ID: CALERT)\n"
        "From: Linear (ID: UBOTID)  [BOT]\n"
        "Message_ts: 1716000099.000000\n"
        "Permalink: [link](https://x.slack.com/archives/CALERT/p1716000099000000?thread_ts=1716000099.000000)\n"
        "Text: \n"
        "<@UME> created issue ABC-123\n"
        "\n"
        "---\n\n"
    )
    p, q, mcp, state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"results": bot_fixture}
    await p._poll_once()
    assert q.all() == []
    # State should still advance so we don't keep re-parsing the same bot
    # message on every poll.
    assert state.last_search_ts() == "1716000099.000000"


@pytest.mark.asyncio
async def test_producer_poll_skips_messages_at_or_before_last_ts(tmp_path: Path):
    fixture = (
        "### Result 1 of 2\n"
        "Channel: #c (ID: C1)\n"
        "From: A (ID: U1) \n"
        "Message_ts: 1716000005.000000\n"
        "Permalink: [link](https://x.slack.com/archives/C1/p1716000005000000?thread_ts=1716000005.000000)\n"
        "Text: \n"
        "<@UME> old\n\n"
        "---\n\n"
        "### Result 2 of 2\n"
        "Channel: #c (ID: C1)\n"
        "From: A (ID: U1) \n"
        "Message_ts: 1716000006.000000\n"
        "Permalink: [link](https://x.slack.com/archives/C1/p1716000006000000?thread_ts=1716000006.000000)\n"
        "Text: \n"
        "<@UME> new\n\n"
        "---\n\n"
    )
    p, q, mcp, state = _producer(tmp_path)
    p._user_id = "UME"
    state.set_last_search_ts("1716000005.000000")
    mcp.call_tool.return_value = {"results": fixture}
    await p._poll_once()
    assert len(q.all()) == 1
    assert "new" in q.all()[0].body


@pytest.mark.asyncio
async def test_producer_poll_dedupes_within_session(tmp_path: Path):
    fixture = (
        "### Result 1 of 1\n"
        "Channel: #c (ID: C1)\n"
        "From: A (ID: U1) \n"
        "Message_ts: 1716000010.000000\n"
        "Permalink: [link](https://x.slack.com/archives/C1/p1716000010000000?thread_ts=1716000010.000000)\n"
        "Text: \n"
        "<@UME> ping\n\n"
        "---\n\n"
    )
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"results": fixture}
    await p._poll_once()
    await p._poll_once()  # same poll result; second pass should be a no-op
    assert len(q.all()) == 1


@pytest.mark.asyncio
async def test_producer_poll_drops_non_mention_results(tmp_path: Path):
    """The claude.ai Slack MCP search isn't strict on `<@USER_ID>` — a
    post-filter drops anything whose raw body doesn't contain the
    user's literal encoded mention. See the Valeria→Molly bug in the
    main branch: a third-party @-mention in a channel-search hit
    surfaced as a task even though the user wasn't named anywhere.
    """
    fixture = (
        "### Result 1 of 2\n"
        "Channel: #channel-a (ID: CA)\n"
        "From: Alice (ID: UALICE) \n"
        "Message_ts: 1716000050.000000\n"
        "Permalink: [link](https://x.slack.com/archives/CA/p1716000050000000?thread_ts=1716000050.000000)\n"
        "Text: \n"
        "<@UME> need your input on the deploy\n\n"
        "---\n\n"
        "### Result 2 of 2\n"
        "Channel: #channel-b (ID: CB)\n"
        "From: Bob (ID: UBOB) \n"
        "Message_ts: 1716000051.000000\n"
        "Permalink: [link](https://x.slack.com/archives/CB/p1716000051000000?thread_ts=1716000051.000000)\n"
        "Text: \n"
        "<@USOMEONE_ELSE> can you look at this\n\n"  # not the user
        "---\n\n"
    )
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"results": fixture}
    await p._poll_once()
    tasks = q.all()
    assert len(tasks) == 1
    assert tasks[0].source["channel_id"] == "CA"
    assert "deploy" in tasks[0].body


@pytest.mark.asyncio
async def test_producer_poll_handles_empty_results(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {"results": "# Search Results for: <@UME>\n\n## Messages (0 results)\n"}
    await p._poll_once()
    assert q.all() == []


@pytest.mark.asyncio
async def test_producer_poll_handles_structured_messages_too(tmp_path: Path):
    """Future-proof: if the MCP ever returns a structured ``messages``
    list instead of markdown, we still cope."""
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {
        "messages": [
            {
                "ts": "1.0",
                "text": "<@UME> ping",
                "channel_id": "C1",
                "channel_name": "c",
                "sender_id": "U1",
                "sender_name": "Alice",
                "thread_ts": "1.0",
            }
        ]
    }
    await p._poll_once()
    assert len(q.all()) == 1


@pytest.mark.asyncio
async def test_producer_poll_resilient_to_mcp_errors(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.side_effect = ClaudeMCPError("network")
    await p._poll_once()  # should not raise
    assert q.all() == []


# --- watched-channel polling ---------------------------------------------


_CHANNEL_READ_FIXTURE = (
    "# Recent messages in #team-ai\n"
    "### Result 1 of 2\n"
    "From: Linear (ID: UBOT) [BOT]\n"
    "Message_ts: 1716000002.000000\n"
    "Permalink: [link](https://x.slack.com/archives/C0TEAMAI/p1716000002000000?thread_ts=1716000002.000000)\n"
    "Text: \n"
    "Created issue ABC-123\n"
    "\n"
    "---\n\n"
    "### Result 2 of 2\n"
    "From: Alice (ID: UALICE) \n"
    "Message_ts: 1716000001.000000\n"
    "Permalink: [link](https://x.slack.com/archives/C0TEAMAI/p1716000001000000?thread_ts=1716000001.000000)\n"
    "Text: \n"
    "morning team\n"
    "\n"
    "---\n\n"
)


@pytest.mark.asyncio
async def test_poll_watched_channel_emits_every_message_including_bots(tmp_path: Path):
    """Unlike the mention search, watched channels include bot messages —
    the user opted into 'every new message in this channel'."""
    p, q, mcp, state = _producer(tmp_path, watch_channels=("team-ai",))
    p._user_id = "UME"
    p._watched = {"team-ai": "C0TEAMAI"}
    mcp.call_tool.return_value = {"results": _CHANNEL_READ_FIXTURE}
    await p._poll_watched_channel("team-ai", "C0TEAMAI")

    tasks = q.all()
    assert len(tasks) == 2  # bot AND human
    by_ts = {t.source["ts"]: t for t in tasks}
    bot_task = by_ts["1716000002.000000"]
    assert "Linear" in bot_task.headline
    assert bot_task.source["channel_id"] == "C0TEAMAI"
    assert bot_task.topic == "team-ai"
    human = by_ts["1716000001.000000"]
    assert "Alice" in human.headline
    assert "morning team" in human.body
    assert state.last_channel_ts("C0TEAMAI") == "1716000002.000000"


@pytest.mark.asyncio
async def test_poll_watched_channel_respects_per_channel_cursor(tmp_path: Path):
    p, q, mcp, state = _producer(tmp_path, watch_channels=("team-ai",))
    p._user_id = "UME"
    p._watched = {"team-ai": "C0TEAMAI"}
    state.set_last_channel_ts("C0TEAMAI", "1716000002.000000")
    mcp.call_tool.return_value = {"results": _CHANNEL_READ_FIXTURE}
    await p._poll_watched_channel("team-ai", "C0TEAMAI")
    # Both messages are at or before the cursor — should emit nothing.
    assert q.all() == []


def test_poll_watched_channel_cursor_is_independent_of_mention_cursor(tmp_path: Path):
    """Mention search and channel poll use different cursors so they
    don't interfere with each other's state."""
    p, _q, _mcp, state = _producer(tmp_path, watch_channels=("team-ai",))
    state.set_last_search_ts("1716000099.000000")
    state.set_last_channel_ts("C0TEAMAI", "1716000050.000000")
    # Each cursor advances independently.
    state.set_last_channel_ts("C0TEAMAI", "1716000060.000000")
    assert state.last_search_ts() == "1716000099.000000"  # unchanged
    assert state.last_channel_ts("C0TEAMAI") == "1716000060.000000"


@pytest.mark.asyncio
async def test_poll_watched_channel_resilient_to_mcp_errors(tmp_path: Path):
    p, q, mcp, _state = _producer(tmp_path, watch_channels=("team-ai",))
    p._watched = {"team-ai": "C0TEAMAI"}
    mcp.call_tool.side_effect = ClaudeMCPError("network down")
    await p._poll_watched_channel("team-ai", "C0TEAMAI")  # should not raise
    assert q.all() == []


@pytest.mark.asyncio
async def test_poll_once_calls_both_paths(tmp_path: Path):
    """_poll_once should run the mention search AND a read per watched
    channel."""
    p, _q, mcp, _state = _producer(tmp_path, watch_channels=("team-ai", "general"))
    p._user_id = "UME"
    p._watched = {"team-ai": "C0TEAMAI", "general": "C0GEN"}
    mcp.call_tool.return_value = {"results": ""}
    await p._poll_once()
    # One mention search + two channel reads.
    tool_names = [c.args[0] for c in mcp.call_tool.call_args_list]
    assert tool_names.count("slack_search_public_and_private") == 1
    assert tool_names.count("slack_read_channel") == 2


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


@pytest.mark.asyncio
async def test_producer_emit_task_cleans_slack_markup(tmp_path: Path):
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
    await p._poll_once()
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
        tts=make_mock_tts(),
        log=create_autospec(SessionLogger, instance=True),
        thinking=create_autospec(Thinking, instance=True),
        slack_mcp=mcp,
    )


@pytest.mark.asyncio
async def test_respond_slack_calls_send_message_via_mcp():
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.call_tool = AsyncMock()
    ctx = _ctx_with_slack_mcp(mcp)
    task = Task(
        kind="slack_msg",
        topic="slack-general",
        source={"channel_id": "C123", "ts": "1.0", "thread_ts": "1.0"},
    )
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_slack(ctx, task, "Yes, ship it.")
    mcp.call_tool.assert_called_once_with(
        "slack_send_message",
        {"channel_id": "C123", "message": "Yes, ship it.", "thread_ts": "1.0"},
    )
    assert ctx.queue.get(task.id).state == "done"
    assert ctx.current_task is None


@pytest.mark.asyncio
async def test_respond_slack_without_mcp_speaks_error():
    ctx = _ctx_with_slack_mcp(mcp=None)
    task = Task(kind="slack_msg",
                source={"channel_id": "C1", "ts": "1.0", "thread_ts": "1.0"})
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_slack(ctx, task, "reply")
    ctx.tts.speak.assert_called_with("Slack MCP is not configured.")
    assert ctx.queue.get(task.id).state != "done"


@pytest.mark.asyncio
async def test_respond_slack_api_error_keeps_task_active():
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.call_tool = AsyncMock(side_effect=ClaudeMCPError("network down"))
    ctx = _ctx_with_slack_mcp(mcp)
    task = Task(kind="slack_msg",
                source={"channel_id": "C1", "ts": "1.0", "thread_ts": "1.0"})
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._respond_slack(ctx, task, "reply")
    assert ctx.queue.get(task.id).state != "done"
    assert ctx.current_task is task


@pytest.mark.asyncio
async def test_dispatch_task_response_routes_slack_kind():
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.call_tool = AsyncMock()
    ctx = _ctx_with_slack_mcp(mcp)
    task = Task(kind="slack_msg",
                source={"channel_id": "C1", "ts": "1.0", "thread_ts": "1.0"})
    ctx.queue.add(task)
    ctx.current_task = task
    await dispatch._dispatch_task_response(ctx, task, "responding now")
    mcp.call_tool.assert_called_once()


# --- thread dedup --------------------------------------------------------


def _thread_fixture(thread_ts: str, msg_ts: str, body: str, sender: str = "Alice") -> str:
    """Build one detailed-markdown result block. The body is prefixed
    with the test user's `<@UME>` mention so the producer's strict
    mention filter doesn't drop it — these fixtures exist to test
    thread-collapse / dedup logic, not the mention filter itself.
    """
    return (
        f"### Result 1 of 1\n"
        f"Channel: #ai-tools (ID: CTHREAD)\n"
        f"From: {sender} (ID: U{sender.upper()}) \n"
        f"Message_ts: {msg_ts}\n"
        f"Permalink: [link](https://x.slack.com/archives/CTHREAD/p{msg_ts.replace('.','')}"
        f"?thread_ts={thread_ts}&cid=CTHREAD)\n"
        f"Text: \n"
        f"<@UME> {body}\n\n"
        f"---\n\n"
    )


@pytest.mark.asyncio
async def test_producer_collapses_thread_replies_into_single_task(tmp_path: Path):
    """Two messages in the same thread should produce ONE pending task
    (the second message updates the first), not two — long Slack threads
    shouldn't pile up dozens of inbox items."""
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"

    # First poll: an initial message in a thread.
    mcp.call_tool.return_value = {
        "results": _thread_fixture("1716000100.000000", "1716000100.000000", "first message")
    }
    await p._poll_once()
    assert len(q.all()) == 1
    first = q.all()[0]
    assert "first message" in first.body

    # Second poll: a follow-up in the same thread (different msg_ts, same thread_ts).
    mcp.call_tool.return_value = {
        "results": _thread_fixture("1716000100.000000", "1716000200.000000", "follow-up message", sender="Bob")
    }
    await p._poll_once()

    pending = q.pending()
    assert len(pending) == 1, "thread reply should update the existing task, not stack"
    assert pending[0].id == first.id
    assert "follow-up message" in pending[0].body
    assert "Bob" in pending[0].headline
    # Both messages in the thread should be recorded in source["messages"]
    # so the TUI can render the initial + follow-ups.
    msgs = pending[0].source["messages"]
    assert [m["sender"] for m in msgs] == ["Alice", "Bob"]
    assert "first message" in msgs[0]["text"]
    assert "follow-up message" in msgs[1]["text"]


@pytest.mark.asyncio
async def test_producer_separate_threads_get_separate_tasks(tmp_path: Path):
    """Two messages with different thread_ts produce two tasks."""
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {
        "results": (
            _thread_fixture("1716000300.000000", "1716000300.000000", "thread one")
            + _thread_fixture("1716000400.000000", "1716000400.000000", "thread two", sender="Charlie")
        )
    }
    await p._poll_once()
    assert len(q.pending()) == 2


@pytest.mark.asyncio
async def test_producer_appends_into_active_thread_task(tmp_path: Path):
    """When the user is viewing a Slack thread (task is ACTIVE), a new
    reply should append to its source["messages"] rather than spawn a
    sibling task in the queue — otherwise the Current task panel stays
    frozen at the original message count."""
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"

    mcp.call_tool.return_value = {
        "results": _thread_fixture("1716001000.000000", "1716001000.000000", "first")
    }
    await p._poll_once()
    [first] = q.all()
    q.mark_active(first.id)  # user opened the task in the Current panel

    mcp.call_tool.return_value = {
        "results": _thread_fixture(
            "1716001000.000000", "1716001100.000000", "reply while viewing", sender="Bob",
        )
    }
    await p._poll_once()

    all_tasks = q.all()
    assert len(all_tasks) == 1, (
        "active task should absorb the reply, not spawn a sibling"
    )
    msgs = all_tasks[0].source["messages"]
    assert [m["sender"] for m in msgs] == ["Alice", "Bob"]


@pytest.mark.asyncio
async def test_producer_does_not_collapse_into_done_threads(tmp_path: Path):
    """If the user has already dismissed (marked done) a thread task,
    a new message in the same thread should create a fresh task — not
    silently update the dismissed one."""
    p, q, mcp, _state = _producer(tmp_path)
    p._user_id = "UME"
    mcp.call_tool.return_value = {
        "results": _thread_fixture("1716000500.000000", "1716000500.000000", "first")
    }
    await p._poll_once()
    [first] = q.all()
    q.mark_done(first.id)

    mcp.call_tool.return_value = {
        "results": _thread_fixture("1716000500.000000", "1716000600.000000", "follow-up after dismiss")
    }
    await p._poll_once()
    pending = q.pending()
    assert len(pending) == 1
    assert pending[0].id != first.id
