"""Tests for slack_state, slack_filter, SlackProducer, and the reply path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from code_trip2 import dispatch, modes
from code_trip2.producers.slack import SlackProducer
from code_trip2.slack_filter import SlackFilter, SlackFilterError
from code_trip2.slack_state import SlackState
from code_trip2.tasks import Task, TaskQueue


# --- SlackState -----------------------------------------------------------


def test_slack_state_persists_roundtrip(tmp_path: Path):
    p = tmp_path / "slack-state.json"
    s = SlackState(path=p)
    s.set_last_ts("C1", "1716000000.000100")
    s.set_last_ts("C2", "1716000005.000200")
    s2 = SlackState(path=p)
    assert s2.last_ts("C1") == "1716000000.000100"
    assert s2.last_ts("C2") == "1716000005.000200"


def test_slack_state_missing_file_is_empty(tmp_path: Path):
    s = SlackState(path=tmp_path / "nope.json")
    assert s.last_ts("anything") is None
    assert s.all() == {}


def test_slack_state_does_not_regress_ts(tmp_path: Path):
    s = SlackState(path=tmp_path / "s.json")
    s.set_last_ts("C1", "1716000100.000000")
    s.set_last_ts("C1", "1716000050.000000")  # older — should be ignored
    assert s.last_ts("C1") == "1716000100.000000"


def test_slack_state_handles_corrupt_file(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text("{ not json")
    s = SlackState(path=p)
    assert s.all() == {}


# --- SlackFilter ----------------------------------------------------------


def _fake_openai_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def test_filter_disabled_without_api_key():
    f = SlackFilter(api_key=None)
    assert f.enabled is False
    with pytest.raises(SlackFilterError):
        f.evaluate(
            text="hello", sender_name="Alice", channel_name="general",
            is_dm=False, user_id="U1",
        )


def test_filter_empty_message_is_irrelevant():
    f = SlackFilter(api_key="sk-test")
    f._client = MagicMock()
    verdict = f.evaluate(
        text="   ", sender_name="A", channel_name="g", is_dm=False, user_id="U1",
    )
    assert verdict["relevant"] is False
    f._client.chat.completions.create.assert_not_called()


def test_filter_parses_clean_json():
    f = SlackFilter(api_key="sk-test")
    f._client = MagicMock()
    f._client.chat.completions.create.return_value = _fake_openai_response(
        '{"relevant": true, "urgency": "interrupt", "headline": "Alice asks about deploy"}'
    )
    verdict = f.evaluate(
        text="hey @user", sender_name="Alice", channel_name="general",
        is_dm=False, user_id="U1",
    )
    assert verdict == {
        "relevant": True,
        "urgency": "interrupt",
        "headline": "Alice asks about deploy",
    }


def test_filter_extracts_json_from_prose():
    f = SlackFilter(api_key="sk-test")
    f._client = MagicMock()
    f._client.chat.completions.create.return_value = _fake_openai_response(
        'Here is the verdict: {"relevant": false, "urgency": "background", "headline": ""}'
    )
    verdict = f.evaluate(
        text="lol", sender_name="A", channel_name="g", is_dm=False, user_id="U1",
    )
    assert verdict["relevant"] is False
    assert verdict["urgency"] == "background"


def test_filter_normalizes_unknown_urgency():
    f = SlackFilter(api_key="sk-test")
    f._client = MagicMock()
    f._client.chat.completions.create.return_value = _fake_openai_response(
        '{"relevant": true, "urgency": "RED ALERT", "headline": "x"}'
    )
    verdict = f.evaluate(
        text="t", sender_name="A", channel_name="g", is_dm=False, user_id="U1",
    )
    assert verdict["urgency"] == "normal"


def test_filter_api_error_raises():
    f = SlackFilter(api_key="sk-test")
    f._client = MagicMock()
    f._client.chat.completions.create.side_effect = RuntimeError("boom")
    with pytest.raises(SlackFilterError):
        f.evaluate(
            text="t", sender_name="A", channel_name="g", is_dm=False, user_id="U1",
        )


def test_filter_includes_extra_prompt_when_set():
    f = SlackFilter(api_key="sk-test", extra_prompt="My handle is @henry.")
    f._client = MagicMock()
    f._client.chat.completions.create.return_value = _fake_openai_response(
        '{"relevant": true, "urgency": "normal", "headline": "x"}'
    )
    f.evaluate(text="t", sender_name="A", channel_name="g", is_dm=False, user_id="U1")
    sent_messages = f._client.chat.completions.create.call_args.kwargs["messages"]
    assert "@henry" in sent_messages[0]["content"]


# --- SlackProducer --------------------------------------------------------


def _producer(
    *,
    channels=("general",),
    state=None,
    client=None,
    filter_enabled=True,
    user_id="UME",
    poll_interval=30.0,
):
    cfg = SimpleNamespace(
        slack_channels=tuple(channels),
        slack_user_id=user_id,
        slack_poll_interval=poll_interval,
    )
    if client is None:
        client = MagicMock()
        client.auth_test.return_value = {"user_id": user_id}
        client.conversations_list.return_value = {
            "channels": [
                {"id": "C123", "name": "general"},
                {"id": "C999", "name": "other"},
            ],
            "response_metadata": {"next_cursor": ""},
        }
    f = MagicMock(spec=SlackFilter)
    f.enabled = filter_enabled
    q = TaskQueue()
    p = SlackProducer(config=cfg, queue=q, client=client, filter_=f, state=state)
    return p, q, client, f


def test_producer_skips_when_no_client():
    cfg = SimpleNamespace(slack_channels=("general",), slack_user_id="UME")
    p = SlackProducer(config=cfg, queue=TaskQueue(), client=None, filter_=MagicMock())
    p.start()
    assert p._thread is None


def test_producer_skips_when_filter_disabled():
    p, _q, _client, _filter = _producer(filter_enabled=False)
    p.start()
    assert p._thread is None


def test_producer_setup_resolves_channel_names(tmp_path: Path):
    state = SlackState(path=tmp_path / "s.json")
    p, _q, client, _filter = _producer(channels=("general",), state=state)
    p._setup()
    assert p._channels == {"C123": "general"}


def test_producer_handle_message_emits_task_when_relevant(tmp_path: Path):
    state = SlackState(path=tmp_path / "s.json")
    p, q, client, filt = _producer(state=state)
    p._channels = {"C123": "general"}
    p._user_id = "UME"
    filt.evaluate.return_value = {
        "relevant": True,
        "urgency": "interrupt",
        "headline": "Alice: deploy looks broken",
    }
    client.users_info.return_value = {
        "user": {"profile": {"display_name": "Alice"}}
    }
    p._handle_message(
        {"user": "UALICE", "text": "deploy looks broken @UME", "ts": "1716000000.000100"},
        "C123",
        "general",
    )
    tasks = q.all()
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == "slack_msg"
    assert t.topic == "slack-general"
    assert t.headline == "Alice: deploy looks broken"
    assert t.urgency == "interrupt"
    assert t.source["channel_id"] == "C123"
    assert t.source["ts"] == "1716000000.000100"
    assert t.source["thread_ts"] == "1716000000.000100"
    assert t.source["sender_name"] == "Alice"


def test_producer_handle_message_drops_when_irrelevant(tmp_path: Path):
    state = SlackState(path=tmp_path / "s.json")
    p, q, client, filt = _producer(state=state)
    p._channels = {"C123": "general"}
    p._user_id = "UME"
    filt.evaluate.return_value = {"relevant": False, "urgency": "background", "headline": ""}
    client.users_info.return_value = {"user": {"profile": {"display_name": "Bob"}}}
    p._handle_message(
        {"user": "UBOB", "text": "morning everyone", "ts": "1716000001.000100"},
        "C123",
        "general",
    )
    assert q.all() == []


def test_producer_handle_message_skips_own_posts():
    p, q, _client, filt = _producer()
    p._channels = {"C123": "general"}
    p._user_id = "UME"
    p._handle_message(
        {"user": "UME", "text": "talking to myself", "ts": "1.0"}, "C123", "general"
    )
    filt.evaluate.assert_not_called()
    assert q.all() == []


def test_producer_handle_message_skips_bots():
    p, q, _client, filt = _producer()
    p._channels = {"C123": "general"}
    p._user_id = "UME"
    p._handle_message(
        {"bot_id": "B1", "text": "deploy: ok", "ts": "1.0"}, "C123", "general"
    )
    filt.evaluate.assert_not_called()
    assert q.all() == []


def test_producer_poll_channel_advances_last_ts(tmp_path: Path):
    state = SlackState(path=tmp_path / "s.json")
    p, q, client, filt = _producer(state=state)
    p._channels = {"C123": "general"}
    p._user_id = "UME"
    filt.evaluate.return_value = {
        "relevant": True, "urgency": "normal", "headline": "Alice: hi",
    }
    client.users_info.return_value = {"user": {"profile": {"display_name": "Alice"}}}
    client.conversations_history.return_value = {
        "messages": [
            {"user": "UA", "text": "third", "ts": "1716000003.000000"},
            {"user": "UA", "text": "second", "ts": "1716000002.000000"},
            {"user": "UA", "text": "first", "ts": "1716000001.000000"},
        ]
    }
    p._poll_channel("C123", "general")
    assert state.last_ts("C123") == "1716000003.000000"
    # All 3 messages were emitted as tasks.
    assert len(q.all()) == 3


# --- reply path ----------------------------------------------------------


def _ctx_with_slack(client):
    cfg = SimpleNamespace(
        ssh_host="", ssh_options=(), tmux_session="main",
        work_window="work", linear_window="linear", terminal_apps=("kitty",),
    )
    return modes.Context(
        config=cfg,  # type: ignore[arg-type]
        tts=MagicMock(),
        log=MagicMock(),
        thinking=MagicMock(),
        slack_client=client,
    )


def test_respond_slack_posts_to_thread():
    client = MagicMock()
    ctx = _ctx_with_slack(client)
    task = Task(
        kind="slack_msg",
        topic="slack-general",
        headline="alice: deploy?",
        source={"channel_id": "C123", "ts": "1.0", "thread_ts": "1.0"},
    )
    ctx.queue.add(task)
    ctx.current_task = task
    dispatch._respond_slack(ctx, task, "Yes, ship it.")
    client.chat_postMessage.assert_called_once_with(
        channel="C123", thread_ts="1.0", text="Yes, ship it."
    )
    assert ctx.queue.get(task.id).state == "done"
    assert ctx.current_task is None


def test_respond_slack_without_client_speaks_error():
    ctx = _ctx_with_slack(client=None)
    task = Task(
        kind="slack_msg",
        source={"channel_id": "C1", "ts": "1.0", "thread_ts": "1.0"},
    )
    ctx.queue.add(task)
    ctx.current_task = task
    dispatch._respond_slack(ctx, task, "reply")
    ctx.tts.speak.assert_called_with("Slack client is not configured.")
    # Task should still be active since we didn't actually send.
    assert ctx.queue.get(task.id).state != "done"


def test_respond_slack_api_error_keeps_task_active():
    client = MagicMock()
    client.chat_postMessage.side_effect = RuntimeError("network down")
    ctx = _ctx_with_slack(client)
    task = Task(
        kind="slack_msg",
        source={"channel_id": "C1", "ts": "1.0", "thread_ts": "1.0"},
    )
    ctx.queue.add(task)
    ctx.current_task = task
    dispatch._respond_slack(ctx, task, "reply")
    assert ctx.queue.get(task.id).state != "done"
    assert ctx.current_task is task  # still active so user can retry


def test_dispatch_task_response_routes_slack_kind():
    """End-to-end: PTTing while a slack_msg is active calls the reply path."""
    client = MagicMock()
    ctx = _ctx_with_slack(client)
    task = Task(
        kind="slack_msg",
        source={"channel_id": "C1", "ts": "1.0", "thread_ts": "1.0"},
    )
    ctx.queue.add(task)
    ctx.current_task = task
    dispatch._dispatch_task_response(ctx, task, "responding now")
    client.chat_postMessage.assert_called_once()
