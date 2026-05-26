"""Unit tests for the Summarizer wrapper + WORK-flow fallback."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from code_trip2 import modes
from code_trip2.summarizer import Summarizer, SummarizerError


# --- Summarizer itself -----------------------------------------------------


def test_summarizer_disabled_without_api_key():
    s = Summarizer(api_key=None)
    assert s.enabled is False
    with pytest.raises(SummarizerError):
        s.summarize("anything")


def test_summarizer_empty_input_returns_empty():
    s = Summarizer(api_key="sk-test")
    s._client = MagicMock()
    assert s.summarize("") == ""
    assert s.summarize("   \n\n   ") == ""
    s._client.chat.completions.create.assert_not_called()


def test_summarizer_calls_chat_completions_with_prompt():
    s = Summarizer(api_key="sk-test", model="gpt-4o-mini")
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Tests passed."))]
    )
    s._client = client

    out = s.summarize("raw output here", context={"user_prompt": "run tests"})

    assert out == "Tests passed."
    args, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    msgs = kwargs["messages"]
    assert msgs[0]["role"] == "system"
    assert "spoken audio" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "run tests" in msgs[1]["content"]
    assert "raw output here" in msgs[1]["content"]


def test_summarizer_caps_output_length():
    s = Summarizer(api_key="sk-test", max_chars=20)
    long = "word " * 100
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=long))]
    )
    s._client = client
    out = s.summarize("anything")
    assert len(out) <= 21  # 20 chars + the ellipsis we append


def test_summarizer_truncates_long_input():
    s = Summarizer(api_key="sk-test", max_input_chars=100)
    big_raw = "X" * 5000 + "Y" * 50  # tail is the meaningful part
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    )
    s._client = client
    s.summarize(big_raw)
    msgs = client.chat.completions.create.call_args.kwargs["messages"]
    body = msgs[1]["content"]
    assert "truncated" in body
    assert "Y" * 50 in body


def test_summarizer_api_error_raises():
    s = Summarizer(api_key="sk-test")
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("boom")
    s._client = client
    with pytest.raises(SummarizerError):
        s.summarize("raw")


# --- modes._summarize_or_strip fallback ------------------------------------


def _make_ctx_for_strip(summarizer=None):
    tts = MagicMock()
    tts.is_playing.return_value = False
    cfg = SimpleNamespace(
        ssh_host="",
        ssh_options=(),
        tmux_session="main",
        work_window="work",
        linear_window="linear",
        terminal_apps=("kitty",),
    )
    return modes.Context(
        config=cfg,  # type: ignore[arg-type]
        tts=tts,
        log=MagicMock(),
        thinking=MagicMock(),
        summarizer=summarizer,
    )


def test_summarize_or_strip_no_summarizer_falls_back():
    ctx = _make_ctx_for_strip(summarizer=None)
    out, used_llm = modes._summarize_or_strip(ctx, "some text\n>", "the prompt")
    assert used_llm is False
    assert ">" not in out


def test_summarize_or_strip_uses_summarizer_when_enabled():
    summarizer = MagicMock()
    summarizer.enabled = True
    summarizer.summarize.return_value = "Summary."
    ctx = _make_ctx_for_strip(summarizer=summarizer)
    out, used_llm = modes._summarize_or_strip(ctx, "raw", "ask")
    assert out == "Summary."
    assert used_llm is True
    summarizer.summarize.assert_called_once()
    kwargs = summarizer.summarize.call_args.kwargs
    assert kwargs["context"]["user_prompt"] == "ask"
    assert kwargs["context"]["kind"] == "claude_reply"


def test_summarize_or_strip_falls_back_on_summarizer_error():
    summarizer = MagicMock()
    summarizer.enabled = True
    summarizer.summarize.side_effect = SummarizerError("api 500")
    ctx = _make_ctx_for_strip(summarizer=summarizer)
    out, used_llm = modes._summarize_or_strip(ctx, "raw text", "ask")
    assert used_llm is False
    assert out


def test_summarize_or_strip_falls_back_on_empty_summary():
    summarizer = MagicMock()
    summarizer.enabled = True
    summarizer.summarize.return_value = ""
    ctx = _make_ctx_for_strip(summarizer=summarizer)
    out, used_llm = modes._summarize_or_strip(ctx, "raw text", "ask")
    assert used_llm is False


# --- ClaudeProducer summarization ------------------------------------------


@pytest.mark.asyncio
async def test_claude_producer_summarizes_pane():
    from code_trip2.producers.claude import ClaudeProducer
    from code_trip2.tasks import TaskQueue

    cfg = SimpleNamespace(ssh_host="remote", ssh_options=(), tmux_session="main")
    summarizer = MagicMock()
    summarizer.enabled = True
    summarizer.summarize.return_value = "Tests passed in two files."
    q = TaskQueue()

    p = ClaudeProducer(config=cfg, queue=q, summarizer=summarizer)  # type: ignore[arg-type]

    with patch("code_trip2.producers.claude.remote.capture") as cap:
        cap.return_value = "raw pane text"
        await p._emit(
            {"window": "ticket-42", "finished_at": 1000.0, "last_user_msg": "run tests"},
            "remote", (),
        )

    tasks = q.all()
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == "claude_reply"
    assert t.topic == "ticket-42"
    assert t.body == "Tests passed in two files."
    summarizer.summarize.assert_called_once()


@pytest.mark.asyncio
async def test_claude_producer_no_summarizer_leaves_body_none():
    from code_trip2.producers.claude import ClaudeProducer
    from code_trip2.tasks import TaskQueue

    cfg = SimpleNamespace(ssh_host="remote", ssh_options=(), tmux_session="main")
    q = TaskQueue()

    p = ClaudeProducer(config=cfg, queue=q, summarizer=None)  # type: ignore[arg-type]
    await p._emit({"window": "w", "finished_at": 1000.0}, "remote", ())

    tasks = q.all()
    assert len(tasks) == 1
    assert tasks[0].body is None
