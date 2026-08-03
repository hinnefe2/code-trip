"""Unit tests for the Summarizer wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from code_trip2.summarizer import Summarizer, SummarizerError


# --- Summarizer itself -----------------------------------------------------


@pytest.mark.asyncio
async def test_summarizer_disabled_without_api_key():
    s = Summarizer(api_key=None)
    assert s.enabled is False
    with pytest.raises(SummarizerError):
        await s.summarize("anything")


@pytest.mark.asyncio
async def test_summarizer_empty_input_returns_empty():
    s = Summarizer(api_key="sk-test")
    s._client = MagicMock()
    assert await s.summarize("") == ""
    assert await s.summarize("   \n\n   ") == ""
    s._client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_summarizer_calls_chat_completions_with_prompt():
    s = Summarizer(api_key="sk-test", model="gpt-4o-mini")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Tests passed."))]
    ))
    s._client = client

    out = await s.summarize("raw output here", context={"user_prompt": "run tests"})

    assert out == "Tests passed."
    args, kwargs = client.chat.completions.create.call_args
    assert kwargs["model"] == "gpt-4o-mini"
    msgs = kwargs["messages"]
    assert msgs[0]["role"] == "system"
    assert "spoken audio" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"
    assert "run tests" in msgs[1]["content"]
    assert "raw output here" in msgs[1]["content"]


@pytest.mark.asyncio
async def test_summarizer_caps_output_length():
    s = Summarizer(api_key="sk-test", max_chars=20)
    long = "word " * 100
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=long))]
    ))
    s._client = client
    out = await s.summarize("anything")
    assert len(out) <= 21  # 20 chars + the ellipsis we append


@pytest.mark.asyncio
async def test_summarizer_truncates_long_input():
    s = Summarizer(api_key="sk-test", max_input_chars=100)
    big_raw = "X" * 5000 + "Y" * 50  # tail is the meaningful part
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
    ))
    s._client = client
    await s.summarize(big_raw)
    msgs = client.chat.completions.create.call_args.kwargs["messages"]
    body = msgs[1]["content"]
    assert "truncated" in body
    assert "Y" * 50 in body


@pytest.mark.asyncio
async def test_summarizer_api_error_raises():
    s = Summarizer(api_key="sk-test")
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    s._client = client
    with pytest.raises(SummarizerError):
        await s.summarize("raw")

