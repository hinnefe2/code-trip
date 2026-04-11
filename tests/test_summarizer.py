"""Unit tests for Summarizer with mocked OpenAI client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from code_trip.summarizer import (
    DEFAULT_MODEL,
    ENV_API_KEY,
    SUMMARIZER_PROMPT,
    VERBOSITY_INSTRUCTIONS,
    Summarizer,
    SummarizerError,
    Verbosity,
)


@pytest.fixture
def mock_openai():
    with patch("code_trip.summarizer.openai") as m:
        mock_client = MagicMock()
        m.OpenAI.return_value = mock_client
        # Default: successful summary
        mock_choice = MagicMock()
        mock_choice.message.content = "Tests passed successfully with no errors."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )
        yield m


@pytest.fixture
def summarizer(mock_openai, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    return Summarizer(api_key="test-key")


# --- summarize happy path -------------------------------------------------


def test_summarize_returns_text(summarizer, mock_openai):
    result = summarizer.summarize("raw output here")
    assert result == "Tests passed successfully with no errors."


def test_summarize_passes_model(summarizer, mock_openai):
    summarizer.summarize("raw output")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_MODEL


def test_summarize_sends_system_and_user_messages(summarizer, mock_openai):
    summarizer.summarize("raw output")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


# --- mode and user_request context ----------------------------------------


def test_mode_context_prepended(summarizer, mock_openai):
    summarizer.summarize("raw output", mode="WORK")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    user_msg = call_kwargs["messages"][1]["content"]
    assert "[Mode: WORK]" in user_msg
    assert "raw output" in user_msg


def test_user_request_context_prepended(summarizer, mock_openai):
    summarizer.summarize("raw output", user_request="run the tests")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    user_msg = call_kwargs["messages"][1]["content"]
    assert '[User asked: "run the tests"]' in user_msg
    assert "raw output" in user_msg


def test_both_context_fields(summarizer, mock_openai):
    summarizer.summarize("raw output", mode="REVIEW", user_request="check types")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    user_msg = call_kwargs["messages"][1]["content"]
    assert "[Mode: REVIEW]" in user_msg
    assert '[User asked: "check types"]' in user_msg


def test_no_context_when_none(summarizer, mock_openai):
    summarizer.summarize("raw output")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    user_msg = call_kwargs["messages"][1]["content"]
    assert user_msg == "raw output"
    assert "[Mode:" not in user_msg
    assert "[User asked:" not in user_msg


# --- verbosity levels -----------------------------------------------------


def test_verbosity_brief(mock_openai, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    s = Summarizer(api_key="test-key", verbosity=Verbosity.BRIEF)
    s.summarize("raw output")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]["content"]
    assert VERBOSITY_INSTRUCTIONS["brief"] in system_msg


def test_verbosity_detailed(summarizer, mock_openai):
    summarizer.summarize("raw output")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]["content"]
    assert VERBOSITY_INSTRUCTIONS["detailed"] in system_msg


def test_verbosity_verbose(mock_openai, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    s = Summarizer(api_key="test-key", verbosity=Verbosity.VERBOSE)
    s.summarize("raw output")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]["content"]
    assert VERBOSITY_INSTRUCTIONS["verbose"] in system_msg


# --- system prompt contains rules -----------------------------------------


def test_system_prompt_contains_summarizer_rules(summarizer, mock_openai):
    summarizer.summarize("raw output")

    call_kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    system_msg = call_kwargs["messages"][0]["content"]
    assert "NEVER include code" in system_msg
    assert SUMMARIZER_PROMPT in system_msg


# --- API key configuration ------------------------------------------------


def test_api_key_from_env(mock_openai, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "env-key")
    Summarizer()
    mock_openai.OpenAI.assert_called_with(api_key="env-key")


def test_explicit_key_takes_precedence(mock_openai, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "env-key")
    Summarizer(api_key="explicit-key")
    mock_openai.OpenAI.assert_called_with(api_key="explicit-key")


def test_missing_api_key_raises(mock_openai, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(SummarizerError, match="No API key"):
        Summarizer()


# --- error handling -------------------------------------------------------


def test_empty_response_raises(summarizer, mock_openai):
    mock_choice = MagicMock()
    mock_choice.message.content = "   "
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = (
        MagicMock(choices=[mock_choice])
    )
    with pytest.raises(SummarizerError, match="Empty summary"):
        summarizer.summarize("raw output")


def test_auth_error_raises(summarizer, mock_openai):
    import openai as _openai

    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
        _openai.AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body=None,
        )
    )
    mock_openai.AuthenticationError = _openai.AuthenticationError

    with pytest.raises(SummarizerError, match="Authentication failed"):
        summarizer.summarize("raw output")


def test_network_error_raises(summarizer, mock_openai):
    import openai as _openai

    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
        _openai.APIConnectionError(request=MagicMock())
    )
    mock_openai.APIConnectionError = _openai.APIConnectionError

    with pytest.raises(SummarizerError, match="Network error"):
        summarizer.summarize("raw output")


def test_api_error_raises(summarizer, mock_openai):
    import openai as _openai

    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
        _openai.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )
    )
    mock_openai.APIError = _openai.APIError

    with pytest.raises(SummarizerError, match="API error"):
        summarizer.summarize("raw output")
