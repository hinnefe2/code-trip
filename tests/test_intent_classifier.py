"""Unit tests for IntentClassifier with mocked OpenAI client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from code_trip.intent_classifier import (
    ENV_API_KEY,
    Intent,
    IntentClassifier,
    IntentClassifierError,
)


def _response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    return MagicMock(choices=[choice])


@pytest.fixture
def mock_openai():
    with patch("code_trip.intent_classifier.openai") as m:
        client = MagicMock()
        m.OpenAI.return_value = client
        client.chat.completions.create.return_value = _response(
            json.dumps({"intent": None, "arg": None})
        )
        yield m


@pytest.fixture
def classifier(mock_openai, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    return IntentClassifier(api_key="test-key")


def _set_response(mock_openai, content: str) -> None:
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = (
        _response(content)
    )


def test_list_tickets(classifier, mock_openai):
    _set_response(mock_openai, json.dumps({"intent": "list_tickets", "arg": None}))
    result = classifier.classify("show my tickets")
    assert result is not None
    assert result.intent is Intent.LIST_TICKETS
    assert result.arg is None


def test_switch_ticket_parses_int(classifier, mock_openai):
    _set_response(mock_openai, json.dumps({"intent": "switch_ticket", "arg": 3}))
    result = classifier.classify("switch to ticket three")
    assert result is not None
    assert result.intent is Intent.SWITCH_TICKET
    assert result.arg == 3


def test_switch_ticket_string_int_coerced(classifier, mock_openai):
    _set_response(mock_openai, json.dumps({"intent": "switch_ticket", "arg": "5"}))
    result = classifier.classify("five")
    assert result is not None
    assert result.arg == 5


def test_switch_ticket_missing_arg_returns_none(classifier, mock_openai):
    _set_response(mock_openai, json.dumps({"intent": "switch_ticket", "arg": None}))
    assert classifier.classify("switch") is None


def test_status(classifier, mock_openai):
    _set_response(mock_openai, json.dumps({"intent": "status", "arg": None}))
    result = classifier.classify("what's the status")
    assert result is not None
    assert result.intent is Intent.STATUS


def test_ship_it(classifier, mock_openai):
    _set_response(mock_openai, json.dumps({"intent": "ship_it", "arg": None}))
    result = classifier.classify("ship it")
    assert result is not None
    assert result.intent is Intent.SHIP_IT


def test_set_verbosity_brief(classifier, mock_openai):
    _set_response(
        mock_openai, json.dumps({"intent": "set_verbosity", "arg": "brief"})
    )
    result = classifier.classify("be brief")
    assert result is not None
    assert result.intent is Intent.SET_VERBOSITY
    assert result.arg == "brief"


def test_set_verbosity_invalid_arg_returns_none(classifier, mock_openai):
    _set_response(
        mock_openai, json.dumps({"intent": "set_verbosity", "arg": "loud"})
    )
    assert classifier.classify("be loud") is None


def test_null_intent_returns_none(classifier, mock_openai):
    _set_response(mock_openai, json.dumps({"intent": None, "arg": None}))
    assert classifier.classify("refactor this function") is None


def test_unknown_intent_name_returns_none(classifier, mock_openai):
    _set_response(mock_openai, json.dumps({"intent": "make_coffee", "arg": None}))
    assert classifier.classify("brew") is None


def test_malformed_json_returns_none(classifier, mock_openai):
    _set_response(mock_openai, "this is not json")
    assert classifier.classify("anything") is None


def test_empty_transcript_skips_api(classifier, mock_openai):
    assert classifier.classify("   ") is None
    mock_openai.OpenAI.return_value.chat.completions.create.assert_not_called()


def test_uses_json_response_format(classifier, mock_openai):
    _set_response(mock_openai, json.dumps({"intent": "status", "arg": None}))
    classifier.classify("status")
    kwargs = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_missing_api_key_raises(mock_openai, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(IntentClassifierError, match="No API key"):
        IntentClassifier()


def test_auth_error_raises(classifier, mock_openai):
    import openai as _openai

    mock_openai.OpenAI.return_value.chat.completions.create.side_effect = (
        _openai.AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body=None,
        )
    )
    mock_openai.AuthenticationError = _openai.AuthenticationError

    with pytest.raises(IntentClassifierError, match="Authentication failed"):
        classifier.classify("status")
