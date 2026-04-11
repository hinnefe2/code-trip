"""Unit tests for STTClient with mocked OpenAI client."""

from __future__ import annotations

import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code_trip.stt_client import (
    DEFAULT_MODEL,
    ENV_API_KEY,
    STTClient,
    STTClientError,
)


@pytest.fixture
def mock_openai():
    with patch("code_trip.stt_client.openai") as m:
        mock_client = MagicMock()
        m.OpenAI.return_value = mock_client
        # Default: successful transcription
        mock_client.audio.transcriptions.create.return_value = MagicMock(
            text="hello world"
        )
        yield m


@pytest.fixture
def client(mock_openai, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    return STTClient(api_key="test-key")


@pytest.fixture
def wav_file(tmp_path):
    """Write a minimal valid WAV file."""
    path = tmp_path / "test.wav"
    data = np.zeros(160, dtype="int16")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(data.tobytes())
    return path


# --- transcribe happy path ------------------------------------------------


def test_transcribe_returns_text(client, mock_openai, wav_file):
    result = client.transcribe(wav_file)
    assert result == "hello world"


def test_transcribe_passes_model_and_file(client, mock_openai, wav_file):
    client.transcribe(wav_file)

    call_kwargs = mock_openai.OpenAI.return_value.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_MODEL
    assert hasattr(call_kwargs["file"], "read")


def test_transcribe_passes_language(mock_openai, monkeypatch, wav_file):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    c = STTClient(api_key="test-key", language="en")
    c.transcribe(wav_file)

    call_kwargs = mock_openai.OpenAI.return_value.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["language"] == "en"


def test_transcribe_omits_language_when_none(client, mock_openai, wav_file):
    client.transcribe(wav_file)

    call_kwargs = mock_openai.OpenAI.return_value.audio.transcriptions.create.call_args.kwargs
    assert "language" not in call_kwargs


# --- API key configuration ------------------------------------------------


def test_api_key_from_env(mock_openai, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "env-key")
    c = STTClient()
    mock_openai.OpenAI.assert_called_with(api_key="env-key")


def test_explicit_key_takes_precedence(mock_openai, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "env-key")
    c = STTClient(api_key="explicit-key")
    mock_openai.OpenAI.assert_called_with(api_key="explicit-key")


def test_missing_api_key_raises(mock_openai, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(STTClientError, match="No API key"):
        STTClient()


# --- error handling -------------------------------------------------------


def test_file_not_found_raises(client):
    with pytest.raises(STTClientError, match="Audio file not found"):
        client.transcribe("/nonexistent/path.wav")


def test_empty_transcription_raises(client, mock_openai, wav_file):
    mock_openai.OpenAI.return_value.audio.transcriptions.create.return_value = (
        MagicMock(text="   ")
    )
    with pytest.raises(STTClientError, match="Empty transcription"):
        client.transcribe(wav_file)


def test_auth_error_raises(client, mock_openai, wav_file):
    import openai as _openai

    mock_openai.OpenAI.return_value.audio.transcriptions.create.side_effect = (
        _openai.AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body=None,
        )
    )
    # Re-assign so the except clause matches
    mock_openai.AuthenticationError = _openai.AuthenticationError

    with pytest.raises(STTClientError, match="Authentication failed"):
        client.transcribe(wav_file)


def test_network_error_raises(client, mock_openai, wav_file):
    import openai as _openai

    mock_openai.OpenAI.return_value.audio.transcriptions.create.side_effect = (
        _openai.APIConnectionError(request=MagicMock())
    )
    mock_openai.APIConnectionError = _openai.APIConnectionError

    with pytest.raises(STTClientError, match="Network error"):
        client.transcribe(wav_file)


def test_api_error_raises(client, mock_openai, wav_file):
    import openai as _openai

    mock_openai.OpenAI.return_value.audio.transcriptions.create.side_effect = (
        _openai.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )
    )
    mock_openai.APIError = _openai.APIError

    with pytest.raises(STTClientError, match="API error"):
        client.transcribe(wav_file)
