"""Unit tests for TTSClient with mocked OpenAI client and subprocess."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from code_trip.tts_client import (
    DEFAULT_MODEL,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    ENV_API_KEY,
    PLAYER_CMD,
    TTSClient,
    TTSClientError,
)


@pytest.fixture
def mock_openai():
    with patch("code_trip.tts_client.openai") as m:
        mock_client = MagicMock()
        m.OpenAI.return_value = mock_client
        mock_client.audio.speech.create.return_value = MagicMock(
            content=b"fake-wav-bytes"
        )
        yield m


@pytest.fixture
def mock_popen():
    with patch("code_trip.tts_client.subprocess.Popen") as m:
        proc = MagicMock()
        proc.stdin = MagicMock()
        proc.poll.return_value = 0  # finished by default
        proc.wait.return_value = 0
        m.return_value = proc
        yield m


@pytest.fixture
def client(mock_openai, mock_popen, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    return TTSClient(api_key="test-key")


# --- speak happy path -----------------------------------------------------


def test_speak_calls_api_with_defaults(client, mock_openai, mock_popen):
    client.speak("hello world")

    kwargs = mock_openai.OpenAI.return_value.audio.speech.create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["voice"] == DEFAULT_VOICE
    assert kwargs["speed"] == DEFAULT_SPEED
    assert kwargs["input"] == "hello world"
    assert kwargs["response_format"] == "wav"


def test_speak_passes_custom_voice_speed_model(mock_openai, mock_popen, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    c = TTSClient(api_key="k", voice="alloy", speed=1.5, model="custom-model")
    c.speak("hi")

    kwargs = mock_openai.OpenAI.return_value.audio.speech.create.call_args.kwargs
    assert kwargs["voice"] == "alloy"
    assert kwargs["speed"] == 1.5
    assert kwargs["model"] == "custom-model"


def test_speak_strips_text(client, mock_openai, mock_popen):
    client.speak("   hello   ")
    kwargs = mock_openai.OpenAI.return_value.audio.speech.create.call_args.kwargs
    assert kwargs["input"] == "hello"


def test_speak_writes_audio_to_player(client, mock_openai, mock_popen):
    client.speak("hello")

    mock_popen.assert_called_once_with(
        PLAYER_CMD,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    proc = mock_popen.return_value
    proc.stdin.write.assert_called_once_with(b"fake-wav-bytes")
    proc.stdin.close.assert_called_once()
    proc.wait.assert_called_once()


def test_speak_clears_playback_after_finish(client, mock_popen):
    client.speak("hello")
    assert client._playback is None


# --- API key configuration ------------------------------------------------


def test_api_key_from_env(mock_openai, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "env-key")
    TTSClient()
    mock_openai.OpenAI.assert_called_with(api_key="env-key")


def test_explicit_key_takes_precedence(mock_openai, monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "env-key")
    TTSClient(api_key="explicit-key")
    mock_openai.OpenAI.assert_called_with(api_key="explicit-key")


def test_missing_api_key_raises(mock_openai, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    with pytest.raises(TTSClientError, match="No API key"):
        TTSClient()


# --- input/output validation ----------------------------------------------


def test_empty_text_raises(client):
    with pytest.raises(TTSClientError, match="Empty text"):
        client.speak("")


def test_whitespace_text_raises(client):
    with pytest.raises(TTSClientError, match="Empty text"):
        client.speak("   \n\t  ")


def test_empty_audio_raises(client, mock_openai):
    mock_openai.OpenAI.return_value.audio.speech.create.return_value = MagicMock(
        content=b""
    )
    with pytest.raises(TTSClientError, match="Empty audio"):
        client.speak("hello")


# --- error handling -------------------------------------------------------


def test_auth_error_raises(client, mock_openai):
    import openai as _openai

    mock_openai.OpenAI.return_value.audio.speech.create.side_effect = (
        _openai.AuthenticationError(
            message="bad key",
            response=MagicMock(status_code=401),
            body=None,
        )
    )
    mock_openai.AuthenticationError = _openai.AuthenticationError

    with pytest.raises(TTSClientError, match="Authentication failed"):
        client.speak("hello")


def test_network_error_raises(client, mock_openai):
    import openai as _openai

    mock_openai.OpenAI.return_value.audio.speech.create.side_effect = (
        _openai.APIConnectionError(request=MagicMock())
    )
    mock_openai.APIConnectionError = _openai.APIConnectionError

    with pytest.raises(TTSClientError, match="Network error"):
        client.speak("hello")


def test_api_error_raises(client, mock_openai):
    import openai as _openai

    mock_openai.OpenAI.return_value.audio.speech.create.side_effect = (
        _openai.APIError(
            message="server error",
            request=MagicMock(),
            body=None,
        )
    )
    mock_openai.APIError = _openai.APIError

    with pytest.raises(TTSClientError, match="API error"):
        client.speak("hello")


def test_missing_player_raises(client, mock_openai, mock_popen):
    mock_popen.side_effect = FileNotFoundError("aplay not found")
    with pytest.raises(TTSClientError, match="Audio player not found"):
        client.speak("hello")


# --- stop() interruption --------------------------------------------------


def test_stop_terminates_running_playback(client, mock_popen):
    proc = MagicMock()
    proc.poll.return_value = None  # still running
    client._playback = proc

    client.stop()

    proc.terminate.assert_called_once()
    proc.wait.assert_called()
    assert client._playback is None


def test_stop_skips_terminate_if_finished(client):
    proc = MagicMock()
    proc.poll.return_value = 0  # finished
    client._playback = proc

    client.stop()

    proc.terminate.assert_not_called()
    assert client._playback is None


def test_stop_noop_when_nothing_playing(client):
    assert client._playback is None
    client.stop()  # should not raise
    assert client._playback is None


def test_stop_force_kills_on_timeout(client):
    import subprocess as _sp

    proc = MagicMock()
    proc.poll.return_value = None
    proc.wait.side_effect = [_sp.TimeoutExpired(cmd="aplay", timeout=1.0), 0]
    client._playback = proc

    client.stop()

    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert client._playback is None
