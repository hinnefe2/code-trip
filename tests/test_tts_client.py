"""Unit tests for TTSClient with mocked OpenAI client and sounddevice."""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code_trip.tts_client import (
    DEFAULT_MODEL,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    ENV_API_KEY,
    TTSClient,
    TTSClientError,
)


def _make_wav_bytes(
    samples: np.ndarray | None = None,
    sample_rate: int = 24_000,
    channels: int = 1,
) -> bytes:
    """Build a valid WAV byte string for mocking the API response."""
    if samples is None:
        samples = np.zeros(480, dtype=np.int16)  # 20ms at 24kHz
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


@pytest.fixture
def mock_openai():
    with patch("code_trip.tts_client.openai") as m:
        mock_client = MagicMock()
        m.OpenAI.return_value = mock_client
        mock_client.audio.speech.create.return_value = MagicMock(
            content=_make_wav_bytes()
        )
        yield m


@pytest.fixture
def mock_sd():
    with patch("code_trip.tts_client.sd") as m:
        yield m


@pytest.fixture
def client(mock_openai, mock_sd, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    return TTSClient(api_key="test-key")


# --- speak happy path -----------------------------------------------------


def test_speak_calls_api_with_defaults(client, mock_openai, mock_sd):
    client.speak("hello world")

    kwargs = mock_openai.OpenAI.return_value.audio.speech.create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["voice"] == DEFAULT_VOICE
    assert kwargs["speed"] == DEFAULT_SPEED
    assert kwargs["input"] == "hello world"
    assert kwargs["response_format"] == "wav"


def test_speak_passes_custom_voice_speed_model(mock_openai, mock_sd, monkeypatch):
    monkeypatch.delenv(ENV_API_KEY, raising=False)
    c = TTSClient(api_key="k", voice="alloy", speed=1.5, model="custom-model")
    c.speak("hi")

    kwargs = mock_openai.OpenAI.return_value.audio.speech.create.call_args.kwargs
    assert kwargs["voice"] == "alloy"
    assert kwargs["speed"] == 1.5
    assert kwargs["model"] == "custom-model"


def test_speak_strips_text(client, mock_openai, mock_sd):
    client.speak("   hello   ")
    kwargs = mock_openai.OpenAI.return_value.audio.speech.create.call_args.kwargs
    assert kwargs["input"] == "hello"


def test_speak_plays_decoded_audio(client, mock_openai, mock_sd):
    samples = np.array([100, 200, 300, 400], dtype=np.int16)
    mock_openai.OpenAI.return_value.audio.speech.create.return_value = MagicMock(
        content=_make_wav_bytes(samples, sample_rate=22_050)
    )

    client.speak("hello")

    # stop before play, then wait
    mock_sd.stop.assert_called()
    played_args, _ = mock_sd.play.call_args
    played_samples, played_rate = played_args
    assert played_rate == 22_050
    np.testing.assert_array_equal(played_samples, samples)
    mock_sd.wait.assert_called_once()


def test_speak_handles_stereo_audio(client, mock_openai, mock_sd):
    # Interleaved stereo: 3 frames × 2 channels
    samples = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)
    mock_openai.OpenAI.return_value.audio.speech.create.return_value = MagicMock(
        content=_make_wav_bytes(samples, channels=2)
    )

    client.speak("hello")

    played_samples = mock_sd.play.call_args[0][0]
    assert played_samples.shape == (3, 2)


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


# --- stop() interruption --------------------------------------------------


def test_stop_calls_sounddevice_stop(client, mock_sd):
    client.stop()
    mock_sd.stop.assert_called_once()
