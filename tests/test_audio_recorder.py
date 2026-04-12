"""Unit tests for AudioRecorder with mocked sounddevice."""

from __future__ import annotations

import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from code_trip.audio_recorder import (
    AudioRecorder,
    AudioRecorderError,
    DEFAULT_CHANNELS,
    DEFAULT_DTYPE,
    DEFAULT_SAMPLE_RATE,
)


@pytest.fixture
def recorder(tmp_path):
    return AudioRecorder(output_dir=tmp_path)


@pytest.fixture
def mock_sd():
    with patch("code_trip.audio_recorder.sd") as m:
        mock_stream = MagicMock()
        m.InputStream.return_value = mock_stream
        yield m


# --- start ----------------------------------------------------------------


def test_start_opens_stream(recorder, mock_sd):
    recorder.start()

    mock_sd.InputStream.assert_called_once()
    kwargs = mock_sd.InputStream.call_args.kwargs
    assert kwargs["samplerate"] == DEFAULT_SAMPLE_RATE
    assert kwargs["channels"] == DEFAULT_CHANNELS
    assert kwargs["dtype"] == DEFAULT_DTYPE
    assert kwargs["device"] is None
    assert kwargs["callback"] is not None
    mock_sd.InputStream.return_value.start.assert_called_once()


def test_start_while_recording_raises(recorder, mock_sd):
    recorder.start()
    with pytest.raises(AudioRecorderError, match="Already recording"):
        recorder.start()


def test_custom_device_passed_to_stream(tmp_path, mock_sd):
    rec = AudioRecorder(device=3, output_dir=tmp_path)
    rec.start()
    assert mock_sd.InputStream.call_args.kwargs["device"] == 3


def test_start_device_error_raises(recorder, mock_sd):
    mock_sd.InputStream.side_effect = OSError("No such device")
    with pytest.raises(AudioRecorderError, match="Failed to open audio stream"):
        recorder.start()


# --- stop -----------------------------------------------------------------


def test_stop_writes_wav_and_returns_path(recorder, mock_sd):
    recorder.start()

    # Simulate two frames arriving via callback
    callback = mock_sd.InputStream.call_args.kwargs["callback"]
    frame = np.zeros((160, 1), dtype="int16")
    callback(frame, 160, None, None)
    callback(frame, 160, None, None)

    path = recorder.stop()

    assert path.exists()
    assert path.suffix == ".wav"
    assert path.parent == recorder.output_dir

    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == DEFAULT_CHANNELS
        assert wf.getsampwidth() == 2  # int16
        assert wf.getframerate() == DEFAULT_SAMPLE_RATE
        assert wf.getnframes() == 320  # 160 * 2


def test_stop_without_start_raises(recorder):
    with pytest.raises(AudioRecorderError, match="Not currently recording"):
        recorder.stop()


def test_wav_content_matches_frames(recorder, mock_sd):
    recorder.start()

    callback = mock_sd.InputStream.call_args.kwargs["callback"]
    data = np.arange(100, dtype="int16").reshape(-1, 1)
    callback(data, 100, None, None)

    path = recorder.stop()

    with wave.open(str(path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    result = np.frombuffer(raw, dtype="int16")
    np.testing.assert_array_equal(result, data.flatten())


def test_stop_with_no_frames_writes_empty_wav(recorder, mock_sd):
    recorder.start()
    path = recorder.stop()

    with wave.open(str(path), "rb") as wf:
        assert wf.getnframes() == 0


# --- output_dir -----------------------------------------------------------


def test_output_dir_created_on_stop(tmp_path, mock_sd):
    nested = tmp_path / "sub" / "dir"
    rec = AudioRecorder(output_dir=nested)
    rec.start()
    path = rec.stop()
    assert nested.is_dir()
    assert path.parent == nested


# --- is_recording ---------------------------------------------------------


def test_is_recording_property(recorder, mock_sd):
    assert recorder.is_recording is False
    recorder.start()
    assert recorder.is_recording is True
    recorder.stop()
    assert recorder.is_recording is False


# --- list_devices ---------------------------------------------------------


def test_list_devices(mock_sd):
    mock_sd.query_devices.return_value = [
        {"name": "Mic", "index": 0},
        {"name": "Speaker", "index": 1},
    ]
    devices = AudioRecorder.list_devices()
    assert len(devices) == 2
    assert devices[0]["name"] == "Mic"
