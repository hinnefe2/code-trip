"""Unit tests for PushToTalk with mocked pynput and AudioRecorder."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_trip.audio_recorder import AudioRecorderError
from code_trip.push_to_talk import PTTState, PushToTalk, PushToTalkError


@pytest.fixture
def mock_recorder():
    rec = MagicMock()
    rec.stop.return_value = Path("/tmp/code-trip-audio/recording_test.wav")
    return rec


@pytest.fixture
def mock_keyboard():
    with patch("code_trip.push_to_talk.keyboard") as m:
        # Make Key.f13 and Key.f14 comparable sentinel objects
        m.Key.f13 = "f13"
        m.Key.f14 = "f14"
        yield m


@pytest.fixture
def ptt(mock_recorder, mock_keyboard):
    return PushToTalk(recorder=mock_recorder, hotkey="f13")


def _get_callbacks(mock_keyboard):
    """Extract on_press and on_release from the Listener constructor call."""
    kwargs = mock_keyboard.Listener.call_args.kwargs
    return kwargs["on_press"], kwargs["on_release"]


# --- start / stop ---------------------------------------------------------


def test_start_creates_listener(ptt, mock_keyboard):
    ptt.start()

    mock_keyboard.Listener.assert_called_once()
    listener = mock_keyboard.Listener.return_value
    listener.start.assert_called_once()


def test_start_twice_raises(ptt, mock_keyboard):
    ptt.start()
    with pytest.raises(PushToTalkError, match="already running"):
        ptt.start()


def test_stop_stops_listener(ptt, mock_keyboard):
    ptt.start()
    ptt.stop()

    mock_keyboard.Listener.return_value.stop.assert_called_once()
    assert ptt.state == PTTState.IDLE


# --- press hotkey ---------------------------------------------------------


def test_press_hotkey_starts_recording(ptt, mock_recorder, mock_keyboard):
    ptt.start()
    on_press, _ = _get_callbacks(mock_keyboard)

    on_press("f13")

    mock_recorder.start.assert_called_once()
    assert ptt.state == PTTState.RECORDING


def test_press_wrong_key_ignored(ptt, mock_recorder, mock_keyboard):
    ptt.start()
    on_press, _ = _get_callbacks(mock_keyboard)

    on_press("f14")

    mock_recorder.start.assert_not_called()
    assert ptt.state == PTTState.IDLE


def test_repeated_press_ignored(ptt, mock_recorder, mock_keyboard):
    ptt.start()
    on_press, _ = _get_callbacks(mock_keyboard)

    on_press("f13")
    on_press("f13")  # key-repeat

    mock_recorder.start.assert_called_once()


# --- release hotkey -------------------------------------------------------


def test_release_hotkey_stops_and_saves(ptt, mock_recorder, mock_keyboard):
    ptt.start()
    on_press, on_release = _get_callbacks(mock_keyboard)

    on_press("f13")
    on_release("f13")

    mock_recorder.stop.assert_called_once()
    assert ptt.state == PTTState.IDLE


def test_release_calls_on_recording_complete(mock_recorder, mock_keyboard):
    callback = MagicMock()
    ptt = PushToTalk(
        recorder=mock_recorder, hotkey="f13", on_recording_complete=callback
    )
    ptt.start()
    on_press, on_release = _get_callbacks(mock_keyboard)

    on_press("f13")
    on_release("f13")

    callback.assert_called_once_with(mock_recorder.stop.return_value)


def test_release_without_press_ignored(ptt, mock_recorder, mock_keyboard):
    ptt.start()
    _, on_release = _get_callbacks(mock_keyboard)

    on_release("f13")

    mock_recorder.stop.assert_not_called()
    assert ptt.state == PTTState.IDLE


def test_release_wrong_key_ignored(ptt, mock_recorder, mock_keyboard):
    ptt.start()
    on_press, on_release = _get_callbacks(mock_keyboard)

    on_press("f13")
    on_release("f14")  # wrong key

    mock_recorder.stop.assert_not_called()
    assert ptt.state == PTTState.RECORDING


# --- custom hotkey --------------------------------------------------------


def test_custom_hotkey(mock_recorder, mock_keyboard):
    ptt = PushToTalk(recorder=mock_recorder, hotkey="f14")
    ptt.start()
    on_press, _ = _get_callbacks(mock_keyboard)

    on_press("f13")
    assert ptt.state == PTTState.IDLE

    on_press("f14")
    assert ptt.state == PTTState.RECORDING


# --- error handling -------------------------------------------------------


def test_recorder_error_during_stop_returns_to_idle(
    ptt, mock_recorder, mock_keyboard
):
    mock_recorder.stop.side_effect = AudioRecorderError("write failed")
    ptt.start()
    on_press, on_release = _get_callbacks(mock_keyboard)

    on_press("f13")
    on_release("f13")

    assert ptt.state == PTTState.IDLE


def test_recorder_error_during_start_stays_idle(
    ptt, mock_recorder, mock_keyboard
):
    mock_recorder.start.side_effect = AudioRecorderError("no device")
    ptt.start()
    on_press, _ = _get_callbacks(mock_keyboard)

    on_press("f13")

    assert ptt.state == PTTState.IDLE


def test_stop_during_recording_stops_recorder(ptt, mock_recorder, mock_keyboard):
    ptt.start()
    on_press, _ = _get_callbacks(mock_keyboard)

    on_press("f13")
    assert ptt.state == PTTState.RECORDING

    ptt.stop()

    mock_recorder.stop.assert_called_once()
    assert ptt.state == PTTState.IDLE
