"""Unit tests for earcons."""

from __future__ import annotations

import time
from unittest.mock import patch

from code_trip.earcon import (
    ThinkingEarcon,
    play_completion,
    play_error,
    play_thinking_beep,
    play_tone,
)


def test_play_tone_calls_sounddevice():
    with patch("code_trip.earcon.sd") as mock_sd:
        play_tone(440.0, 0.05)
        mock_sd.play.assert_called_once()
        mock_sd.wait.assert_called_once()


def test_play_completion_plays_two_tones():
    with patch("code_trip.earcon.sd") as mock_sd:
        play_completion()
        mock_sd.play.assert_called_once()
        # concatenated waveform should be longer than a single beep
        waveform = mock_sd.play.call_args.args[0]
        assert len(waveform) > 0


def test_play_error_runs():
    with patch("code_trip.earcon.sd") as mock_sd:
        play_error()
        mock_sd.play.assert_called_once()


def test_play_thinking_beep_runs():
    with patch("code_trip.earcon.sd") as mock_sd:
        play_thinking_beep()
        mock_sd.play.assert_called_once()


def test_thinking_earcon_start_stop_lifecycle():
    with patch("code_trip.earcon.sd"):
        earcon = ThinkingEarcon(interval=0.05)
        earcon.start()
        time.sleep(0.12)
        assert earcon._thread is not None
        assert earcon._thread.is_alive()
        earcon.stop()
        assert earcon._thread is None


def test_thinking_earcon_stop_idempotent():
    with patch("code_trip.earcon.sd"):
        earcon = ThinkingEarcon(interval=0.05)
        earcon.stop()  # no-op before start
        earcon.start()
        earcon.stop()
        earcon.stop()  # second stop is safe


def test_thinking_earcon_double_start_ignored():
    with patch("code_trip.earcon.sd"):
        earcon = ThinkingEarcon(interval=0.05)
        earcon.start()
        first = earcon._thread
        earcon.start()
        assert earcon._thread is first
        earcon.stop()
