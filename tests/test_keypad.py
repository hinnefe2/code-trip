"""Unit tests for KeypadListener with mocked pynput + controllable clock."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_trip.mode_fsm import Gesture, Key


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def mock_recorder():
    rec = MagicMock()
    rec.stop.return_value = Path("/tmp/code-trip-audio/recording_test.wav")
    return rec


@pytest.fixture
def mock_keyboard():
    with patch("code_trip.keypad.keyboard") as m:
        m.Key.f13 = "f13"
        m.Key.f14 = "f14"
        m.Key.f15 = "f15"
        m.Key.f16 = "f16"
        m.Key.f17 = "f17"
        yield m


@pytest.fixture
def captured_timers(monkeypatch):
    """Replace threading.Timer in keypad with an inspectable fake."""
    timers: list["FakeTimer"] = []

    class FakeTimer:
        def __init__(self, interval, function, args=(), kwargs=None):
            self.interval = interval
            self.function = function
            self.args = args
            self.kwargs = kwargs or {}
            self.started = False
            self.cancelled = False
            self.daemon = False
            timers.append(self)

        def start(self) -> None:
            self.started = True

        def cancel(self) -> None:
            self.cancelled = True

        def fire(self) -> None:
            self.function(*self.args, **self.kwargs)

    monkeypatch.setattr("code_trip.keypad.threading.Timer", FakeTimer)
    return timers


@pytest.fixture
def gestures():
    return []


@pytest.fixture
def listener(mock_recorder, mock_keyboard, clock, gestures, captured_timers):
    from code_trip.keypad import KeypadListener

    return KeypadListener(
        recorder=mock_recorder,
        on_gesture=lambda k, g: gestures.append((k, g)),
        clock=clock,
    )


def _callbacks(mock_keyboard):
    kwargs = mock_keyboard.Listener.call_args.kwargs
    return kwargs["on_press"], kwargs["on_release"]


# --- lifecycle ------------------------------------------------------------


def test_start_creates_listener(listener, mock_keyboard):
    listener.start()
    mock_keyboard.Listener.assert_called_once()
    mock_keyboard.Listener.return_value.start.assert_called_once()


def test_double_start_raises(listener, mock_keyboard):
    from code_trip.keypad import KeypadError

    listener.start()
    with pytest.raises(KeypadError, match="already running"):
        listener.start()


def test_stop_cancels_pending_timers(listener, mock_keyboard, captured_timers):
    listener.start()
    on_press, _ = _callbacks(mock_keyboard)
    on_press("f14")
    assert len(captured_timers) == 1
    listener.stop()
    assert captured_timers[0].cancelled


# --- PTT path -------------------------------------------------------------


def test_ptt_press_starts_recorder_release_saves(
    listener, mock_recorder, mock_keyboard
):
    callback = MagicMock()
    listener.on_recording_complete = callback
    listener.start()
    on_press, on_release = _callbacks(mock_keyboard)

    on_press("f13")
    mock_recorder.start.assert_called_once()

    on_release("f13")
    mock_recorder.stop.assert_called_once()
    callback.assert_called_once_with(mock_recorder.stop.return_value)


def test_ptt_repeated_press_ignored(listener, mock_recorder, mock_keyboard):
    listener.start()
    on_press, _ = _callbacks(mock_keyboard)
    on_press("f13")
    on_press("f13")
    assert mock_recorder.start.call_count == 1


def test_ptt_does_not_fire_gesture(listener, mock_keyboard, gestures):
    listener.start()
    on_press, on_release = _callbacks(mock_keyboard)
    on_press("f13")
    on_release("f13")
    assert gestures == []


# --- gesture detection ----------------------------------------------------


def test_short_press_fires_short(listener, mock_keyboard, gestures, clock):
    listener.start()
    on_press, on_release = _callbacks(mock_keyboard)
    on_press("f14")
    clock.advance(0.2)
    on_release("f14")
    assert gestures == [(Key.NAV, Gesture.SHORT)]


def test_long_press_fires_long(listener, mock_keyboard, gestures, clock):
    listener.start()
    on_press, on_release = _callbacks(mock_keyboard)
    on_press("f14")
    clock.advance(0.8)  # > 0.5, < 1.5
    on_release("f14")
    assert gestures == [(Key.NAV, Gesture.LONG)]


def test_hold_fires_via_timer(
    listener, mock_keyboard, gestures, captured_timers
):
    listener.start()
    on_press, on_release = _callbacks(mock_keyboard)
    on_press("f17")
    assert len(captured_timers) == 1
    assert captured_timers[0].interval == 1.5
    captured_timers[0].fire()
    assert gestures == [(Key.NO, Gesture.HOLD)]

    # Release after HOLD must NOT fire a second gesture.
    on_release("f17")
    assert gestures == [(Key.NO, Gesture.HOLD)]


def test_release_before_hold_cancels_timer(
    listener, mock_keyboard, gestures, captured_timers, clock
):
    listener.start()
    on_press, on_release = _callbacks(mock_keyboard)
    on_press("f14")
    clock.advance(0.1)
    on_release("f14")
    assert captured_timers[0].cancelled
    assert gestures == [(Key.NAV, Gesture.SHORT)]


def test_repeated_press_does_not_reset_timer(
    listener, mock_keyboard, captured_timers
):
    listener.start()
    on_press, _ = _callbacks(mock_keyboard)
    on_press("f14")
    on_press("f14")
    assert len(captured_timers) == 1


# --- key routing ----------------------------------------------------------


@pytest.mark.parametrize(
    "pynput_key,logical",
    [
        ("f14", Key.NAV),
        ("f15", Key.ACT),
        ("f16", Key.OK),
        ("f17", Key.NO),
    ],
)
def test_each_key_routes_to_logical(
    listener, mock_keyboard, gestures, clock, pynput_key, logical
):
    listener.start()
    on_press, on_release = _callbacks(mock_keyboard)
    on_press(pynput_key)
    clock.advance(0.1)
    on_release(pynput_key)
    assert gestures == [(logical, Gesture.SHORT)]


def test_unmapped_key_ignored(listener, mock_keyboard, gestures, mock_recorder):
    listener.start()
    on_press, on_release = _callbacks(mock_keyboard)
    on_press("unknown")
    on_release("unknown")
    assert gestures == []
    mock_recorder.start.assert_not_called()
