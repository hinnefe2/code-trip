"""Keypad listener: maps F13-F17 to logical keys with gesture detection.

Provides a single pynput listener for the full 5-key layout (PTT, NAV,
ACT, OK, NO). PTT stays hold-to-record and drives the :class:`AudioRecorder`
directly; the other four keys emit ``(Key, Gesture)`` events to a caller-
supplied handler, distinguishing short press, long press, and hold based
on configurable thresholds.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pynput import keyboard

from code_trip.audio_recorder import AudioRecorder, AudioRecorderError
from code_trip.mode_fsm import Gesture, Key

logger = logging.getLogger(__name__)


# --- Exceptions -----------------------------------------------------------


class KeypadError(Exception):
    """Raised when a keypad operation fails."""


# --- Config ---------------------------------------------------------------


def _default_key_map() -> dict[Key, keyboard.Key]:
    return {
        Key.PTT: keyboard.Key.f13,
        Key.NAV: keyboard.Key.f14,
        Key.ACT: keyboard.Key.f15,
        Key.OK: keyboard.Key.f16,
        Key.NO: keyboard.Key.f17,
    }


@dataclass
class KeypadConfig:
    """Keypad mapping and gesture timing thresholds.

    Args:
        key_map: logical ``Key`` -> pynput key binding.
        long_threshold: minimum hold duration (s) to count as LONG.
        hold_threshold: hold duration (s) after which HOLD fires while
            the key is still pressed.
    """

    key_map: dict[Key, keyboard.Key] = field(default_factory=_default_key_map)
    long_threshold: float = 0.5
    hold_threshold: float = 1.5


# --- Per-key state --------------------------------------------------------


@dataclass
class _KeyState:
    press_time: float
    timer: threading.Timer | None
    hold_fired: bool = False


# --- KeypadListener -------------------------------------------------------


@dataclass
class KeypadListener:
    """Single pynput listener driving PTT + gesture-aware dispatch.

    Args:
        recorder: AudioRecorder used by the PTT key.
        on_gesture: called ``(Key, Gesture)`` for NAV/ACT/OK/NO.
        on_recording_complete: optional callback fired with the WAV
            path after each PTT release.
        config: key map and gesture thresholds.
        clock: monotonic clock (override for testing).
    """

    recorder: AudioRecorder
    on_gesture: Callable[[Key, Gesture], None]
    on_recording_complete: Callable[[Path], None] | None = None
    config: KeypadConfig = field(default_factory=KeypadConfig)
    clock: Callable[[], float] = time.monotonic

    _listener: keyboard.Listener | None = field(default=None, init=False, repr=False)
    _reverse_map: dict[object, Key] = field(default_factory=dict, init=False, repr=False)
    _states: dict[Key, _KeyState] = field(default_factory=dict, init=False, repr=False)
    _ptt_recording: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._reverse_map = {
            pynput_key: logical for logical, pynput_key in self.config.key_map.items()
        }

    # --- public API -------------------------------------------------------

    def start(self) -> None:
        """Start listening for keypad events.

        Raises:
            KeypadError: if the listener is already running.
        """
        if self._listener is not None:
            raise KeypadError("Listener already running")
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        """Stop the listener and cancel any pending hold timers."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        for state in self._states.values():
            if state.timer is not None:
                state.timer.cancel()
        self._states.clear()
        if self._ptt_recording:
            try:
                self.recorder.stop()
            except AudioRecorderError:
                pass
            self._ptt_recording = False

    # --- pynput callbacks -------------------------------------------------

    def _on_press(self, pynput_key: object) -> None:
        logical = self._reverse_map.get(pynput_key)
        if logical is None:
            return
        if logical is Key.PTT:
            self._ptt_press()
            return
        # Ignore key-repeat while already tracking this key.
        if logical in self._states:
            return
        timer = threading.Timer(
            self.config.hold_threshold, self._fire_hold, args=(logical,)
        )
        timer.daemon = True
        self._states[logical] = _KeyState(press_time=self.clock(), timer=timer)
        timer.start()

    def _on_release(self, pynput_key: object) -> None:
        logical = self._reverse_map.get(pynput_key)
        if logical is None:
            return
        if logical is Key.PTT:
            self._ptt_release()
            return
        state = self._states.pop(logical, None)
        if state is None:
            return
        if state.timer is not None:
            state.timer.cancel()
        if state.hold_fired:
            return
        duration = self.clock() - state.press_time
        gesture = Gesture.SHORT if duration < self.config.long_threshold else Gesture.LONG
        self._dispatch(logical, gesture)

    # --- PTT path ---------------------------------------------------------

    def _ptt_press(self) -> None:
        if self._ptt_recording:
            return
        try:
            self.recorder.start()
            self._ptt_recording = True
        except AudioRecorderError:
            logger.exception("Failed to start recording")

    def _ptt_release(self) -> None:
        if not self._ptt_recording:
            return
        self._ptt_recording = False
        try:
            path = self.recorder.stop()
        except AudioRecorderError:
            logger.exception("Failed to stop recording")
            return
        if self.on_recording_complete is not None:
            self.on_recording_complete(path)

    # --- hold firing ------------------------------------------------------

    def _fire_hold(self, logical: Key) -> None:
        state = self._states.get(logical)
        if state is None or state.hold_fired:
            return
        state.hold_fired = True
        self._dispatch(logical, Gesture.HOLD)

    def _dispatch(self, logical: Key, gesture: Gesture) -> None:
        try:
            self.on_gesture(logical, gesture)
        except Exception:
            logger.exception("Gesture handler raised for %s.%s", logical.name, gesture.name)
