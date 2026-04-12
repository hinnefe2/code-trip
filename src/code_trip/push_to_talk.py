"""Push-to-talk: hold a key to record, release to save.

Wires a pynput keyboard listener to an :class:`AudioRecorder` so that
holding a hotkey captures audio and releasing it writes a WAV file.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from pynput import keyboard

from code_trip.audio_recorder import AudioRecorder, AudioRecorderError

logger = logging.getLogger(__name__)

# --- Module-level defaults ------------------------------------------------

DEFAULT_HOTKEY = keyboard.Key.f13


# --- State ----------------------------------------------------------------


class PTTState(enum.Enum):
    """Push-to-talk state machine states."""

    IDLE = "idle"
    RECORDING = "recording"
    SAVING = "saving"


# --- Exceptions -----------------------------------------------------------


class PushToTalkError(Exception):
    """Raised when a push-to-talk operation fails."""


# --- PushToTalk -----------------------------------------------------------


@dataclass
class PushToTalk:
    """Push-to-talk controller: hold hotkey to record, release to save WAV.

    Args:
        recorder: AudioRecorder instance to use for capture.
        hotkey: The key that activates recording (default F13).
        on_recording_complete: Optional callback invoked with the WAV file
            path after each recording is saved.  Runs on the pynput
            listener thread.
    """

    recorder: AudioRecorder
    hotkey: keyboard.Key | keyboard.KeyCode = DEFAULT_HOTKEY
    on_recording_complete: Callable[[Path], None] | None = None

    _state: PTTState = field(default=PTTState.IDLE, init=False, repr=False)
    _listener: keyboard.Listener | None = field(default=None, init=False, repr=False)

    # --- public API -------------------------------------------------------

    def start(self) -> None:
        """Start listening for the hotkey.

        Creates a :class:`pynput.keyboard.Listener` that runs in a daemon
        thread.

        Raises:
            PushToTalkError: if the listener is already running.
        """
        if self._listener is not None:
            raise PushToTalkError("Listener already running")
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        """Stop the hotkey listener.

        If a recording is in progress it is discarded.
        """
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        if self._state == PTTState.RECORDING:
            try:
                self.recorder.stop()
            except AudioRecorderError:
                pass
            self._state = PTTState.IDLE

    @property
    def state(self) -> PTTState:
        """Current push-to-talk state."""
        return self._state

    # --- pynput callbacks -------------------------------------------------

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key != self.hotkey:
            return
        if self._state != PTTState.IDLE:
            return  # ignore key-repeat while already recording
        try:
            self.recorder.start()
            self._state = PTTState.RECORDING
        except AudioRecorderError:
            logger.exception("Failed to start recording")
            self._state = PTTState.IDLE

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key != self.hotkey:
            return
        if self._state != PTTState.RECORDING:
            return
        self._state = PTTState.SAVING
        try:
            path = self.recorder.stop()
            if self.on_recording_complete is not None:
                self.on_recording_complete(path)
        except AudioRecorderError:
            logger.exception("Failed to stop recording")
        finally:
            self._state = PTTState.IDLE
