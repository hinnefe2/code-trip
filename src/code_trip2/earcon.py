"""Short audio cues (earcons) routed through the shared OutputStream.

Earcons used to call ``sd.play()`` per beep, which opens and tears
down a fresh CoreAudio stream every time. Repeated open/close
thrashes the device and shows up as a click/chop at the start of
the next sound — including in unrelated apps. We now write into the
same persistent stream as TTS, so the device stays warm.
"""

from __future__ import annotations

import threading

import numpy as np

from code_trip2 import audio_out


class EarconError(Exception):
    pass


# Module-level mute. ``main.py`` flips this on for ``--silent``; every
# play call early-returns and ``Thinking`` declines to spawn its
# background tone thread. Module global rather than a per-callsite
# flag because earcons fire from many places (chords, dispatch,
# modes) — gating them all by parameter would be invasive.
_silent = False


def set_silent(value: bool) -> None:
    """Mute (or unmute) all earcon playback. Process-wide."""
    global _silent
    _silent = bool(value)


def is_silent() -> bool:
    return _silent


def _tone(freq: float, duration: float, amplitude: float = 0.2) -> np.ndarray:
    sr = audio_out.stream_rate()
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    fade = int(sr * 0.01)
    if fade * 2 < len(wave):
        env = np.ones_like(wave)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        wave *= env
    return wave.astype(np.float32)


def _play(samples: np.ndarray) -> None:
    if _silent:
        return
    try:
        audio_out.play_blocking(samples)
    except Exception as exc:
        raise EarconError(f"Playback failed: {exc}") from exc


def thinking() -> None:
    _play(_tone(660.0, 0.08))


def completion() -> None:
    _play(np.concatenate([_tone(660.0, 0.08), _tone(880.0, 0.12)]))


def error() -> None:
    _play(_tone(220.0, 0.25))


MODE_CHIMES: dict[str, float] = {
    "IDLE": 440.0,
    "NAVIGATING": 523.25,
    "WORK": 587.33,
    "LINEAR": 659.25,
    "SLACK": 783.99,
    # Queue / focused-app app-modes for the task-queue dispatch.
    "queue": 698.46,    # F5
    "focused": 392.00,  # G4 — distinctly lower so direction of flip is audible
}


def mode_chime(mode: str) -> None:
    base = MODE_CHIMES.get(mode)
    if base is None:
        raise EarconError(f"Unknown mode: {mode}")
    _play(np.concatenate([_tone(base, 0.09), _tone(base * 1.5, 0.12)]))


def new_task() -> None:
    """A task just arrived on an empty/idle queue."""
    _play(np.concatenate([_tone(523.25, 0.07), _tone(659.25, 0.10)]))


class Thinking:
    """Periodic beep on a background thread while Claude is working."""

    def __init__(self, interval: float = 2.5) -> None:
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if _silent:
            # Don't bother spawning the periodic-beep thread when
            # earcons are muted — it would just wake up every interval
            # to call a no-op _play.
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                thinking()
            except EarconError:
                return
            if self._stop.wait(self._interval):
                return
