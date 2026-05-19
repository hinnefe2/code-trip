"""Short audio cues (earcons) via sounddevice + numpy."""

from __future__ import annotations

import threading

import numpy as np

try:
    import sounddevice as sd
except OSError:
    sd = None  # type: ignore[assignment]


class EarconError(Exception):
    pass


_SR = 44_100


def _tone(freq: float, duration: float, amplitude: float = 0.2) -> np.ndarray:
    t = np.linspace(0, duration, int(_SR * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    fade = int(_SR * 0.01)
    if fade * 2 < len(wave):
        env = np.ones_like(wave)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        wave *= env
    return wave.astype(np.float32)


def _play(samples: np.ndarray) -> None:
    try:
        sd.play(samples, samplerate=_SR)
        sd.wait()
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
