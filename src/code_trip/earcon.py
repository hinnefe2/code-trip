"""Short audio cues (earcons) using sounddevice + numpy.

Cross-platform (Linux + macOS); no OS-specific binaries.
"""

from __future__ import annotations

import threading

import numpy as np

try:
    import sounddevice as sd
except OSError:  # pragma: no cover - PortAudio missing; tests mock sd
    sd = None  # type: ignore[assignment]


class EarconError(Exception):
    """Raised when earcon playback fails."""


_SAMPLE_RATE = 44_100


def _tone(freq: float, duration: float, amplitude: float = 0.2) -> np.ndarray:
    t = np.linspace(0, duration, int(_SAMPLE_RATE * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    # Short fade in/out to avoid clicks.
    fade = int(_SAMPLE_RATE * 0.01)
    if fade * 2 < len(wave):
        env = np.ones_like(wave)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        wave = wave * env
    return wave.astype(np.float32)


def play_tone(freq: float, duration: float, *, blocking: bool = True) -> None:
    try:
        sd.play(_tone(freq, duration), samplerate=_SAMPLE_RATE)
        if blocking:
            sd.wait()
    except Exception as exc:  # pragma: no cover - device-dependent
        raise EarconError(f"Failed to play tone: {exc}") from exc


def play_thinking_beep() -> None:
    play_tone(660.0, 0.08)


def play_completion() -> None:
    try:
        sd.play(
            np.concatenate([_tone(660.0, 0.08), _tone(880.0, 0.12)]),
            samplerate=_SAMPLE_RATE,
        )
        sd.wait()
    except Exception as exc:  # pragma: no cover
        raise EarconError(f"Failed to play completion: {exc}") from exc


def play_error() -> None:
    play_tone(220.0, 0.25)


MODE_CHIME_FREQS: dict[str, float] = {
    "IDLE": 440.0,
    "BROWSE": 523.25,
    "WORK": 587.33,
    "REVIEW": 659.25,
    "SHIP": 783.99,
}


def play_mode_chime(mode_name: str) -> None:
    """Play a distinct two-tone chime for ``mode_name`` (e.g. ``"WORK"``)."""
    try:
        base = MODE_CHIME_FREQS[mode_name]
    except KeyError as exc:
        raise EarconError(f"Unknown mode chime: {mode_name}") from exc
    try:
        sd.play(
            np.concatenate([_tone(base, 0.09), _tone(base * 1.5, 0.12)]),
            samplerate=_SAMPLE_RATE,
        )
        sd.wait()
    except Exception as exc:  # pragma: no cover - device-dependent
        raise EarconError(f"Failed to play mode chime: {exc}") from exc


class ThinkingEarcon:
    """Periodic beep played on a background thread while Claude is working."""

    def __init__(self, interval: float = 2.5) -> None:
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                play_thinking_beep()
            except EarconError:
                return
            if self._stop_event.wait(self._interval):
                return
