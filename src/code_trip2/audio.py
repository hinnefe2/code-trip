"""Push-to-talk: mic recorder + hotkey listener in one class."""

from __future__ import annotations

import logging
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from pynput import keyboard

try:
    import sounddevice as sd
except OSError:
    sd = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("/tmp/code-trip-audio")


class AudioError(Exception):
    pass


@dataclass
class PushToTalk:
    """Hold hotkey → record; release → save WAV → invoke callback(path).

    Callbacks run on the pynput listener thread; keep them fast or hand
    off to another thread.
    """

    hotkey: keyboard.Key | keyboard.KeyCode
    on_audio: Callable[[Path], None]
    sample_rate: int = 16_000
    device: int | str | None = None
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    _stream: "sd.InputStream | None" = field(default=None, init=False, repr=False)
    _frames: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _listener: keyboard.Listener | None = field(default=None, init=False, repr=False)

    def start(self) -> None:
        if self._listener is not None:
            raise AudioError("Already listening")
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                self._recording = False

    # --- key callbacks ----------------------------------------------------

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key != self.hotkey:
            return
        with self._lock:
            if self._recording:
                return
            self._frames = []
            try:
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="int16",
                    device=self.device,
                    callback=self._audio_cb,
                )
                self._stream.start()
                self._recording = True
            except Exception:
                logger.exception("Failed to start mic")
                self._stream = None

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key != self.hotkey:
            return
        with self._lock:
            if not self._recording or self._stream is None:
                return
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._recording = False
            frames = self._frames
            self._frames = []

        audio = np.concatenate(frames) if frames else np.empty(0, dtype="int16")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"rec_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())

        try:
            self.on_audio(path)
        except Exception:
            logger.exception("on_audio callback failed")

    def _audio_cb(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        self._frames.append(indata.copy())


def resolve_hotkey(name: str) -> keyboard.Key | keyboard.KeyCode:
    try:
        return getattr(keyboard.Key, name)
    except AttributeError as exc:
        raise AudioError(f"Unknown hotkey: {name}") from exc
