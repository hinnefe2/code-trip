"""Macropad listener: PTT recording + NAV-modifier chord dispatch.

Replaces the old single-key ``PushToTalk``. One pynput listener watches
all five logical keys (PTT, ACT, YES, NO, NAV) and tracks which are
currently held. Behavior:

  - PTT pressed alone       → start mic stream.
  - PTT released            → close stream, write WAV, call ``on_audio``.
  - NAV held + PTT tap      → ``on_chord("nav+ptt")`` (no recording).
  - NAV held + YES/NO/ACT   → ``on_chord("nav+yes" | "nav+no" | "nav+act")``.
  - YES or NO tapped alone  → ``on_tap("yes" | "no")``.
  - Everything else         → no-op.

Callbacks run on the pynput listener thread; keep them fast or hand off
to another thread.
"""

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

LOGICAL_KEYS = ("ptt", "act", "yes", "no", "nav")


class MacropadError(Exception):
    pass


def resolve_key(name: str) -> keyboard.Key | keyboard.KeyCode:
    try:
        return getattr(keyboard.Key, name)
    except AttributeError as exc:
        raise MacropadError(f"Unknown key: {name}") from exc


@dataclass
class Macropad:
    keymap: dict[str, keyboard.Key | keyboard.KeyCode]
    on_audio: Callable[[Path], None]
    on_chord: Callable[[str], None]
    on_tap: Callable[[str], None] | None = None
    sample_rate: int = 16_000
    device: int | str | None = None
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    _stream: "sd.InputStream | None" = field(default=None, init=False, repr=False)
    _frames: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _listener: keyboard.Listener | None = field(default=None, init=False, repr=False)
    _reverse: dict[object, str] = field(default_factory=dict, init=False, repr=False)
    _held: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        missing = set(LOGICAL_KEYS) - set(self.keymap)
        if missing:
            raise MacropadError(f"Missing keymap entries: {sorted(missing)}")
        self._reverse = {pk: name for name, pk in self.keymap.items()}

    def start(self) -> None:
        if self._listener is not None:
            raise MacropadError("Already listening")
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        with self._lock:
            self._stop_stream_locked()
        self._held.clear()

    # --- key callbacks ----------------------------------------------------

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        name = self._reverse.get(key)
        if name is None:
            return
        if name in self._held:
            return
        self._held.add(name)

        nav_modifier = "nav" in self._held and name != "nav"
        if name == "ptt":
            if nav_modifier:
                self._fire_chord("nav+ptt")
            else:
                self._start_recording()
        elif name in ("yes", "no", "act") and nav_modifier:
            self._fire_chord(f"nav+{name}")
        elif name in ("yes", "no") and self.on_tap is not None:
            self._fire_tap(name)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        name = self._reverse.get(key)
        if name is None:
            return
        self._held.discard(name)
        if name == "ptt":
            self._finish_recording()

    # --- audio ------------------------------------------------------------

    def _start_recording(self) -> None:
        with self._lock:
            if self._recording:
                return
            if sd is None:
                logger.warning("sounddevice unavailable; PTT ignored")
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

    def _finish_recording(self) -> None:
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

    def _stop_stream_locked(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                logger.exception("Failed to close mic stream")
            self._stream = None
            self._recording = False

    def _audio_cb(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        self._frames.append(indata.copy())

    # --- chord ------------------------------------------------------------

    def _fire_chord(self, name: str) -> None:
        try:
            self.on_chord(name)
        except Exception:
            logger.exception("on_chord(%s) failed", name)

    def _fire_tap(self, name: str) -> None:
        try:
            assert self.on_tap is not None
            self.on_tap(name)
        except Exception:
            logger.exception("on_tap(%s) failed", name)
