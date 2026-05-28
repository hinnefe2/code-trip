"""Macropad listener: PTT recording + NAV-modifier chord dispatch.

Replaces the old single-key ``PushToTalk``. One pynput listener watches
all five logical keys (PTT, ACT, YES, NO, NAV) and tracks which are
currently held. Behavior:

  - PTT pressed alone       → start mic stream (or press ``ptt_forward_key``
                              if set, letting a local STT like Superwhisper
                              handle the recording).
  - PTT released            → close stream + call ``on_audio`` (or release
                              ``ptt_forward_key``).
  - NAV held + PTT tap      → ``on_chord("nav+ptt")`` (no recording/forward).
  - NAV held + YES/NO/ACT   → ``on_chord("nav+yes" | "nav+no" | "nav+act")``.
  - NAV tapped alone        → ``on_tap("nav")`` (fires on release if NAV was
                              not used as a modifier).
  - ACT held + NO           → ``on_chord("act+no")`` (clear-line Ctrl+U).
  - YES or NO tapped alone  → ``on_tap("yes" | "no")``.
  - Everything else         → no-op.

  NAV takes precedence over ACT when both are held.

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

try:
    from Quartz import CGEventGetIntegerValueField, kCGKeyboardEventKeycode
except ImportError:  # non-darwin or Quartz unavailable
    CGEventGetIntegerValueField = None
    kCGKeyboardEventKeycode = None

logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_DIR = Path("/tmp/code-trip-audio")

LOGICAL_KEYS = ("ptt", "yes", "no", "act", "nav")

MIN_RECORDING_SECONDS = 0.2


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
    on_audio: Callable[..., None]  # (path, *, skill_mode: bool)
    on_chord: Callable[[str], None]
    on_tap: Callable[[str], None] | None = None
    on_ptt_press: Callable[[], None] | None = None
    # Fires on PTT release when the forward-key path was used (local STT
    # mode — no audio captured here, the external STT tool will paste
    # its transcript into the focused app). The orchestrator wires this
    # to a stdin paste reader so it knows whether the next pasted
    # transcript should route to handle_skill vs handle_voice.
    on_ptt_release: Callable[[bool], None] | None = None
    ptt_forward_key: keyboard.Key | keyboard.KeyCode | None = None
    sample_rate: int = 16_000
    device: int | str | None = None
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    _stream: "sd.InputStream | None" = field(default=None, init=False, repr=False)
    _frames: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _recording: bool = field(default=False, init=False, repr=False)
    _forwarding: bool = field(default=False, init=False, repr=False)
    _controller: keyboard.Controller | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _listener: keyboard.Listener | None = field(default=None, init=False, repr=False)
    _reverse: dict[object, str] = field(default_factory=dict, init=False, repr=False)
    _held: set[str] = field(default_factory=set, init=False, repr=False)
    _nav_chorded: bool = field(default=False, init=False, repr=False)
    _act_chorded: bool = field(default=False, init=False, repr=False)
    # ACT+PTT (ACT held when PTT pressed) flags the current recording as
    # "skill mode" — the transcript gets routed to a free-form Claude
    # invocation against the active task instead of the per-kind reply
    # path. Cleared each time recording finishes.
    _skill_mode: bool = field(default=False, init=False, repr=False)
    _suppress_vks: set[int] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        missing = set(LOGICAL_KEYS) - set(self.keymap)
        if missing:
            raise MacropadError(f"Missing keymap entries: {sorted(missing)}")
        self._reverse = {pk: name for name, pk in self.keymap.items()}
        for k in self.keymap.values():
            val = getattr(k, "value", k)
            vk = getattr(val, "vk", None)
            if vk is not None:
                self._suppress_vks.add(vk)

    def start(self) -> None:
        if self._listener is not None:
            raise MacropadError("Already listening")
        kwargs: dict = {"on_press": self._on_press, "on_release": self._on_release}
        if CGEventGetIntegerValueField is not None:
            kwargs["darwin_intercept"] = self._darwin_intercept
        self._listener = keyboard.Listener(**kwargs)
        self._listener.daemon = True
        self._listener.start()

    def _darwin_intercept(self, event_type: object, event: object) -> object | None:
        # Drop macropad keys at the OS event tap so the focused app never sees
        # them (kitty's CSI-u protocol would otherwise spam escape sequences
        # while NAV/PTT is held).
        try:
            vk = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
        except Exception:
            return event
        return None if vk in self._suppress_vks else event

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
        with self._lock:
            self._stop_stream_locked()
        self._release_forward_if_held()
        self._held.clear()

    # --- key callbacks ----------------------------------------------------

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        name = self._reverse.get(key)
        if name is None:
            return
        if name in self._held:
            return
        self._held.add(name)

        # A modifier key is "chorded" if it participated in a chord at all
        # — either by being held when another key was pressed, OR by being
        # the secondary key pressed while another modifier was held.
        # Without the second case, NAV+ACT would fire "nav+act" on press
        # AND fire "act" solo on release, because ACT's chord state would
        # have been reset to False when ACT was pressed.
        other_keys_held = bool(self._held - {name})
        if name == "nav":
            self._nav_chorded = other_keys_held
        elif "nav" in self._held:
            self._nav_chorded = True

        if name == "act":
            self._act_chorded = other_keys_held
        elif "act" in self._held:
            self._act_chorded = True

        nav_modifier = "nav" in self._held and name != "nav"
        act_modifier = "act" in self._held and name != "act"
        if name == "ptt":
            if nav_modifier:
                self._fire_chord("nav+ptt")
            else:
                # ACT held during PTT = skill mode: the transcript
                # routes to a free-form Claude call instead of the
                # source reply path. The audio capture path is decided
                # by STT config (openai-capture vs key-forwarded local
                # STT) — skill mode only affects dispatch, not capture.
                self._skill_mode = "act" in self._held
                if self.on_ptt_press is not None:
                    try:
                        self.on_ptt_press()
                    except Exception:
                        logger.exception("on_ptt_press failed")
                if self.ptt_forward_key is not None:
                    self._press_forward()
                else:
                    self._start_recording()
        elif name in ("yes", "no", "act") and nav_modifier:
            self._fire_chord(f"nav+{name}")
        elif name in ("yes", "no") and act_modifier:
            self._fire_chord(f"act+{name}")
        elif name in ("yes", "no") and self.on_tap is not None:
            self._fire_tap(name)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        name = self._reverse.get(key)
        if name is None:
            return
        self._held.discard(name)
        if name == "ptt":
            if self._forwarding:
                # Capture-and-clear ``_skill_mode`` here too (parallel to
                # ``_finish_recording``) so the next PTT cycle starts
                # clean. The flag rides along to ``on_ptt_release`` so
                # the orchestrator can match the upcoming pasted
                # transcript to the right dispatch path.
                skill_mode = self._skill_mode
                self._skill_mode = False
                self._release_forward_if_held()
                logger.info("PTT released in forward mode (skill_mode=%s)", skill_mode)
                if self.on_ptt_release is not None:
                    try:
                        self.on_ptt_release(skill_mode)
                    except Exception:
                        logger.exception("on_ptt_release failed")
            else:
                self._finish_recording()
        elif name == "nav":
            if not self._nav_chorded and self.on_tap is not None:
                self._fire_tap("nav")
        elif name == "act":
            if not self._act_chorded and self.on_tap is not None:
                self._fire_tap("act")

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
            skill_mode = self._skill_mode
            self._skill_mode = False

        audio = np.concatenate(frames) if frames else np.empty(0, dtype="int16")
        min_samples = int(MIN_RECORDING_SECONDS * self.sample_rate)
        if audio.size < min_samples:
            logger.warning(
                "PTT recording too short (%.3fs); ignoring. Hold PTT for at least %.2fs.",
                audio.size / self.sample_rate,
                MIN_RECORDING_SECONDS,
            )
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"rec_{time.strftime('%Y%m%d-%H%M%S')}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())
        try:
            self.on_audio(path, skill_mode=skill_mode)
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

    # --- forward-key (local STT) -----------------------------------------

    def _get_controller(self) -> keyboard.Controller:
        if self._controller is None:
            self._controller = keyboard.Controller()
        return self._controller

    def _press_forward(self) -> None:
        if self._forwarding or self.ptt_forward_key is None:
            return
        try:
            self._get_controller().press(self.ptt_forward_key)
            self._forwarding = True
        except Exception:
            logger.exception("Failed to press forward key")

    def _release_forward_if_held(self) -> None:
        if not self._forwarding or self.ptt_forward_key is None:
            return
        try:
            self._get_controller().release(self.ptt_forward_key)
        except Exception:
            logger.exception("Failed to release forward key")
        finally:
            self._forwarding = False

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
