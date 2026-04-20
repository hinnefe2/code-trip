"""macOS helpers: frontmost app, app activation, synthesized keystrokes.

Shells out to ``osascript`` for app detection/activation and uses
``pynput.keyboard.Controller`` to synthesize key combos. Everything here
is macOS-only; on other platforms the functions raise WindowError.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass

from pynput import keyboard

logger = logging.getLogger(__name__)


class WindowError(Exception):
    pass


@dataclass(frozen=True)
class Chord:
    """A single modifier+key keystroke."""

    modifiers: tuple[keyboard.Key, ...] = ()
    key: str | keyboard.Key = ""


@dataclass(frozen=True)
class KeyStroke:
    """One or more chords sent back-to-back with a short delay between them."""

    chords: tuple[Chord, ...]
    delay: float = 0.05


_OSASCRIPT_TIMEOUT = 2.0


def active_app() -> str:
    if sys.platform != "darwin":
        raise WindowError("active_app is macOS-only")
    script = (
        'tell application "System Events" to return name of first '
        "process whose frontmost is true"
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=_OSASCRIPT_TIMEOUT,
            check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise WindowError(f"Could not read frontmost app: {exc}") from exc
    return result.stdout.strip()


def activate_app(name: str) -> None:
    if sys.platform != "darwin":
        raise WindowError("activate_app is macOS-only")
    script = f'tell application "{name}" to activate'
    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=_OSASCRIPT_TIMEOUT,
            check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise WindowError(f"Could not activate {name}: {exc}") from exc


_controller: keyboard.Controller | None = None


def _get_controller() -> keyboard.Controller:
    global _controller
    if _controller is None:
        _controller = keyboard.Controller()
    return _controller


def send_keystroke(stroke: KeyStroke) -> None:
    controller = _get_controller()
    for i, chord in enumerate(stroke.chords):
        if i > 0:
            time.sleep(stroke.delay)
        _send_chord(controller, chord)


def _send_chord(controller: keyboard.Controller, chord: Chord) -> None:
    for m in chord.modifiers:
        controller.press(m)
    try:
        if chord.key:
            controller.tap(chord.key)
    finally:
        for m in reversed(chord.modifiers):
            controller.release(m)


_PASTE_STROKE = KeyStroke(chords=(Chord(modifiers=(keyboard.Key.cmd,), key="v"),))


def paste_text(text: str) -> None:
    """Copy text to clipboard, then Cmd+V into the frontmost app (macOS)."""
    if sys.platform != "darwin":
        raise WindowError("paste_text is macOS-only")
    if not text:
        return
    try:
        subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=True,
            timeout=_OSASCRIPT_TIMEOUT,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        raise WindowError(f"Could not copy to clipboard: {exc}") from exc
    time.sleep(0.05)
    send_keystroke(_PASTE_STROKE)
