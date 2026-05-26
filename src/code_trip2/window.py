"""macOS helpers: frontmost app, app activation, synthesized keystrokes.

Shells out to ``osascript`` (async) for app detection/activation and
uses ``pynput.keyboard.Controller`` to synthesize key combos.
Everything here is macOS-only; on other platforms the functions raise
WindowError.
"""

from __future__ import annotations

import asyncio
import logging
import sys
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


async def _run_capture(
    argv: list[str], *, input_bytes: bytes | None = None, timeout: float = _OSASCRIPT_TIMEOUT,
) -> str:
    """Run a short subprocess and return its stdout as text.

    Raises WindowError on failure (spawn error, timeout, non-zero exit).
    Helper used by every osascript / pbcopy call in this module.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if input_bytes is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, FileNotFoundError) as exc:
        raise WindowError(f"Could not spawn {argv[0]}: {exc}") from exc
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input_bytes), timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        try:
            await proc.wait()
        except Exception:
            pass
        raise WindowError(f"{argv[0]} timed out") from exc
    if proc.returncode != 0:
        stderr = (stderr_b or b"").decode(errors="replace").strip()
        raise WindowError(f"{argv[0]} exited {proc.returncode}: {stderr}")
    return (stdout_b or b"").decode(errors="replace")


async def active_app() -> str:
    if sys.platform != "darwin":
        raise WindowError("active_app is macOS-only")
    script = (
        'tell application "System Events" to return name of first '
        "process whose frontmost is true"
    )
    out = await _run_capture(["osascript", "-e", script])
    return out.strip()


async def activate_app(name: str) -> None:
    if sys.platform != "darwin":
        raise WindowError("activate_app is macOS-only")
    script = f'tell application "{name}" to activate'
    await _run_capture(["osascript", "-e", script])


_controller: keyboard.Controller | None = None


def _get_controller() -> keyboard.Controller:
    global _controller
    if _controller is None:
        _controller = keyboard.Controller()
    return _controller


async def send_keystroke(stroke: KeyStroke) -> None:
    controller = _get_controller()
    for i, chord in enumerate(stroke.chords):
        if i > 0:
            await asyncio.sleep(stroke.delay)
        _send_chord(controller, chord)


def _send_chord(controller: keyboard.Controller, chord: Chord) -> None:
    """Sync — pynput press/tap/release are instant Quartz calls."""
    for m in chord.modifiers:
        controller.press(m)
    try:
        if chord.key:
            controller.tap(chord.key)
    finally:
        for m in reversed(chord.modifiers):
            controller.release(m)


_PASTE_STROKE = KeyStroke(chords=(Chord(modifiers=(keyboard.Key.cmd,), key="v"),))


async def paste_text(text: str) -> None:
    """Copy text to clipboard, then Cmd+V into the frontmost app (macOS)."""
    if sys.platform != "darwin":
        raise WindowError("paste_text is macOS-only")
    if not text:
        return
    await _run_capture(["pbcopy"], input_bytes=text.encode("utf-8"))
    await asyncio.sleep(0.05)
    await send_keystroke(_PASTE_STROKE)
