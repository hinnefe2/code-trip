"""Chord dispatch: NAV-modifier combos trigger per-app actions.

Four chords, all triggered by tapping a key while NAV is held:

  - ``nav+yes`` — forward in the focused app's natural unit
  - ``nav+no``  — backward in the focused app's natural unit
  - ``nav+act`` — rotate to the next app in ``config.app_cycle``
  - ``nav+ptt`` — speak the frontmost app's name via TTS

"Natural unit" per app is defined in ``APP_NAV`` as a (yes, no) pair of
:class:`KeyStroke` values. YES and NO do not have to be symmetric — Slack
deliberately pairs "next unread" with "history back".
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pynput import keyboard

from code_trip2 import earcon, window
from code_trip2.tts_client import TTSClientError
from code_trip2.window import Chord, KeyStroke

if TYPE_CHECKING:
    from code_trip2.modes import Context

logger = logging.getLogger(__name__)


_KITTY_NEXT = KeyStroke(
    chords=(
        Chord(modifiers=(keyboard.Key.ctrl,), key="b"),
        Chord(key="n"),
    )
)
_KITTY_PREV = KeyStroke(
    chords=(
        Chord(modifiers=(keyboard.Key.ctrl,), key="b"),
        Chord(key="p"),
    )
)

_CHROME_NEXT_TAB = KeyStroke(
    chords=(Chord(modifiers=(keyboard.Key.cmd, keyboard.Key.alt), key=keyboard.Key.right),)
)
_CHROME_PREV_TAB = KeyStroke(
    chords=(Chord(modifiers=(keyboard.Key.cmd, keyboard.Key.alt), key=keyboard.Key.left),)
)

_SLACK_NEXT_UNREAD = KeyStroke(
    chords=(Chord(modifiers=(keyboard.Key.alt, keyboard.Key.shift), key=keyboard.Key.down),)
)
_SLACK_BACK_HISTORY = KeyStroke(
    chords=(Chord(modifiers=(keyboard.Key.cmd,), key="["),)
)


APP_NAV: dict[str, tuple[KeyStroke, KeyStroke]] = {
    "kitty": (_KITTY_NEXT, _KITTY_PREV),
    "Google Chrome": (_CHROME_NEXT_TAB, _CHROME_PREV_TAB),
    "Slack": (_SLACK_NEXT_UNREAD, _SLACK_BACK_HISTORY),
}


# Solo taps — typed into whatever app currently has focus.
# YES = Enter (default-accept in most prompts); NO = Esc (cancel).
_TAP_YES = KeyStroke(chords=(Chord(key=keyboard.Key.enter),))
_TAP_NO = KeyStroke(chords=(Chord(key=keyboard.Key.esc),))

# ACT+NO = Ctrl+U (clear line to start in shell / readline inputs).
_ACT_NO_CLEAR_LINE = KeyStroke(chords=(Chord(modifiers=(keyboard.Key.ctrl,), key="u"),))

TAP_STROKES: dict[str, KeyStroke] = {
    "yes": _TAP_YES,
    "no": _TAP_NO,
}


def handle_chord(ctx: "Context", name: str) -> None:
    if name == "nav+yes":
        _nav(ctx, forward=True)
    elif name == "nav+no":
        _nav(ctx, forward=False)
    elif name == "nav+act":
        _cycle_app(ctx)
    elif name == "nav+ptt":
        _speak_active_app(ctx)
    elif name == "act+no":
        _send_stroke(ctx, _ACT_NO_CLEAR_LINE)
    else:
        logger.warning("Unknown chord: %s", name)


def _send_stroke(ctx: "Context", stroke: KeyStroke) -> None:
    try:
        window.send_keystroke(stroke)
    except Exception as exc:
        _speak_error(ctx, f"Could not send keystroke: {exc}")


def handle_tap(ctx: "Context", name: str) -> None:
    stroke = TAP_STROKES.get(name)
    if stroke is None:
        logger.warning("Unknown tap: %s", name)
        return
    try:
        window.send_keystroke(stroke)
    except Exception as exc:
        _speak_error(ctx, f"Could not send keystroke: {exc}")


def _nav(ctx: "Context", forward: bool) -> None:
    try:
        app = window.active_app()
    except window.WindowError as exc:
        _speak_error(ctx, f"Could not read active app: {exc}")
        return
    pair = APP_NAV.get(app)
    if pair is None:
        _speak_error(ctx, f"No navigation for {app}.")
        return
    stroke = pair[0] if forward else pair[1]
    try:
        window.send_keystroke(stroke)
    except Exception as exc:
        _speak_error(ctx, f"Could not send keystroke: {exc}")


def _cycle_app(ctx: "Context") -> None:
    apps = tuple(ctx.config.app_cycle)
    if not apps:
        return
    try:
        current = window.active_app()
    except window.WindowError as exc:
        _speak_error(ctx, f"Could not read active app: {exc}")
        return
    try:
        idx = apps.index(current)
        next_app = apps[(idx + 1) % len(apps)]
    except ValueError:
        next_app = apps[0]
    try:
        window.activate_app(next_app)
    except window.WindowError as exc:
        _speak_error(ctx, f"Could not activate {next_app}: {exc}")


def _speak_active_app(ctx: "Context") -> None:
    try:
        app = window.active_app()
    except window.WindowError as exc:
        _speak_error(ctx, f"Could not read active app: {exc}")
        return
    _speak(ctx, app)


def _speak(ctx: "Context", text: str) -> None:
    if not text:
        return
    try:
        ctx.tts.speak(text)
    except TTSClientError:
        logger.exception("TTS failed for: %s", text)


def _speak_error(ctx: "Context", message: str) -> None:
    try:
        earcon.error()
    except earcon.EarconError:
        pass
    _speak(ctx, message)
