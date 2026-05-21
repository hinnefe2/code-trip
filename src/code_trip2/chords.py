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
import re
import subprocess
from typing import TYPE_CHECKING

from pynput import keyboard

from code_trip2 import earcon, remote, window
from code_trip2.tts_client import TTSClientError
from code_trip2.window import Chord, KeyStroke

if TYPE_CHECKING:
    from code_trip2.modes import Context

logger = logging.getLogger(__name__)

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_URL_RE = re.compile(r"https?://[^\s\)\]\}\"\'>`<]+")
_URL_TRIM = ".,;:!?"
_CHROME_APP = "Google Chrome"


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


# Solo taps that synthesize a keystroke into the focused app. NAV solo
# tap is no longer in here — NAV now flips app-mode (see handle_tap).
# ACT solo tap is also handled inline (stop playback / per-app handler).
_TAP_YES = KeyStroke(chords=(Chord(key=keyboard.Key.enter),))
_TAP_NO = KeyStroke(chords=(Chord(key=keyboard.Key.esc),))

# Cmd+T = open new tab (focuses the URL bar by default in Chrome).
_CHROME_NEW_TAB = KeyStroke(chords=(Chord(modifiers=(keyboard.Key.cmd,), key="t"),))

# ACT+NO = Ctrl+U (clear line to start in shell / readline inputs).
_ACT_NO_CLEAR_LINE = KeyStroke(chords=(Chord(modifiers=(keyboard.Key.ctrl,), key="u"),))

TAP_STROKES: dict[str, KeyStroke] = {
    "yes": _TAP_YES,
    "no": _TAP_NO,
}


def handle_chord(ctx: "Context", name: str) -> None:
    from code_trip2 import dispatch  # local import to avoid cycle

    if name == "nav+yes":
        _nav(ctx, forward=True)
    elif name == "nav+no":
        _nav(ctx, forward=False)
    elif name == "nav+act":
        _cycle_app(ctx)
    elif name == "nav+ptt":
        _speak_active_app(ctx)
    elif name == "act+no":
        # ACT+NO is mode-dependent: in queue mode it dismisses the
        # current task (the "permanent skip" the user reaches for when
        # NO-tap-as-defer keeps re-surfacing the same message). In
        # focused mode it stays as Ctrl+U for shell input.
        if ctx.app_mode == dispatch.MODE_QUEUE:
            dispatch.dismiss_current_task(ctx)
        else:
            _send_stroke(ctx, _ACT_NO_CLEAR_LINE)
    else:
        logger.warning("Unknown chord: %s", name)


def _send_stroke(ctx: "Context", stroke: KeyStroke) -> None:
    if _keystroke_targets_tui_host(ctx):
        # The TUI's alternate-screen buffer would scroll on every Enter/Esc/Down.
        # Better to silently swallow than to corrupt the dashboard.
        logger.debug("Suppressing keystroke targeting TUI host app")
        return
    try:
        window.send_keystroke(stroke)
    except Exception as exc:
        _speak_error(ctx, f"Could not send keystroke: {exc}")


def _keystroke_targets_tui_host(ctx: "Context") -> bool:
    """True when the frontmost app is the terminal hosting the TUI."""
    host = getattr(ctx, "tui_host_app", None)
    if not host:
        return False
    try:
        return window.active_app() == host
    except window.WindowError:
        return False


def handle_tap(ctx: "Context", name: str) -> None:
    """Dispatch a macropad solo tap.

    Key bindings (post-revamp):

    - **NAV**: flip app-mode (queue ↔ focused). Mode-independent.
    - **ACT**: stop TTS playback if speaking; otherwise per-app handler
      (open URL in tmux, new tab in Chrome, etc).
    - **YES/NO** in queue mode: drive the queue (accept/expand, skip).
    - **YES/NO** in focused mode: synthesize Enter / Esc into the
      focused app.
    - **PTT**: handled at the macropad layer, not here.

    Playback chunks auto-advance through ``_playback_loop``, so there's
    no "next chunk" tap binding any more — ACT just stops playback
    outright.
    """
    from code_trip2 import dispatch, modes  # local import to avoid cycle

    # NAV solo: app-mode flip.
    if name == "nav":
        dispatch.flip_mode(ctx)
        return

    # ACT solo: interrupt playback if speaking. Otherwise, if we're in
    # focused mode, run the per-app handler (open URL / Cmd+T / etc).
    # In queue mode with no playback it's a silent no-op — the user is
    # away from the screen, so a focused-app keystroke would be wrong.
    if name == "act":
        if modes.is_playback_active(ctx):
            modes.stop_playback(ctx)
            return
        if ctx.app_mode == dispatch.MODE_FOCUSED:
            _act_tap_app_aware(ctx)
        return

    # Queue mode: YES/NO drive the queue rather than the focused app.
    if ctx.app_mode == dispatch.MODE_QUEUE:
        if name == "yes":
            dispatch.queue_yes_tap(ctx)
            return
        if name == "no":
            dispatch.queue_no_tap(ctx)
            return

    stroke = TAP_STROKES.get(name)
    if stroke is None:
        logger.warning("Unknown tap: %s", name)
        return
    _send_stroke(ctx, stroke)


def _act_tap_app_aware(ctx: "Context") -> bool:
    """Per-app ACT-tap behavior. Returns True if handled (else no-op)."""
    try:
        app = window.active_app()
    except window.WindowError:
        return False
    if app in ctx.config.terminal_apps:
        _open_last_pane_url(ctx)
        return True
    if app == _CHROME_APP:
        _send_stroke(ctx, _CHROME_NEW_TAB)
        return True
    return False


def _open_last_pane_url(ctx: "Context") -> None:
    """Capture the active tmux pane, find the most recent URL, open in Chrome."""
    host, opts = ctx.ssh
    try:
        raw = remote.capture(
            host, opts, ctx.config.tmux_session, ctx.active_window, lines=200
        )
    except remote.RemoteError as exc:
        _speak_error(ctx, f"Could not read pane: {exc}")
        return
    text = _ANSI_RE.sub("", raw)
    matches = _URL_RE.findall(text)
    if not matches:
        _speak_error(ctx, "No URL in pane.")
        return
    url = matches[-1].rstrip(_URL_TRIM)
    try:
        subprocess.run(
            ["open", "-a", _CHROME_APP, url],
            check=True,
            capture_output=True,
            timeout=5.0,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        _speak_error(ctx, f"Could not open URL: {exc}")
        return
    try:
        earcon.completion()
    except earcon.EarconError:
        pass


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
    _send_stroke(ctx, stroke)


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
