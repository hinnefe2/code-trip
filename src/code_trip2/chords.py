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

import asyncio
import logging
import re
from typing import TYPE_CHECKING

from pynput import keyboard

from code_trip2 import dispatch, earcon, modes, remote, window
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


async def handle_chord(ctx: "Context", name: str) -> None:
    if name == "nav+yes":
        await _nav(ctx, forward=True)
    elif name == "nav+no":
        await _nav(ctx, forward=False)
    elif name == "nav+act":
        await _cycle_app(ctx)
    elif name == "nav+ptt":
        await _speak_active_app(ctx)
    elif name == "act+no":
        # ACT+NO is mode-dependent: in queue mode it dismisses the
        # current task (the "permanent skip" the user reaches for when
        # NO-tap-as-defer keeps re-surfacing the same message). In
        # focused mode it stays as Ctrl+U for shell input.
        logger.info("Chord act+no fired in app_mode=%r", ctx.app_mode)
        if ctx.app_mode == dispatch.MODE_QUEUE:
            await dispatch.dismiss_current_task(ctx)
        else:
            await _send_stroke(ctx, _ACT_NO_CLEAR_LINE)
    elif name == "act+yes":
        # ACT+YES in queue mode opens the active task in a browser
        # when the task has an obvious URL (email_msg → Gmail thread).
        # No-op in focused mode and for task kinds without a natural
        # browser landing.
        if ctx.app_mode == dispatch.MODE_QUEUE:
            await _open_current_task_in_browser(ctx)
    else:
        logger.warning("Unknown chord: %s", name)


_GMAIL_THREAD_URL = "https://mail.google.com/mail/u/0/#all/{thread_id}"
_SLACK_APP = "Slack"

# Per-kind app target for ACT+YES open-in-app. Slack-hosted permalinks
# (https://workspace.slack.com/archives/…) get intercepted by the Slack
# desktop app when handed to ``open -a Slack``, jumping straight to
# the message in-context. Email + Linear stay in the browser.
_TASK_OPEN_APPS = {
    "email_msg": _CHROME_APP,
    "linear_issue": _CHROME_APP,
    "slack_msg": _SLACK_APP,
}


def _task_browser_url(task) -> str | None:
    """URL to open for this task, or ``None`` when there's no natural landing.

    ``email_msg`` builds a Gmail thread URL; ``linear_issue`` and
    ``slack_msg`` use the URL the producer captured in
    ``source["url"]`` (Linear's MCP response field; Slack's
    workspace-qualified permalink from the message's metadata).
    """
    src = task.source or {}
    if task.kind == "email_msg":
        thread_id = src.get("thread_id") or ""
        return _GMAIL_THREAD_URL.format(thread_id=thread_id) if thread_id else None
    if task.kind in ("linear_issue", "slack_msg"):
        url = src.get("url") or ""
        return url or None
    return None


async def _open_current_task_in_browser(ctx: "Context") -> None:
    """Open the active task's source URL in its natural app.

    Resolves the URL via :func:`_task_browser_url` and looks up the
    target app in :data:`_TASK_OPEN_APPS` (Slack permalinks go to the
    Slack desktop app; everything else goes to Chrome). Kinds without
    a natural URL fall through with a spoken hint so the chord doesn't
    silently appear broken.
    """
    task = ctx.current_task
    if task is None:
        await _speak_error(ctx, "Nothing active.")
        return
    url = _task_browser_url(task)
    if url is None:
        await _speak_error(ctx, f"Can't open {task.kind}.")
        return
    app = _TASK_OPEN_APPS.get(task.kind, _CHROME_APP)
    try:
        proc = await asyncio.create_subprocess_exec(
            "open", "-a", app, url,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode != 0:
            err = (stderr_b or b"").decode(errors="replace").strip()
            raise RuntimeError(err or f"open exited {proc.returncode}")
    except (asyncio.TimeoutError, OSError, RuntimeError) as exc:
        await _speak_error(ctx, f"Could not open in {app}: {exc}")
        return
    try:
        earcon.completion()
    except earcon.EarconError:
        pass


async def _send_stroke(ctx: "Context", stroke: KeyStroke) -> None:
    if await _keystroke_targets_tui_host(ctx):
        # The TUI's alternate-screen buffer would scroll on every Enter/Esc/Down.
        # Better to silently swallow than to corrupt the dashboard.
        logger.debug("Suppressing keystroke targeting TUI host app")
        return
    try:
        await window.send_keystroke(stroke)
    except Exception as exc:
        await _speak_error(ctx, f"Could not send keystroke: {exc}")


async def _keystroke_targets_tui_host(ctx: "Context") -> bool:
    """True when the frontmost app is the terminal hosting the TUI."""
    host = getattr(ctx, "tui_host_app", None)
    if not host:
        return False
    try:
        return await window.active_app() == host
    except window.WindowError:
        return False


async def handle_tap(ctx: "Context", name: str) -> None:
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
            await _act_tap_app_aware(ctx)
        return

    # Queue mode: YES/NO drive the queue rather than the focused app.
    if ctx.app_mode == dispatch.MODE_QUEUE:
        if name == "yes":
            await dispatch.queue_yes_tap(ctx)
            return
        if name == "no":
            await dispatch.queue_no_tap(ctx)
            return

    stroke = TAP_STROKES.get(name)
    if stroke is None:
        logger.warning("Unknown tap: %s", name)
        return
    await _send_stroke(ctx, stroke)


async def _act_tap_app_aware(ctx: "Context") -> bool:
    """Per-app ACT-tap behavior. Returns True if handled (else no-op)."""
    try:
        app = await window.active_app()
    except window.WindowError:
        return False
    if app in ctx.config.terminal_apps:
        await _open_last_pane_url(ctx)
        return True
    if app == _CHROME_APP:
        await _send_stroke(ctx, _CHROME_NEW_TAB)
        return True
    return False


async def _open_last_pane_url(ctx: "Context") -> None:
    """Capture the active tmux pane, find the most recent URL, open in Chrome."""
    host, opts = ctx.ssh
    try:
        raw = await remote.capture(
            host, opts, ctx.config.tmux_session, ctx.active_window, lines=200,
        )
    except remote.RemoteError as exc:
        await _speak_error(ctx, f"Could not read pane: {exc}")
        return
    text = _ANSI_RE.sub("", raw)
    matches = _URL_RE.findall(text)
    if not matches:
        await _speak_error(ctx, "No URL in pane.")
        return
    url = matches[-1].rstrip(_URL_TRIM)
    try:
        proc = await asyncio.create_subprocess_exec(
            "open", "-a", _CHROME_APP, url,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode != 0:
            err = (stderr_b or b"").decode(errors="replace").strip()
            raise RuntimeError(err or f"open exited {proc.returncode}")
    except (asyncio.TimeoutError, OSError, RuntimeError) as exc:
        await _speak_error(ctx, f"Could not open URL: {exc}")
        return
    try:
        earcon.completion()
    except earcon.EarconError:
        pass


async def _nav(ctx: "Context", forward: bool) -> None:
    try:
        app = await window.active_app()
    except window.WindowError as exc:
        await _speak_error(ctx, f"Could not read active app: {exc}")
        return
    pair = APP_NAV.get(app)
    if pair is None:
        await _speak_error(ctx, f"No navigation for {app}.")
        return
    stroke = pair[0] if forward else pair[1]
    await _send_stroke(ctx, stroke)


async def _cycle_app(ctx: "Context") -> None:
    apps = tuple(ctx.config.app_cycle)
    if not apps:
        return
    try:
        current = await window.active_app()
    except window.WindowError as exc:
        await _speak_error(ctx, f"Could not read active app: {exc}")
        return
    try:
        idx = apps.index(current)
        next_app = apps[(idx + 1) % len(apps)]
    except ValueError:
        next_app = apps[0]
    try:
        await window.activate_app(next_app)
    except window.WindowError as exc:
        await _speak_error(ctx, f"Could not activate {next_app}: {exc}")


async def _speak_active_app(ctx: "Context") -> None:
    try:
        app = await window.active_app()
    except window.WindowError as exc:
        await _speak_error(ctx, f"Could not read active app: {exc}")
        return
    await _speak(ctx, app)


async def _speak(ctx: "Context", text: str) -> None:
    if not text:
        return
    try:
        await ctx.tts.speak(text)
    except TTSClientError:
        logger.exception("TTS failed for: %s", text)


async def _speak_error(ctx: "Context", message: str) -> None:
    try:
        earcon.error()
    except earcon.EarconError:
        pass
    await _speak(ctx, message)
