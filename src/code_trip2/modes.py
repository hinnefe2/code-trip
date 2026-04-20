"""Modes and voice dispatch.

Modes reflect the *surface being interacted with*, not workflow phase:

- IDLE        — no mode active; PTT routes by keyword (mode changes, status)
- DICTATE     — paste transcript into the frontmost macOS app
- NAVIGATING  — choosing tmux windows/panes
- WORK        — voice ↔ remote Claude in the active pane
- LINEAR      — ticket list via a dedicated pane running `claude -p ...`
- SLACK       — stubbed

Only one entry point matters today: ``handle_voice(ctx, transcript)``.
Keyed gestures (NAV/OK/NO/ACT) will be added when the macro pad is ready.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from code_trip2 import earcon, remote, window
from code_trip2.config import Config
from code_trip2.session_log import SessionLogger
from code_trip2.tts_client import TTSClient, TTSClientError

logger = logging.getLogger(__name__)


# --- Context ---------------------------------------------------------------

MODES = ("IDLE", "DICTATE", "NAVIGATING", "WORK", "LINEAR", "SLACK")


@dataclass
class Context:
    config: Config
    tts: TTSClient
    log: SessionLogger
    thinking: earcon.Thinking
    mode: str = ""
    # WORK state
    active_window: str = ""
    # LINEAR state
    tickets: list[dict] = field(default_factory=list)
    ticket_index: int = 0

    def __post_init__(self) -> None:
        if not self.mode:
            self.mode = self.config.default_mode
        if not self.active_window:
            self.active_window = self.config.work_window

    @property
    def ssh(self) -> tuple[str, tuple[str, ...]]:
        return self.config.ssh_host, self.config.ssh_options


# --- top-level dispatch ----------------------------------------------------


def handle_voice(ctx: Context, transcript: str) -> None:
    """Route a PTT transcript based on the current mode."""
    t = transcript.strip()
    if not t:
        return

    if _try_mode_switch(ctx, t):
        return
    if _try_global_commands(ctx, t):
        return

    if ctx.mode == "WORK":
        _work_voice(ctx, t)
    elif ctx.mode == "DICTATE":
        _dictate_voice(ctx, t)
    elif ctx.mode == "NAVIGATING":
        _nav_voice(ctx, t)
    elif ctx.mode == "LINEAR":
        _linear_voice(ctx, t)
    elif ctx.mode == "SLACK":
        _speak(ctx, "Slack mode is not yet implemented.")
    else:
        # IDLE: no mode → assume WORK for this turn
        _enter_mode(ctx, "WORK")
        _work_voice(ctx, t)


# --- mode switching --------------------------------------------------------

_MODE_PHRASES: dict[str, tuple[str, ...]] = {
    "WORK": ("work mode", "switch to work", "start working"),
    "DICTATE": ("dictation mode", "dictate mode", "switch to dictate"),
    "NAVIGATING": ("navigation mode", "switch windows"),
    "LINEAR": ("linear mode", "list tickets", "show tickets", "my tickets"),
    "SLACK": ("slack mode", "check slack", "read slack"),
    "IDLE": ("idle mode", "go idle", "exit mode"),
}


def _try_mode_switch(ctx: Context, t: str) -> bool:
    low = t.lower()
    for mode, phrases in _MODE_PHRASES.items():
        if any(p in low for p in phrases):
            _enter_mode(ctx, mode)
            # "list tickets" should also refresh on entry
            if mode == "LINEAR" and "ticket" in low:
                _linear_refresh(ctx)
            return True
    return False


def _enter_mode(ctx: Context, mode: str) -> None:
    if mode not in MODES:
        return
    if ctx.mode == mode:
        return
    prev, ctx.mode = ctx.mode, mode
    ctx.log.event("mode", **{"from": prev, "to": mode})
    try:
        earcon.mode_chime(mode)
    except earcon.EarconError:
        logger.exception("Mode chime failed")
    _speak(ctx, f"{mode.lower()} mode")


# --- global commands -------------------------------------------------------


def _try_global_commands(ctx: Context, t: str) -> bool:
    low = t.lower()
    if low in ("what", "repeat", "say that again"):
        # No history buffer yet; tell user.
        _speak(ctx, "Nothing to repeat.")
        return True
    if low.startswith("status"):
        _speak(ctx, f"{ctx.mode.lower()} mode. Active window: {ctx.active_window}.")
        return True
    return False


# --- DICTATE ---------------------------------------------------------------


def _dictate_voice(ctx: Context, t: str) -> None:
    try:
        window.paste_text(t)
    except window.WindowError as exc:
        _report_error(ctx, f"Could not paste: {exc}")
        return
    ctx.log.event("turn", mode="DICTATE", transcript=t)


# --- WORK ------------------------------------------------------------------


def _work_voice(ctx: Context, t: str) -> None:
    host, opts = ctx.ssh
    win = ctx.active_window
    cfg = ctx.config
    try:
        remote.send(host, opts, cfg.tmux_session, win, t)
    except remote.RemoteError as exc:
        _report_error(ctx, f"Could not reach Claude: {exc}")
        return

    ctx.thinking.start()
    raw: str | None = None
    try:
        try:
            remote.wait_done(host, opts, win, timeout=cfg.wait_timeout)
        except remote.WaitTimeout:
            _report_error(ctx, "Claude did not respond in time.")
            return
        except remote.RemoteError as exc:
            _report_error(ctx, f"Lost connection to Claude: {exc}")
            return
        try:
            raw = remote.capture(host, opts, cfg.tmux_session, win, lines=200)
        except remote.RemoteError as exc:
            _report_error(ctx, f"Could not read Claude's response: {exc}")
            return
    finally:
        ctx.thinking.stop()

    spoken = clean_output(raw or "")
    _speak(ctx, spoken or "No output.")
    try:
        earcon.completion()
    except earcon.EarconError:
        pass
    ctx.log.event("turn", mode="WORK", user=t, remote_output=raw, spoken=spoken)


# Strip ANSI + box-drawing, collapse whitespace, truncate code blocks.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_BOX_RE = re.compile(r"[╭╮╰╯│─┌┐└┘├┤┬┴┼━┃┏┓┗┛┣┫┳┻╋]")


def clean_output(raw: str) -> str:
    """Minimal Python-side cleanup of Claude's terminal output.

    Goal: extract the most recent assistant message from the captured
    pane. Claude Code prints output above its input prompt (the trailing
    `>` line). We strip ANSI / box-drawing chars, drop the input prompt
    block, and return the last non-empty segment.
    """
    if not raw:
        return ""
    s = _ANSI_RE.sub("", raw)
    s = _BOX_RE.sub("", s)
    # Drop trailing "> " prompt lines
    lines = [ln.rstrip() for ln in s.splitlines()]
    while lines and (not lines[-1].strip() or lines[-1].strip().startswith(">")):
        lines.pop()
    # Take the tail: last ~60 non-empty lines
    tail = [ln for ln in lines if ln.strip()][-60:]
    cleaned = "\n".join(tail).strip()
    # Collapse runs of blank lines (already stripped) and long whitespace runs
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    # Cap total length
    if len(cleaned) > 4000:
        cleaned = cleaned[-4000:]
    return cleaned


# --- NAVIGATING ------------------------------------------------------------


def _nav_voice(ctx: Context, t: str) -> None:
    """Voice commands for window navigation.

    Recognized patterns (keyword-based):
      - "list windows" / "what windows"
      - "switch to <name>" / "go to <name>"
      - "new window <name>"
    Anything else: speak the window list.
    """
    low = t.lower()
    host, opts = ctx.ssh
    session = ctx.config.tmux_session

    if "list" in low or "what" in low:
        _announce_windows(ctx)
        return

    m = re.match(r"(?:switch(?:\s+to)?|go\s+to|select)\s+(.+)", low)
    if m:
        name = m.group(1).strip().strip(".")
        try:
            remote.select_window(host, opts, session, name)
            ctx.active_window = name
            _speak(ctx, f"Switched to {name}.")
            _enter_mode(ctx, "WORK")
        except remote.RemoteError as exc:
            _report_error(ctx, f"Could not switch: {exc}")
        return

    m = re.match(r"new\s+window\s+(.+)", low)
    if m:
        name = m.group(1).strip().strip(".")
        try:
            remote.new_window(host, opts, session, name)
            ctx.active_window = name
            _speak(ctx, f"Created window {name}.")
            _enter_mode(ctx, "WORK")
        except remote.RemoteError as exc:
            _report_error(ctx, f"Could not create window: {exc}")
        return

    _announce_windows(ctx)


def _announce_windows(ctx: Context) -> None:
    host, opts = ctx.ssh
    try:
        rows = remote.list_windows(host, opts, ctx.config.tmux_session)
    except remote.RemoteError as exc:
        _report_error(ctx, f"Could not list windows: {exc}")
        return
    if not rows:
        _speak(ctx, "No windows.")
        return
    names = ", ".join(name for _idx, name, _cwd in rows)
    _speak(ctx, f"{len(rows)} windows: {names}.")


# --- LINEAR ----------------------------------------------------------------

_LINEAR_REFRESH_PROMPT = (
    "List my assigned Linear tickets using the Linear MCP. "
    "Respond with ONLY a JSON array (no prose, no code fences). "
    'Each object: {"id","title","priority","assignee","branch"}. '
    'Use empty string for missing fields.'
)

_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)


def _linear_voice(ctx: Context, t: str) -> None:
    low = t.lower()
    if "refresh" in low or "reload" in low or "list" in low:
        _linear_refresh(ctx)
        return
    if low in ("next", "next ticket"):
        _linear_step(ctx, +1)
        return
    if low in ("previous", "prev", "back", "previous ticket"):
        _linear_step(ctx, -1)
        return
    m = re.match(r"(?:select|work on|start)\s+(?:ticket\s+)?(\d+|this)", low)
    if m:
        token = m.group(1)
        idx: int | None
        if token == "this":
            idx = ctx.ticket_index
        else:
            idx = int(token) - 1
        _linear_select(ctx, idx)
        return
    # Otherwise treat as a filter / free-text query on cached tickets
    if ctx.tickets:
        hits = [
            t_ for t_ in ctx.tickets
            if low in str(t_.get("title", "")).lower()
            or low in str(t_.get("id", "")).lower()
        ]
        if hits:
            ctx.tickets = hits
            ctx.ticket_index = 0
            _speak(ctx, f"{len(hits)} tickets match.")
            _linear_announce(ctx)
            return
    _speak(ctx, "Say 'list', 'next', 'previous', or 'select <n>'.")


def _linear_refresh(ctx: Context) -> None:
    host, opts = ctx.ssh
    cfg = ctx.config
    win = cfg.linear_window
    prompt = f"claude -p {json.dumps(_LINEAR_REFRESH_PROMPT)}"
    try:
        remote.send(host, opts, cfg.tmux_session, win, prompt)
        remote.wait_done(host, opts, win, timeout=cfg.wait_timeout)
        raw = remote.capture(host, opts, cfg.tmux_session, win, lines=400)
    except (remote.RemoteError, remote.WaitTimeout) as exc:
        _report_error(ctx, f"Could not refresh tickets: {exc}")
        return
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        _report_error(ctx, "No ticket JSON found.")
        return
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        _report_error(ctx, f"Invalid ticket JSON: {exc}")
        return
    tickets: list[dict] = [t for t in payload if isinstance(t, dict) and t.get("id")]
    ctx.tickets = tickets
    ctx.ticket_index = 0
    _speak(ctx, f"{len(tickets)} tickets.")
    _linear_announce(ctx)


def _linear_step(ctx: Context, delta: int) -> None:
    if not ctx.tickets:
        _speak(ctx, "No tickets. Say 'list' to refresh.")
        return
    ctx.ticket_index = (ctx.ticket_index + delta) % len(ctx.tickets)
    _linear_announce(ctx)


def _linear_announce(ctx: Context) -> None:
    if not ctx.tickets:
        _speak(ctx, "No tickets.")
        return
    t = ctx.tickets[ctx.ticket_index]
    tid = str(t.get("id", "?"))
    title = str(t.get("title", ""))
    prio = str(t.get("priority", "")) or "no priority"
    assignee = str(t.get("assignee", "")) or "unassigned"
    _speak(ctx, f"{ctx.ticket_index + 1} of {len(ctx.tickets)}. {tid}. {title}. Priority {prio}. Assigned to {assignee}.")


def _linear_select(ctx: Context, idx: int | None) -> None:
    if not ctx.tickets or idx is None or idx < 0 or idx >= len(ctx.tickets):
        _speak(ctx, "No such ticket.")
        return
    t = ctx.tickets[idx]
    tid = str(t.get("id", ""))
    branch = str(t.get("branch", "")) or tid.lower()
    ctx.ticket_index = idx
    # Open a window for the ticket; rely on the user's existing shell
    # to have the right setup. Domain logic stays on the remote side.
    host, opts = ctx.ssh
    try:
        remote.new_window(host, opts, ctx.config.tmux_session, branch)
        remote.send(host, opts, ctx.config.tmux_session, branch, "claude")
    except remote.RemoteError as exc:
        _report_error(ctx, f"Could not open window for {tid}: {exc}")
        return
    ctx.active_window = branch
    _speak(ctx, f"Opened {tid} in window {branch}.")
    _enter_mode(ctx, "WORK")


# --- helpers ---------------------------------------------------------------


def _speak(ctx: Context, text: str) -> None:
    if not text:
        return
    try:
        ctx.tts.speak(text)
    except TTSClientError:
        logger.exception("TTS failed for: %s", text)


def _report_error(ctx: Context, message: str) -> None:
    ctx.thinking.stop()
    try:
        earcon.error()
    except earcon.EarconError:
        pass
    _speak(ctx, message)
    ctx.log.event("error", message=message, mode=ctx.mode)
