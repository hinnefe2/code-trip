"""Textual TUI for the orchestrator.

Enabled by ``--tui``. The app owns the foreground terminal; logs go to a
file instead. Panels render the orchestrator state (mode, active window,
summarizer, current task, queue, recent topics, keymap, producers) at
~2 Hz so a queue mutation is visible within ~500 ms.

The :class:`CodeTripApp` also exposes an :class:`Input` widget used in
local-STT mode (Superwhisper / clipboard-paste) — the macropad bridges
PTT release into :class:`PttReleased`, which clears the input field and
arms an auto-submit timer so a pasted transcript dispatches without
needing a trailing newline.

Panel-builders (``_header``, ``_current_task_panel``, …) stay as pure
Rich-returning functions so the existing assertions still apply.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING

from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Input, Static

if TYPE_CHECKING:
    from code_trip2.modes import Context
    from code_trip2.producers import ProducerSupervisor

logger = logging.getLogger(__name__)


# --- host-terminal detection ----------------------------------------------

# Maps ``TERM_PROGRAM`` env-var values to the macOS application name that
# ``window.active_app()`` returns. Used to suppress synthesized keystrokes
# that would otherwise land in the terminal hosting the TUI (causing the
# alternate-buffer to scroll when YES/NO/NAV taps fire Enter/Esc/Down).
_TERM_PROGRAM_TO_APP: dict[str, str] = {
    "Apple_Terminal": "Terminal",
    "iTerm.app": "iTerm2",
    "kitty": "kitty",
    "WezTerm": "WezTerm",
    "Tabby": "Tabby",
    "vscode": "Code",
    "ghostty": "Ghostty",
    "Hyper": "Hyper",
    "alacritty": "Alacritty",
}


def detect_tui_host_app() -> str | None:
    """Best-effort identify the terminal app hosting this process.

    Reads ``TERM_PROGRAM`` (set by most macOS terminals) and maps to the
    app name ``window.active_app()`` would return. Falls back to the raw
    env-var value so unknown terminals still get *some* coverage.
    Returns ``None`` when the env var isn't set.
    """
    term = os.environ.get("TERM_PROGRAM", "").strip()
    if not term:
        return None
    return _TERM_PROGRAM_TO_APP.get(term, term)


# --- formatting helpers ---------------------------------------------------


def _format_age(seconds: float) -> str:
    """Compact age: 9s / 12m / 1h / 2d."""
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}m"
    if seconds < 86_400:
        return f"{int(seconds / 3600)}h"
    return f"{int(seconds / 86_400)}d"


def _truncate(s: str, n: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


_STATE_COLOR = {
    "running": "green",
    "ready": "green",
    "polling": "cyan",
    "idle": "yellow",
    "stopped": "red",
}


# --- panel builders (pure Rich; used by Static widgets) -------------------


def _header(ctx: "Context") -> Panel:
    mode_color = "cyan" if ctx.app_mode == "queue" else "blue"
    mode_str = Text(ctx.app_mode.upper(), style=f"bold {mode_color}")
    win_str = Text(ctx.active_window or "(no window)", style="white")
    summ_ok = ctx.summarizer is not None and ctx.summarizer.enabled
    summ_color = "green" if summ_ok else "yellow"
    summ_model = getattr(ctx.summarizer, "_model", "off") if summ_ok else "off"
    summ_str = Text(summ_model, style=summ_color)
    line = Text.assemble(
        "mode ",
        mode_str,
        "   window ",
        win_str,
        "   summarizer ",
        summ_str,
    )
    return Panel(line, title="code-trip", border_style="bright_black")


# Roughly 4x what the panel showed in single-line-preview mode. Hard
# cap so a long-bodied task (Linear ticket descriptions, email
# threads) can't push the Queue panel below it off-screen.
_CURRENT_TASK_BODY_MAX_LINES = 16


def _clip_body(body: str, max_lines: int) -> tuple[str, bool]:
    """Cap ``body`` at ``max_lines`` lines, preserving newlines.

    Returns ``(clipped, truncated)``. Long single lines are NOT
    hard-wrapped here — Rich's Text renders them with soft-wrap, which
    can make the visual height exceed ``max_lines``. That's a tradeoff
    in favor of structure preservation; markdown bodies usually have
    reasonable line lengths.
    """
    if not body:
        return "", False
    lines = body.splitlines()
    if len(lines) <= max_lines:
        return body.rstrip(), False
    return "\n".join(lines[:max_lines]).rstrip(), True


def _current_task_panel(ctx: "Context") -> Panel:
    t = ctx.current_task
    if t is None:
        body = Text("(idle — say 'next' or tap YES)", style="dim italic")
        return Panel(body, title="Current task", border_style="bright_black")
    now = time.time()
    head = Text.assemble(
        Text(t.kind, style="bold magenta"),
        "  ",
        Text(t.topic, style="cyan"),
        Text(f"  {_format_age(now - t.created_at)} old", style="dim"),
    )
    headline = Text(t.headline or "(no headline)", style="white")
    parts = [head, headline]
    if t.body:
        clipped, truncated = _clip_body(t.body, _CURRENT_TASK_BODY_MAX_LINES)
        if clipped:
            parts.append(Text(clipped, style="dim"))
        if truncated:
            parts.append(Text("… (truncated — body shown above)", style="dim italic"))
    return Panel(
        Group(*parts),
        title="Current task",
        border_style="magenta",
    )


def _queue_table(ctx: "Context") -> Panel:
    pending = ctx.queue.pending()
    if not pending:
        empty = Text("(queue empty)", style="dim italic")
        return Panel(empty, title="Queue (0)", border_style="bright_black")
    # Rank for display so the order matches what the next-pull would pick.
    ranked = ctx.queue.ranked(now=time.time(), recent=ctx.recent_topics)
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=2)  # marker
    table.add_column("kind", style="magenta", width=14)
    table.add_column("topic", style="cyan", width=18)
    table.add_column("age", style="white", width=5)
    table.add_column("headline", overflow="ellipsis")
    now = time.time()
    cursor_id = ctx.current_task.id if ctx.current_task is not None else None
    for i, (t, _score) in enumerate(ranked[:10]):
        # The ▶ marker follows the cursor (not the top-ranked row) so
        # arrow-key navigation moves visibly without reshuffling the
        # queue. Falls back to the top row when there's no cursor —
        # matches the auto-announce target.
        if cursor_id is not None:
            is_marked = t.id == cursor_id
        else:
            is_marked = i == 0
        marker = "▶" if is_marked else " "
        table.add_row(
            Text(marker, style="bold green"),
            t.kind,
            t.topic,
            _format_age(now - t.created_at),
            _truncate(t.headline or "", 80),
        )
    title = f"Queue ({len(pending)} pending)"
    if len(ranked) > 10:
        title += f"  (+{len(ranked) - 10} more)"
    return Panel(table, title=title, border_style="green")


def _topics_panel(ctx: "Context") -> Panel:
    items = ctx.recent_topics.as_list()
    if not items:
        body = Text("(no recent topics)", style="dim italic")
    else:
        now = time.time()
        # Reversed so most-recent is on the left.
        parts: list[Text] = []
        for topic, when in reversed(items):
            parts.append(Text(topic, style="cyan"))
            parts.append(Text(f" ({_format_age(now - when)})  ", style="dim"))
        body = Text.assemble(*parts)
    return Panel(body, title="Recent topics", border_style="bright_black")


def _keymap_panel(ctx: "Context") -> Panel:
    """Mode-aware macropad reference, rendered as a grid.

    Columns are the five physical keys; rows are the chord prefixes
    (none / NAV / ACT). Each cell shows what that key does under that
    prefix, or "—" when the combination is unbound (or is itself the
    held modifier).

    - **Queue mode** (away from screen, audio-driven): solo YES/NO
      drive the queue, ACT stops audio, NAV flips back to focused.
      ``ACT+NO`` dismisses the current task. ``ACT+NO`` Ctrl+U is
      omitted — it's purely a shell-input affordance.
    - **Focused mode** (at the screen): solo YES/NO synthesize
      Enter/Esc, ACT does per-app navigation, NAV flips to queue.
      ``ACT+NO`` is the Ctrl+U "clear line" binding.
    """
    if ctx.app_mode == "queue":
        bindings: dict[tuple[str | None, str], str] = {
            (None, "TALK"): "hold to talk",
            (None, "YES"): "submit (Enter)",
            (None, "NO"): "skip task",
            (None, "NAV"): "→ focused",
            (None, "ACT"): "stop audio",
            ("NAV", "TALK"): "speak app",
            ("NAV", "YES"): "next",
            ("NAV", "NO"): "prev",
            ("NAV", "ACT"): "cycle app",
            ("ACT", "TALK"): "skill mode",
            ("ACT", "YES"): "open in app",
            ("ACT", "NO"): "dismiss task",
        }
    else:
        bindings = {
            (None, "TALK"): "hold to talk",
            (None, "YES"): "Enter",
            (None, "NO"): "Esc",
            (None, "NAV"): "→ queue",
            (None, "ACT"): "per-app",
            ("NAV", "TALK"): "speak app",
            ("NAV", "YES"): "next",
            ("NAV", "NO"): "prev",
            ("NAV", "ACT"): "cycle app",
            ("ACT", "NO"): "Ctrl+U (clear line)",
        }

    columns = ("NAV", "ACT", "NO", "YES", "TALK")
    rows: tuple[str | None, ...] = (None, "NAV", "ACT")

    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(width=5, no_wrap=True)
    for _ in columns:
        table.add_column(justify="center", overflow="fold", ratio=1)

    header = [Text("")]
    for col in columns:
        header.append(Text(col, style="bold cyan"))
    table.add_row(*header)

    for prefix in rows:
        label = Text("") if prefix is None else Text(f"{prefix}+", style="bold magenta")
        cells: list[Text] = [label]
        for col in columns:
            action = bindings.get((prefix, col))
            if action is None:
                cells.append(Text("—", style="dim"))
            else:
                cells.append(Text(action, style="white"))
        table.add_row(*cells)

    return Panel(table, title="Macropad", border_style="bright_black")


def _keymap_panel_size(ctx: "Context") -> int:
    """Total layout rows the macropad panel should reserve.

    Grid has a header row plus three prefix rows (none / NAV+ / ACT+),
    so 4 content rows plus 2 for the panel border.
    """
    return 6


_AUTOHANDLE_LABEL_STYLE = {
    "HANDLED": "bold green",
    "FAILED": "bold red",
    "DISMISSED": "bold bright_black",
    "DRY-RUN": "bold yellow",
}


def _autohandle_panel(ctx: "Context") -> Panel:
    """One line per recent screener outcome, newest at top.

    Only screener outcomes that represent a real or would-be action
    land in ``ctx.autohandle_log`` (the main.py callback filters out
    plain pass-throughs), so every visible row corresponds to a skill
    that ran, failed, or — in dry-run mode — would have run.
    """
    entries = list(getattr(ctx, "autohandle_log", ()) or ())
    enabled = bool(getattr(ctx.config, "autohandle_enabled", False))
    kinds = tuple(getattr(ctx.config, "autohandle_kinds", ()) or ())
    if not entries:
        if not enabled or not kinds:
            body = Text("(auto-handle disabled)", style="dim italic")
        else:
            body = Text("(no recent actions)", style="dim italic")
        return Panel(body, title="Auto-handle log", border_style="bright_black")

    now = time.time()
    lines: list[Text] = []
    # deque has oldest at left; reversed gives newest-first display.
    for entry in reversed(entries):
        out = entry.outcome
        if out.dry_run_nominated:
            label = "DRY-RUN"
        else:
            label = out.action.upper()
        style = _AUTOHANDLE_LABEL_STYLE.get(label, "white")
        age = _format_age(now - entry.ts)
        skill = out.skill or "?"
        headline = _truncate(out.task.headline or "", 60)
        suffix_text = ""
        if out.action == "failed" and out.error:
            suffix_text = f"  ({_truncate(out.error, 40)})"
        elif out.action == "handled" and out.summary:
            suffix_text = f"  — {_truncate(out.summary, 40)}"
        lines.append(Text.assemble(
            Text(f"{age:>4}  ", style="dim"),
            Text(f"{label:<8}", style=style),
            Text(f"{skill}: ", style="cyan"),
            Text(headline, style="white"),
            Text(suffix_text, style="dim"),
        ))

    visible = lines[:10]  # deque holds 20; cap display so the panel doesn't grow
    title = f"Auto-handle log ({len(entries)})"
    if len(entries) > len(visible):
        title += f"  (+{len(entries) - len(visible)} older)"
    return Panel(Group(*visible), title=title, border_style="bright_black")


def _producers_panel(supervisor: "ProducerSupervisor | None") -> Panel:
    if supervisor is None:
        return Panel(Text("(no supervisor)", style="dim"), title="Producers")
    parts: list[Text] = []
    for name, state in supervisor.status():
        color = _STATE_COLOR.get(state, "white")
        if parts:
            parts.append(Text("   "))
        parts.append(Text(name, style="bold"))
        parts.append(Text(f": {state}", style=color))
    body = Text.assemble(*parts) if parts else Text("(none)", style="dim")
    return Panel(body, title="Producers", border_style="bright_black")


# --- Textual messages from the macropad bridge ---------------------------


class PttReleased(Message):
    """Posted from the macropad's pynput thread when PTT is released in
    local-STT (key-forwarding) mode. ``skill_mode`` rides along so the
    next submitted Input value dispatches to ``handle_skill`` instead of
    ``handle_voice``.
    """

    def __init__(self, skill_mode: bool) -> None:
        super().__init__()
        self.skill_mode = skill_mode


# --- the app --------------------------------------------------------------


# Periodic refresh of the dynamic panels (current task, queue, topics,
# producers). The orchestrator pushes via message in a few spots but most
# state mutates without a TUI hook; a poll at 2 Hz keeps it close enough.
_REFRESH_HZ = 2.0


class CodeTripApp(App):
    """Textual app that owns the foreground terminal during ``--tui``.

    Read-only display of orchestrator state plus an ``Input`` widget for
    local-STT mode. The macropad still drives chords/taps/voice via the
    bridge in ``main.py``; the app only routes PTT-release-then-paste
    and Input-Enter through ``handle_voice`` / ``handle_skill``.
    """

    CSS = """
    Screen {
        layout: vertical;
    }
    #body {
        height: 1fr;
    }
    #left {
        width: 2fr;
    }
    #right {
        width: 1fr;
    }
    #voice_input {
        height: 3;
    }
    #voice_input.hidden {
        display: none;
    }
    Static {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        # Arrow nav through the queue. ``priority=True`` so they fire
        # even when the Input widget has focus (Input doesn't use up/
        # down itself, so we're not stealing a real edit affordance).
        Binding("up", "queue_prev", "Prev task", priority=True),
        Binding("down", "queue_next", "Next task", priority=True),
    ]

    def __init__(
        self,
        ctx: "Context",
        supervisor: "ProducerSupervisor | None" = None,
        *,
        local_stt: bool = False,
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.supervisor = supervisor
        self.local_stt = local_stt
        # Cleared on each PTT release; submit_input / on_input_submitted
        # read it to decide between handle_voice and handle_skill. Stays
        # set across edits so the user can tweak the pasted transcript
        # before submitting.
        self._pending_skill_mode = False

    def compose(self) -> ComposeResult:
        yield Static(_header(self.ctx), id="header")
        with Horizontal(id="body"):
            with Vertical(id="left"):
                yield Static(_current_task_panel(self.ctx), id="current_task")
                yield Static(_queue_table(self.ctx), id="queue")
                yield Static(_autohandle_panel(self.ctx), id="autohandle")
            yield Static(_topics_panel(self.ctx), id="right")
        input_widget = Input(
            placeholder="type or wait for paste — Enter submits, Esc clears",
            id="voice_input",
        )
        if not self.local_stt:
            input_widget.add_class("hidden")
        yield input_widget
        yield Static(_keymap_panel(self.ctx), id="keymap")
        yield Static(_producers_panel(self.supervisor), id="producers")

    def on_mount(self) -> None:
        self.set_interval(1.0 / _REFRESH_HZ, self._refresh_panels)

    def _refresh_panels(self) -> None:
        try:
            self.query_one("#header", Static).update(_header(self.ctx))
            self.query_one("#current_task", Static).update(_current_task_panel(self.ctx))
            self.query_one("#queue", Static).update(_queue_table(self.ctx))
            self.query_one("#autohandle", Static).update(_autohandle_panel(self.ctx))
            self.query_one("#right", Static).update(_topics_panel(self.ctx))
            self.query_one("#keymap", Static).update(_keymap_panel(self.ctx))
            self.query_one("#producers", Static).update(_producers_panel(self.supervisor))
        except Exception:
            logger.exception("TUI refresh failed")

    # --- macropad bridge --------------------------------------------------

    def on_ptt_released(self, message: PttReleased) -> None:
        """Wipe + focus the Input so the pasted transcript lands cleanly.

        Auto-submit is intentionally absent — the user wants to read /
        edit / approve the pasted transcript before it dispatches. They
        submit it manually with Enter (typed) or the macropad YES key
        (which calls :meth:`submit_input`).
        """
        if not self.local_stt:
            return
        self._pending_skill_mode = message.skill_mode
        try:
            input_widget = self.query_one("#voice_input", Input)
        except Exception:
            return
        input_widget.value = ""
        input_widget.focus()
        logger.info(
            "on_ptt_released — input cleared/focused (skill_mode=%s)",
            message.skill_mode,
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "voice_input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            self._pending_skill_mode = False
            return
        self._dispatch_transcript(text)

    def submit_input(self) -> bool:
        """Submit the Input widget's current value as if Enter was pressed.

        Returns ``True`` if there was text to submit (so the caller can
        decide whether to fall back to another action), ``False`` when
        the Input was empty / missing. Called by the macropad YES tap
        in queue mode via ``dispatch.queue_yes_tap``.
        """
        try:
            input_widget = self.query_one("#voice_input", Input)
        except Exception:
            return False
        text = (input_widget.value or "").strip()
        if not text:
            return False
        input_widget.value = ""
        self._dispatch_transcript(text)
        return True

    def _dispatch_transcript(self, text: str) -> None:
        skill = self._pending_skill_mode
        self._pending_skill_mode = False
        # Local imports to keep tui.py importable in tests that stub
        # the dispatch module — and to keep the loop happy if Textual
        # ever decides to import tui before dispatch wires up.
        from code_trip2.dispatch import handle_skill, handle_voice

        logger.info("submit transcript (skill_mode=%s): %s", skill, text)
        if skill:
            asyncio.create_task(handle_skill(self.ctx, text))
        else:
            asyncio.create_task(handle_voice(self.ctx, text))

    # --- queue navigation actions -----------------------------------------

    async def action_queue_prev(self) -> None:
        await self._queue_arrow(-1)

    async def action_queue_next(self) -> None:
        await self._queue_arrow(+1)

    async def _queue_arrow(self, direction: int) -> None:
        """Forward up/down to dispatch.queue_navigate, queue-mode only.

        Local import mirrors :meth:`_dispatch_transcript` — keeps tui
        importable in tests that stub the dispatch module.
        """
        from code_trip2 import dispatch

        if self.ctx.app_mode != dispatch.MODE_QUEUE:
            return
        await dispatch.queue_navigate(self.ctx, direction=direction)

    # --- thread-safe entry from the pynput listener -----------------------

    def post_ptt_release_from_thread(self, skill_mode: bool) -> None:
        """Schedule a :class:`PttReleased` post from the macropad's thread."""
        try:
            self.call_from_thread(self.post_message, PttReleased(skill_mode))
        except Exception:
            logger.exception("Failed to post PttReleased from thread")
