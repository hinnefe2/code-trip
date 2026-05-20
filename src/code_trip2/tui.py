"""Live status dashboard for the orchestrator.

Enabled by ``--tui``. Suppresses normal Python logging so the live
display owns the terminal. The dashboard is read-only: it shows
mode, active window, summarizer state, current task, queue contents,
recent topics, and per-producer health. No input handling — taps and
voice still go through the macropad / voice loop.

Refreshes at ``refresh_hz`` (default 2 Hz, so the longest a queue mutation
sits invisible is ~500 ms). The render function is pure (reads context
fields, builds a Rich Layout) so it can be tested without starting Live.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
    "idle": "yellow",
    "stopped": "red",
}


# --- render --------------------------------------------------------------


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
    body_preview = Text("")
    if t.body:
        body_preview = Text(_truncate(t.body, 160), style="dim")
    return Panel(
        Group(head, headline, body_preview),
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
    for i, (t, _score) in enumerate(ranked[:10]):
        marker = "▶" if i == 0 else " "
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
    """Mode-aware reminder of what each macropad key does.

    YES/NO solo tap meaning depends on app-mode (queue vs focused), so
    the panel re-renders that row when the mode flips. Chord rows
    (NAV+x, ACT+NO) are mode-independent and shown verbatim.
    """
    if ctx.app_mode == "queue":
        yes_solo = "accept / expand"
        no_solo = "skip task"
        act_solo = "→ focused"
    else:
        yes_solo = "Enter"
        no_solo = "Esc"
        act_solo = "→ queue"

    def _key(name: str) -> Text:
        return Text(name, style="bold cyan")

    def _act(text: str) -> Text:
        return Text(text, style="white")

    sep = Text("   ")

    solo = Text.assemble(
        _key("PTT"), " ", _act("hold to talk"), sep,
        _key("YES"), " ", _act(yes_solo), sep,
        _key("NO"), " ", _act(no_solo), sep,
        _key("ACT"), " ", _act(act_solo), sep,
        _key("NAV"), " ", _act("per-app"),
    )
    nav_chords = Text.assemble(
        Text("NAV+", style="bold magenta"),
        _key("PTT"), " ", _act("speak app"), sep,
        _key("YES"), " ", _act("next"), sep,
        _key("NO"), " ", _act("prev"), sep,
        _key("ACT"), " ", _act("cycle app"),
    )
    act_chords = Text.assemble(
        Text("ACT+", style="bold magenta"),
        _key("NO"), " ", _act("Ctrl+U (clear line)"),
    )
    body = Group(solo, nav_chords, act_chords)
    return Panel(body, title="Macropad", border_style="bright_black")


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


def render(ctx: "Context", supervisor: "ProducerSupervisor | None") -> Layout:
    """Build the dashboard layout. Pure function; safe to call from tests."""
    layout = Layout()
    layout.split_column(
        Layout(_header(ctx), name="header", size=3),
        Layout(name="body"),
        Layout(_keymap_panel(ctx), name="keymap", size=5),
        Layout(_producers_panel(supervisor), name="producers", size=3),
    )
    layout["body"].split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=1),
    )
    layout["left"].split_column(
        Layout(_current_task_panel(ctx), size=7),
        Layout(_queue_table(ctx)),
    )
    layout["right"].update(_topics_panel(ctx))
    return layout


# --- live dashboard --------------------------------------------------------


class Dashboard:
    """Background thread that drives a Rich ``Live`` display.

    Lifecycle: construct → :meth:`start` (enters Live) → polling thread
    re-renders every ``1/refresh_hz`` seconds → :meth:`stop` (exits Live,
    restores terminal). Safe to call ``stop`` multiple times.
    """

    def __init__(
        self,
        ctx: "Context",
        supervisor: "ProducerSupervisor | None" = None,
        *,
        refresh_hz: float = 2.0,
    ) -> None:
        self._ctx = ctx
        self._supervisor = supervisor
        self._refresh_hz = refresh_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._live = None  # type: ignore[var-annotated]

    def start(self) -> None:
        from rich.console import Console
        from rich.live import Live

        if self._live is not None:
            return
        console = Console()
        self._live = Live(
            render(self._ctx, self._supervisor),
            console=console,
            screen=True,
            refresh_per_second=self._refresh_hz,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.__enter__()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._live is not None:
            try:
                self._live.__exit__(None, None, None)
            except Exception:
                logger.exception("Live exit failed")
            self._live = None

    def _run(self) -> None:
        interval = 1.0 / max(0.5, self._refresh_hz)
        while not self._stop.is_set():
            try:
                self._live.update(render(self._ctx, self._supervisor))
            except Exception:
                logger.exception("Dashboard render failed")
            if self._stop.wait(interval):
                return
