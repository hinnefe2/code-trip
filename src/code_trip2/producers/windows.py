"""WindowProducer: state-driven tasks for remote ticket tmux windows.

Polls the remote tmux session and tracks windows whose names look like
Linear ticket identifiers (``ENGAGE-1234``) — the worktree windows the
remote /do-ticket skill creates. Each poll captures every ticket
window's pane and classifies it via :mod:`code_trip2.claude_screen`:

- **waiting** (idle prompt or permission dialog) → a
  ``Task(kind="remote_window")`` sits in the queue until answered.
- **running** → the task is retired; the queue only ever shows sessions
  that need the user.
- window gone → task retired.

The captured pane text goes into the shared ``mirrors`` dict (the TUI's
live mirror view) on **every** tick, but the queue is only touched on
state *transitions* — steady-state polling fires zero queue events, so
the JSONL queue log doesn't grow per-poll.

Replaces the old event-driven ClaudeProducer (Stop-hook JSON files +
LLM pane summaries). A single fast lane for the currently-viewed window
was considered and skipped: at a 1.5s tick the mirror already outpaces
the TUI's 2 Hz refresh.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Callable

from code_trip2 import claude_screen, remote
from code_trip2._async_utils import event_or_timeout
from code_trip2.config import Config
from code_trip2.tasks import Task, TaskQueue

logger = logging.getLogger(__name__)

# Remote tmux windows named like Linear ticket identifiers, as created
# by the /do-ticket skill.
_TICKET_WINDOW_RE = re.compile(r"^[A-Z]+-\d+$")

_HEADLINE_QUESTION_MAX = 80


def _origin_key(window: str) -> str:
    # Same namespace the old ClaudeProducer used, so replayed history
    # and window tasks share one identity per window.
    return f"claude:{window}"


class WindowProducer:
    name = "windows"
    has_background_work = True

    def __init__(
        self,
        *,
        config: Config,
        queue: TaskQueue,
        submit: "Callable[..., Task] | None" = None,
        mirrors: dict[str, str] | None = None,
        poll_interval: float | None = None,
    ) -> None:
        self._config = config
        self._queue = queue
        self._submit: "Callable[..., Task]" = submit or (
            lambda task, *, if_terminal="new": queue.upsert(
                task, if_terminal=if_terminal
            )
        )
        # Shared with Context.window_mirrors — the TUI reads it directly.
        self._mirrors = mirrors if mirrors is not None else {}
        self._poll = poll_interval or config.window_poll_interval
        self._stop = asyncio.Event()
        self._last_state: dict[str, str] = {}

    def request_stop(self) -> None:
        self._stop.set()

    # --- internals --------------------------------------------------------

    async def run(self) -> None:
        if not self._config.ssh_host:
            logger.info("WindowProducer: no ssh_host configured; not starting.")
            return
        host, opts = self._config.ssh_host, self._config.ssh_options
        while not self._stop.is_set():
            try:
                await self._poll_once(host, opts)
            except remote.RemoteError as exc:
                logger.warning("WindowProducer poll failed: %s", exc)
                if await event_or_timeout(self._stop, self._poll * 4):
                    return
                continue
            except Exception:
                logger.exception("WindowProducer unexpected error")
            if await event_or_timeout(self._stop, self._poll):
                return

    async def _poll_once(self, host: str, opts: tuple[str, ...]) -> None:
        session = self._config.tmux_session
        rows = await remote.list_windows(host, opts, session)
        names = [name for _idx, name, _path in rows if _TICKET_WINDOW_RE.match(name)]

        for gone in set(self._last_state) - set(names):
            self._queue.resolve_by_origin(_origin_key(gone))
            self._mirrors.pop(gone, None)
            self._last_state.pop(gone, None)

        caps = await asyncio.gather(
            *(
                remote.capture(
                    host, opts, session, w,
                    lines=self._config.window_capture_lines, ansi=True,
                )
                for w in names
            ),
            return_exceptions=True,
        )
        for window, cap in zip(names, caps):
            if isinstance(cap, BaseException):
                # Keep the last mirror and state; a transient capture
                # failure must not flap the task.
                logger.warning(
                    "WindowProducer: capture failed for %s: %s", window, cap
                )
                continue
            self._mirrors[window] = cap
            self._observe(window, cap)

    def _observe(self, window: str, cap: str) -> None:
        state = claude_screen.detect_state(cap)
        prev = self._last_state.get(window)
        self._last_state[window] = state
        if state == prev:
            return
        if state == claude_screen.RUNNING:
            self._queue.resolve_by_origin(_origin_key(window))
            return

        question = (
            claude_screen.permission_question(cap)
            if state == claude_screen.WAITING_PERMISSION
            else None
        )
        if question:
            headline = f"needs permission: {question[:_HEADLINE_QUESTION_MAX]}"
        elif state == claude_screen.WAITING_PERMISSION:
            headline = "needs permission"
        else:
            headline = "waiting for input"

        task = Task(
            kind="remote_window",
            topic=window,
            headline=headline,
            body=question,
            source={"window": window, "claude_state": state},
            created_at=time.time(),
            # Clusters adjacent to the linear_issue task for the ticket.
            subject_key=f"linear:{window.upper()}",
            origin_key=_origin_key(window),
        )
        # Re-mint after terminal only when a run actually ended — a
        # waiting_input↔waiting_permission flap on a task the user
        # already answered must not resurrect it.
        if_terminal = "new" if prev in (None, claude_screen.RUNNING) else "skip"
        self._submit(task, if_terminal=if_terminal)
