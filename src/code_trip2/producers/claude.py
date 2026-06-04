"""ClaudeProducer: emits a task each time a Claude session finishes.

Uses an SSH+ssh-control directory watch on the **remote** host. The Stop
hook (installed via ``scripts/setup-stop-hook.sh``) writes a JSON event
to ``/tmp/claude-events/<window>-<timestamp>.json`` every time Claude
emits a Stop event. This producer polls that directory over SSH (cheap
with ControlMaster), parses any new files, and pushes a
``Task(kind="claude_reply", topic=window)``.

The legacy ``/tmp/claude-done-<window>`` touch-file is still emitted by
the hook so :func:`remote.wait_done` keeps working in focused mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import time

from code_trip2 import remote
from code_trip2._async_utils import event_or_timeout
from code_trip2.config import Config
from code_trip2.summarizer import Summarizer, SummarizerError
from code_trip2.tasks import Task, TaskQueue

logger = logging.getLogger(__name__)


EVENTS_DIR = "/tmp/claude-events"

# Matches Linear-style ticket identifiers used as remote tmux window
# names by the /do-ticket skill (e.g. ``ENGAGE-1234``). When a Stop event
# arrives for one of these windows we tag the resulting claude_reply
# with a subject_key so the queue clusters it adjacent to the original
# linear_issue task that prompted the work.
_TICKET_WINDOW_RE = re.compile(r"^[A-Z]+-\d+$")


class ClaudeProducer:
    name = "claude"

    def __init__(
        self,
        *,
        config: Config,
        queue: TaskQueue,
        summarizer: Summarizer | None = None,
        poll_interval: float = 1.5,
    ) -> None:
        self._config = config
        self._queue = queue
        self._summarizer = summarizer
        self._poll = poll_interval
        self._stop = asyncio.Event()
        self._seen: set[str] = set()

    def request_stop(self) -> None:
        self._stop.set()

    # --- internals --------------------------------------------------------

    async def run(self) -> None:
        if not self._config.ssh_host:
            logger.info("ClaudeProducer: no ssh_host configured; not starting.")
            return
        host, opts = self._config.ssh_host, self._config.ssh_options
        await self._ensure_remote_dir(host, opts)
        while not self._stop.is_set():
            try:
                await self._poll_once(host, opts)
            except remote.RemoteError as exc:
                logger.warning("ClaudeProducer poll failed: %s", exc)
                if await event_or_timeout(self._stop, self._poll * 4):
                    return
                continue
            except Exception:
                logger.exception("ClaudeProducer unexpected error")
            if await event_or_timeout(self._stop, self._poll):
                return

    async def _poll_once(self, host: str, opts: tuple[str, ...]) -> None:
        files = await self._list_remote_events(host, opts)
        for filename in files:
            if filename in self._seen:
                continue
            self._seen.add(filename)
            try:
                payload = await self._read_and_clear(host, opts, filename)
            except remote.RemoteError as exc:
                logger.warning("ClaudeProducer read failed for %s: %s", filename, exc)
                continue
            if payload is None:
                continue
            await self._emit(payload, host, opts)

    async def _ensure_remote_dir(self, host: str, opts: tuple[str, ...]) -> None:
        cmd = f"mkdir -p {shlex.quote(EVENTS_DIR)}"
        try:
            await remote._ssh(host, opts, cmd, capture=False)
        except remote.RemoteError:
            logger.warning("Could not create %s on remote", EVENTS_DIR)

    async def _list_remote_events(self, host: str, opts: tuple[str, ...]) -> list[str]:
        cmd = (
            f"ls -1 {shlex.quote(EVENTS_DIR)} 2>/dev/null | grep -E '\\.json$' || true"
        )
        out = await remote._ssh(host, opts, cmd)
        return [line.strip() for line in out.splitlines() if line.strip()]

    async def _read_and_clear(
        self, host: str, opts: tuple[str, ...], filename: str
    ) -> dict | None:
        path = f"{EVENTS_DIR}/{filename}"
        cmd = f"cat {shlex.quote(path)} && rm -f {shlex.quote(path)}"
        try:
            raw = await remote._ssh(host, opts, cmd)
        except remote.RemoteError:
            return None
        raw = raw.strip()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("ClaudeProducer: bad JSON in %s", filename)
            return None

    async def _emit(self, payload: dict, host: str, opts: tuple[str, ...]) -> None:
        window = str(payload.get("window") or "unknown")
        finished_at = float(payload.get("finished_at") or time.time())
        last_user_msg = payload.get("last_user_msg") or ""
        headline = self._make_headline(window, last_user_msg)
        body = await self._summarize_pane(window, last_user_msg, host, opts)
        source = {"window": window, "finished_at": finished_at}

        # Collapse: only one pending claude_reply per window. A long
        # remote session that hits Stop multiple times (or a /do-ticket
        # run that stops, then resumes after a follow-up) should keep
        # rewriting the same queue task rather than stacking duplicates.
        # Same pattern as LinearProducer._find_pending_issue_task.
        existing = self._find_pending_window_task(window)
        if existing is not None:
            self._queue.update_task(
                existing.id,
                headline=headline,
                body=body,
                source=source,
                created_at=time.time(),
            )
            return

        subject_key = (
            f"linear:{window.upper()}"
            if _TICKET_WINDOW_RE.match(window)
            else None
        )
        task = Task(
            kind="claude_reply",
            topic=window,
            headline=headline,
            body=body,
            source=source,
            created_at=time.time(),
            subject_key=subject_key,
        )
        self._queue.add(task)

    def _find_pending_window_task(self, window: str) -> Task | None:
        if not window:
            return None
        for task in self._queue.pending():
            if task.kind != "claude_reply":
                continue
            if (task.source or {}).get("window") == window:
                return task
        return None

    async def _summarize_pane(
        self,
        window: str,
        last_user_msg: str,
        host: str,
        opts: tuple[str, ...],
    ) -> str | None:
        """Capture the pane and run it through the summarizer.

        Returns the summary (audio-ready) or None if no summarizer is
        configured or the call fails. The producer doesn't fall back to
        ``clean_output`` here — the *headline* is already a usable
        announcement; an absent body just means the user has nothing to
        expand on tap. Better than reading raw ANSI gunk aloud.
        """
        if self._summarizer is None or not self._summarizer.enabled:
            return None
        if not host:
            return None
        try:
            raw = await remote.capture(
                host, opts, self._config.tmux_session, window, lines=400,
            )
        except remote.RemoteError as exc:
            logger.warning("ClaudeProducer: capture-pane failed for %s: %s", window, exc)
            return None
        if not raw or not raw.strip():
            return None
        try:
            return await self._summarizer.summarize(
                raw, context={"kind": "claude_reply", "user_prompt": last_user_msg},
            )
        except SummarizerError as exc:
            logger.warning("ClaudeProducer: summarize failed for %s: %s", window, exc)
            return None

    def _make_headline(self, window: str, last_user_msg: str) -> str:
        if last_user_msg:
            snippet = last_user_msg.strip().splitlines()[0][:80]
            return f"replied to: {snippet}"
        return "finished"
