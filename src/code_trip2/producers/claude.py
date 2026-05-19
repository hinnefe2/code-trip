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

import json
import logging
import shlex
import threading
import time

from code_trip2 import remote
from code_trip2.config import Config
from code_trip2.tasks import Task, TaskQueue

logger = logging.getLogger(__name__)


EVENTS_DIR = "/tmp/claude-events"


class ClaudeProducer:
    name = "claude"

    def __init__(
        self,
        *,
        config: Config,
        queue: TaskQueue,
        poll_interval: float = 1.5,
    ) -> None:
        self._config = config
        self._queue = queue
        self._poll = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen: set[str] = set()

    def start(self) -> None:
        if not self._config.ssh_host:
            logger.info("ClaudeProducer: no ssh_host configured; not starting.")
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # --- internals --------------------------------------------------------

    def _run(self) -> None:
        host, opts = self._config.ssh_host, self._config.ssh_options
        # Make sure the dir exists on the remote so the first poll doesn't fail.
        self._ensure_remote_dir(host, opts)
        while not self._stop.is_set():
            try:
                files = self._list_remote_events(host, opts)
            except remote.RemoteError as exc:
                logger.warning("ClaudeProducer poll failed: %s", exc)
                if self._stop.wait(self._poll * 4):
                    return
                continue
            for filename in files:
                if filename in self._seen:
                    continue
                self._seen.add(filename)
                try:
                    payload = self._read_and_clear(host, opts, filename)
                except remote.RemoteError as exc:
                    logger.warning("ClaudeProducer read failed for %s: %s", filename, exc)
                    continue
                if payload is None:
                    continue
                self._emit(payload)
            if self._stop.wait(self._poll):
                return

    def _ensure_remote_dir(self, host: str, opts: tuple[str, ...]) -> None:
        cmd = f"mkdir -p {shlex.quote(EVENTS_DIR)}"
        try:
            remote._ssh(host, opts, cmd, capture=False)  # type: ignore[attr-defined]
        except remote.RemoteError:
            logger.warning("Could not create %s on remote", EVENTS_DIR)

    def _list_remote_events(self, host: str, opts: tuple[str, ...]) -> list[str]:
        cmd = (
            f"ls -1 {shlex.quote(EVENTS_DIR)} 2>/dev/null | grep -E '\\.json$' || true"
        )
        out = remote._ssh(host, opts, cmd)  # type: ignore[attr-defined]
        return [line.strip() for line in out.splitlines() if line.strip()]

    def _read_and_clear(
        self, host: str, opts: tuple[str, ...], filename: str
    ) -> dict | None:
        path = f"{EVENTS_DIR}/{filename}"
        cmd = f"cat {shlex.quote(path)} && rm -f {shlex.quote(path)}"
        try:
            raw = remote._ssh(host, opts, cmd)  # type: ignore[attr-defined]
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

    def _emit(self, payload: dict) -> None:
        window = str(payload.get("window") or "unknown")
        finished_at = float(payload.get("finished_at") or time.time())
        last_user_msg = payload.get("last_user_msg") or ""
        headline = self._make_headline(window, last_user_msg)
        body = payload.get("preview") or None
        task = Task(
            kind="claude_reply",
            topic=window,
            headline=headline,
            body=body,
            source={"window": window, "finished_at": finished_at},
            created_at=time.time(),
        )
        self._queue.add(task)

    def _make_headline(self, window: str, last_user_msg: str) -> str:
        if last_user_msg:
            snippet = last_user_msg.strip().splitlines()[0][:80]
            return f"replied to: {snippet}"
        return "finished"
