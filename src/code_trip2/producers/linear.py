"""LinearProducer: polls Linear via a locally-hosted Linear MCP server.

v1 is a skeleton. Same shape as :class:`SlackProducer`: thread polls
periodically, MCP client does the real work once wired. Replaces
``_linear_refresh`` in :mod:`code_trip2.modes` once functional — the
``claude -p`` + regex JSON extraction path can be removed when this
producer reliably surfaces tickets as tasks.
"""

from __future__ import annotations

import logging
import threading

from code_trip2.config import Config
from code_trip2.producers.mcp_client import MCPClient, MCPClientError, MCPServerSpec
from code_trip2.tasks import TaskQueue

logger = logging.getLogger(__name__)


class LinearProducer:
    name = "linear"

    def __init__(
        self,
        *,
        config: Config,
        queue: TaskQueue,
        poll_interval: float = 120.0,
    ) -> None:
        self._config = config
        self._queue = queue
        self._poll = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: MCPClient | None = None

    def start(self) -> None:
        if not self._config.linear_mcp_command:
            logger.info("LinearProducer: no linear_mcp_command configured; not starting.")
            return
        spec = MCPServerSpec(
            command=self._config.linear_mcp_command,
            args=tuple(self._config.linear_mcp_args),
        )
        self._client = MCPClient(spec)
        try:
            self._client.start()
        except MCPClientError as exc:
            logger.warning("LinearProducer: MCP client failed to start (%s); idling.", exc)
            self._client = None
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
        if self._client is not None:
            self._client.stop()
            self._client = None

    def _run(self) -> None:
        # TODO: per poll tick, list_issues via MCP and emit tasks for
        # high-priority new/changed tickets. Skeleton ticks idle.
        while not self._stop.is_set():
            if self._stop.wait(self._poll):
                return
