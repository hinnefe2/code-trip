"""SlackProducer: polls Slack via a locally-hosted Slack MCP server.

v1 is a skeleton — the producer thread runs and logs, but
:class:`mcp_client.MCPClient` is not yet wired up to an actual MCP
implementation. The shape is in place so once a Slack MCP server is
picked, the polling body is the only thing that needs filling in.

Producer pattern: every ``poll_interval`` seconds, query the MCP server
for unread messages / mentions across ``slack_channels`` from config.
For each new unread, push a ``Task(kind="slack_msg", topic=f"slack-{channel}")``.
"""

from __future__ import annotations

import logging
import threading

from code_trip2.config import Config
from code_trip2.producers.mcp_client import MCPClient, MCPClientError, MCPServerSpec
from code_trip2.tasks import TaskQueue

logger = logging.getLogger(__name__)


class SlackProducer:
    name = "slack"

    def __init__(
        self,
        *,
        config: Config,
        queue: TaskQueue,
        poll_interval: float = 30.0,
    ) -> None:
        self._config = config
        self._queue = queue
        self._poll = poll_interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: MCPClient | None = None

    def start(self) -> None:
        if not self._config.slack_channels:
            logger.info("SlackProducer: no slack_channels configured; not starting.")
            return
        if not self._config.slack_mcp_command:
            logger.info("SlackProducer: no slack_mcp_command configured; not starting.")
            return
        spec = MCPServerSpec(
            command=self._config.slack_mcp_command,
            args=tuple(self._config.slack_mcp_args),
        )
        self._client = MCPClient(spec)
        try:
            self._client.start()
        except MCPClientError as exc:
            logger.warning("SlackProducer: MCP client failed to start (%s); idling.", exc)
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
        # TODO: per poll tick, call MCP tools to fetch unreads + emit tasks.
        # Skeleton: just sit and tick so the supervisor sees us as alive.
        while not self._stop.is_set():
            if self._stop.wait(self._poll):
                return
