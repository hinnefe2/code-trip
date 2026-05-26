"""LinearProducer: polls Linear via a locally-hosted Linear MCP server.

v1 is a skeleton. Same shape as :class:`SlackProducer`: an async task
polls periodically, MCP client does the real work once wired. Replaces
``_linear_refresh`` in :mod:`code_trip2.modes` once functional — the
``claude -p`` + regex JSON extraction path can be removed when this
producer reliably surfaces tickets as tasks.
"""

from __future__ import annotations

import asyncio
import logging

from code_trip2._async_utils import event_or_timeout
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
        self._stop = asyncio.Event()
        self._client: MCPClient | None = None

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        if not self._config.linear_mcp_command:
            logger.info("LinearProducer: no linear_mcp_command configured; not starting.")
            return
        spec = MCPServerSpec(
            command=self._config.linear_mcp_command,
            args=tuple(self._config.linear_mcp_args),
        )
        self._client = MCPClient(spec)
        try:
            # MCPClient.start spins up a stdio subprocess; sync today.
            # Bridged via to_thread until the MCP client is itself async.
            await asyncio.to_thread(self._client.start)
        except MCPClientError as exc:
            logger.warning("LinearProducer: MCP client failed to start (%s); idling.", exc)
            self._client = None
            return
        try:
            # TODO: per poll tick, list_issues via MCP and emit tasks for
            # high-priority new/changed tickets. Skeleton ticks idle.
            while not self._stop.is_set():
                if await event_or_timeout(self._stop, self._poll):
                    return
        finally:
            if self._client is not None:
                await asyncio.to_thread(self._client.stop)
                self._client = None
