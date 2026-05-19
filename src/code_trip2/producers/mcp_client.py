"""Thin wrapper around the official Python MCP SDK (``pip install mcp``).

Each producer that needs an MCP server constructs an :class:`MCPClient`,
spawns the server as a stdio child process, and calls tools via
``call_tool(name, args)``. The wrapper is intentionally minimal:
producers do their own error handling and tool-arg shaping.

Skeleton only for v1 — the actual server choice (which Slack MCP
implementation, which Linear MCP implementation) is deferred. The
imports are lazy so the orchestrator can start without the ``mcp``
package installed; producers that try to use it fail at start().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MCPServerSpec:
    """How to launch an MCP server subprocess."""

    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None


class MCPClientError(Exception):
    pass


class MCPClient:
    """Wrapper around an ``mcp`` SDK ClientSession.

    v1 surface: ``start`` / ``stop`` / ``call_tool``. The implementation
    is stubbed until producers actually need it; raise loudly on use so
    nothing silently no-ops.
    """

    def __init__(self, spec: MCPServerSpec) -> None:
        self._spec = spec
        self._session = None  # type: ignore[var-annotated]

    def start(self) -> None:
        try:
            import mcp  # noqa: F401  (lazy import; see module docstring)
        except ImportError as exc:
            raise MCPClientError(
                "Python 'mcp' package not installed. "
                "Add it to pyproject.toml before enabling MCP producers."
            ) from exc
        # TODO: spawn subprocess + open mcp.ClientSession against stdio.
        raise MCPClientError(
            f"MCPClient.start() not yet implemented for spec={self._spec!r}"
        )

    def stop(self) -> None:
        if self._session is None:
            return
        # TODO: close session + reap subprocess.
        self._session = None

    def call_tool(self, name: str, args: dict | None = None) -> dict:
        raise MCPClientError("MCPClient.call_tool() not yet implemented")
