"""Background producers that push :class:`Task` objects into the queue.

Each producer is an :class:`asyncio.Task` owned by
:class:`ProducerSupervisor`, which is in turn owned by ``main.py``.
Public surface:

- :class:`Producer` — protocol any producer satisfies
- :class:`ProducerSupervisor` — starts/stops a collection of producers

Concrete producers live in sibling modules:

- :mod:`code_trip2.producers.claude`  — watches the Stop-hook event dir
- :mod:`code_trip2.producers.manual`  — voice-triggered manual adds
- :mod:`code_trip2.producers.slack`   — Slack mention search + watched channels via claude.ai MCP
- :mod:`code_trip2.producers.email`   — Gmail inbox poller via claude.ai MCP
- :mod:`code_trip2.producers.linear`  — Linear MCP (stub)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Producer(Protocol):
    name: str

    async def run(self) -> None: ...
    def request_stop(self) -> None: ...


class ProducerSupervisor:
    """Owns a set of producers; starts and stops them together as asyncio tasks."""

    def __init__(self) -> None:
        self._producers: list[Producer] = []
        self._tasks: dict[str, asyncio.Task] = {}

    def add(self, producer: Producer) -> None:
        self._producers.append(producer)

    def start_all(self) -> None:
        for p in self._producers:
            try:
                task = asyncio.create_task(p.run(), name=f"producer:{p.name}")
                self._tasks[p.name] = task
                logger.info("Started producer %s", p.name)
            except Exception:
                logger.exception("Failed to start producer %s", p.name)

    async def stop_all(self) -> None:
        # Ask everyone to exit voluntarily first.
        for p in self._producers:
            try:
                p.request_stop()
            except Exception:
                logger.exception("Failed to request stop for %s", p.name)
        if not self._tasks:
            return
        tasks = list(self._tasks.values())
        # Grace period for voluntary exit; then cancel anything still running.
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=2.0,
            )
        except asyncio.TimeoutError:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for name, t in self._tasks.items():
            exc = t.exception() if t.done() and not t.cancelled() else None
            if exc is not None:
                logger.warning("Producer %s exited with: %s", name, exc)
        self._tasks.clear()

    def status(self) -> list[tuple[str, str]]:
        """Best-effort ``(name, state)`` per producer for the TUI / debug.

        State is one of ``running`` (task alive), ``stopped`` (task ended),
        ``ready`` (manual producer — no background work expected), or
        ``idle`` (never started, e.g. because the relevant config was
        empty or start_all hasn't been called).
        """
        out: list[tuple[str, str]] = []
        for p in self._producers:
            task = self._tasks.get(p.name)
            if task is None:
                out.append((p.name, "ready" if p.name == "manual" else "idle"))
            elif not task.done():
                out.append((p.name, "running"))
            else:
                out.append((p.name, "ready" if p.name == "manual" else "stopped"))
        return out

    def __iter__(self):
        return iter(self._producers)
