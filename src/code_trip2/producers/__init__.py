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
- :mod:`code_trip2.producers.linear`  — Linear issue poller via claude.ai MCP
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Producer(Protocol):
    name: str
    # False for producers that have nothing to run in the background
    # (e.g. ManualProducer, which only exists so the dispatch surface is
    # uniform). Drives the "ready" vs "stopped" distinction in status().
    has_background_work: bool
    # Optional. True while the producer is blocked on an external call
    # (e.g. ``claude --print`` for MCP-backed producers). When True,
    # status() returns "polling" instead of "running" so the TUI can
    # show what we're actually doing. Producers that don't set this
    # default to False via ``getattr``.
    is_polling: bool

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

        State is one of:
        - ``polling`` — alive AND mid-external-call (``is_polling`` True)
        - ``running`` — alive, between calls
        - ``stopped`` — task ended
        - ``ready`` — no background work expected; task done is normal
        - ``idle`` — never started (start_all hasn't been called yet)
        """
        out: list[tuple[str, str]] = []
        for p in self._producers:
            task = self._tasks.get(p.name)
            has_bg = getattr(p, "has_background_work", True)
            if task is None:
                out.append((p.name, "ready" if not has_bg else "idle"))
            elif not task.done():
                if getattr(p, "is_polling", False):
                    out.append((p.name, "polling"))
                else:
                    out.append((p.name, "running"))
            else:
                out.append((p.name, "ready" if not has_bg else "stopped"))
        return out

    def __iter__(self):
        return iter(self._producers)
