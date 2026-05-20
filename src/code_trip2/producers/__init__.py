"""Background producers that push :class:`Task` objects into the queue.

Each producer runs on its own daemon thread and is owned by ``main.py``.
The public surface is:

- :class:`Producer` — protocol any producer satisfies (``start`` / ``stop``)
- :class:`ProducerSupervisor` — starts/stops a collection of producers

Concrete producers live in sibling modules:

- :mod:`code_trip2.producers.claude`  — watches the Stop-hook event dir
- :mod:`code_trip2.producers.manual`  — voice-triggered manual adds
- :mod:`code_trip2.producers.slack`   — Slack MCP (stub)
- :mod:`code_trip2.producers.linear`  — Linear MCP (stub)
"""

from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class Producer(Protocol):
    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...


class ProducerSupervisor:
    """Owns a set of producers; starts and stops them together."""

    def __init__(self) -> None:
        self._producers: list[Producer] = []

    def add(self, producer: Producer) -> None:
        self._producers.append(producer)

    def start_all(self) -> None:
        for p in self._producers:
            try:
                p.start()
                logger.info("Started producer %s", p.name)
            except Exception:
                logger.exception("Failed to start producer %s", p.name)

    def stop_all(self) -> None:
        for p in self._producers:
            try:
                p.stop()
            except Exception:
                logger.exception("Failed to stop producer %s", p.name)

    def status(self) -> list[tuple[str, str]]:
        """Best-effort ``(name, state)`` per producer for the TUI / debug.

        State is one of ``running`` (thread alive), ``stopped`` (thread
        existed but exited), ``ready`` (manual-style producer with no
        background work), or ``idle`` (never started, e.g. because the
        relevant config was empty).
        """
        out: list[tuple[str, str]] = []
        for p in self._producers:
            thread = getattr(p, "_thread", None)
            if thread is None:
                out.append((p.name, "ready" if p.name == "manual" else "idle"))
            elif thread.is_alive():
                out.append((p.name, "running"))
            else:
                out.append((p.name, "stopped"))
        return out

    def __iter__(self):
        return iter(self._producers)
