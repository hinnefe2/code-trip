"""ManualProducer: voice-triggered manual task adds.

There's no background work here — the producer is just a thin helper
that gets called from the voice handler when the user says
``"add a task X"`` / ``"remind me to read this"``. Dispatch already lives
in :mod:`code_trip2.dispatch`; this module exists so the producer
surface is uniform and so future variants (auto-clip-from-clipboard,
hotkey-driven adds) have a home.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ManualProducer:
    """No-op run/stop; manual adds are pull-driven from dispatch."""

    name = "manual"
    has_background_work = False

    def request_stop(self) -> None:
        return

    async def run(self) -> None:
        return
