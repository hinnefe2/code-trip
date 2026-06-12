"""Small asyncio helpers shared across the orchestrator.

Internal — the underscore prefix marks this as an implementation
detail of the package, not part of any public surface.
"""

from __future__ import annotations

import asyncio
import time


async def event_or_timeout(event: asyncio.Event, timeout: float) -> bool:
    """Wait for ``event`` for up to ``timeout`` seconds.

    Returns True if the event fired before the timeout, False if the
    timeout elapsed first. The common idiom in poll loops:

        if await event_or_timeout(self._stop, self._poll_interval):
            return  # stop signal received
    """
    try:
        await asyncio.wait_for(event.wait(), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        return False


def next_tick_delay(interval: float, *, now: float | None = None) -> float:
    """Seconds until the next wall-clock multiple of ``interval``.

    Producers that sleep on aligned ticks instead of free-running
    intervals fire at the same instants (slack@120 and email@120
    always coincide; linear@180 joins every lcm), which lets the MCP
    batcher coalesce their poll calls into one claude session.
    Returns a value in (0, interval].
    """
    ts = time.time() if now is None else now
    delay = interval - (ts % interval)
    # On an exact boundary (or within scheduler jitter of one), wait a
    # full interval rather than re-firing immediately.
    if delay < 0.05:
        delay = interval
    return delay
