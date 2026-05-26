"""Small asyncio helpers shared across the orchestrator.

Internal — the underscore prefix marks this as an implementation
detail of the package, not part of any public surface.
"""

from __future__ import annotations

import asyncio


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
