"""ReconcileProducer: periodically retires queued tasks handled elsewhere.

The other producers are *additive* — each poll only pushes new work into
the queue; none of them (except LinearProducer, which retires closed
tickets on its own incremental poll) walks the *standing* queue to check
whether an already-queued task has since been dealt with outside the
orchestrator. That gap is why the queue accumulates stale entries: you
archive a mail in Gmail, close a thread in Slack, and the task lingers
until you hand-dismiss it.

This producer closes the gap. On a timer (``reconcile_interval``) it runs
each registered reconciler — plain async callables that re-check a
source's current state against the pending queue and ``mark_done`` the
tasks that no longer belong. It owns no source state of its own; the
reconcilers (e.g. :meth:`EmailProducer.reconcile_inbox`) do the work and
report how many tasks they retired.

Satisfies the :class:`~code_trip2.producers.Producer` protocol so
:class:`ProducerSupervisor` starts and stops it like any other producer.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Sequence

from code_trip2 import config as config_mod
from code_trip2._async_utils import event_or_timeout, next_tick_delay
from code_trip2.config import Config

logger = logging.getLogger(__name__)

# A reconciler re-checks one source against the queue and returns how many
# pending tasks it retired this pass.
Reconciler = Callable[[], Awaitable[int]]


class ReconcileProducer:
    name = "reconcile"
    has_background_work = True

    # Initial stagger so the first reconcile doesn't collide with the
    # producers' own startup polls (and their MCP calls).
    _STARTUP_DELAY_S = 30.0

    def __init__(
        self,
        *,
        config: Config,
        reconcilers: Sequence[Reconciler],
    ) -> None:
        self._config = config
        self._reconcilers = tuple(reconcilers)
        self._stop = asyncio.Event()
        self.is_polling = False

    def request_stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        if not self._reconcilers:
            logger.info("ReconcileProducer: no reconcilers registered; not starting.")
            return
        if await event_or_timeout(self._stop, self._STARTUP_DELAY_S):
            return
        was_active = True
        while not self._stop.is_set():
            if config_mod.polling_active(self._config):
                if not was_active:
                    logger.info("ReconcileProducer: active hours resumed; reconciling")
                    was_active = True
                await self._reconcile_once()
            elif was_active:
                logger.info("ReconcileProducer: outside active hours; reconcile paused")
                was_active = False
            delay = next_tick_delay(self._config.reconcile_interval)
            if await event_or_timeout(self._stop, delay):
                return

    async def _reconcile_once(self) -> None:
        self.is_polling = True
        total = 0
        try:
            for reconcile in self._reconcilers:
                try:
                    total += await reconcile()
                except Exception:
                    logger.exception("ReconcileProducer: a reconciler raised; continuing")
        finally:
            self.is_polling = False
        if total:
            logger.info("ReconcileProducer: retired %d stale task(s) this pass", total)
