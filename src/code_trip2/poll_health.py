"""Consecutive-poll-failure surfacing.

A failed producer poll is a WARNING log line and a retry next tick —
right for transients, but it silently ate a 10-day Gmail outage
(expired claude CLI auth, 2026-07-24 → 2026-08-03): every poll failed,
nothing surfaced, and a week of email tasks was simply missing until
the user went digging in the logs.

:class:`PollHealth` counts consecutive failures per producer. Past a
threshold it puts ONE ``producer_health`` task on the queue — the
outage shows up exactly where the user already looks, at interrupt
urgency so it sorts to the front. The first successful poll afterwards
retires the task via ``resolve_by_origin`` and resets the streak, so
recovery needs no user action; a task the user dismissed mid-outage is
not re-spawned until the producer has actually recovered once
(``_surfaced`` stays latched for the rest of the streak).

Tasks are added straight to the queue, not through the screening gate:
no skill handles this kind, and during the exact failure being
reported the screener's own ``claude --print`` calls are likely broken
too.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from code_trip2.tasks import URGENCY_INTERRUPT, Task, TaskQueue

logger = logging.getLogger(__name__)


def _origin_key(producer: str) -> str:
    return f"health:{producer}"


@dataclass
class PollHealth:
    """Per-producer consecutive-failure counter with queue surfacing."""

    producer: str
    queue: TaskQueue
    threshold: int = 10

    _failures: int = field(default=0, init=False)
    _streak_started: float = field(default=0.0, init=False)
    _surfaced: bool = field(default=False, init=False)

    def record_failure(self, error: str) -> None:
        if self._failures == 0:
            self._streak_started = time.time()
        self._failures += 1
        if self._failures < self.threshold or self._surfaced:
            return
        self._surfaced = True
        minutes = max(1, int((time.time() - self._streak_started) / 60))
        logger.warning(
            "PollHealth: %s crossed %d consecutive poll failures; "
            "surfacing health task",
            self.producer, self._failures,
        )
        self.queue.upsert(Task(
            kind="producer_health",
            topic="health",
            headline=(
                f"{self.producer} polling down — "
                f"{self._failures} straight failures (~{minutes} min)"
            ),
            body=(
                f"Every {self.producer} poll for the last ~{minutes} minutes "
                f"has failed. Latest error: {error}\n"
                "New items from this source are NOT reaching the queue. "
                "Check ~/.code-trip/logs/orchestrator.log; if the error is "
                "'claude exited 1', re-authenticating the claude CLI usually "
                "fixes it. This task clears itself when polling recovers."
            ),
            urgency=URGENCY_INTERRUPT,
            origin_key=_origin_key(self.producer),
        ))

    def record_success(self) -> None:
        if self._surfaced:
            if self.queue.resolve_by_origin(_origin_key(self.producer)) is not None:
                logger.info(
                    "PollHealth: %s recovered after %d failures; retired "
                    "health task", self.producer, self._failures,
                )
        self._failures = 0
        self._surfaced = False
