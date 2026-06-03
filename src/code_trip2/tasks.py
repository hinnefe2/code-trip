"""Task queue: the unit of work for queue-mode interaction.

Mental model: an inbox. Producers push :class:`Task` objects in. The
consumer (the voice loop) pulls the highest-scoring pending task and
announces it. Scoring favors:

- tasks whose topic matches what the user has recently been working on
  (so unrelated work doesn't constantly interrupt focused threads)
- older tasks over newer (so nothing sits forever)
- ``interrupt`` urgency over normal; ``background`` is deprioritized

The queue is flat by design. ``Task.topic`` is a free-form string
(``"ticket-42"``, ``"slack-general"``, ``"inbox"``). A future tree
retrofit would add ``parent_id`` semantics; the field is already
reserved.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


# --- scoring constants ----------------------------------------------------
# First-pass guesses. The plan is to tune these from logged events offline,
# not in-process. See docs/task-queue-design.md for the methodology.

_BASE_AGE_PER_SECOND = 1.0
_TOPIC_AFFINITY_MAX = 1000.0
_TOPIC_AFFINITY_DECAY_S = 60.0
_URGENCY_INTERRUPT = 1_000_000.0
_URGENCY_BACKGROUND = -1_000_000.0
_NOT_READY = -math.inf
_NOT_PENDING = -math.inf

# Per-topic soft cap; older tasks beyond this collapse into a digest.
_PER_TOPIC_CAP = 5

# State values are strings (not an Enum) because they end up in JSONL logs
# and we want them human-readable without a custom encoder.
STATE_PENDING = "pending"
STATE_ACTIVE = "active"
STATE_SNOOZED = "snoozed"
STATE_DONE = "done"
STATE_DROPPED = "dropped"

URGENCY_INTERRUPT = "interrupt"
URGENCY_NORMAL = "normal"
URGENCY_BACKGROUND = "background"


@dataclass
class Task:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    kind: str = "note"
    topic: str = "inbox"
    headline: str = ""
    body: str | None = None
    source: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    ready_at: float = 0.0
    urgency: str = URGENCY_NORMAL
    state: str = STATE_PENDING
    parent_id: str | None = None
    # Canonical identifier for the real-world subject this task is
    # about (e.g. ``"linear:ENGAGE-3991"``). Producers populate it when
    # they can name the subject; tasks without one are singletons.
    # Used by :meth:`TaskQueue.ranked` to cluster cross-producer tasks
    # that refer to the same thing — so a Linear issue task and a
    # Gmail notification about a comment on that issue end up adjacent
    # in the queue. The key is a free-form string by convention namespaced
    # as ``<system>:<identifier>``.
    subject_key: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "topic": self.topic,
            "headline": self.headline,
            "body": self.body,
            "source": self.source,
            "created_at": self.created_at,
            "ready_at": self.ready_at,
            "urgency": self.urgency,
            "state": self.state,
            "parent_id": self.parent_id,
            "subject_key": self.subject_key,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            id=d["id"],
            kind=d.get("kind", "note"),
            topic=d.get("topic", "inbox"),
            headline=d.get("headline", ""),
            body=d.get("body"),
            source=d.get("source") or {},
            created_at=float(d.get("created_at", time.time())),
            ready_at=float(d.get("ready_at", 0.0)),
            urgency=d.get("urgency", URGENCY_NORMAL),
            state=d.get("state", STATE_PENDING),
            parent_id=d.get("parent_id"),
            subject_key=d.get("subject_key"),
        )


# --- scheduler state ------------------------------------------------------


@dataclass
class RecentTopics:
    """Most-recently-touched topics with timestamps.

    Used by :func:`score` to apply an affinity bonus to tasks tagged with a
    topic the user has been working on recently. Capped at a small N so the
    scheduler doesn't bias forever toward stale topics.
    """

    _entries: deque = field(default_factory=lambda: deque(maxlen=4))

    def touch(self, topic: str, *, now: float | None = None) -> None:
        if not topic:
            return
        ts = time.time() if now is None else now
        # Drop any prior entry for this topic so the most recent time wins.
        self._entries = deque(
            (t, when) for t, when in self._entries if t != topic
        )
        self._entries.append((topic, ts))
        # Re-cap; deque preserves maxlen only on append, not after slicing.
        while len(self._entries) > 4:
            self._entries.popleft()

    def best_match(self, topic: str) -> float | None:
        """Return the most-recent touch time for ``topic``, or None."""
        best: float | None = None
        for t, when in self._entries:
            if t == topic and (best is None or when > best):
                best = when
        return best

    def as_list(self) -> list[tuple[str, float]]:
        return list(self._entries)


# --- scoring --------------------------------------------------------------


def score(task: Task, *, now: float, recent: RecentTopics) -> float:
    """Rank a task. Higher is more important. Pure function."""
    if task.state != STATE_PENDING:
        return _NOT_PENDING
    if task.ready_at > now:
        return _NOT_READY

    age = max(0.0, now - task.created_at)
    s = age * _BASE_AGE_PER_SECOND

    last = recent.best_match(task.topic)
    if last is not None:
        elapsed = max(0.0, now - last)
        s += _TOPIC_AFFINITY_MAX * math.exp(-elapsed / _TOPIC_AFFINITY_DECAY_S)

    if task.urgency == URGENCY_INTERRUPT:
        s += _URGENCY_INTERRUPT
    elif task.urgency == URGENCY_BACKGROUND:
        s += _URGENCY_BACKGROUND

    return s


def _cluster_by_subject(
    scored: list[tuple[Task, float]],
) -> list[tuple[Task, float]]:
    """Reorder a score-descending list so tasks sharing a
    ``subject_key`` cluster adjacent.

    Assumes ``scored`` is already sorted by score descending — that
    guarantee lets us pick cluster ordering by first-arrival without a
    second sort. The first time a subject_key is seen, the cluster's
    top score is fixed (it's that task's score); subsequent members
    are appended in score order. Singleton tasks (``subject_key`` is
    ``None``) slot in by score relative to the cluster heads they sit
    between.
    """
    clusters: dict[object, list[tuple[Task, float]]] = {}
    cluster_order: list[object] = []
    singleton_counter = 0
    for pair in scored:
        sk = pair[0].subject_key
        if sk:
            key: object = sk
        else:
            # Each subject_key-less task is its own cluster so it
            # keeps its score-determined position.
            key = ("__singleton__", singleton_counter)
            singleton_counter += 1
        if key not in clusters:
            clusters[key] = []
            cluster_order.append(key)
        clusters[key].append(pair)
    out: list[tuple[Task, float]] = []
    for key in cluster_order:
        out.extend(clusters[key])
    return out


# --- queue ----------------------------------------------------------------


class TaskQueue:
    """Collection of tasks keyed by id.

    Single-event-loop discipline replaces the previous ``threading.Lock``:
    every public method is a non-awaiting compute body, so they're atomic
    with respect to other coroutines on the loop. Tests can also drive
    the queue synchronously without a loop at all.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._listeners: list = []

    def add_listener(self, fn) -> None:
        """Subscribe to mutations: ``fn(event_kind: str, task: Task)``.

        Listeners must be sync. If a listener needs to do async work, it
        should call ``asyncio.create_task(...)`` itself — keeping that
        explicit is clearer than having the queue auto-schedule.
        """
        self._listeners.append(fn)

    # ----- mutations ------------------------------------------------------

    def add(self, task: Task) -> Task:
        """Add a task, applying per-topic backpressure if needed."""
        self._tasks[task.id] = task
        self._enforce_topic_cap(task.topic)
        self._fire("add", task)
        return task

    def _enforce_topic_cap(self, topic: str) -> None:
        """Drop oldest pending tasks for ``topic`` beyond the soft cap.

        v1 just drops the oldest; an actual digest task is a follow-up.
        The state-change still propagates via the listener callback.
        """
        pending = [
            t for t in self._tasks.values()
            if t.topic == topic and t.state == STATE_PENDING
        ]
        if len(pending) <= _PER_TOPIC_CAP:
            return
        pending.sort(key=lambda t: t.created_at)
        overflow = pending[: len(pending) - _PER_TOPIC_CAP]
        for t in overflow:
            t.state = STATE_DROPPED
            self._fire("drop", t)

    def update_task(
        self,
        task_id: str,
        *,
        headline: str | None = None,
        body: str | None = None,
        source: dict | None = None,
        created_at: float | None = None,
    ) -> Task | None:
        """Mutate fields on an existing task and fire an ``update`` event.

        Used by producers that collapse a stream of messages in the same
        thread into a single live task (e.g. SlackProducer): when a new
        message arrives for an already-pending thread task, the producer
        rewrites the body/headline rather than queueing a duplicate task.
        """
        t = self._tasks.get(task_id)
        if t is None:
            return None
        if headline is not None:
            t.headline = headline
        if body is not None:
            t.body = body
        if source is not None:
            t.source = source
        if created_at is not None:
            t.created_at = created_at
        self._fire("update", t)
        return t

    def set_state(self, task_id: str, state: str) -> Task | None:
        t = self._tasks.get(task_id)
        if t is None:
            return None
        t.state = state
        self._fire("state", t)
        return t

    def mark_active(self, task_id: str) -> Task | None:
        return self.set_state(task_id, STATE_ACTIVE)

    def mark_done(self, task_id: str) -> Task | None:
        return self.set_state(task_id, STATE_DONE)

    def mark_dropped(self, task_id: str) -> Task | None:
        return self.set_state(task_id, STATE_DROPPED)

    def defer(self, task_id: str, seconds: float, *, now: float | None = None) -> Task | None:
        ts = time.time() if now is None else now
        t = self._tasks.get(task_id)
        if t is None:
            return None
        t.ready_at = ts + max(0.0, seconds)
        t.state = STATE_PENDING
        self._fire("defer", t)
        return t

    # ----- reads ----------------------------------------------------------

    def pending(self) -> list[Task]:
        return [t for t in self._tasks.values() if t.state == STATE_PENDING]

    def all(self) -> list[Task]:
        return list(self._tasks.values())

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def count_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self._tasks.values():
            if t.state == STATE_PENDING:
                out[t.kind] = out.get(t.kind, 0) + 1
        return out

    # ----- scoring views --------------------------------------------------

    def ranked(self, *, now: float, recent: RecentTopics) -> list[tuple[Task, float]]:
        """Return all pending tasks sorted by score descending, then
        clustered so tasks sharing a ``subject_key`` are adjacent.

        Cluster placement is by the cluster's top-scoring member, so the
        head of the list is still the global top-score task (``peek`` /
        ``pull`` semantics unchanged). The trade-off is that the second
        row may be a lower-scored sibling instead of the next-highest
        unrelated task — that's the point of clustering.
        """
        pending = self.pending()
        scored = [(t, score(t, now=now, recent=recent)) for t in pending]
        scored = [(t, s) for t, s in scored if s > _NOT_READY]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return _cluster_by_subject(scored)

    def peek(self, *, now: float, recent: RecentTopics) -> Task | None:
        scored = self.ranked(now=now, recent=recent)
        return scored[0][0] if scored else None

    def pull(self, *, now: float, recent: RecentTopics) -> Task | None:
        """Mark the highest-scoring pending task active and return it."""
        t = self.peek(now=now, recent=recent)
        if t is None:
            return None
        self.set_state(t.id, STATE_ACTIVE)
        return t

    # ----- bulk load (for replay) -----------------------------------------

    def load(self, tasks: Iterable[Task]) -> None:
        """Replace contents wholesale. Used by JSONL replay on startup."""
        self._tasks = {t.id: t for t in tasks}

    # ----- listeners ------------------------------------------------------

    def _fire(self, kind: str, task: Task) -> None:
        # Snapshot so a listener that itself mutates the listener list
        # (none today, but cheap insurance) can't break this iteration.
        for fn in list(self._listeners):
            try:
                fn(kind, task)
            except Exception:
                # Listener errors must not corrupt queue state.
                logger.exception("Listener %r failed for %s event", fn, kind)
