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

import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable


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


# --- queue ----------------------------------------------------------------


class TaskQueue:
    """Thread-safe collection of tasks keyed by id.

    Producers call :meth:`add` from background threads; the consumer
    (voice loop) calls :meth:`peek` / :meth:`pull`. All public operations
    acquire an internal lock.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()
        self._listeners: list = []

    def add_listener(self, fn) -> None:
        """Subscribe to mutations: fn(event_kind: str, task: Task)."""
        with self._lock:
            self._listeners.append(fn)

    # ----- mutations ------------------------------------------------------

    def add(self, task: Task) -> Task:
        """Add a task, applying per-topic backpressure if needed."""
        with self._lock:
            self._tasks[task.id] = task
            self._enforce_topic_cap(task.topic)
        self._fire("add", task)
        return task

    def _enforce_topic_cap(self, topic: str) -> None:
        """Drop oldest pending tasks for ``topic`` beyond the soft cap.

        Called with ``_lock`` held. v1 just drops the oldest; an actual
        digest task is a follow-up. The state-change still propagates via
        the listener callback.
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
            self._fire_locked("drop", t)

    def set_state(self, task_id: str, state: str) -> Task | None:
        with self._lock:
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
        with self._lock:
            t = self._tasks.get(task_id)
            if t is None:
                return None
            t.ready_at = ts + max(0.0, seconds)
            t.state = STATE_PENDING
        self._fire("defer", t)
        return t

    # ----- reads ----------------------------------------------------------

    def pending(self) -> list[Task]:
        with self._lock:
            return [t for t in self._tasks.values() if t.state == STATE_PENDING]

    def all(self) -> list[Task]:
        with self._lock:
            return list(self._tasks.values())

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    def count_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for t in self._tasks.values():
                if t.state == STATE_PENDING:
                    out[t.kind] = out.get(t.kind, 0) + 1
        return out

    # ----- scoring views --------------------------------------------------

    def ranked(self, *, now: float, recent: RecentTopics) -> list[tuple[Task, float]]:
        """Return all pending tasks sorted by score descending."""
        pending = self.pending()
        scored = [(t, score(t, now=now, recent=recent)) for t in pending]
        scored = [(t, s) for t, s in scored if s > _NOT_READY]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored

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
        with self._lock:
            self._tasks = {t.id: t for t in tasks}

    # ----- listeners ------------------------------------------------------

    def _fire(self, kind: str, task: Task) -> None:
        # Snapshot listeners under the lock to avoid mutation during iteration.
        with self._lock:
            listeners = list(self._listeners)
        for fn in listeners:
            try:
                fn(kind, task)
            except Exception:
                # Listener errors must not corrupt queue state.
                pass

    def _fire_locked(self, kind: str, task: Task) -> None:
        # Variant used while already holding the lock (e.g. during cap
        # enforcement). Snapshots a copy of the listener list.
        listeners = list(self._listeners)
        # Release-and-reacquire pattern would be safer, but for our usage
        # listeners are cheap (just JSONL appends); inline-firing is fine.
        for fn in listeners:
            try:
                fn(kind, task)
            except Exception:
                pass
