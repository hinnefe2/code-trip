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
STATE_SCREENING = "screening"
STATE_DONE = "done"
STATE_DROPPED = "dropped"

# States where the task is still "the" task for its real-world object:
# a re-sighting of the same object updates it in place rather than
# minting a sibling. Terminal states (done/dropped) are the complement.
LIVE_STATES = frozenset(
    {STATE_PENDING, STATE_ACTIVE, STATE_SNOOZED, STATE_SCREENING}
)

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
    # Identity of the real-world object that produced this task —
    # "which thing is this", where ``subject_key`` is "what is it
    # about". Producer-owned and namespaced: ``email:<thread_id>``,
    # ``slack:<channel_id>:<thread_ts>``, ``linear:<IDENTIFIER>``,
    # ``claude:<window>``, ``followup:<thread_id>:<headline>``. Unlike
    # ``id`` (a fresh uuid every mint) it is derived from durable
    # content, so re-emits and restarts produce the same key.
    # :meth:`TaskQueue.upsert` uses it to collapse re-sightings into
    # the live task and to suppress or re-surface after a terminal
    # one. ``None`` (manual notes) means singleton — never collapsed.
    origin_key: str | None = None

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
            "origin_key": self.origin_key,
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
            # Legacy JSONL records wrote the field as ``dedup_key``.
            origin_key=d.get("origin_key") or d.get("dedup_key"),
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
        # origin_key -> id of the most recent task minted for that key.
        # The single identity authority: producers resolve "is there
        # already a task for this object?" through this index (via
        # ``upsert`` / ``get_by_origin``) instead of scanning the queue.
        self._origin: dict[str, str] = {}
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
        """Add a task unconditionally, applying per-topic backpressure.

        No identity checks — callers that want collapse/suppression
        semantics go through :meth:`upsert`. ``add`` still registers
        the task's ``origin_key`` in the index so a later ``upsert``
        for the same object finds it.
        """
        self._tasks[task.id] = task
        if task.origin_key:
            self._origin[task.origin_key] = task.id
        self._enforce_topic_cap(task.topic)
        self._fire("add", task)
        return task

    def upsert(self, task: Task, *, if_terminal: str = "new") -> Task:
        """Add ``task`` unless a task for the same ``origin_key`` exists.

        The one collapse/suppress/re-surface rule for all producers:

        - no ``origin_key`` → plain :meth:`add` (singleton).
        - existing task is live (pending / active / snoozed /
          screening) → update it in place (headline, body, source,
          subject_key, created_at) and return it; never a sibling.
          ``created_at`` deliberately refreshes: post-cursor-fix a
          re-sighting means the object genuinely changed, so its age
          resets.
        - existing task is terminal (done / dropped):
          ``if_terminal="new"`` mints a fresh task (the object came
          back — e.g. a reopened ticket); ``if_terminal="skip"``
          returns the terminal task unchanged, no event (a respawned
          follow-up the user already filed or dismissed).

        Returns the resident task — the caller can test ``result is
        task`` to learn whether the passed task actually entered.
        """
        key = task.origin_key
        if key is None:
            return self.add(task)
        existing_id = self._origin.get(key)
        existing = self._tasks.get(existing_id) if existing_id else None
        if existing is None:
            return self.add(task)
        if existing.state in LIVE_STATES:
            existing.headline = task.headline
            existing.body = task.body
            existing.source = task.source
            existing.subject_key = task.subject_key
            existing.created_at = task.created_at
            self._fire("update", existing)
            return existing
        if if_terminal == "skip":
            return existing
        return self.add(task)  # re-surface fresh; add() re-points the index

    def get_by_origin(self, origin_key: str) -> Task | None:
        """Most recent task minted for ``origin_key``, any state.

        For producers whose update-in-place is richer than ``upsert``'s
        field overwrite (Slack merges message histories) and need to
        look before they merge.
        """
        task_id = self._origin.get(origin_key)
        return self._tasks.get(task_id) if task_id else None

    def resolve_by_origin(self, origin_key: str) -> Task | None:
        """The source says the object is resolved — retire its task.

        Marks done iff the task is pending or snoozed and returns it;
        otherwise returns None. ACTIVE is deliberately untouched: if
        the user is mid-conversation with a task whose object just
        closed under them, yanking it away is worse than letting the
        stale state linger until they finish. SCREENING is also left
        alone — the screener will surface or retire it on its own.
        """
        t = self.get_by_origin(origin_key)
        if t is not None and t.state in (STATE_PENDING, STATE_SNOOZED):
            self.mark_done(t.id)
            return t
        return None

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
        prior = t.state
        t.state = state
        self._fire("state", t)
        # A task leaving screening only now counts toward its topic's
        # pending cap — enforce on release or the cap can be exceeded
        # by however many siblings were in screening at once.
        if prior == STATE_SCREENING and state == STATE_PENDING:
            self._enforce_topic_cap(t.topic)
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
        # Rebuild the origin index. When a key maps to both a terminal
        # record (kept for cross-restart follow-up suppression) and a
        # live task, the live one wins.
        self._origin = {}
        for t in self._tasks.values():
            if not t.origin_key:
                continue
            cur = self._tasks.get(self._origin.get(t.origin_key, ""))
            if cur is None or cur.state not in LIVE_STATES:
                self._origin[t.origin_key] = t.id

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
