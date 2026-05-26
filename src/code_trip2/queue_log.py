"""JSONL persistence for the task queue and scoring events.

Two logs under ``~/.code-trip/queue/``:

- ``queue-YYYY-MM-DD.jsonl`` — every mutation (add / state-change / defer
  / drop / done). Replayed on startup to rebuild :class:`TaskQueue` state,
  filtering to events from the last 24 hours.
- ``scoring-YYYY-MM-DD.jsonl`` — scheduler decisions (peek / pull with
  score breakdowns). Read offline to tune scoring weights; not replayed.

Append-only. Inspectable with ``cat`` / ``jq``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from code_trip2.tasks import RecentTopics, Task, TaskQueue, score

logger = logging.getLogger(__name__)


_REPLAY_WINDOW_S = 24 * 60 * 60  # Drop events older than 24h on startup.


def default_queue_dir() -> Path:
    return Path.home() / ".code-trip" / "queue"


def _today_path(prefix: str, dir_: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return dir_ / f"{prefix}-{stamp}.jsonl"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


class QueueLog:
    """Append-only JSONL log for queue mutations + scoring decisions.

    Two file handles, two daily rotations. Writes are guarded by a lock so
    producer threads can call concurrently.
    """

    def __init__(self, dir_: Path | None = None) -> None:
        self.dir = dir_ or default_queue_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._queue_path = _today_path("queue", self.dir)
        self._scoring_path = _today_path("scoring", self.dir)

    # ----- queue mutations ------------------------------------------------

    def record(self, kind: str, task: Task) -> None:
        """Append a queue-mutation event."""
        rec = {"ts": _iso_now(), "event": kind, "task": task.to_dict()}
        self._append(self._queue_path, rec)

    def attach(self, queue: TaskQueue) -> None:
        """Wire this log up as a listener for queue mutations."""
        queue.add_listener(self.record)

    # ----- scoring --------------------------------------------------------

    def record_scoring(
        self,
        *,
        action: str,
        chosen: Task | None,
        ranked: list[tuple[Task, float]],
        recent: RecentTopics,
    ) -> None:
        """Append a scoring decision (peek/pull) with score breakdown."""
        rec = {
            "ts": _iso_now(),
            "event": action,
            "chosen_id": chosen.id if chosen else None,
            "recent_topics": recent.as_list(),
            "ranked": [
                {"id": t.id, "topic": t.topic, "kind": t.kind, "score": s}
                for t, s in ranked[:10]  # cap to avoid huge entries
            ],
        }
        self._append(self._scoring_path, rec)

    # ----- replay ---------------------------------------------------------

    def replay(self) -> list[Task]:
        """Reconstruct queue state from the most-recent queue log file(s).

        Reads up to the last 2 daily files (to handle the midnight boundary),
        applies events in order, and drops anything older than 24 hours by
        ``created_at``. Tasks marked done/dropped during the replay window
        are not returned — they're terminal.

        Tasks left in ``active`` state at shutdown are demoted back to
        ``pending``. The active state only means anything while the
        orchestrator is running and ``ctx.current_task`` points at it;
        across a restart there's no one "holding" the task, so leaving
        it active makes it invisible to both ``TaskQueue.pending()``
        and the TUI's current-task panel.
        """
        cutoff = time.time() - _REPLAY_WINDOW_S
        state: dict[str, Task] = {}
        for path in self._recent_queue_files():
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        td = rec.get("task")
                        if not isinstance(td, dict):
                            continue
                        try:
                            t = Task.from_dict(td)
                        except (KeyError, TypeError):
                            continue
                        if t.created_at < cutoff:
                            continue
                        state[t.id] = t
            except OSError:
                logger.warning("Could not read %s for replay", path, exc_info=True)
        out: list[Task] = []
        for t in state.values():
            if t.state == "active":
                t.state = "pending"
                out.append(t)
            elif t.state in ("pending", "snoozed"):
                out.append(t)
            # done / dropped are terminal — skip.
        return out

    def _recent_queue_files(self) -> list[Path]:
        try:
            files = sorted(
                p for p in self.dir.iterdir() if p.name.startswith("queue-")
            )
        except OSError:
            return []
        return files[-2:]  # today and yesterday is enough for a 24h window

    # ----- internals ------------------------------------------------------

    def _append(self, path: Path, record: dict) -> None:
        try:
            line = json.dumps(record, ensure_ascii=False) + "\n"
        except (TypeError, ValueError):
            logger.warning("Failed to encode log record", exc_info=True)
            return
        with self._lock:
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(line)
            except OSError:
                logger.warning("Failed to write %s", path, exc_info=True)


def ranked_with_logging(
    queue: TaskQueue,
    *,
    now: float,
    recent: RecentTopics,
    log: QueueLog | None,
    action: str,
) -> list[tuple[Task, float]]:
    """Compute ranked tasks and log the scoring decision if a log is given."""
    ranked = queue.ranked(now=now, recent=recent)
    chosen = ranked[0][0] if ranked else None
    if log is not None:
        log.record_scoring(action=action, chosen=chosen, ranked=ranked, recent=recent)
    return ranked
