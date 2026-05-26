"""Unit tests for JSONL queue persistence + replay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from code_trip2.queue_log import QueueLog, _REPLAY_WINDOW_S
from code_trip2.tasks import STATE_DONE, RecentTopics, Task, TaskQueue


def test_record_appends_line_per_event(tmp_path: Path):
    log = QueueLog(dir_=tmp_path)
    q = TaskQueue()
    log.attach(q)
    t = q.add(Task(headline="hello"))
    q.mark_done(t.id)
    # Daily file should exist with 2 lines.
    files = sorted(tmp_path.glob("queue-*.jsonl"))
    assert len(files) == 1
    lines = [
        json.loads(line)
        for line in files[0].read_text().splitlines()
        if line.strip()
    ]
    assert [r["event"] for r in lines] == ["add", "state"]
    assert lines[0]["task"]["headline"] == "hello"


def test_replay_rebuilds_pending(tmp_path: Path):
    log = QueueLog(dir_=tmp_path)
    q1 = TaskQueue()
    log.attach(q1)
    a = q1.add(Task(headline="alive"))
    b = q1.add(Task(headline="finished"))
    q1.mark_done(b.id)

    # Fresh log instance to simulate restart.
    log2 = QueueLog(dir_=tmp_path)
    tasks = log2.replay()
    ids = {t.id for t in tasks}
    assert a.id in ids
    assert b.id not in ids  # terminal state filtered out


def test_replay_drops_old_events(tmp_path: Path):
    # Hand-write a stale event older than the 24h window.
    stale = {
        "ts": "2020-01-01T00:00:00Z",
        "event": "add",
        "task": {
            "id": "stale-1",
            "kind": "note",
            "topic": "inbox",
            "headline": "ancient",
            "body": None,
            "source": {},
            "created_at": time.time() - _REPLAY_WINDOW_S - 3600,
            "ready_at": 0.0,
            "urgency": "normal",
            "state": "pending",
            "parent_id": None,
        },
    }
    from datetime import datetime, timezone
    name = f"queue-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
    (tmp_path / name).write_text(json.dumps(stale) + "\n")

    log = QueueLog(dir_=tmp_path)
    assert log.replay() == []


def test_replay_demotes_active_to_pending(tmp_path: Path):
    """A task left ``state=active`` at shutdown (e.g. orchestrator was
    announcing it when the user hit Ctrl-C) should come back as
    ``pending`` on replay. ``current_task`` is in-memory only, so an
    active task with no live owner is invisible — pending makes it
    rejoin the queue."""
    log = QueueLog(dir_=tmp_path)
    q = TaskQueue()
    log.attach(q)
    t = q.add(Task(headline="was being announced"))
    q.mark_active(t.id)

    fresh = QueueLog(dir_=tmp_path).replay()
    assert len(fresh) == 1
    assert fresh[0].id == t.id
    assert fresh[0].state == "pending"


def test_record_scoring_writes_separate_file(tmp_path: Path):
    log = QueueLog(dir_=tmp_path)
    t = Task(headline="x")
    log.record_scoring(action="pull", chosen=t, ranked=[(t, 12.5)], recent=RecentTopics())
    files = sorted(tmp_path.glob("scoring-*.jsonl"))
    assert len(files) == 1
    rec = json.loads(files[0].read_text().strip())
    assert rec["event"] == "pull"
    assert rec["chosen_id"] == t.id
    assert rec["ranked"][0]["score"] == 12.5
