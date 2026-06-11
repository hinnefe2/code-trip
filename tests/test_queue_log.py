"""Unit tests for JSONL queue persistence + replay."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from datetime import datetime, timedelta, timezone

from code_trip2.queue_log import _DURABLE_KINDS, QueueLog, _REPLAY_WINDOW_S
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


def _stale_record(kind: str, task_id: str = "stale-1") -> dict:
    return {
        "ts": "2020-01-01T00:00:00Z",
        "event": "add",
        "task": {
            "id": task_id,
            "kind": kind,
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


def _write_log_for_date(dir_: Path, date: datetime, records: list[dict]) -> Path:
    path = dir_ / f"queue-{date.strftime('%Y-%m-%d')}.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def test_replay_drops_old_events_for_non_durable_kinds(tmp_path: Path):
    """Non-durable kinds (slack_msg, etc.) older than 24h still drop —
    the cutoff only relaxes for the explicitly-durable set."""
    _write_log_for_date(
        tmp_path, datetime.now(timezone.utc), [_stale_record("slack_msg")],
    )
    log = QueueLog(dir_=tmp_path)
    assert log.replay() == []


def test_replay_keeps_durable_kinds_past_cutoff(tmp_path: Path):
    """Meeting follow-ups and notes outlive the 24h cutoff — the queue
    log IS the source of truth for them, so they shouldn't silently
    vanish across restarts."""
    records = [
        _stale_record("meeting_followup", task_id="followup-1"),
        _stale_record("note", task_id="note-1"),
    ]
    _write_log_for_date(tmp_path, datetime.now(timezone.utc), records)
    log = QueueLog(dir_=tmp_path)
    surviving = {t.id for t in log.replay()}
    assert surviving == {"followup-1", "note-1"}


def test_durable_kinds_constant_covers_expected_kinds():
    """Lock the set so a future kind addition forces an intentional
    decision about durability — easy thing to forget."""
    assert _DURABLE_KINDS == frozenset({"meeting_followup", "note"})


def test_replay_reads_files_older_than_yesterday_for_durable_tasks(tmp_path: Path):
    """The old ``[-2:]`` file window made meeting_followups invisible
    after 36-ish hours regardless of the cutoff. Replay should scan
    further back so durable tasks from a week ago still load."""
    week_ago = datetime.now(timezone.utc) - timedelta(days=7)
    rec = _stale_record("meeting_followup", task_id="week-old")
    # Re-anchor created_at to "7 days ago" so the durable-kind path is
    # what saves it, not happenstance.
    rec["task"]["created_at"] = time.time() - 7 * 24 * 60 * 60
    _write_log_for_date(tmp_path, week_ago, [rec])
    # Add a few intermediate empty files so this isn't trivially the
    # most-recent file.
    for offset in (1, 3, 5):
        d = datetime.now(timezone.utc) - timedelta(days=offset)
        _write_log_for_date(tmp_path, d, [])

    log = QueueLog(dir_=tmp_path)
    tasks = log.replay()
    assert [t.id for t in tasks] == ["week-old"]


def test_replay_honors_terminal_state_in_older_files(tmp_path: Path):
    """A meeting_followup created last week and marked done last week
    should NOT come back — terminal state is honored even when reading
    older files."""
    last_week = datetime.now(timezone.utc) - timedelta(days=5)
    rec_add = _stale_record("meeting_followup", task_id="done-task")
    rec_add["task"]["created_at"] = time.time() - 5 * 24 * 60 * 60
    rec_done = json.loads(json.dumps(rec_add))  # deep copy
    rec_done["event"] = "state"
    rec_done["task"]["state"] = "done"
    _write_log_for_date(tmp_path, last_week, [rec_add, rec_done])

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
