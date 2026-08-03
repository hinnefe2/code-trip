"""Tests for PollHealth: consecutive-poll-failure surfacing."""

from __future__ import annotations

from code_trip2.poll_health import PollHealth, _origin_key
from code_trip2.tasks import URGENCY_INTERRUPT, TaskQueue


def _health(threshold: int = 3) -> tuple[PollHealth, TaskQueue]:
    q = TaskQueue()
    return PollHealth(producer="email", queue=q, threshold=threshold), q


def test_below_threshold_stays_silent():
    h, q = _health(threshold=3)
    h.record_failure("boom")
    h.record_failure("boom")
    assert q.get_by_origin(_origin_key("email")) is None


def test_crossing_threshold_surfaces_one_interrupt_task():
    h, q = _health(threshold=3)
    for _ in range(3):
        h.record_failure("claude exited 1")
    t = q.get_by_origin(_origin_key("email"))
    assert t is not None
    assert t.kind == "producer_health"
    assert t.urgency == URGENCY_INTERRUPT
    assert "claude exited 1" in (t.body or "")
    assert "email" in t.headline


def test_continued_failures_do_not_duplicate():
    h, q = _health(threshold=2)
    for _ in range(10):
        h.record_failure("boom")
    assert sum(1 for t in q.pending() if t.kind == "producer_health") == 1


def test_success_retires_task_and_resets_streak():
    h, q = _health(threshold=2)
    h.record_failure("boom")
    h.record_failure("boom")
    h.record_success()
    t = q.get_by_origin(_origin_key("email"))
    assert t.state == "done"
    # A fresh streak must re-cross the full threshold before surfacing.
    h.record_failure("boom")
    assert q.get_by_origin(_origin_key("email")).state == "done"
    h.record_failure("boom")
    assert q.get_by_origin(_origin_key("email")).state == "pending"


def test_user_dismissal_mid_outage_is_not_respawned():
    """A health task the user marked done stays done for the rest of the
    streak — no nagging while the outage persists."""
    h, q = _health(threshold=2)
    h.record_failure("boom")
    h.record_failure("boom")
    task = q.get_by_origin(_origin_key("email"))
    q.mark_done(task.id)
    h.record_failure("boom")
    h.record_failure("boom")
    assert q.get_by_origin(_origin_key("email")).state == "done"


def test_success_without_surfacing_is_a_noop():
    h, q = _health(threshold=3)
    h.record_failure("boom")
    h.record_success()
    assert q.get_by_origin(_origin_key("email")) is None
