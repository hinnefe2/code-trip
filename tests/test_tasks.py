"""Unit tests for the task queue: Task, TaskQueue, RecentTopics, score()."""

from __future__ import annotations

import time

import pytest

from code_trip2.tasks import (
    STATE_DONE,
    STATE_DROPPED,
    STATE_PENDING,
    URGENCY_BACKGROUND,
    URGENCY_INTERRUPT,
    RecentTopics,
    Task,
    TaskQueue,
    score,
)


# --- Task --------------------------------------------------------------------


def test_task_defaults_unique_ids():
    a = Task()
    b = Task()
    assert a.id != b.id
    assert a.kind == "note"
    assert a.topic == "inbox"
    assert a.state == STATE_PENDING


def test_task_roundtrip_dict():
    t = Task(kind="claude_reply", topic="ticket-42", headline="ready", body="ok")
    d = t.to_dict()
    back = Task.from_dict(d)
    assert back.id == t.id
    assert back.kind == "claude_reply"
    assert back.topic == "ticket-42"
    assert back.headline == "ready"
    assert back.body == "ok"
    assert back.state == STATE_PENDING


# --- RecentTopics ------------------------------------------------------------


def test_recent_topics_touch_overwrites_prior():
    r = RecentTopics()
    r.touch("a", now=100.0)
    r.touch("a", now=200.0)
    assert r.best_match("a") == 200.0


def test_recent_topics_returns_none_for_unseen():
    r = RecentTopics()
    r.touch("a", now=100.0)
    assert r.best_match("b") is None


# --- score -------------------------------------------------------------------


def test_score_pending_age_wins_over_younger():
    now = 1000.0
    old = Task(topic="t", created_at=now - 100)
    young = Task(topic="t", created_at=now - 10)
    r = RecentTopics()
    assert score(old, now=now, recent=r) > score(young, now=now, recent=r)


def test_score_not_pending_is_minus_infinity():
    now = 1000.0
    t = Task(state=STATE_DONE, created_at=now - 100)
    assert score(t, now=now, recent=RecentTopics()) == float("-inf")


def test_score_not_ready_is_filtered():
    now = 1000.0
    t = Task(created_at=now - 100, ready_at=now + 60)
    assert score(t, now=now, recent=RecentTopics()) == float("-inf")


def test_score_topic_affinity_boosts_match():
    now = 1000.0
    r = RecentTopics()
    r.touch("hot", now=now - 5)
    on_topic = Task(topic="hot", created_at=now - 10)
    off_topic = Task(topic="cold", created_at=now - 10)
    assert score(on_topic, now=now, recent=r) > score(off_topic, now=now, recent=r)


def test_score_topic_affinity_decays_over_time():
    now = 1000.0
    fresh = RecentTopics()
    fresh.touch("hot", now=now - 1)
    stale = RecentTopics()
    stale.touch("hot", now=now - 300)
    t = Task(topic="hot", created_at=now - 10)
    assert score(t, now=now, recent=fresh) > score(t, now=now, recent=stale)


def test_score_interrupt_dominates():
    now = 1000.0
    urgent = Task(topic="x", created_at=now - 1, urgency=URGENCY_INTERRUPT)
    old = Task(topic="x", created_at=now - 10_000)
    assert score(urgent, now=now, recent=RecentTopics()) > score(
        old, now=now, recent=RecentTopics()
    )


def test_score_background_is_penalized():
    now = 1000.0
    bg = Task(topic="x", created_at=now - 1000, urgency=URGENCY_BACKGROUND)
    normal = Task(topic="x", created_at=now - 1)
    assert score(normal, now=now, recent=RecentTopics()) > score(
        bg, now=now, recent=RecentTopics()
    )


# --- TaskQueue ---------------------------------------------------------------


def test_queue_add_and_pending():
    q = TaskQueue()
    t = q.add(Task(headline="x"))
    assert q.pending() == [t]


def test_queue_ranked_orders_by_score():
    q = TaskQueue()
    now = 1000.0
    a = Task(topic="t", headline="a", created_at=now - 10)
    b = Task(topic="t", headline="b", created_at=now - 100)
    q.add(a)
    q.add(b)
    ranked = q.ranked(now=now, recent=RecentTopics())
    assert ranked[0][0].id == b.id  # older wins on age


def test_queue_ranked_clusters_tasks_by_subject_key():
    """A Linear task and a Linear-notification email for the same issue
    should end up adjacent in the ranked output, even when an unrelated
    higher-aged task would otherwise sit between them."""
    q = TaskQueue()
    now = 1000.0
    sk = "linear:ENGAGE-3991"
    # Order in queue: linear is oldest (top score), unrelated is middle,
    # email about the same issue is newest. Without clustering we'd see
    # [linear, unrelated, email]; with clustering [linear, email, unrelated].
    linear = q.add(Task(
        kind="linear_issue", topic="engage-3991", headline="L",
        created_at=now - 300, subject_key=sk,
    ))
    unrelated = q.add(Task(
        kind="note", topic="other", headline="U", created_at=now - 200,
    ))
    email = q.add(Task(
        kind="email_msg", topic="email-foo", headline="E",
        created_at=now - 100, subject_key=sk,
    ))
    out = q.ranked(now=now, recent=RecentTopics())
    ids = [t.id for t, _ in out]
    assert ids == [linear.id, email.id, unrelated.id]


def test_queue_ranked_singletons_keep_score_order():
    """Tasks without subject_key shouldn't be reshuffled — they slot in
    by score relative to each cluster's head."""
    q = TaskQueue()
    now = 1000.0
    older = q.add(Task(topic="t1", headline="older", created_at=now - 300))
    newer = q.add(Task(topic="t2", headline="newer", created_at=now - 10))
    out = q.ranked(now=now, recent=RecentTopics())
    assert [t.id for t, _ in out] == [older.id, newer.id]


def test_queue_ranked_cluster_position_uses_top_member_score():
    """The cluster sits where its hottest member would, not where its
    coolest member would. Singletons cluster around it accordingly."""
    q = TaskQueue()
    now = 1000.0
    sk = "linear:AI-42"
    # Highest-scoring (oldest) singleton wins the head; cluster's head
    # member is second; the cluster's lower-scoring sibling lands
    # adjacent rather than after the third singleton.
    head = q.add(Task(topic="h", headline="head", created_at=now - 500))
    cluster_top = q.add(Task(
        kind="linear_issue", topic="ai-42", headline="ct",
        created_at=now - 400, subject_key=sk,
    ))
    middle = q.add(Task(topic="m", headline="mid", created_at=now - 300))
    cluster_tail = q.add(Task(
        kind="email_msg", topic="email-x", headline="cl-tail",
        created_at=now - 50, subject_key=sk,
    ))
    out = q.ranked(now=now, recent=RecentTopics())
    assert [t.id for t, _ in out] == [
        head.id, cluster_top.id, cluster_tail.id, middle.id,
    ]


def test_queue_pull_marks_active_and_returns_top():
    q = TaskQueue()
    t = q.add(Task(headline="x"))
    out = q.pull(now=time.time(), recent=RecentTopics())
    assert out is t
    assert q.get(t.id).state == "active"
    assert q.pending() == []


def test_queue_defer_resets_pending_with_future_ready_at():
    q = TaskQueue()
    t = q.add(Task(headline="x"))
    q.mark_active(t.id)
    q.defer(t.id, 60.0, now=1000.0)
    assert q.get(t.id).state == STATE_PENDING
    assert q.get(t.id).ready_at == 1060.0
    # Score should now exclude it.
    assert score(q.get(t.id), now=1000.0, recent=RecentTopics()) == float("-inf")


def test_queue_topic_cap_drops_oldest_pending():
    q = TaskQueue()
    # Default cap is 5. Add 7, oldest 2 should be dropped.
    for i in range(7):
        q.add(Task(topic="t", headline=str(i), created_at=float(i)))
    pending = q.pending()
    assert len(pending) == 5
    pending_headlines = sorted(p.headline for p in pending)
    assert pending_headlines == ["2", "3", "4", "5", "6"]
    dropped = [t for t in q.all() if t.state == STATE_DROPPED]
    assert len(dropped) == 2


def test_queue_listener_fires_on_add():
    q = TaskQueue()
    events: list[tuple[str, str]] = []
    q.add_listener(lambda kind, t: events.append((kind, t.id)))
    t = q.add(Task())
    assert events == [("add", t.id)]


def test_queue_listener_fires_on_state_change():
    q = TaskQueue()
    t = q.add(Task())
    events: list[str] = []
    q.add_listener(lambda kind, _t: events.append(kind))
    q.mark_done(t.id)
    assert events == ["state"]


def test_queue_load_replaces_state():
    q = TaskQueue()
    q.add(Task())
    q.add(Task())
    fresh = [Task(), Task(), Task()]
    q.load(fresh)
    assert {t.id for t in q.all()} == {t.id for t in fresh}


def test_queue_count_by_kind_only_counts_pending():
    q = TaskQueue()
    a = q.add(Task(kind="claude_reply"))
    q.add(Task(kind="slack_msg"))
    q.mark_done(a.id)
    counts = q.count_by_kind()
    assert counts == {"slack_msg": 1}


def test_queue_update_task_mutates_fields_and_fires_event():
    q = TaskQueue()
    events: list[tuple[str, str]] = []
    q.add_listener(lambda kind, t: events.append((kind, t.id)))
    t = q.add(Task(headline="old", body="old body"))
    out = q.update_task(
        t.id,
        headline="new",
        body="new body",
        source={"channel_id": "C1"},
    )
    assert out is not None
    stored = q.get(t.id)
    assert stored.headline == "new"
    assert stored.body == "new body"
    assert stored.source == {"channel_id": "C1"}
    assert ("update", t.id) in events


def test_queue_update_task_unknown_id_returns_none():
    q = TaskQueue()
    assert q.update_task("nope", headline="x") is None
