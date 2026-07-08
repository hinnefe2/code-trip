"""Unit tests for the task queue: Task, TaskQueue, RecentTopics, score()."""

from __future__ import annotations

import time

import pytest

from code_trip2.tasks import (
    STATE_ACTIVE,
    STATE_DONE,
    STATE_DROPPED,
    STATE_PENDING,
    STATE_SCREENING,
    STATE_SNOOZED,
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
    t = Task(
        kind="claude_reply", topic="ticket-42", headline="ready", body="ok",
        origin_key="followup:abc:ready",
    )
    d = t.to_dict()
    back = Task.from_dict(d)
    assert back.id == t.id
    assert back.kind == "claude_reply"
    assert back.topic == "ticket-42"
    assert back.headline == "ready"
    assert back.body == "ok"
    assert back.state == STATE_PENDING
    # origin_key survives the log round-trip, so a done follow-up replayed on
    # restart can still suppress a re-emitted duplicate.
    assert back.origin_key == "followup:abc:ready"


def test_task_from_dict_reads_legacy_dedup_key():
    """Pre-refactor JSONL records wrote the identity field as
    ``dedup_key`` — replay must still pick it up."""
    back = Task.from_dict({"id": "x", "dedup_key": "followup:abc:ready"})
    assert back.origin_key == "followup:abc:ready"


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


# --- upsert / origin_key -------------------------------------------------


def test_upsert_without_origin_key_never_collapses():
    """Tasks with no origin_key (manual notes) are singletons — added
    unconditionally even when headline/topic collide."""
    q = TaskQueue()
    a = q.upsert(Task(headline="same", topic="t"))
    b = q.upsert(Task(headline="same", topic="t"))
    assert a.id != b.id
    assert len(q.all()) == 2


def test_upsert_new_origin_key_adds():
    q = TaskQueue()
    t = q.upsert(Task(headline="x", origin_key="linear:AI-1"))
    assert q.pending() == [t]


def test_upsert_live_origin_key_updates_in_place():
    """A re-sighting of the same object updates the live task's fields
    (including created_at — the object genuinely changed) instead of
    minting a sibling."""
    q = TaskQueue()
    events: list[str] = []
    q.add_listener(lambda kind, _t: events.append(kind))
    first = q.upsert(Task(
        headline="old", body="old body", origin_key="linear:AI-1",
        created_at=100.0,
    ))
    dup = Task(
        headline="new", body="new body", origin_key="linear:AI-1",
        source={"identifier": "AI-1"}, created_at=200.0,
    )
    returned = q.upsert(dup)
    assert returned is first
    assert len(q.all()) == 1
    assert first.headline == "new"
    assert first.body == "new body"
    assert first.source == {"identifier": "AI-1"}
    assert first.created_at == 200.0
    assert events == ["add", "update"]


@pytest.mark.parametrize(
    "state", [STATE_PENDING, STATE_ACTIVE, STATE_SNOOZED, STATE_SCREENING],
)
def test_upsert_collapses_into_every_live_state(state: str):
    """Live spans pending/active/snoozed AND screening — a task parked
    in the screener's screening state still collapses re-sightings
    (the TRI-240 duplication window)."""
    q = TaskQueue()
    first = q.upsert(Task(headline="a", origin_key="linear:AI-1"))
    first.state = state
    returned = q.upsert(Task(headline="b", origin_key="linear:AI-1"))
    assert returned is first
    assert len(q.all()) == 1
    assert first.state == state  # collapse never disturbs state


def test_upsert_terminal_default_resurfaces_fresh_task():
    """A sighting after the task went terminal means the object came
    back (e.g. reopened ticket) — mint a fresh task."""
    q = TaskQueue()
    first = q.upsert(Task(headline="a", origin_key="linear:AI-1"))
    q.mark_done(first.id)
    fresh = q.upsert(Task(headline="b", origin_key="linear:AI-1"))
    assert fresh is not first
    assert q.pending() == [fresh]
    # The index re-points: a third sighting collapses into the fresh task.
    third = q.upsert(Task(headline="c", origin_key="linear:AI-1"))
    assert third is fresh


def test_upsert_terminal_skip_suppresses():
    """if_terminal="skip": a re-spawned follow-up whose original was
    already filed (done) or dismissed stays gone — no new entry, no
    event, nothing resurfaced."""
    q = TaskQueue()
    first = q.upsert(Task(
        headline="Send doc to Anna", origin_key="followup:abc:send doc",
    ))
    q.mark_done(first.id)  # filed via ACT+YES -> done

    dup = Task(headline="Send doc to Anna", origin_key="followup:abc:send doc")
    returned = q.upsert(dup, if_terminal="skip")

    assert returned is first
    assert dup.id not in {t.id for t in q.all()}
    assert len(q.all()) == 1
    assert q.pending() == []


def test_get_by_origin_returns_indexed_task_any_state():
    q = TaskQueue()
    t = q.upsert(Task(headline="a", origin_key="slack:C1:123"))
    assert q.get_by_origin("slack:C1:123") is t
    q.mark_done(t.id)
    assert q.get_by_origin("slack:C1:123") is t
    assert q.get_by_origin("slack:C1:999") is None


# --- resolve_by_origin -----------------------------------------------------


def test_resolve_by_origin_marks_pending_done():
    q = TaskQueue()
    t = q.upsert(Task(headline="a", origin_key="linear:AI-1"))
    out = q.resolve_by_origin("linear:AI-1")
    assert out is t
    assert t.state == STATE_DONE


def test_resolve_by_origin_marks_snoozed_done():
    q = TaskQueue()
    t = q.upsert(Task(headline="a", origin_key="linear:AI-1"))
    q.set_state(t.id, STATE_SNOOZED)
    assert q.resolve_by_origin("linear:AI-1") is t
    assert t.state == STATE_DONE


def test_resolve_by_origin_never_touches_active():
    """The user is mid-conversation with the task — leave it alone even
    though the source says the object is resolved."""
    q = TaskQueue()
    t = q.upsert(Task(headline="a", origin_key="linear:AI-1"))
    q.mark_active(t.id)
    assert q.resolve_by_origin("linear:AI-1") is None
    assert t.state == STATE_ACTIVE


def test_resolve_by_origin_unknown_key_is_noop():
    q = TaskQueue()
    assert q.resolve_by_origin("linear:AI-404") is None


# --- screening state -------------------------------------------------------


def test_screening_tasks_are_not_pending():
    q = TaskQueue()
    t = Task(headline="a", origin_key="email:T1", state=STATE_SCREENING)
    q.add(t)
    assert q.pending() == []
    assert score(t, now=time.time(), recent=RecentTopics()) == float("-inf")
    q.set_state(t.id, STATE_PENDING)
    assert q.pending() == [t]


def test_topic_cap_enforced_on_screening_release():
    """Screening tasks don't count toward the per-topic pending cap, so
    the cap re-runs when one is released to pending."""
    q = TaskQueue()
    for i in range(5):  # fill the cap (default 5)
        q.add(Task(topic="t", headline=str(i), created_at=float(i)))
    parked = Task(
        topic="t", headline="parked", state=STATE_SCREENING, created_at=6.0,
    )
    q.add(parked)
    assert len(q.pending()) == 5  # screening task not counted
    q.set_state(parked.id, STATE_PENDING)
    pending = q.pending()
    assert len(pending) == 5  # cap re-enforced on release
    assert sum(1 for t in q.all() if t.state == STATE_DROPPED) == 1


def test_load_rebuilds_origin_index_live_wins_over_terminal():
    """Replay can hand load() both a terminal durable record and a live
    task for the same key — the live one must own the index entry."""
    q = TaskQueue()
    done = Task(headline="old", origin_key="followup:abc:x", state=STATE_DONE)
    live = Task(headline="new", origin_key="followup:abc:x")
    q.load([done, live])
    assert q.get_by_origin("followup:abc:x") is live
    # And a terminal-only key still resolves for if_terminal="skip".
    q2 = TaskQueue()
    q2.load([done])
    assert q2.get_by_origin("followup:abc:x") is done
    suppressed = q2.upsert(
        Task(headline="dup", origin_key="followup:abc:x"), if_terminal="skip",
    )
    assert suppressed is done


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
