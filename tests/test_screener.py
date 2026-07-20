"""Tests for the task screener pipeline."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.screener import (
    ScreeningOutcome,
    _next_or_stop,
    _one_line,
    candidates_for,
    parse_classifier_reply,
    parse_follow_up_tasks,
    parse_skill_status,
    run_screener_loop,
    screen,
    summary_indicates_failure,
)
from code_trip2.skills import SkillManifest
from code_trip2.tasks import (
    STATE_DONE,
    STATE_DROPPED,
    STATE_PENDING,
    STATE_SCREENING,
    Task,
    TaskQueue,
)


# --- fixtures --------------------------------------------------------------


def _manifest(
    name: str,
    *,
    auto_handle: bool = True,
    kinds: tuple[str, ...] = ("email_msg",),
    tools: tuple[str, ...] = ("mcp__some__tool",),
    description: str = "test skill",
    verify: str = "",
) -> SkillManifest:
    return SkillManifest(
        name=name,
        description=description,
        allowed_tools=tools,
        auto_handle=auto_handle,
        auto_handle_kinds=frozenset(kinds),
        verify=verify,
    )


def _task(
    kind: str = "email_msg",
    *,
    headline: str = "Test headline",
    body: str | None = "Test body",
    source: dict | None = None,
) -> Task:
    return Task(
        kind=kind,
        topic="t",
        headline=headline,
        body=body,
        source=source or {"thread_id": "abc"},
    )


def _mcp(*, agent_reply: str | Exception = "") -> Any:
    """A fake ClaudeMCPClient with ``run_agent`` mocked.

    Pass an Exception instance for ``agent_reply`` to make
    ``run_agent`` raise; otherwise it returns the string. Spec'd
    against the real class so accessing an attribute that doesn't
    exist (or calling a method with the wrong shape) fails at test
    time instead of silently no-op'ing.
    """
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    if isinstance(agent_reply, Exception):
        mcp.run_agent = AsyncMock(side_effect=agent_reply)
    else:
        mcp.run_agent = AsyncMock(return_value=agent_reply)
    return mcp


def _loop_env(
    *tasks: Task, mode: str = "intake",
) -> tuple["asyncio.Queue[tuple[str, str]]", TaskQueue]:
    """Park ``tasks`` in a real TaskQueue the way main.py's submit gate
    would (``screening`` state for intake items; whatever they already
    hold for reconsider) and enqueue their work items."""
    work: "asyncio.Queue[tuple[str, str]]" = asyncio.Queue()
    q = TaskQueue()
    for t in tasks:
        if mode == "intake":
            t.state = STATE_SCREENING
        q.add(t)
        work.put_nowait((t.id, mode))
    return work, q


# --- pure helpers ----------------------------------------------------------


def test_candidates_for_filters_by_auto_handle_and_kind():
    matching = _manifest("a", kinds=("email_msg",), auto_handle=True)
    wrong_kind = _manifest("b", kinds=("slack_msg",), auto_handle=True)
    not_auto = _manifest("c", kinds=("email_msg",), auto_handle=False)
    out = candidates_for(_task("email_msg"), [matching, wrong_kind, not_auto])
    assert out == [matching]


def test_candidates_for_returns_empty_when_no_match():
    assert candidates_for(_task("note"), [_manifest("a")]) == []


def test_parse_classifier_reply_handle_line():
    cands = [_manifest("accept-invite")]
    assert parse_classifier_reply("HANDLE: accept-invite", cands) is cands[0]
    assert parse_classifier_reply("HANDLE:accept-invite", cands) is cands[0]
    assert parse_classifier_reply("HANDLE = accept-invite", cands) is cands[0]


def test_parse_classifier_reply_handles_prose_wrapping():
    cands = [_manifest("accept-invite")]
    reply = "Sure, I think this is an invite.\nHANDLE: accept-invite\n"
    assert parse_classifier_reply(reply, cands) is cands[0]


def test_parse_classifier_reply_none_returns_none():
    assert parse_classifier_reply("NONE", [_manifest("a")]) is None
    assert parse_classifier_reply("", [_manifest("a")]) is None
    assert parse_classifier_reply(
        "I don't think any skill applies here.", [_manifest("a")],
    ) is None


def test_parse_classifier_reply_unknown_name_returns_none():
    """Defensive: the model could name a skill that wasn't in the list."""
    cands = [_manifest("accept-invite")]
    assert parse_classifier_reply("HANDLE: imaginary-skill", cands) is None


# --- screen() --------------------------------------------------------------


@pytest.mark.asyncio
async def test_screen_no_candidates_forwards_without_mcp_call():
    mcp = _mcp(agent_reply="HANDLE: nope")  # should not be called
    outcome = await screen(_task("note"), [_manifest("a")], mcp)
    assert outcome.action == "forward"
    assert outcome.skill is None
    mcp.run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_screen_classifier_declines_forwards():
    mcp = _mcp(agent_reply="NONE")
    outcome = await screen(_task("email_msg"), [_manifest("accept-invite")], mcp)
    assert outcome.action == "forward"
    assert outcome.skill is None
    assert mcp.run_agent.await_count == 1  # classifier only


@pytest.mark.asyncio
async def test_screen_classifier_picks_then_executor_succeeds():
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        "Accepted the invite and archived the email.",
    ])
    outcome = await screen(
        _task("email_msg"), [_manifest("accept-invite")], mcp,
    )
    assert outcome.action == "handled"
    assert outcome.skill == "accept-invite"
    assert outcome.summary == "Accepted the invite and archived the email."
    assert mcp.run_agent.await_count == 2


@pytest.mark.asyncio
async def test_screen_executor_raises_returns_failed_with_annotated_body():
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        RuntimeError("MCP timeout"),
    ])
    task = _task("email_msg", body="Original body")
    outcome = await screen(task, [_manifest("accept-invite")], mcp)
    assert outcome.action == "failed"
    assert outcome.skill == "accept-invite"
    assert "MCP timeout" in (outcome.error or "")
    assert "Original body" in (outcome.task.body or "")
    assert "auto-handle attempted (accept-invite)" in (outcome.task.body or "")


@pytest.mark.asyncio
async def test_screen_classifier_raises_forwards():
    """Classifier exception is fail-safe: forward, no executor call."""
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=ClaudeMCPError("subprocess died"))
    outcome = await screen(
        _task("email_msg"), [_manifest("accept-invite")], mcp,
    )
    assert outcome.action == "forward"
    assert outcome.skill is None
    assert mcp.run_agent.await_count == 1  # only the classifier ran


@pytest.mark.asyncio
async def test_screen_dry_run_logs_pick_but_forwards():
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(return_value="HANDLE: accept-invite")
    outcome = await screen(
        _task("email_msg"),
        [_manifest("accept-invite")],
        mcp,
        dry_run=True,
    )
    assert outcome.action == "forward"
    assert outcome.skill == "accept-invite"
    assert outcome.dry_run_nominated is True
    # Classifier ran, executor did NOT.
    assert mcp.run_agent.await_count == 1


# --- dismiss skills --------------------------------------------------------


def _dismiss_manifest(
    name: str,
    *,
    kinds: tuple[str, ...] = ("slack_msg",),
    description: str = "dismiss noise",
) -> SkillManifest:
    return SkillManifest(
        name=name,
        description=description,
        allowed_tools=(),
        auto_handle=False,
        auto_handle_kinds=frozenset(),
        dismiss=True,
        dismiss_kinds=frozenset(kinds),
    )


@pytest.mark.asyncio
async def test_screen_dismiss_skill_returns_dismissed_outcome():
    """A dismiss skill matched → outcome `dismissed`, no executor call."""
    mcp = _mcp(agent_reply="DISMISS: drop-standups")
    outcome = await screen(
        _task("slack_msg"),
        [_dismiss_manifest("drop-standups")],
        mcp,
    )
    assert outcome.action == "dismissed"
    assert outcome.skill == "drop-standups"
    # Only the classifier ran; executor was skipped.
    assert mcp.run_agent.await_count == 1


@pytest.mark.asyncio
async def test_screen_dismiss_skill_prefix_mismatch_still_dispatches_by_flag():
    """Classifier said HANDLE: for a dismiss-only skill — we trust the
    skill's flag, dispatch as dismiss anyway."""
    mcp = _mcp(agent_reply="HANDLE: drop-standups")  # wrong prefix
    outcome = await screen(
        _task("slack_msg"),
        [_dismiss_manifest("drop-standups")],
        mcp,
    )
    assert outcome.action == "dismissed"


@pytest.mark.asyncio
async def test_screen_dismiss_skill_in_dry_run_forwards():
    """Dry-run forwards even for dismiss outcomes, with the pick logged."""
    mcp = _mcp(agent_reply="DISMISS: drop-standups")
    outcome = await screen(
        _task("slack_msg"),
        [_dismiss_manifest("drop-standups")],
        mcp,
        dry_run=True,
    )
    assert outcome.action == "forward"
    assert outcome.skill == "drop-standups"
    assert outcome.dry_run_nominated is True


@pytest.mark.asyncio
async def test_screen_mixed_candidates_classifier_chooses_dismiss():
    """When both handle and dismiss skills are candidates, the
    classifier's pick determines which fires."""
    mcp = _mcp(agent_reply="DISMISS: drop-standups")
    outcome = await screen(
        _task("slack_msg"),
        [
            _manifest("handle-slack", kinds=("slack_msg",)),
            _dismiss_manifest("drop-standups"),
        ],
        mcp,
    )
    assert outcome.action == "dismissed"
    assert outcome.skill == "drop-standups"


@pytest.mark.asyncio
async def test_loop_dismissed_outcome_drops_task():
    """``dismissed`` outcomes suppress the task just like ``handled`` —
    the resident task transitions to dropped, never pending."""
    task = _task("slack_msg")
    work, q = _loop_env(task)
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = _mcp(agent_reply="DISMISS: drop-standups")

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_dismiss_manifest("drop-standups"),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.pending() == []  # dismissed, not surfaced
    assert q.get(task.id).state == STATE_DROPPED
    assert [o.action for o in outcomes] == ["dismissed"]


def test_parse_classifier_reply_accepts_dismiss_prefix():
    cands = [_manifest("h")]
    assert parse_classifier_reply("DISMISS: h", cands) is cands[0]
    assert parse_classifier_reply("DISMISS:h", cands) is cands[0]


def test_candidates_for_includes_dismiss_skills():
    handle = _manifest("h", kinds=("slack_msg",))
    dismiss = _dismiss_manifest("d", kinds=("slack_msg",))
    out = candidates_for(_task("slack_msg"), [handle, dismiss])
    assert handle in out
    assert dismiss in out
    # Other kinds: neither applies.
    assert candidates_for(_task("note"), [handle, dismiss]) == []


# --- run_screener_loop ----------------------------------------------------


@pytest.mark.asyncio
async def test_loop_forwards_when_no_candidates():
    task = _task("note")
    work, q = _loop_env(task)
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    async def driver() -> None:
        # Let one task drain, then stop the loop.
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    mcp = _mcp(agent_reply="HANDLE: should-not-run")
    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_manifest("only-email", kinds=("email_msg",)),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.pending() == [task]  # released to the user queue
    assert q.get(task.id).state == STATE_PENDING
    assert [o.action for o in outcomes] == ["forward"]
    mcp.run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_loop_handled_outcome_marks_task_done():
    task = _task("email_msg")
    work, q = _loop_env(task)
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        "Accepted and archived.",
    ])

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_manifest("accept-invite", kinds=("email_msg",)),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.pending() == []  # handled, not surfaced
    assert q.get(task.id).state == STATE_DONE
    assert [o.action for o in outcomes] == ["handled"]
    assert outcomes[0].skill == "accept-invite"


@pytest.mark.asyncio
async def test_loop_failed_outcome_still_surfaces_task():
    task = _task("email_msg")
    work, q = _loop_env(task)
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        RuntimeError("boom"),
    ])

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_manifest("accept-invite", kinds=("email_msg",)),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.pending() == [task]
    # The error annotation lands on the resident task's body.
    assert "auto-handle attempted" in (q.get(task.id).body or "")
    assert outcomes[0].action == "failed"


@pytest.mark.asyncio
async def test_loop_allowed_kinds_gate_short_circuits():
    """Even if a skill opts into a kind, allowed_kinds gates execution."""
    task = _task("slack_msg")
    work, q = _loop_env(task)
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = _mcp(agent_reply="HANDLE: handle-slack")

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_manifest("handle-slack", kinds=("slack_msg",)),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=frozenset({"email_msg"}),  # slack_msg NOT allowed
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.pending() == [task]
    assert outcomes[0].action == "forward"
    mcp.run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_loop_exits_on_stop_event_without_pending_task():
    work, q = _loop_env()
    stop = asyncio.Event()

    async def driver() -> None:
        # Loop is idle (nothing queued); set stop and ensure it exits.
        await asyncio.sleep(0.01)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(),
            mcp=_mcp(),
            on_outcome=lambda _o: None,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    # No assertions other than "we didn't hang."


@pytest.mark.asyncio
async def test_loop_on_outcome_exception_does_not_crash_loop():
    """A buggy logger shouldn't take down screening."""
    a, b = _task("note"), _task("note")
    work, q = _loop_env(a, b)
    stop = asyncio.Event()

    def explosive_outcome(_o: ScreeningOutcome) -> None:
        raise RuntimeError("logger broken")

    async def driver() -> None:
        while len(q.pending()) < 2:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(),
            mcp=_mcp(),
            on_outcome=explosive_outcome,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert len(q.pending()) == 2  # both tasks still made it through


@pytest.mark.asyncio
async def test_loop_stale_verdict_does_not_override_user_action():
    """If the task left the screening state mid-screen (user acted, a
    resolve sweep retired it), the screener's verdict is stale and must
    not clobber the newer state."""
    task = _task("email_msg")
    work, q = _loop_env(task)
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = create_autospec(ClaudeMCPClient, instance=True)

    async def classify_then_yield(*_a, **_k):
        # Simulate the user retiring the task while the classifier runs.
        q.mark_done(task.id)
        return "NONE"

    mcp.run_agent = AsyncMock(side_effect=classify_then_yield)

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_manifest("accept-invite", kinds=("email_msg",)),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.get(task.id).state == STATE_DONE  # user's action stands


# --- _next_or_stop --------------------------------------------------------


@pytest.mark.asyncio
async def test_next_or_stop_returns_work_item_when_available():
    q: "asyncio.Queue[tuple[str, str]]" = asyncio.Queue()
    stop = asyncio.Event()
    q.put_nowait(("task-1", "intake"))
    out = await _next_or_stop(q, stop)
    assert out == ("task-1", "intake")


@pytest.mark.asyncio
async def test_next_or_stop_returns_none_when_stop_fires_first():
    q: "asyncio.Queue[tuple[str, str]]" = asyncio.Queue()
    stop = asyncio.Event()

    async def setter():
        await asyncio.sleep(0.01)
        stop.set()

    asyncio.create_task(setter())
    out = await asyncio.wait_for(_next_or_stop(q, stop), timeout=1.0)
    assert out is None


# --- follow-up tasks ------------------------------------------------------


def test_parse_follow_up_tasks_extracts_meeting_followups():
    summary = (
        "FOLLOWUP_TASK: {\"headline\": \"Draft retention doc\", "
        "\"body\": \"From planning sync: draft retention metrics doc.\", "
        "\"topic\": \"planning-sync\"}\n"
        "FOLLOWUP_TASK: {\"headline\": \"Reply to Anna re: schema\"}\n"
        "Archived Gemini meeting notes: Planning sync."
    )
    out = parse_follow_up_tasks(summary)
    assert len(out) == 2
    assert out[0].kind == "meeting_followup"
    assert out[0].headline == "Draft retention doc"
    assert out[0].topic == "planning-sync"
    assert out[0].body and "retention metrics" in out[0].body
    # Missing topic falls back to "inbox".
    assert out[1].topic == "inbox"
    assert out[1].headline == "Reply to Anna re: schema"


def test_parse_follow_up_tasks_skips_malformed_lines():
    summary = (
        "FOLLOWUP_TASK: not-json-at-all\n"
        "FOLLOWUP_TASK: {\"body\": \"no headline\"}\n"
        "FOLLOWUP_TASK: {\"headline\": \"\"}\n"
        "FOLLOWUP_TASK: {\"headline\": \"keep me\", \"body\": \"yep\"}\n"
    )
    out = parse_follow_up_tasks(summary)
    assert [t.headline for t in out] == ["keep me"]


def test_parse_follow_up_tasks_handles_no_summary():
    assert parse_follow_up_tasks(None) == ()
    assert parse_follow_up_tasks("") == ()
    assert parse_follow_up_tasks("plain summary, no follow-ups") == ()


def test_parse_follow_up_tasks_tolerates_backtick_wrapping():
    """LLMs sometimes wrap the line in code fences."""
    summary = "`FOLLOWUP_TASK: {\"headline\": \"a\"}`"
    out = parse_follow_up_tasks(summary)
    assert len(out) == 1 and out[0].headline == "a"


@pytest.mark.asyncio
async def test_screen_attaches_follow_up_tasks_from_summary():
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: archive-gemini-meeting-notes",
        (
            "FOLLOWUP_TASK: {\"headline\": \"Send doc to Anna\", "
            "\"body\": \"context\", \"topic\": \"docs\"}\n"
            "Archived Gemini meeting notes: Planning sync."
        ),
    ])
    parent = _task("email_msg")
    outcome = await screen(
        parent, [_manifest("archive-gemini-meeting-notes")], mcp,
    )
    assert outcome.action == "handled"
    assert len(outcome.follow_up_tasks) == 1
    spawned = outcome.follow_up_tasks[0]
    assert spawned.kind == "meeting_followup"
    assert spawned.headline == "Send doc to Anna"
    assert spawned.topic == "docs"
    # Spawned tasks reference the parent so the queue log can show lineage.
    assert spawned.parent_id == parent.id
    # And carry a stable origin key anchored on the parent's thread id, so a
    # re-screen of the same email doesn't respawn a duplicate follow-up.
    assert spawned.origin_key == "followup:abc:send doc to anna"


@pytest.mark.asyncio
async def test_screen_origin_key_survives_parent_id_change():
    """The follow-up's origin key anchors on the parent's thread id (stable),
    not its task id (a fresh uuid each poll), so two screens of the same
    meeting-notes email produce the *same* key even though parent ids differ."""
    def _run():
        mcp = create_autospec(ClaudeMCPClient, instance=True)
        mcp.run_agent = AsyncMock(side_effect=[
            "HANDLE: archive-gemini-meeting-notes",
            (
                "FOLLOWUP_TASK: {\"headline\": \"Send doc to Anna\"}\n"
                "Archived Gemini meeting notes: Planning sync."
            ),
        ])
        return mcp

    src = {"thread_id": "thread-xyz"}
    out1 = await screen(_task("email_msg", source=src), [_manifest("archive-gemini-meeting-notes")], _run())
    out2 = await screen(_task("email_msg", source=src), [_manifest("archive-gemini-meeting-notes")], _run())
    a, b = out1.follow_up_tasks[0], out2.follow_up_tasks[0]
    assert a.parent_id != b.parent_id  # fresh uuid each screen
    assert a.origin_key == b.origin_key == "followup:thread-xyz:send doc to anna"


@pytest.mark.asyncio
async def test_loop_adds_follow_up_tasks_even_when_parent_handled():
    """A handled meeting-notes email suppresses the parent but still
    needs to enqueue the spawned meeting_followup tasks."""
    parent = _task("email_msg")
    work, q = _loop_env(parent)
    submitted: list[Task] = []
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: archive-gemini-meeting-notes",
        (
            "FOLLOWUP_TASK: {\"headline\": \"Draft retention doc\"}\n"
            "FOLLOWUP_TASK: {\"headline\": \"Reply re schema\"}\n"
            "Archived Gemini meeting notes: Planning sync."
        ),
    ])

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    def submit_follow_up(t: Task) -> None:
        submitted.append(t)
        q.upsert(t, if_terminal="skip")

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_manifest(
                "archive-gemini-meeting-notes", kinds=("email_msg",),
            ),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
            submit_follow_up=submit_follow_up,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    # Parent was handled (suppressed); both follow-ups routed through
    # the submit gate and now pending.
    assert q.get(parent.id).state == STATE_DONE
    assert [t.kind for t in submitted] == ["meeting_followup", "meeting_followup"]
    assert sorted(t.headline for t in q.pending()) == [
        "Draft retention doc", "Reply re schema",
    ]
    assert outcomes[0].action == "handled"
    assert len(outcomes[0].follow_up_tasks) == 2


@pytest.mark.asyncio
async def test_loop_follow_ups_default_to_terminal_skip_upsert():
    """Without an explicit submit_follow_up, follow-ups land via
    ``queue.upsert(if_terminal="skip")`` — a respawned follow-up whose
    original was already filed stays gone."""
    parent = _task("email_msg")
    work, q = _loop_env(parent)
    # The user already filed this follow-up (terminal record present).
    filed = q.add(Task(
        kind="meeting_followup",
        headline="Draft retention doc",
        origin_key="followup:abc:draft retention doc",
    ))
    q.mark_done(filed.id)
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: archive-gemini-meeting-notes",
        (
            "FOLLOWUP_TASK: {\"headline\": \"Draft retention doc\"}\n"
            "Archived Gemini meeting notes: Planning sync."
        ),
    ])

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_manifest(
                "archive-gemini-meeting-notes", kinds=("email_msg",),
            ),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.pending() == []  # respawned follow-up suppressed
    assert q.get(filed.id).state == STATE_DONE


# --- reconsider path ------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_reconsider_dismissed_marks_existing_done():
    """The interesting case: a task already visible in the user queue
    arrives with mode="reconsider", the classifier picks a dismiss
    skill, and the loop marks the *existing* task done."""
    task = _task("slack_msg")
    work, q = _loop_env(task, mode="reconsider")
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = _mcp(agent_reply="DISMISS: dismiss-resolved-slack-thread")

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_dismiss_manifest("dismiss-resolved-slack-thread"),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.get(task.id).state == STATE_DONE
    assert outcomes[0].action == "dismissed"


@pytest.mark.asyncio
async def test_loop_reconsider_classifier_declines_is_noop():
    """Classifier says NONE on a reconsider task → task stays where it
    is in the user queue."""
    task = _task("slack_msg")
    work, q = _loop_env(task, mode="reconsider")
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = _mcp(agent_reply="NONE")

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_dismiss_manifest("dismiss-resolved-slack-thread"),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.get(task.id).state == STATE_PENDING
    assert outcomes[0].action == "forward"


@pytest.mark.asyncio
async def test_loop_reconsider_dry_run_logs_pick_but_does_not_mark_done():
    """Dry-run is a hard gate even for reconsider — the user can validate
    the dismiss judgement before letting it actually fire."""
    task = _task("slack_msg")
    work, q = _loop_env(task, mode="reconsider")
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = _mcp(agent_reply="DISMISS: dismiss-resolved-slack-thread")

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_dismiss_manifest("dismiss-resolved-slack-thread"),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=True,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    # Dry-run forwards (no dismiss), and in reconsider mode forward is a
    # no-op. The pick is captured in the outcome for visibility.
    assert q.get(task.id).state == STATE_PENDING
    assert outcomes[0].action == "forward"
    assert outcomes[0].dry_run_nominated is True
    assert outcomes[0].skill == "dismiss-resolved-slack-thread"


@pytest.mark.asyncio
async def test_loop_intake_still_works_alongside_reconsider():
    """Intake and reconsider items share one work queue — a regular
    intake task still gets released to pending when no skill applies,
    and a reconsider item on the same queue is judged independently."""
    intake_task = _task("note")
    work, q = _loop_env(intake_task)
    reconsider_task = q.add(_task("slack_msg"))
    work.put_nowait((reconsider_task.id, "reconsider"))
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    async def driver() -> None:
        while len(outcomes) < 2:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(),
            mcp=_mcp(),
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.get(intake_task.id).state == STATE_PENDING
    assert q.get(reconsider_task.id).state == STATE_PENDING  # untouched
    assert [o.action for o in outcomes] == ["forward", "forward"]


# --- self-reported failure detection --------------------------------------


def test_summary_indicates_failure_catches_observed_phrases():
    """Real-world failure prose pulled from session logs the subagent
    audited (May 27 – Jun 8, 2026). Every one of these summaries left
    a task marked ``handled`` while the side effect didn't happen."""
    observed = [
        # accept-invite: couldn't find event, skipped archive
        "I cannot complete the acceptance without finding the event first, "
        "so I'm skipping the archive step",
        # archive-github-bot-notification: tool failed
        "authentication errors... unable to complete the archive",
        # archive-gemini-meeting-notes: tool failure phrasing
        "Not archived — Gmail tools encountered errors",
        # accept-invite, variant phrasing
        "I could not find the calendar event for this invitation, skipping the archive",
        # generic agent abdication
        "I'll skip this one — the email looks personal",
        "Failed to archive: thread not found",
        "I couldn't archive the email because the thread ID was missing.",
    ]
    for summary in observed:
        assert summary_indicates_failure(summary), (
            f"Should flag as failure: {summary!r}"
        )


def test_summary_indicates_failure_passes_clean_success_summaries():
    """Canonical success summaries from each skill — must NOT flag as
    failure. False positives just annoy the user; false negatives are
    silent data loss, which is worse, but no need to be paranoid."""
    successes = [
        "Archived vendor update from notify@cobalt.io: credit reminder.",
        "Accepted 'Lunch' and archived the email.",
        "Archived GitHub bot notification: linear-code[bot] comment on PR #11086.",
        "Archived office-hours invite \"Skip-level Q&A\" without RSVPing.",
        "Archived RSVP-acceptance notification for \"Standup\".",
        "Archived cancellation notification for \"Recruitment Solutions\".",
        "Archived Gemini meeting notes: Sprint Planning. Spawned 3 follow-ups for Henry.",
        "Sent /do-ticket ENGAGE-4010 to dev:0.",
    ]
    for summary in successes:
        assert not summary_indicates_failure(summary), (
            f"Should NOT flag as failure: {summary!r}"
        )


def test_summary_indicates_failure_handles_none_and_empty():
    assert summary_indicates_failure(None) is False
    assert summary_indicates_failure("") is False
    assert summary_indicates_failure("   ") is False


@pytest.mark.asyncio
async def test_screen_downgrades_handled_to_failed_when_summary_admits_failure():
    """End-to-end: the agent returned a normal summary saying it
    declined to act, and the screener routes the task back to the user
    queue (``failed``) instead of vanishing it (``handled``)."""
    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        # The exact failure summary from a real May 27 session.
        "I cannot complete the acceptance without finding the event first, "
        "so I'm skipping the archive step.",
    ])
    parent = _task("email_msg", body="Original body")
    outcome = await screen(parent, [_manifest("accept-invite")], mcp)
    assert outcome.action == "failed"
    assert outcome.skill == "accept-invite"
    assert "self-reported failure" in (outcome.error or "")
    # The task body gains an annotation so the user understands why
    # this task reappeared instead of being silently archived.
    assert "auto-handle declined (accept-invite)" in (outcome.task.body or "")
    assert "Original body" in (outcome.task.body or "")


def test_one_line_flattens_multiline_summary_preserving_status_reason():
    # A real rate-limit decline: preamble + working note + trailing STATUS.
    # The trailing reason is the diagnostic payload and must survive.
    summary = (
        "I'll find and execute the archive-gemini-meeting-notes skill.\n"
        "I've hit a rate limit trying to fetch the email content.\n"
        "STATUS: declined: Unable to fetch full email content due to rate limit."
    )
    out = _one_line(summary)
    assert "\n" not in out
    assert "STATUS: declined: Unable to fetch full email content" in out


def test_one_line_caps_and_ellipsizes():
    out = _one_line("word " * 400, limit=50)
    assert len(out) == 50
    assert out.endswith("…")
    assert _one_line(None) == ""


@pytest.mark.asyncio
async def test_declined_body_annotation_is_single_line_with_reason():
    """The queue-task annotation and log line must flatten newlines so
    line-based tooling doesn't lose the trailing decline reason."""
    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: archive-gemini-meeting-notes",
        "I'll execute the skill.\nI've hit a rate limit.\n"
        "STATUS: declined: rate limit — could not extract action items.",
    ])
    parent = _task("email_msg", body="Original body")
    outcome = await screen(
        parent, [_manifest("archive-gemini-meeting-notes")], mcp,
    )
    assert outcome.action == "failed"
    body = outcome.task.body or ""
    annotation = body.split("[auto-handle declined", 1)[1]
    assert "\n" not in annotation
    assert "could not extract action items" in annotation


async def _handling_mcp():
    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: archive-x",
        "Archived the email. STATUS: handled",
    ])
    return mcp


@pytest.mark.asyncio
async def test_verify_false_surfaces_task_instead_of_closing():
    """The skill reports success but the ground-truth check says the side
    effect didn't happen → the task is surfaced (failed), not closed."""
    mcp = await _handling_mcp()
    calls = []

    async def verify(task, verify_type):
        calls.append((task.id, verify_type))
        return False  # side effect NOT observed

    outcome = await screen(
        _task("email_msg"), [_manifest("archive-x", verify="left-inbox")], mcp,
        verify_side_effect=verify,
    )
    assert outcome.action == "failed"
    assert outcome.error == "unverified side-effect"
    assert "unverified" in (outcome.task.body or "")
    assert calls and calls[0][1] == "left-inbox"


@pytest.mark.asyncio
async def test_verify_true_keeps_handled():
    mcp = await _handling_mcp()

    async def verify(task, verify_type):
        return True

    outcome = await screen(
        _task("email_msg"), [_manifest("archive-x", verify="left-inbox")], mcp,
        verify_side_effect=verify,
    )
    assert outcome.action == "handled"


@pytest.mark.asyncio
async def test_verify_none_falls_back_to_trusting_skill():
    """Inconclusive check (e.g. Gmail read failed) must not nag the user —
    fall back to the skill's self-report."""
    mcp = await _handling_mcp()

    async def verify(task, verify_type):
        return None

    outcome = await screen(
        _task("email_msg"), [_manifest("archive-x", verify="left-inbox")], mcp,
        verify_side_effect=verify,
    )
    assert outcome.action == "handled"


@pytest.mark.asyncio
async def test_verify_exception_falls_back_to_handled():
    mcp = await _handling_mcp()

    async def verify(task, verify_type):
        raise RuntimeError("gmail down")

    outcome = await screen(
        _task("email_msg"), [_manifest("archive-x", verify="left-inbox")], mcp,
        verify_side_effect=verify,
    )
    assert outcome.action == "handled"


@pytest.mark.asyncio
async def test_verify_skipped_when_skill_declares_none():
    """A skill with no ``verify`` never triggers the check, even when a
    verifier is wired in."""
    mcp = await _handling_mcp()
    called = False

    async def verify(task, verify_type):
        nonlocal called
        called = True
        return False

    outcome = await screen(
        _task("email_msg"), [_manifest("archive-x", verify="")], mcp,
        verify_side_effect=verify,
    )
    assert outcome.action == "handled"
    assert called is False


@pytest.mark.asyncio
async def test_screen_keeps_handled_for_clean_success_summary():
    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        "Accepted 'Standup' and archived the email.",
    ])
    outcome = await screen(_task("email_msg"), [_manifest("accept-invite")], mcp)
    assert outcome.action == "handled"


@pytest.mark.asyncio
async def test_screen_routes_classify_and_execute_to_separate_clients():
    """Classification uses ``classifier_mcp``; execution uses ``mcp``.
    Lets the nomination step run a stronger model than the executor."""
    classifier_mcp = MagicMock(spec=ClaudeMCPClient)
    classifier_mcp.run_agent = AsyncMock(return_value="HANDLE: accept-invite")
    executor_mcp = MagicMock(spec=ClaudeMCPClient)
    executor_mcp.run_agent = AsyncMock(
        return_value="Accepted 'Standup' and archived the email.",
    )
    outcome = await screen(
        _task("email_msg"), [_manifest("accept-invite")], executor_mcp,
        classifier_mcp=classifier_mcp,
    )
    assert outcome.action == "handled"
    classifier_mcp.run_agent.assert_awaited_once()
    executor_mcp.run_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_screen_defaults_classifier_to_executor_client():
    """Omitting ``classifier_mcp`` keeps the single-client behavior."""
    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        "Accepted 'Standup' and archived the email.",
    ])
    outcome = await screen(_task("email_msg"), [_manifest("accept-invite")], mcp)
    assert outcome.action == "handled"
    assert mcp.run_agent.await_count == 2


@pytest.mark.asyncio
async def test_loop_self_reported_failure_surfaces_task_to_user_queue():
    """Regression: the bug was that ``handled`` outcomes suppress the
    task. After the fix, a self-reported-failure summary should route
    through the ``failed`` branch, which releases to pending."""
    task = _task("email_msg")
    work, q = _loop_env(task)
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    mcp = MagicMock(spec=ClaudeMCPClient)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        "Not archived — Gmail tools encountered errors.",
    ])

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            work=work,
            queue=q,
            manifests=(_manifest("accept-invite", kinds=("email_msg",)),),
            mcp=mcp,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert q.pending() == [task]
    assert outcomes[0].action == "failed"
    assert "auto-handle declined" in (q.get(task.id).body or "")


# --- STATUS exit signal --------------------------------------------------


def test_parse_skill_status_handled():
    assert parse_skill_status("did the thing.\nSTATUS: handled") == "handled"


def test_parse_skill_status_declined_with_reason():
    assert (
        parse_skill_status("STATUS: declined: couldn't find the event")
        == "declined"
    )


def test_parse_skill_status_skipped_folds_into_declined():
    """``skipped`` is a softer phrasing but the user-visible disposition
    is the same as ``declined`` — the task surfaces back to the queue."""
    assert parse_skill_status("STATUS: skipped: not a vendor email") == "declined"


def test_parse_skill_status_missing_returns_none():
    assert parse_skill_status(None) is None
    assert parse_skill_status("") is None
    assert parse_skill_status("Archived office-hours invite.") is None


def test_parse_skill_status_malformed_returns_none():
    """No recognised verb after STATUS: → no signal."""
    assert parse_skill_status("STATUS: ok") is None
    assert parse_skill_status("STATUS:") is None
    assert parse_skill_status("status maybe handled") is None


def test_parse_skill_status_tolerates_code_fence_wrapping():
    """LLMs sometimes wrap the line in backticks — mirror the
    `FOLLOWUP_TASK` parser's tolerance."""
    assert parse_skill_status("`STATUS: handled`") == "handled"
    assert (
        parse_skill_status("```\nSTATUS: declined: tool errored\n```")
        == "declined"
    )


def test_parse_skill_status_case_insensitive():
    assert parse_skill_status("status: Handled") == "handled"
    assert parse_skill_status("Status: DECLINED: nope") == "declined"


def test_parse_skill_status_last_match_wins():
    """If the model emits two STATUS lines (e.g. quoted earlier output
    plus a final decision), the final one is authoritative."""
    summary = "STATUS: declined: first pass said no\nSTATUS: handled"
    assert parse_skill_status(summary) == "handled"


@pytest.mark.asyncio
async def test_screen_explicit_status_handled_overrides_failure_heuristic():
    """A summary that the heuristic would flag as failure is still
    treated as ``handled`` when the skill explicitly says STATUS: handled.
    Lets a skill opt out of the heuristic when it knows its prose is
    benign (e.g. mentioning 'unable to' inside a quoted message)."""
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        # Heuristic would normally flag "unable to" — STATUS overrides.
        "Sender was unable to attend, but I archived the cancellation.\n"
        "STATUS: handled",
    ])
    outcome = await screen(_task("email_msg"), [_manifest("accept-invite")], mcp)
    assert outcome.action == "handled"
    assert outcome.skill == "accept-invite"


@pytest.mark.asyncio
async def test_screen_explicit_status_declined_flips_success_summary():
    """A success-looking summary is still treated as ``failed`` when the
    skill explicitly says STATUS: declined. Lets a skill opt into the
    failure path when its prose looks clean but the side effect didn't
    happen."""
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        # No heuristic trigger words; only the explicit signal flips it.
        "Reviewed the invite.\nSTATUS: declined: organizer added a real note",
    ])
    outcome = await screen(
        _task("email_msg", body="Original body"),
        [_manifest("accept-invite")],
        mcp,
    )
    assert outcome.action == "failed"
    assert outcome.skill == "accept-invite"
    assert "self-reported failure" in (outcome.error or "")
    assert "auto-handle declined (accept-invite)" in (outcome.task.body or "")
    assert "Original body" in (outcome.task.body or "")


@pytest.mark.asyncio
async def test_screen_missing_status_falls_back_to_heuristic_handled():
    """Without STATUS, a clean summary stays ``handled`` (unchanged)."""
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        "Accepted 'Standup' and archived the email.",
    ])
    outcome = await screen(_task("email_msg"), [_manifest("accept-invite")], mcp)
    assert outcome.action == "handled"


@pytest.mark.asyncio
async def test_screen_missing_status_falls_back_to_heuristic_failed():
    """Without STATUS, a heuristic-tripping summary still flips to
    ``failed`` — backwards-compat path for un-updated skills."""
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: accept-invite",
        "I cannot complete the acceptance without finding the event first, "
        "so I'm skipping the archive step.",
    ])
    outcome = await screen(_task("email_msg"), [_manifest("accept-invite")], mcp)
    assert outcome.action == "failed"
    assert "self-reported failure" in (outcome.error or "")


@pytest.mark.asyncio
async def test_screen_status_handled_coexists_with_followup_task():
    """STATUS and FOLLOWUP_TASK live in the same summary — both parse
    independently."""
    mcp = create_autospec(ClaudeMCPClient, instance=True)
    mcp.run_agent = AsyncMock(side_effect=[
        "HANDLE: archive-gemini-meeting-notes",
        (
            "FOLLOWUP_TASK: {\"headline\": \"Draft retention doc\"}\n"
            "Archived Gemini meeting notes: Planning sync.\n"
            "STATUS: handled"
        ),
    ])
    outcome = await screen(
        _task("email_msg"), [_manifest("archive-gemini-meeting-notes")], mcp,
    )
    assert outcome.action == "handled"
    assert len(outcome.follow_up_tasks) == 1
    assert outcome.follow_up_tasks[0].headline == "Draft retention doc"
