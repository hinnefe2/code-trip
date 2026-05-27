"""Tests for the task screener pipeline."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from code_trip2.producers.claude_mcp import ClaudeMCPError
from code_trip2.screener import (
    ScreeningOutcome,
    _next_or_stop,
    candidates_for,
    parse_classifier_reply,
    run_screener_loop,
    screen,
)
from code_trip2.skills import SkillManifest
from code_trip2.tasks import Task


# --- fixtures --------------------------------------------------------------


def _manifest(
    name: str,
    *,
    auto_handle: bool = True,
    kinds: tuple[str, ...] = ("email_msg",),
    tools: tuple[str, ...] = ("mcp__some__tool",),
    description: str = "test skill",
) -> SkillManifest:
    return SkillManifest(
        name=name,
        description=description,
        allowed_tools=tools,
        auto_handle=auto_handle,
        auto_handle_kinds=frozenset(kinds),
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
    ``run_agent`` raise; otherwise it returns the string.
    """
    mcp = MagicMock()
    if isinstance(agent_reply, Exception):
        mcp.run_agent = AsyncMock(side_effect=agent_reply)
    else:
        mcp.run_agent = AsyncMock(return_value=agent_reply)
    return mcp


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
    mcp = MagicMock()
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
    mcp = MagicMock()
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
    mcp = MagicMock()
    mcp.run_agent = AsyncMock(side_effect=ClaudeMCPError("subprocess died"))
    outcome = await screen(
        _task("email_msg"), [_manifest("accept-invite")], mcp,
    )
    assert outcome.action == "forward"
    assert outcome.skill is None
    assert mcp.run_agent.await_count == 1  # only the classifier ran


@pytest.mark.asyncio
async def test_screen_dry_run_logs_pick_but_forwards():
    mcp = MagicMock()
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
async def test_loop_dismissed_outcome_does_not_add_to_queue():
    """``dismissed`` outcomes suppress the task just like ``handled``."""
    intake: "asyncio.Queue[Task]" = asyncio.Queue()
    added: list[Task] = []
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    intake.put_nowait(_task("slack_msg"))

    mcp = _mcp(agent_reply="DISMISS: drop-standups")

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            intake=intake,
            manifests=(_dismiss_manifest("drop-standups"),),
            mcp=mcp,
            add_to_queue=added.append,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert added == []  # dismissed, not forwarded
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
    intake: "asyncio.Queue[Task]" = asyncio.Queue()
    added: list[Task] = []
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    intake.put_nowait(_task("note"))

    async def driver() -> None:
        # Let one task drain, then stop the loop.
        while not added and not outcomes:
            await asyncio.sleep(0)
        stop.set()

    mcp = _mcp(agent_reply="HANDLE: should-not-run")
    loop_task = asyncio.create_task(
        run_screener_loop(
            intake=intake,
            manifests=(_manifest("only-email", kinds=("email_msg",)),),
            mcp=mcp,
            add_to_queue=added.append,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert len(added) == 1
    assert added[0].kind == "note"
    assert [o.action for o in outcomes] == ["forward"]
    mcp.run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_loop_handled_outcome_does_not_add_to_queue():
    intake: "asyncio.Queue[Task]" = asyncio.Queue()
    added: list[Task] = []
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    intake.put_nowait(_task("email_msg"))

    mcp = MagicMock()
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
            intake=intake,
            manifests=(_manifest("accept-invite", kinds=("email_msg",)),),
            mcp=mcp,
            add_to_queue=added.append,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert added == []  # handled, not forwarded
    assert [o.action for o in outcomes] == ["handled"]
    assert outcomes[0].skill == "accept-invite"


@pytest.mark.asyncio
async def test_loop_failed_outcome_still_forwards_to_queue():
    intake: "asyncio.Queue[Task]" = asyncio.Queue()
    added: list[Task] = []
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    intake.put_nowait(_task("email_msg"))

    mcp = MagicMock()
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
            intake=intake,
            manifests=(_manifest("accept-invite", kinds=("email_msg",)),),
            mcp=mcp,
            add_to_queue=added.append,
            on_outcome=outcomes.append,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert len(added) == 1
    assert "auto-handle attempted" in (added[0].body or "")
    assert outcomes[0].action == "failed"


@pytest.mark.asyncio
async def test_loop_allowed_kinds_gate_short_circuits():
    """Even if a skill opts into a kind, allowed_kinds gates execution."""
    intake: "asyncio.Queue[Task]" = asyncio.Queue()
    added: list[Task] = []
    outcomes: list[ScreeningOutcome] = []
    stop = asyncio.Event()

    intake.put_nowait(_task("slack_msg"))

    mcp = _mcp(agent_reply="HANDLE: handle-slack")

    async def driver() -> None:
        while not outcomes:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            intake=intake,
            manifests=(_manifest("handle-slack", kinds=("slack_msg",)),),
            mcp=mcp,
            add_to_queue=added.append,
            on_outcome=outcomes.append,
            allowed_kinds=frozenset({"email_msg"}),  # slack_msg NOT allowed
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert len(added) == 1
    assert added[0].kind == "slack_msg"
    assert outcomes[0].action == "forward"
    mcp.run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_loop_exits_on_stop_event_without_pending_task():
    intake: "asyncio.Queue[Task]" = asyncio.Queue()
    stop = asyncio.Event()

    async def driver() -> None:
        # Loop is idle (nothing in intake); set stop and ensure it exits.
        await asyncio.sleep(0.01)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            intake=intake,
            manifests=(),
            mcp=_mcp(),
            add_to_queue=lambda _t: None,
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
    intake: "asyncio.Queue[Task]" = asyncio.Queue()
    added: list[Task] = []
    stop = asyncio.Event()

    intake.put_nowait(_task("note"))
    intake.put_nowait(_task("note"))

    def explosive_outcome(_o: ScreeningOutcome) -> None:
        raise RuntimeError("logger broken")

    async def driver() -> None:
        while len(added) < 2:
            await asyncio.sleep(0)
        stop.set()

    loop_task = asyncio.create_task(
        run_screener_loop(
            intake=intake,
            manifests=(),
            mcp=_mcp(),
            add_to_queue=added.append,
            on_outcome=explosive_outcome,
            allowed_kinds=None,
            dry_run=False,
            stop=stop,
        )
    )
    await asyncio.wait_for(asyncio.gather(driver(), loop_task), timeout=2.0)
    assert len(added) == 2  # both tasks still made it through


# --- _next_or_stop --------------------------------------------------------


@pytest.mark.asyncio
async def test_next_or_stop_returns_task_when_available():
    q: "asyncio.Queue[Task]" = asyncio.Queue()
    stop = asyncio.Event()
    task = _task("note")
    q.put_nowait(task)
    out = await _next_or_stop(q, stop)
    assert out is task


@pytest.mark.asyncio
async def test_next_or_stop_returns_none_when_stop_fires_first():
    q: "asyncio.Queue[Task]" = asyncio.Queue()
    stop = asyncio.Event()

    async def setter():
        await asyncio.sleep(0.01)
        stop.set()

    asyncio.create_task(setter())
    out = await asyncio.wait_for(_next_or_stop(q, stop), timeout=1.0)
    assert out is None
