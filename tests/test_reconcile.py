"""Tests for ReconcileProducer — the timer-driven stale-task sweep."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from code_trip2.producers.reconcile import ReconcileProducer


def _cfg(interval: float = 300.0) -> SimpleNamespace:
    # Only the fields ReconcileProducer / polling_active read.
    return SimpleNamespace(
        reconcile_interval=interval,
        poll_start_hour=0, poll_end_hour=0,  # equal → always active
        poll_ignore_active_hours=True,
    )


@pytest.mark.asyncio
async def test_reconcile_once_runs_all_and_sums():
    calls: list[str] = []

    async def r_email() -> int:
        calls.append("email")
        return 2

    async def r_other() -> int:
        calls.append("other")
        return 3

    p = ReconcileProducer(config=_cfg(), reconcilers=[r_email, r_other])
    await p._reconcile_once()
    assert calls == ["email", "other"]
    assert p.is_polling is False  # cleared after the pass


@pytest.mark.asyncio
async def test_reconcile_once_isolates_a_raising_reconciler():
    ran: list[str] = []

    async def boom() -> int:
        raise RuntimeError("nope")

    async def ok() -> int:
        ran.append("ok")
        return 1

    p = ReconcileProducer(config=_cfg(), reconcilers=[boom, ok])
    await p._reconcile_once()  # must not raise
    assert ran == ["ok"]       # the surviving reconciler still ran


@pytest.mark.asyncio
async def test_run_returns_immediately_with_no_reconcilers():
    p = ReconcileProducer(config=_cfg(), reconcilers=[])
    await asyncio.wait_for(p.run(), timeout=1.0)


@pytest.mark.asyncio
async def test_request_stop_ends_the_loop():
    async def r() -> int:
        return 0

    p = ReconcileProducer(config=_cfg(interval=0.01), reconcilers=[r])
    p._STARTUP_DELAY_S = 0.01
    task = asyncio.create_task(p.run())
    await asyncio.sleep(0.05)
    p.request_stop()
    await asyncio.wait_for(task, timeout=1.0)
