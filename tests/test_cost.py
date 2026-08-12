"""Tests for run-spend accounting (``code_trip2.cost``).

Covers the stream-json extraction, the accumulator, and the wiring at
the ``claude --print`` chokepoint that books every session's cost.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock

import pytest

from code_trip2 import cost
from code_trip2.producers.claude_mcp import ClaudeMCPClient


@pytest.fixture(autouse=True)
def _fresh_counter():
    cost.reset()
    yield
    cost.reset()


def _result_event(usd: float) -> str:
    """A trimmed copy of the terminal event claude --print emits."""
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "session_id": "abc",
        "total_cost_usd": usd,
        "result": "done",
    })


# --- extraction -----------------------------------------------------------


def test_extract_cost_reads_result_event():
    stdout = "\n".join([
        '{"type": "system", "subtype": "init"}',
        '{"type": "assistant", "message": {"role": "assistant"}}',
        _result_event(0.0135829),
    ])
    assert cost.extract_cost_usd(stdout) == pytest.approx(0.0135829)


def test_extract_cost_ignores_non_result_events_carrying_the_key():
    """Only the terminal ``result`` event is authoritative — a tool
    payload that happens to mention the field must not be counted."""
    stdout = "\n".join([
        json.dumps({
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": '{"total_cost_usd": 99}'}],
            },
        }),
        _result_event(0.02),
    ])
    assert cost.extract_cost_usd(stdout) == pytest.approx(0.02)


def test_extract_cost_missing_or_unparseable_returns_none():
    assert cost.extract_cost_usd("") is None
    assert cost.extract_cost_usd('{"type": "result", "is_error": false}') is None
    assert cost.extract_cost_usd("not json at all") is None


# --- accumulator ----------------------------------------------------------


def test_record_accumulates_total_and_calls():
    cost.record(0.01)
    cost.record(0.02)
    snap = cost.snapshot()
    assert snap.total_usd == pytest.approx(0.03)
    assert snap.calls == 2


def test_reset_zeroes_the_run():
    cost.record(0.5)
    cost.reset()
    snap = cost.snapshot()
    assert snap.total_usd == 0.0
    assert snap.calls == 0


def test_usd_per_hour_extrapolates_from_elapsed():
    snap = cost.CostSnapshot(total_usd=0.75, calls=3, started_at=time.time() - 1800)
    assert snap.usd_per_hour == pytest.approx(1.5, rel=1e-3)


def test_by_label_attributes_spend_to_the_caller():
    cost.record(0.09, what="executor:archive-github-bot-notification")
    cost.record(0.05, what="classifier")
    cost.record(0.03, what="classifier")
    snap = cost.snapshot()
    assert snap.by_label["classifier"] == (pytest.approx(0.08), 2)
    # Biggest spender leads the breakdown.
    assert cost.format_breakdown(snap).startswith(
        "executor:archive-github-bot-notification $0.0900/1"
    )


def test_extract_model_names_the_costliest_model():
    stdout = json.dumps({
        "type": "result",
        "total_cost_usd": 0.08,
        "modelUsage": {
            "claude-haiku-4-5-20251001": {"costUSD": 0.001},
            "claude-sonnet-4-5": {"costUSD": 0.079},
        },
    })
    assert cost.extract_model(stdout) == "claude-sonnet-4-5"
    assert cost.extract_model('{"type": "result"}') is None


def test_record_stream_books_the_session():
    assert cost.record_stream(_result_event(0.04)) == pytest.approx(0.04)
    assert cost.snapshot().calls == 1
    # A stream with no result event contributes nothing.
    assert cost.record_stream('{"type": "system"}') is None
    assert cost.snapshot().calls == 1


# --- wiring ---------------------------------------------------------------


def _patch_exec(monkeypatch, *, stdout: str, returncode: int = 0):
    async def fake_exec(*argv, **kwargs):
        proc = MagicMock()

        async def communicate(input=None):
            return (stdout.encode("utf-8"), b"")

        proc.communicate = communicate
        proc.returncode = returncode
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


@pytest.mark.asyncio
async def test_claude_call_records_run_cost(monkeypatch):
    """Every claude invocation books its cost, even the failing ones —
    tokens burned before a nonzero exit are still billed."""
    c = ClaudeMCPClient()
    c._available = True
    tool_result = json.dumps({
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "content": '{"ok": true}'}],
        },
    })
    _patch_exec(monkeypatch, stdout="\n".join([tool_result, _result_event(0.013)]))
    await c.call_tool("slack_read_user_profile", {})
    assert cost.snapshot().total_usd == pytest.approx(0.013)

    _patch_exec(monkeypatch, stdout=_result_event(0.007), returncode=1)
    with pytest.raises(Exception):
        await c.call_tool("slack_read_user_profile", {})
    snap = cost.snapshot()
    assert snap.total_usd == pytest.approx(0.020)
    assert snap.calls == 2
