"""Spend accounting for the current orchestrator run.

Every dollar the orchestrator spends on inference goes through a
``claude --print`` subprocess — producer polls (Slack / Gmail / Linear
MCP calls), the screener, and skill runs. Those all funnel through
:func:`code_trip2.producers.claude_mcp._run_subprocess`, which records
here, so this module sees the whole bill from one chokepoint.

The number is Claude's own, not an estimate: ``--output-format
stream-json`` ends every session with a terminal ``result`` event
carrying ``total_cost_usd``. We read that field and add it up. A
session that dies before emitting the event (timeout, spawn failure)
contributes nothing — under-counting a killed run is better than
guessing at it.

State is process-global on purpose: "the current run" *is* the process,
and the recording sites (``ClaudeMCPClient``, ``MCPBatcher``) are
constructed independently of :class:`~code_trip2.modes.Context`, so
threading an accumulator through them would mean plumbing a shared
object into every producer for a display-only counter.

NOT counted: the OpenAI STT / TTS / summarizer calls. Those APIs bill
per token rather than reporting a per-call price, so including them
would mean hard-coding a price table that silently goes stale. The TUI
labels the figure ``claude`` for that reason.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CostSnapshot:
    """Immutable read of the run's spend so far."""

    total_usd: float
    calls: int
    started_at: float
    # label ("classifier", "executor:archive-…", "batch:3", "tool:…")
    # → (usd, calls). Attribution matters because the callers differ by
    # an order of magnitude: a Haiku producer poll and a Sonnet skill
    # run both count as one "call" but not as one dollar.
    by_label: dict[str, tuple[float, int]] = field(default_factory=dict)

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.time() - self.started_at)

    @property
    def usd_per_hour(self) -> float:
        """Burn rate, or 0.0 before any measurable time has passed."""
        elapsed = self.elapsed_s
        if elapsed <= 0:
            return 0.0
        return self.total_usd * 3600.0 / elapsed


# Mutable process-global accumulator. Single-loop discipline (same as
# ``modes.Context`` playback state): every mutation happens on the
# event-loop thread, so no lock.
_total_usd: float = 0.0
_calls: int = 0
_started_at: float = time.time()
_by_label: dict[str, tuple[float, int]] = {}


def reset(*, now: float | None = None) -> None:
    """Zero the accumulator and restart the clock. Called at startup."""
    global _total_usd, _calls, _started_at
    _total_usd = 0.0
    _calls = 0
    _started_at = time.time() if now is None else now
    _by_label.clear()


def record(usd: float, *, what: str = "claude", model: str | None = None) -> None:
    """Add one billed session to the run total.

    Logged at INFO, not DEBUG: at ~$0.01–0.15 a session this is the
    only per-call record of where the money went, and the orchestrator
    normally runs at INFO.
    """
    global _total_usd, _calls
    if usd < 0:
        return
    _total_usd += usd
    _calls += 1
    prev_usd, prev_calls = _by_label.get(what, (0.0, 0))
    _by_label[what] = (prev_usd + usd, prev_calls + 1)
    logger.info(
        "cost: %s [%s] +$%.4f (run $%.4f over %d calls)",
        what, model or "?", usd, _total_usd, _calls,
    )


def snapshot() -> CostSnapshot:
    return CostSnapshot(
        total_usd=_total_usd,
        calls=_calls,
        started_at=_started_at,
        by_label=dict(_by_label),
    )


def format_breakdown(snap: CostSnapshot, *, limit: int = 8) -> str:
    """``classifier $0.83/6, batch:3 $0.41/5, …`` — biggest spender first."""
    rows = sorted(snap.by_label.items(), key=lambda kv: kv[1][0], reverse=True)
    return ", ".join(
        f"{label} ${usd:.4f}/{calls}" for label, (usd, calls) in rows[:limit]
    )


def extract_cost_usd(stdout: str) -> float | None:
    """Pull ``total_cost_usd`` out of a stream-json stdout.

    The terminal ``result`` event is the last line of a healthy run, so
    scan backwards and skip any line that can't carry the field — this
    runs on every claude invocation, alongside two other full passes
    over the same stdout, and the tool-result payloads can be large.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line or "total_cost_usd" not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "result":
            continue
        usd = event.get("total_cost_usd")
        if isinstance(usd, (int, float)) and not isinstance(usd, bool):
            return float(usd)
    return None


def extract_model(stdout: str) -> str | None:
    """Name the model that billed, from the result event's ``modelUsage``.

    Only for the log line — a session that somehow used two models
    reports the costlier one rather than inventing a combined label.
    """
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line or "modelUsage" not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("modelUsage")
        if not isinstance(usage, dict) or not usage:
            continue
        return max(
            usage.items(),
            key=lambda kv: (kv[1] or {}).get("costUSD", 0)
            if isinstance(kv[1], dict) else 0,
        )[0]
    return None


def record_stream(stdout: str, *, what: str = "claude") -> float | None:
    """Extract this session's cost and record it. Returns what it found."""
    usd = extract_cost_usd(stdout)
    if usd is not None:
        record(usd, what=what, model=extract_model(stdout))
    return usd
