"""LinearProducer: polls Linear via the claude.ai Linear MCP.

Same auth-passthrough pattern as :class:`EmailProducer` and
:class:`SlackProducer`: goes through :class:`ClaudeMCPClient` pointed at
the claude.ai Linear MCP server, so auth piggy-backs on whatever the
user already authorized in claude.ai (no Linear API token to manage).

Per poll tick we make one ``list_issues`` call constrained to
``assignee: "me"`` and filter the response client-side to issues whose
``statusType`` falls in the configured allow-list (Todo / In Progress /
In Review by default). Wide first poll has no ``updatedAt`` floor;
subsequent polls pass the last seen ``updatedAt`` so we only get
recently-changed issues.

Topic is the lowercase issue identifier (``ai-1389``) so one ticket
maps to one queue task. Repeat sightings of the same identifier carry
the same ``origin_key`` and collapse into the existing live task via
:meth:`TaskQueue.upsert` — same rule as every other producer.

**Reply path**: :func:`dispatch._respond_linear` posts the transcript
as a comment on the issue via ``save_comment``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from code_trip2 import config as config_mod
from code_trip2._async_utils import event_or_timeout, next_tick_delay
from code_trip2.config import Config
from code_trip2.linear_state import LinearState
from code_trip2.poll_health import PollHealth
from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.tasks import Task, TaskQueue

logger = logging.getLogger(__name__)


class LinearProducer:
    name = "linear"

    # Initial stagger before the first poll so producers don't all hit
    # claude --print the instant the orchestrator starts. Class constant
    # so tests can lower it via per-instance override.
    _STARTUP_DELAY_S = 4.0

    def __init__(
        self,
        *,
        config: Config,
        queue: TaskQueue,
        mcp: ClaudeMCPClient | None = None,
        state: LinearState | None = None,
        submit: Callable[[Task], Task] | None = None,
        health: PollHealth | None = None,
    ) -> None:
        self._config = config
        self._queue = queue
        self._mcp = mcp
        self._state = state or LinearState()
        self._health = health
        # ``submit`` is the pipeline entry point (main.py's screening
        # gate). It lands the task via ``queue.upsert``, so a
        # re-sighting of an issue that already has a live task —
        # including one still being screened — updates it in place.
        self._submit: Callable[[Task], Task] = submit or queue.upsert
        self._stop = asyncio.Event()
        # First poll of the session uses a wide query (no ``updatedAt``
        # floor) so all currently-active issues surface, regardless of
        # when they last changed. Subsequent polls revert to the
        # incremental ``updatedAt: <iso>`` query.
        self._first_poll = True
        # True while an MCP call is in flight. The supervisor reads this
        # so the TUI shows "polling" instead of "running" while we're
        # waiting on ``claude --print``.
        self.is_polling = False

    # ---- lifecycle ------------------------------------------------------

    def request_stop(self) -> None:
        self._stop.set()

    # ---- poll loop ------------------------------------------------------

    async def run(self) -> None:
        if self._mcp is None or not self._mcp.enabled:
            logger.info("LinearProducer: ClaudeMCPClient unavailable; not starting.")
            return
        if await event_or_timeout(self._stop, self._STARTUP_DELAY_S):
            return
        was_active = True
        while not self._stop.is_set():
            if config_mod.polling_active(self._config):
                if not was_active:
                    logger.info("LinearProducer: active hours resumed; polling")
                    was_active = True
                try:
                    await self._poll_once()
                except Exception:
                    logger.exception("LinearProducer poll failed")
            elif was_active:
                logger.info("LinearProducer: outside active hours; polling paused")
                was_active = False
            # Sleep to the next wall-clock multiple of the interval so
            # producers with compatible intervals fire at the same
            # instant and the MCP batcher can coalesce their calls
            # into one claude session.
            delay = next_tick_delay(self._config.linear_poll_interval)
            if await event_or_timeout(self._stop, delay):
                return

    async def _poll_once(self) -> None:
        wide_poll = self._first_poll
        allowed = frozenset(self._config.linear_state_types)

        self.is_polling = True
        try:
            if wide_poll:
                issues = await self._wide_pull(allowed)
            else:
                issues = await self._incremental_pull()
        finally:
            self.is_polling = False
        if issues is None:
            # Transient MCP failure already logged inside the pull
            # helper. Don't burn the first-poll wide window — retry
            # next tick.
            if self._health is not None:
                self._health.record_failure(
                    "list_issues call failed (see orchestrator.log)",
                )
            return
        if self._health is not None:
            self._health.record_success()
        emitted = 0
        skipped = 0
        retired = 0
        last_cursor = self._state.last_updated_at() or ""
        # ``updatedAt`` of issues we definitively resolved this poll
        # (emitted, or confirmed out-of-scope) vs. ones we could not
        # (unknown status, emit failure). The cursor may advance past the
        # former but must stay behind the latter — see the barrier logic
        # below.
        resolved_ts: list[str] = []
        unresolved_ts: list[str] = []
        for issue in issues:
            identifier = issue.get("identifier") or ""
            status_type = issue.get("statusType") or ""
            updated_at = issue.get("updatedAt") or ""
            if not identifier:
                continue
            # The incremental ``updatedAt`` filter is inclusive, so the
            # boundary issue comes back every poll. Strict client-side
            # guard (mirrors the email producer's ``ts <= last_ts``
            # check) so an unchanged re-return is never re-processed —
            # without this the boundary ticket's task gets its
            # created_at refreshed each poll and never ages up. Safe
            # because the cursor only ever reflects *resolved* work, so
            # ``updated_at <= last_cursor`` really does mean "already
            # handled" (not merely "already seen").
            if not wide_poll and last_cursor and updated_at and updated_at <= last_cursor:
                skipped += 1
                continue
            if not status_type:
                # Status omitted from this response — happens for a
                # freshly-created issue before Linear's list index
                # settles (this is how TRI-279 was lost: filtered as
                # "not allowed", which also burned the cursor to its
                # updatedAt, after which the boundary guard skipped it
                # forever). We can't tell if it's in scope, so treat it
                # as UNRESOLVED: hold the cursor behind it so the next
                # poll re-pulls and re-evaluates.
                if updated_at:
                    unresolved_ts.append(updated_at)
                skipped += 1
                continue
            if status_type not in allowed:
                # Known terminal/inactive state (completed, canceled,
                # backlog). Genuinely not our concern, so it's resolved —
                # the cursor may advance past it. Mid-session cleanup: if
                # we previously surfaced this ticket, the user just moved
                # it out of the active set in Linear, so retire the task
                # (Linear has no status-change push; this sweep is the
                # only way to sync without a restart).
                if self._mark_closed_task(identifier):
                    retired += 1
                else:
                    skipped += 1
                if updated_at:
                    resolved_ts.append(updated_at)
                continue
            try:
                self._emit_task(issue)
                emitted += 1
                if updated_at:
                    resolved_ts.append(updated_at)
            except Exception:
                logger.exception("Failed to emit Linear task for %s", identifier)
                # Emit failed — leave the cursor behind so we retry
                # instead of stranding the ticket.
                if updated_at:
                    unresolved_ts.append(updated_at)

        # Advance the cursor to the newest resolved issue, but never to or
        # past the oldest unresolved one: the inclusive-boundary skip
        # above would then drop the unresolved issue permanently (it may
        # never get a newer ``updatedAt``). Re-pulling already-resolved
        # issues above the barrier next poll is harmless — emits collapse
        # via ``origin_key`` upsert, retires are idempotent.
        candidate = last_cursor
        for ts in resolved_ts:
            if ts > candidate:
                candidate = ts
        if unresolved_ts:
            barrier = min(unresolved_ts)
            if candidate >= barrier:
                below = [t for t in resolved_ts if t < barrier]
                candidate = max(below) if below else last_cursor
        if candidate and candidate != (self._state.last_updated_at() or ""):
            self._state.set_last_updated_at(candidate)

        logger.info(
            "LinearProducer: %s poll — %d issues (%d emitted, %d retired, "
            "%d filtered out, %d held for retry)",
            "wide" if wide_poll else "incremental",
            len(issues), emitted, retired, skipped, len(unresolved_ts),
        )

        # Wide-poll only happens once per session. Even if it returned
        # no results, flip the flag so we don't keep paying the wider
        # cost on every interval.
        self._first_poll = False

    async def _wide_pull(self, allowed: frozenset[str]) -> list[dict] | None:
        """Initial sync: one MCP call per state in the allow-list.

        Pushing the state filter server-side keeps each response small
        (only matching issues come back) — vital because ``list_issues``
        defaults to ``orderBy: updatedAt`` and an unfiltered call gets
        dominated by recently-completed work, which both bloats the
        response past ``claude --print``'s per-tool token cap and
        pages active issues off the end.

        Failures on a single state are logged-and-skipped: the others
        still populate.
        """
        out: list[dict] = []
        any_call_succeeded = False
        for state_type in allowed:
            args = {
                "assignee": "me",
                "state": state_type,
                "limit": int(self._config.linear_max_results),
                "includeArchived": False,
            }
            try:
                result = await self._mcp.call_tool("list_issues", args)
            except ClaudeMCPError as exc:
                logger.warning(
                    "LinearProducer: state=%s call failed: %s", state_type, exc,
                )
                continue
            any_call_succeeded = True
            out.extend(self._extract_issues(result))
        if not any_call_succeeded:
            return None
        return out

    async def _incremental_pull(self) -> list[dict] | None:
        """Single MCP call with ``updatedAt`` floor.

        Response size is bounded by how much changed since the cursor
        — small in the common case. No server-side state filter; the
        client-side allow-list weeds out completions / cancellations
        that happen during the window.
        """
        args: dict = {
            "assignee": "me",
            "limit": int(self._config.linear_max_results),
            "includeArchived": False,
        }
        last = self._state.last_updated_at()
        if last:
            args["updatedAt"] = last
        try:
            result = await self._mcp.call_tool("list_issues", args)
        except ClaudeMCPError as exc:
            logger.warning("LinearProducer: incremental call failed: %s", exc)
            return None
        return self._extract_issues(result)

    # ---- response shape -------------------------------------------------

    def _extract_issues(self, result: dict) -> list[dict]:
        """Normalize the MCP's response into our internal issue shape.

        Linear's MCP returns ``{"issues": [...], "hasNextPage": bool}``
        with each issue carrying ``id`` (which is the human identifier
        like ``AI-1389``), ``title``, ``description``, ``status``,
        ``statusType``, ``url``, ``updatedAt``, ``assignee``, etc. We
        normalize to a consistent ``identifier`` field so downstream
        code doesn't have to remember that Linear's ``id`` is actually
        the identifier, not a UUID.
        """
        for key in ("issues", "items"):
            value = result.get(key)
            if isinstance(value, list):
                return [self._normalize_structured(i) for i in value if isinstance(i, dict)]
        return []

    def _normalize_structured(self, issue: dict) -> dict:
        # Linear's list_issues returns the human identifier (``AI-1389``)
        # as ``id`` — surprising, but documented in the tool schema.
        # Keep ``identifier`` as our canonical field and fall back
        # through possible alternate keys in case the MCP shape shifts.
        identifier = str(
            issue.get("identifier")
            or issue.get("id")
            or ""
        )
        title = str(issue.get("title") or "")
        description = str(issue.get("description") or "")
        status = str(issue.get("status") or "")
        status_type = str(issue.get("statusType") or "")
        url = str(issue.get("url") or "")
        updated_at = str(issue.get("updatedAt") or "")
        priority_name = ""
        priority = issue.get("priority")
        if isinstance(priority, dict):
            priority_name = str(priority.get("name") or "")
        elif isinstance(priority, str):
            priority_name = priority
        return {
            "identifier": identifier,
            "title": title,
            "description": description,
            "status": status,
            "statusType": status_type,
            "url": url,
            "updatedAt": updated_at,
            "priority": priority_name,
        }

    def _emit_task(self, issue: dict) -> None:
        identifier = issue["identifier"]
        title = issue.get("title") or "(no title)"
        description = issue.get("description") or ""
        status = issue.get("status") or ""
        url = issue.get("url") or ""
        priority = issue.get("priority") or ""

        headline = f"{identifier}: {title[:60]}"
        body = title if not description else f"{title}\n\n{description}"
        # Topic is the identifier so the scheduler treats each ticket
        # as its own thread — recent-topic affinity boosts the same
        # ticket if the user has been working on it.
        topic_key = identifier.lower()
        source = {
            "identifier": identifier,
            "url": url,
            "title": title,
            "status": status,
            "priority": priority,
        }

        task = Task(
            kind="linear_issue",
            topic=topic_key,
            headline=headline,
            body=body,
            source=source,
            created_at=time.time(),
            subject_key=f"linear:{identifier.upper()}",
            origin_key=_origin_key(identifier),
        )
        self._submit(task)

    def _mark_closed_task(self, identifier: str) -> bool:
        """Retire a queue task for a ticket that's left the active set.

        Returns True when a pending/snoozed task existed and was marked
        done; False when there was nothing to clean up (the common case
        — most filtered issues never had a queue task). ACTIVE tasks
        are deliberately left alone — if the user is mid-conversation
        with a ticket that just got closed in Linear, yanking it out
        from under them would be worse than letting the stale state
        linger until they finish.
        """
        if not identifier:
            return False
        return self._queue.resolve_by_origin(_origin_key(identifier)) is not None


def _origin_key(identifier: str) -> str:
    return f"linear:{identifier.upper()}"
