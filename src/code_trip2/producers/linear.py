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
maps to one queue task. Repeat sightings of the same identifier
collapse into the existing task — same pattern as Slack/email thread
collapse.

**Reply path**: :func:`dispatch._respond_linear` posts the transcript
as a comment on the issue via ``save_comment``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

from code_trip2._async_utils import event_or_timeout
from code_trip2.config import Config
from code_trip2.linear_state import LinearState
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
        intake: Callable[[Task], None] | None = None,
    ) -> None:
        self._config = config
        self._queue = queue
        self._mcp = mcp
        self._state = state or LinearState()
        # ``intake`` routes new tasks through the screener (or directly
        # to the queue when no screener is configured). Existing-task
        # updates skip intake — they mutate an already-visible task.
        self._intake: Callable[[Task], None] = intake or queue.add
        self._stop = asyncio.Event()
        # Per-session dedup keyed by issue identifier (e.g. ``AI-1389``).
        # Cross-restart dedup is Linear itself: ``main.py`` drops replayed
        # ``linear_issue`` tasks and the first poll re-populates from the
        # current Linear state.
        self._recent_keys: set[str] = set()
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
        while not self._stop.is_set():
            try:
                await self._poll_once()
            except Exception:
                logger.exception("LinearProducer poll failed")
            if await event_or_timeout(self._stop, self._config.linear_poll_interval):
                return

    async def _poll_once(self) -> None:
        wide_poll = self._first_poll
        args: dict = {
            "assignee": "me",
            "limit": int(self._config.linear_max_results),
            "includeArchived": False,
        }
        if not wide_poll:
            last = self._state.last_updated_at()
            if last:
                args["updatedAt"] = last

        self.is_polling = True
        try:
            result = await self._mcp.call_tool("list_issues", args)
        except ClaudeMCPError as exc:
            logger.warning("LinearProducer: list call failed: %s", exc)
            # Don't burn the first-poll wide window on a transient
            # failure — retry on the next tick.
            return
        finally:
            self.is_polling = False

        issues = self._extract_issues(result)
        allowed = frozenset(self._config.linear_state_types)
        emitted = 0
        skipped = 0
        max_ts_seen = self._state.last_updated_at() or ""
        for issue in issues:
            identifier = issue.get("identifier") or ""
            status_type = issue.get("statusType") or ""
            updated_at = issue.get("updatedAt") or ""
            if not identifier:
                continue
            # Track max ts even for filtered issues so the cursor
            # advances past them and we don't keep re-pulling on
            # incremental polls.
            if updated_at > max_ts_seen:
                max_ts_seen = updated_at
            if status_type not in allowed:
                skipped += 1
                continue
            try:
                self._emit_task(issue)
                emitted += 1
            except Exception:
                logger.exception("Failed to emit Linear task for %s", identifier)
            self._recent_keys.add(identifier)

        if max_ts_seen and max_ts_seen != (self._state.last_updated_at() or ""):
            self._state.set_last_updated_at(max_ts_seen)

        logger.info(
            "LinearProducer: %s poll — %d issues (%d emitted, %d filtered out)",
            "wide" if wide_poll else "incremental",
            len(issues), emitted, skipped,
        )

        # Wide-poll only happens once per session. Even if it returned
        # no results, flip the flag so we don't keep paying the wider
        # cost on every interval.
        self._first_poll = False

        if len(self._recent_keys) > 500:
            self._recent_keys = set(sorted(self._recent_keys)[-250:])

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

        # Same collapse rule as Slack/email: one pending task per
        # issue identifier. Updates to title/description/status replace
        # the existing task's body rather than stacking duplicates.
        existing = self._find_pending_issue_task(identifier)
        if existing is not None:
            self._queue.update_task(
                existing.id,
                headline=headline,
                body=body,
                source=source,
                created_at=time.time(),
            )
            return

        task = Task(
            kind="linear_issue",
            topic=topic_key,
            headline=headline,
            body=body,
            source=source,
            created_at=time.time(),
        )
        self._intake(task)

    def _find_pending_issue_task(self, identifier: str) -> Task | None:
        if not identifier:
            return None
        for task in self._queue.pending():
            if task.kind != "linear_issue":
                continue
            if (task.source or {}).get("identifier") == identifier:
                return task
        return None
