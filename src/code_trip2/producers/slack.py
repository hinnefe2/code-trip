"""SlackProducer: cross-channel @-mention polling via the claude.ai Slack MCP.

Single tool call per poll tick: ``slack_search_public_and_private``
with the user's own ID as the query. That catches @-mentions in any
channel (public, private, DM, group DM) — the producer doesn't need a
per-channel list, doesn't need a workspace install, and doesn't need
its own Slack credentials. All auth is piggy-backed off claude.ai via
:class:`ClaudeMCPClient`.

**Known limitation:** DMs that don't @-mention the user won't surface.
The Slack MCP exposed by claude.ai doesn't have a clean
"list-all-DMs-since-X" tool, and a heuristic search query would either
miss messages or burn tokens. Out of scope for this pass.

On startup the producer makes one ``slack_read_user_profile`` call to
discover the current user's ID, then loops mention searches at the
configured interval.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING

from code_trip2.config import Config
from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.slack_state import SlackState
from code_trip2.tasks import Task, TaskQueue


_USER_ID_RE = re.compile(r"User ID:\s*(U[A-Z0-9]+)", re.IGNORECASE)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Search "from" floor when the state file is empty. One week feels
# right: enough to capture stale-but-unread mentions from a weekend
# away, not so much that startup floods the queue.
_INITIAL_LOOKBACK_S = 7 * 24 * 3600


class SlackProducer:
    name = "slack"

    def __init__(
        self,
        *,
        config: Config,
        queue: TaskQueue,
        mcp: ClaudeMCPClient | None = None,
        state: SlackState | None = None,
    ) -> None:
        self._config = config
        self._queue = queue
        self._mcp = mcp
        self._state = state or SlackState()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._user_id: str = ""
        # Tiny cache to suppress duplicates within a session (state file
        # already handles cross-restart dedup).
        self._recent_ids: set[str] = set()

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._mcp is None or not self._mcp.enabled:
            logger.info("SlackProducer: ClaudeMCPClient unavailable; not starting.")
            return
        try:
            self._user_id = self._fetch_user_id()
        except ClaudeMCPError as exc:
            logger.warning("SlackProducer: could not resolve user_id (%s); not starting.", exc)
            return
        if not self._user_id:
            logger.warning("SlackProducer: empty user_id from slack_read_user_profile; not starting.")
            return
        logger.info("SlackProducer: starting (user_id=%s)", self._user_id)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- setup ----------------------------------------------------------

    def _fetch_user_id(self) -> str:
        """Resolve the current user's Slack ID via the MCP.

        The Slack MCP returns human-readable text wrapped in a ``result``
        key (e.g. ``"User ID: U02L5V8H9RS\\nUsername: henry...\\n..."``).
        We grep for the ID rather than expecting structured JSON. We
        also try a few structured shapes in case a future MCP version
        returns JSON.
        """
        result = self._mcp.call_tool("slack_read_user_profile", {})

        # Structured shapes first, just in case.
        for candidate in (result.get("user"), result):
            if isinstance(candidate, dict):
                uid = candidate.get("id") or candidate.get("user_id")
                if uid:
                    return str(uid)

        # Plain-text "result" field — the common case.
        text = result.get("result") or result.get("_raw") or ""
        if isinstance(text, str):
            m = _USER_ID_RE.search(text)
            if m:
                return m.group(1)
        return ""

    # ---- poll loop ------------------------------------------------------

    def _run(self) -> None:
        # Slight stagger so we don't hammer claude at the same instant the
        # orchestrator starts.
        if self._stop.wait(2.0):
            return
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("SlackProducer poll failed")
            if self._stop.wait(self._config.slack_poll_interval):
                return

    def _poll_once(self) -> None:
        last_ts = self._state.last_search_ts()
        after = self._after_param(last_ts)
        try:
            result = self._mcp.call_tool(
                "slack_search_public_and_private",
                {
                    "query": f"<@{self._user_id}>",
                    "after": after,
                    "sort": "timestamp",
                    "sort_dir": "asc",
                    "limit": 20,
                    "include_context": False,
                    "response_format": "concise",
                },
            )
        except ClaudeMCPError as exc:
            logger.warning("SlackProducer: search call failed: %s", exc)
            return

        for msg in self._extract_messages(result):
            ts = msg.get("ts") or ""
            if not ts:
                continue
            if last_ts and ts <= last_ts:
                continue
            if ts in self._recent_ids:
                continue
            try:
                self._emit_task(msg)
            except Exception:
                logger.exception("Failed to emit Slack task for ts=%s", ts)
            self._recent_ids.add(ts)
            self._state.set_last_search_ts(ts)

        # Cap the in-memory dedup set so it doesn't grow unbounded.
        if len(self._recent_ids) > 500:
            self._recent_ids = set(sorted(self._recent_ids)[-250:])

    def _after_param(self, last_ts: str | None) -> str:
        """Compute the ``after`` argument for the search call.

        Slack's ``after`` is a Unix timestamp in seconds. From a Slack
        message ts string we take the integer portion. From an empty
        state, we look back :data:`_INITIAL_LOOKBACK_S` seconds.
        """
        if last_ts:
            try:
                return str(int(float(last_ts)))
            except ValueError:
                pass
        return str(int(time.time() - _INITIAL_LOOKBACK_S))

    # ---- response shape -------------------------------------------------

    def _extract_messages(self, result: dict) -> list[dict]:
        """Defensive: the Slack MCP returns matches under several keys
        depending on response_format and version. Try the common ones."""
        candidates = (
            result.get("messages"),
            (result.get("messages") or {}).get("matches") if isinstance(result.get("messages"), dict) else None,
            result.get("matches"),
            result.get("results"),
            result.get("items"),
        )
        for c in candidates:
            if isinstance(c, list):
                return c
        return []

    def _emit_task(self, msg: dict) -> None:
        text = (msg.get("text") or "").strip()
        if not text:
            return
        sender_id = msg.get("user") or msg.get("user_id") or ""
        sender_name = (
            msg.get("username")
            or msg.get("user_name")
            or (msg.get("user_profile") or {}).get("display_name")
            or (msg.get("user_profile") or {}).get("real_name")
            or sender_id
            or "someone"
        )
        channel = msg.get("channel") or {}
        if isinstance(channel, str):
            channel_id = channel
            channel_name = "channel"
        else:
            channel_id = channel.get("id") or msg.get("channel_id") or ""
            channel_name = (
                channel.get("name")
                or msg.get("channel_name")
                or "channel"
            )
        ts = msg.get("ts") or ""
        thread_ts = msg.get("thread_ts") or ts
        headline = f"{sender_name}: {text[:60]}"
        task = Task(
            kind="slack_msg",
            topic=f"slack-{channel_name}" if channel_name else "slack",
            headline=headline,
            body=text,
            source={
                "channel_id": channel_id,
                "channel_name": channel_name,
                "ts": ts,
                "thread_ts": thread_ts,
                "sender_id": sender_id,
                "sender_name": sender_name,
            },
            created_at=time.time(),
        )
        self._queue.add(task)
