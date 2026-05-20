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
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from code_trip2.config import Config
from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.slack_state import SlackState
from code_trip2.tasks import Task, TaskQueue


_USER_ID_RE = re.compile(r"User ID:\s*(U[A-Z0-9]+)", re.IGNORECASE)

# Slack message-format markup. The Slack message body is a "mrkdwn"
# string where mentions, channels, links, and broadcasts are encoded
# with angle-bracketed forms. Reading those aloud sounds awful
# ("less-than at U-zero-two-L-five..."), so before we hand text to
# TTS or use it for a task headline we collapse them to natural
# spoken-English equivalents.
#
# Forms handled:
#   <@U123|Alice>           → "Alice"
#   <@U123>                 → "" (no display name to read; drop)
#   <#C123|general>         → "#general"
#   <#C123>                 → "a channel"
#   <!channel>              → "channel"   (also @here, @everyone)
#   <!subteam^S123|@team>   → "team"
#   <https://x.com|label>   → "label"
#   <https://x.com>         → "" (URL alone reads worse than nothing)
_RE_USER_MENTION = re.compile(r"<@[UW][A-Z0-9]+(?:\|([^>]+))?>")
_RE_CHANNEL_MENTION = re.compile(r"<#[CG][A-Z0-9]+(?:\|([^>]+))?>")
_RE_BROADCAST = re.compile(r"<!(channel|here|everyone)>")
_RE_SUBTEAM = re.compile(r"<!subteam\^[A-Z0-9]+(?:\|@?([^>]+))?>")
_RE_LINK = re.compile(r"<(https?://[^|>]+)(?:\|([^>]+))?>")
_RE_WHITESPACE_RUN = re.compile(r"[ \t]{2,}")


def slack_to_plain_text(text: str) -> str:
    """Strip Slack mrkdwn markup so the message reads cleanly aloud."""
    if not text:
        return text
    text = _RE_USER_MENTION.sub(lambda m: (m.group(1) or "").strip(), text)
    text = _RE_CHANNEL_MENTION.sub(
        lambda m: f"#{m.group(1)}" if m.group(1) else "a channel", text
    )
    text = _RE_BROADCAST.sub(lambda m: m.group(1), text)
    text = _RE_SUBTEAM.sub(lambda m: (m.group(1) or "team").strip(), text)
    text = _RE_LINK.sub(lambda m: (m.group(2) or "").strip(), text)
    # Slack escapes &, <, > in message text. Decode for spoken readability.
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    # Collapse the whitespace that markup removal can leave behind.
    text = _RE_WHITESPACE_RUN.sub(" ", text)
    return text.strip()

# Parsers for the Slack MCP's ``response_format=detailed`` markdown output.
# Each result block looks like::
#
#     ### Result 1 of N
#     Channel: #<name> (ID: C0…)
#     From: <Display Name> (ID: U0…)  [BOT]
#     Time: 2026-05-20 13:43:54 CDT
#     Message_ts: 1779302634.710049
#     Reply count: 5             (optional)
#     Permalink: [link](https://…?thread_ts=…&cid=…)
#     Text:
#     <body, possibly multiline>
#
#     ---
_BLOCK_SPLIT_RE = re.compile(r"^### Result \d+ of \d+\s*$", re.MULTILINE)
_CHANNEL_RE = re.compile(r"^Channel:\s+#(\S+)\s+\(ID:\s+(\w+)\)\s*$")
_FROM_RE = re.compile(r"^From:\s+(.+?)\s+\(ID:\s+(\w+)\)(?:\s+\[(BOT)\])?\s*$")
_TS_RE = re.compile(r"^Message_ts:\s+(\d+\.\d+)\s*$")
_THREAD_TS_RE = re.compile(r"thread_ts=(\d+\.\d+)")

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _previous_workday_5pm_unix(now: datetime | None = None) -> int:
    """Unix timestamp for 5pm of the most recent weekday before ``now``.

    Used as the search ``after`` floor on first startup (no saved state).
    Walks back day-by-day from yesterday until we hit a Mon-Fri date.
    Times are interpreted in the system's local timezone.

    Mapping by day-of-week the orchestrator is started:

    - **Mon**: previous workday = Fri (skips weekend) — catches all of
      Fri-after-5pm and the whole weekend.
    - **Tue–Fri**: previous workday = yesterday — catches anything since
      end-of-day yesterday.
    - **Sat/Sun**: previous workday = Fri — catches Fri-after-5pm
      forward.
    """
    if now is None:
        now = datetime.now()
    candidate = (now - timedelta(days=1)).date()
    while candidate.weekday() >= 5:  # Sat=5, Sun=6
        candidate -= timedelta(days=1)
    target = datetime(candidate.year, candidate.month, candidate.day, 17, 0, 0)
    return int(target.timestamp())


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
        # Resolve user_id inside the worker thread, not here — the
        # claude --print call takes ~10 s and would block the
        # orchestrator's whole startup (no logs, no dashboard) until
        # it returns.
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
        if not self._setup_in_thread():
            return
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("SlackProducer poll failed")
            if self._stop.wait(self._config.slack_poll_interval):
                return

    def _setup_in_thread(self) -> bool:
        """Resolve user_id from inside the worker thread.

        Returns ``True`` if setup succeeded and the poll loop should
        proceed, ``False`` otherwise (failed MCP call or empty user_id).
        Split out from :meth:`_run` so tests can drive the setup path
        without sitting through the 2 s startup stagger.
        """
        try:
            self._user_id = self._fetch_user_id()
        except ClaudeMCPError as exc:
            logger.warning("SlackProducer: could not resolve user_id (%s); idling.", exc)
            return False
        if not self._user_id:
            logger.warning("SlackProducer: empty user_id from slack_read_user_profile; idling.")
            return False
        logger.info("SlackProducer: ready (user_id=%s)", self._user_id)
        return True

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
                    "response_format": "detailed",
                },
            )
        except ClaudeMCPError as exc:
            logger.warning("SlackProducer: search call failed: %s", exc)
            return

        messages = self._extract_messages(result)
        emitted = 0
        skipped_bot = 0
        skipped_old = 0
        for msg in messages:
            ts = msg.get("ts") or ""
            if not ts:
                continue
            if last_ts and ts <= last_ts:
                skipped_old += 1
                continue
            if ts in self._recent_ids:
                skipped_old += 1
                continue
            if msg.get("is_bot"):
                skipped_bot += 1
                self._recent_ids.add(ts)
                self._state.set_last_search_ts(ts)
                continue
            try:
                self._emit_task(msg)
                emitted += 1
            except Exception:
                logger.exception("Failed to emit Slack task for ts=%s", ts)
            self._recent_ids.add(ts)
            self._state.set_last_search_ts(ts)

        logger.info(
            "SlackProducer: poll found %d matches (%d emitted, %d bot, %d already-seen)",
            len(messages),
            emitted,
            skipped_bot,
            skipped_old,
        )

        # Cap the in-memory dedup set so it doesn't grow unbounded.
        if len(self._recent_ids) > 500:
            self._recent_ids = set(sorted(self._recent_ids)[-250:])

    def _after_param(self, last_ts: str | None) -> str:
        """Compute the ``after`` argument for the search call.

        Slack's ``after`` is a Unix timestamp in seconds. From a Slack
        message ts string we take the integer portion. From an empty
        state, we floor at 5pm of the previous workday (see
        :func:`_previous_workday_5pm_unix`).
        """
        if last_ts:
            try:
                return str(int(float(last_ts)))
            except ValueError:
                pass
        return str(_previous_workday_5pm_unix())

    # ---- response shape -------------------------------------------------

    def _extract_messages(self, result: dict) -> list[dict]:
        """Parse the Slack MCP's ``response_format=detailed`` markdown.

        The MCP returns ``{"results": "# Search Results for: …\\n### Result
        1 of N\\nChannel: …\\nFrom: …\\n…"}``. We split on ``### Result``
        headers and pull structured fields out of each block.
        """
        if isinstance(result.get("messages"), list):
            # Future-proof: if the MCP ever returns structured JSON instead
            # of markdown, take it.
            return result["messages"]

        text = result.get("results") or result.get("result") or ""
        if not isinstance(text, str) or not text.strip():
            return []

        # The split keeps the chunks between headers; the first one is the
        # preamble (e.g. "# Search Results for: …") and we drop it.
        blocks = _BLOCK_SPLIT_RE.split(text)
        return [parsed for b in blocks[1:] if (parsed := self._parse_block(b))]

    def _parse_block(self, block: str) -> dict | None:
        """Pull (channel, sender, ts, thread_ts, text, is_bot) out of one
        result block. Returns None if any required field is missing."""
        msg: dict = {"is_bot": False}
        text_lines: list[str] = []
        in_text = False
        for line in block.splitlines():
            stripped = line.strip()
            if stripped == "---":
                break
            if in_text:
                text_lines.append(line)
                continue
            if not stripped:
                continue
            m = _CHANNEL_RE.match(stripped)
            if m:
                msg["channel_name"] = m.group(1)
                msg["channel_id"] = m.group(2)
                continue
            m = _FROM_RE.match(stripped)
            if m:
                msg["sender_name"] = m.group(1).strip()
                msg["sender_id"] = m.group(2)
                msg["is_bot"] = m.group(3) == "BOT"
                continue
            m = _TS_RE.match(stripped)
            if m:
                msg["ts"] = m.group(1)
                continue
            if stripped.startswith("Permalink:"):
                t = _THREAD_TS_RE.search(stripped)
                msg["thread_ts"] = t.group(1) if t else msg.get("ts", "")
                continue
            if stripped.startswith("Text:"):
                in_text = True
                continue
            # Time, Reply count, etc. — ignore for now.

        body = "\n".join(text_lines).strip()
        if not msg.get("ts") or not msg.get("channel_id") or not body:
            return None
        msg["text"] = body
        msg.setdefault("thread_ts", msg["ts"])
        return msg

    def _emit_task(self, msg: dict) -> None:
        raw_text = (msg.get("text") or "").strip()
        if not raw_text:
            return
        text = slack_to_plain_text(raw_text)
        if not text:
            return
        channel_id = msg.get("channel_id") or ""
        channel_name = msg.get("channel_name") or "channel"
        sender_id = msg.get("sender_id") or ""
        sender_name = msg.get("sender_name") or sender_id or "someone"
        ts = msg.get("ts") or ""
        thread_ts = msg.get("thread_ts") or ts
        headline = f"{sender_name}: {text[:60]}"
        task = Task(
            kind="slack_msg",
            topic=f"slack-{channel_name}",
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
