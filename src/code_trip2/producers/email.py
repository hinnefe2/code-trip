"""EmailProducer: polls Gmail inbox via the claude.ai Gmail MCP.

Same auth-passthrough pattern as :class:`SlackProducer`: goes through
:class:`ClaudeMCPClient` pointed at the claude.ai Gmail MCP server, so
auth piggy-backs on whatever the user already authorized in claude.ai
(no Google OAuth app to install). Per poll tick we make one
``search_threads`` call with ``after:<unix_ts>`` appended to the user's
configured Gmail search query, parse the returned threads, and emit a
task per new thread.

Topic is ``email-<sender-name>`` so multiple messages from the same
person cluster in the scheduler's topic-affinity bonus.

**Reply path**: dispatch._respond_email creates a *draft* (not a
direct send) via ``create_draft``. The claude.ai Gmail MCP doesn't
expose a send tool, but draft-only is the safer default for voice
anyway — the user can review and send from Gmail.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING

from code_trip2.config import Config
from code_trip2.email_state import EmailState
from code_trip2.producers.claude_mcp import ClaudeMCPClient, ClaudeMCPError
from code_trip2.producers.slack import _previous_workday_5pm_unix
from code_trip2.tasks import Task, TaskQueue

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Fallback markdown-block parser, in case the Gmail MCP ever returns a
# detailed-text format like Slack's. The structured-JSON path is the
# expected default — see ``_extract_threads``.
_BLOCK_SPLIT_RE = re.compile(r"^### (?:Result|Thread) \d+ of \d+\s*$", re.MULTILINE)
_SUBJECT_RE = re.compile(r"^Subject:\s*(.*)$", re.IGNORECASE)
_FROM_RE = re.compile(r"^(?:From|Sender):\s*(.*)$", re.IGNORECASE)
_THREAD_ID_RE = re.compile(r"^(?:Thread[ _]ID|Thread):\s*(\S+)", re.IGNORECASE)
_MESSAGE_ID_RE = re.compile(r"^(?:Message[ _]ID|Message):\s*(\S+)", re.IGNORECASE)
_DATE_RE = re.compile(r"^(?:Date|Time|InternalDate):\s*(.+)$", re.IGNORECASE)
_SNIPPET_RE = re.compile(r"^(?:Snippet|Body|Text):\s*(.*)$", re.IGNORECASE)

# "Alice Smith <alice@example.com>" → name + addr.
_NAME_ADDR_RE = re.compile(r'^\s*"?([^"<]*?)"?\s*<([^>]+)>\s*$')
_BARE_EMAIL_RE = re.compile(r"[\w.+\-]+@[\w.\-]+\.[A-Za-z]{2,}")


class EmailProducer:
    name = "email"

    def __init__(
        self,
        *,
        config: Config,
        queue: TaskQueue,
        mcp: ClaudeMCPClient | None = None,
        state: EmailState | None = None,
    ) -> None:
        self._config = config
        self._queue = queue
        self._mcp = mcp
        self._state = state or EmailState()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Per-session dedup keyed by (thread_id, latest_message_id).
        # State file already handles cross-restart dedup via last_message_ts.
        self._recent_keys: set[str] = set()

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._mcp is None or not self._mcp.enabled:
            logger.info("EmailProducer: ClaudeMCPClient unavailable; not starting.")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- poll loop ------------------------------------------------------

    def _run(self) -> None:
        # Small stagger so producers don't all hit claude --print at once.
        if self._stop.wait(3.0):
            return
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception:
                logger.exception("EmailProducer poll failed")
            if self._stop.wait(self._config.email_poll_interval):
                return

    def _poll_once(self) -> None:
        last_ts = self._state.last_message_ts()
        after = self._after_param(last_ts)
        base_query = (self._config.email_search_query or "").strip()
        query = f"{base_query} after:{after}".strip()
        try:
            result = self._mcp.call_tool(
                "search_threads",
                {
                    "query": query,
                    "pageSize": int(self._config.email_max_results),
                },
            )
        except ClaudeMCPError as exc:
            logger.warning("EmailProducer: search call failed: %s", exc)
            return

        threads = self._extract_threads(result)
        emitted = 0
        skipped_old = 0
        max_ts_seen = last_ts or 0
        for th in threads:
            ts = th.get("ts_unix") or 0
            thread_id = th.get("thread_id") or ""
            msg_id = th.get("message_id") or thread_id
            if not thread_id or not ts:
                continue
            key = f"{thread_id}:{msg_id}"
            if last_ts and ts <= last_ts:
                skipped_old += 1
                continue
            if key in self._recent_keys:
                skipped_old += 1
                continue
            try:
                self._emit_task(th)
                emitted += 1
            except Exception:
                logger.exception("Failed to emit email task for thread=%s", thread_id)
            self._recent_keys.add(key)
            if ts > max_ts_seen:
                max_ts_seen = ts

        if max_ts_seen and (last_ts is None or max_ts_seen > last_ts):
            self._state.set_last_message_ts(max_ts_seen)

        logger.info(
            "EmailProducer: poll — %d threads (%d emitted, %d already-seen)",
            len(threads), emitted, skipped_old,
        )

        if len(self._recent_keys) > 500:
            self._recent_keys = set(sorted(self._recent_keys)[-250:])

    def _after_param(self, last_ts: int | None) -> str:
        """Gmail's ``after:`` operator accepts a unix timestamp. Floor at
        5pm of the previous workday on first run."""
        if last_ts and last_ts > 0:
            return str(int(last_ts))
        return str(_previous_workday_5pm_unix())

    # ---- response shape -------------------------------------------------

    def _extract_threads(self, result: dict) -> list[dict]:
        """Normalize the MCP's response into our internal thread shape.

        The Gmail MCP returns structured thread JSON in the common case;
        we also tolerate a Slack-style markdown ``result`` field in case
        the response format ever changes."""
        # Structured shapes — try a few key names.
        for key in ("threads", "items", "messages"):
            value = result.get(key)
            if isinstance(value, list):
                return [self._normalize_structured(t) for t in value if isinstance(t, dict)]

        # Single-thread response shape: result IS the thread.
        if "id" in result and ("messages" in result or "subject" in result):
            return [self._normalize_structured(result)]

        # Markdown / text fallback.
        text = (
            result.get("result")
            or result.get("results")
            or result.get("_raw")
            or ""
        )
        if not isinstance(text, str) or not text.strip():
            return []
        blocks = _BLOCK_SPLIT_RE.split(text)
        out: list[dict] = []
        for b in blocks[1:]:
            parsed = self._parse_markdown_block(b)
            if parsed is not None and parsed.get("thread_id"):
                out.append(parsed)
        return out

    def _normalize_structured(self, t: dict) -> dict:
        """Coerce one structured thread/message dict into our internal shape.

        The MCP returns either a thread (with a ``messages`` list) or a
        single message; pick out the most recent message either way."""
        thread_id = str(
            t.get("threadId") or t.get("thread_id") or t.get("id") or ""
        )
        messages = t.get("messages")
        latest = t
        if isinstance(messages, list) and messages:
            # The Gmail API returns messages in chronological order;
            # take the last one as "latest". If the source is reversed,
            # the timestamp dedup still picks up the new one next poll.
            for m in messages:
                if isinstance(m, dict):
                    latest = m

        message_id = str(
            latest.get("id") or latest.get("messageId") or latest.get("message_id") or ""
        )
        subject = str(latest.get("subject") or t.get("subject") or "")
        sender_raw = str(
            latest.get("from") or latest.get("sender") or t.get("from") or ""
        )
        snippet = str(
            latest.get("snippet")
            or latest.get("plaintextBody")
            or latest.get("plaintext_body")
            or latest.get("text")
            or t.get("snippet")
            or ""
        ).strip()
        ts_raw = (
            latest.get("internalDate")
            or latest.get("internal_date")
            or latest.get("date")
            or latest.get("ts")
            or t.get("internalDate")
            or 0
        )
        return {
            "thread_id": thread_id,
            "message_id": message_id,
            "subject": subject,
            "sender": sender_raw,
            "snippet": snippet,
            "ts_unix": _parse_ts(ts_raw),
        }

    def _parse_markdown_block(self, block: str) -> dict | None:
        out = {
            "thread_id": "",
            "message_id": "",
            "subject": "",
            "sender": "",
            "snippet": "",
            "ts_unix": 0,
        }
        snippet_lines: list[str] = []
        in_snippet = False
        for line in block.splitlines():
            stripped = line.strip()
            if stripped == "---":
                break
            if in_snippet:
                if not stripped:
                    in_snippet = False
                    continue
                snippet_lines.append(line)
                continue
            if not stripped:
                continue
            m = _SUBJECT_RE.match(stripped)
            if m:
                out["subject"] = m.group(1).strip()
                continue
            m = _FROM_RE.match(stripped)
            if m:
                out["sender"] = m.group(1).strip()
                continue
            m = _THREAD_ID_RE.match(stripped)
            if m:
                out["thread_id"] = m.group(1)
                continue
            m = _MESSAGE_ID_RE.match(stripped)
            if m:
                out["message_id"] = m.group(1)
                continue
            m = _DATE_RE.match(stripped)
            if m:
                out["ts_unix"] = _parse_ts(m.group(1).strip())
                continue
            m = _SNIPPET_RE.match(stripped)
            if m:
                rest = m.group(1).strip()
                if rest:
                    snippet_lines.append(rest)
                in_snippet = True
                continue
        out["snippet"] = "\n".join(snippet_lines).strip()
        return out if out["thread_id"] else None

    def _emit_task(self, th: dict) -> None:
        sender_raw = th.get("sender") or ""
        sender_name, sender_email = _split_name_addr(sender_raw)
        subject = th.get("subject") or "(no subject)"
        snippet = th.get("snippet") or ""
        display_name = sender_name or sender_email or "unknown"
        headline = f"{display_name}: {subject[:60]}"
        body = subject if not snippet else f"{subject}. {snippet}"
        topic_key = (sender_name or sender_email or "inbox").lower().replace(" ", "-")
        thread_id = th.get("thread_id") or ""
        source = {
            "thread_id": thread_id,
            "message_id": th.get("message_id") or "",
            "sender_name": sender_name,
            "sender_email": sender_email,
            "subject": subject,
        }

        # Same thread-collapse rule as Slack: one pending task per thread.
        # A reply in an existing thread updates the live task's body to
        # the latest message rather than stacking duplicates.
        existing = self._find_pending_thread_task(thread_id)
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
            kind="email_msg",
            topic=topic_key,
            headline=headline,
            body=body,
            source=source,
            created_at=time.time(),
        )
        self._queue.add(task)

    def _find_pending_thread_task(self, thread_id: str) -> Task | None:
        if not thread_id:
            return None
        for task in self._queue.pending():
            if task.kind != "email_msg":
                continue
            if (task.source or {}).get("thread_id") == thread_id:
                return task
        return None


# --- helpers ----------------------------------------------------------------


def _parse_ts(raw) -> int:
    """Parse a date-ish value into Unix seconds. Returns 0 on failure.

    Handles:
    - Gmail's ``internalDate`` (epoch ms as int or string)
    - ISO-ish strings ``"2026-05-20 13:43:54"`` (with or without offset)
    - Bare unix-seconds ints/strings
    """
    if raw is None:
        return 0
    if isinstance(raw, bool):
        return 0
    if isinstance(raw, (int, float)):
        v = int(raw)
        if v > 10_000_000_000:  # Looks like epoch ms.
            v //= 1000
        return max(0, v)
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return 0
        try:
            v = int(float(s))
            if v > 10_000_000_000:
                v //= 1000
            return max(0, v)
        except ValueError:
            pass
        from datetime import datetime
        for fmt in (
            "%Y-%m-%d %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S",
        ):
            try:
                return int(datetime.strptime(s, fmt).timestamp())
            except ValueError:
                continue
    return 0


def _split_name_addr(raw: str) -> tuple[str, str]:
    """Parse ``"Name <addr@x.com>"`` into ``("Name", "addr@x.com")``.

    Falls back to ``("", addr)`` for bare addresses and ``(raw, "")``
    for name-only inputs.
    """
    if not raw:
        return "", ""
    raw = raw.strip()
    m = _NAME_ADDR_RE.match(raw)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _BARE_EMAIL_RE.search(raw)
    if m and m.group(0) == raw:
        return "", raw
    if m:
        # Name and address concatenated without angle brackets.
        addr = m.group(0)
        name = raw.replace(addr, "").strip(" ,;")
        return name, addr
    return raw, ""
