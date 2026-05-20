"""SlackProducer: polls Slack via slack_sdk and emits filtered tasks.

Direct Web API integration — no MCP layer. On start, resolves the
configured channel **names** to channel **IDs** (via
``conversations.list``) and looks up the bot's own user_id (via
``auth.test``) if the config didn't provide one.

Per poll tick (``slack_poll_interval``):

1. For each known channel, call ``conversations.history`` with
   ``oldest = SlackState.last_ts(channel_id)`` so we only see new
   messages.
2. Skip messages from bots, the user themselves, and edits/threads we
   don't care about.
3. Pass each survivor through :class:`SlackFilter` (LLM relevance
   check). Only ``relevant: true`` messages become tasks.
4. Emit ``Task(kind="slack_msg", topic=f"slack-{name}")`` with the
   channel id, ts, and thread_ts on ``source`` so the reply path can
   thread back correctly.
5. Persist the new last-seen timestamp for the channel.

Threading model: one daemon thread runs the poll loop. All
``slack_sdk`` calls block, which is fine on the dedicated thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from code_trip2.config import Config
from code_trip2.slack_filter import SlackFilter, SlackFilterError
from code_trip2.slack_state import SlackState
from code_trip2.tasks import (
    URGENCY_BACKGROUND,
    URGENCY_INTERRUPT,
    URGENCY_NORMAL,
    Task,
    TaskQueue,
)

if TYPE_CHECKING:
    from slack_sdk import WebClient

logger = logging.getLogger(__name__)


_URGENCY_MAP = {
    "interrupt": URGENCY_INTERRUPT,
    "normal": URGENCY_NORMAL,
    "background": URGENCY_BACKGROUND,
}


class SlackProducer:
    name = "slack"

    def __init__(
        self,
        *,
        config: Config,
        queue: TaskQueue,
        client: "WebClient | None" = None,
        filter_: SlackFilter | None = None,
        state: SlackState | None = None,
    ) -> None:
        self._config = config
        self._queue = queue
        self._client = client
        self._filter = filter_
        self._state = state or SlackState()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Resolved lazily in _setup. Maps channel_id → display name.
        self._channels: dict[str, str] = {}
        # User cache for sender-name lookups.
        self._users: dict[str, str] = {}
        self._user_id: str = config.slack_user_id

    # ---- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if self._client is None:
            logger.info("SlackProducer: no client configured; not starting.")
            return
        if not self._config.slack_channels:
            logger.info("SlackProducer: no channels configured; not starting.")
            return
        if self._filter is None or not self._filter.enabled:
            logger.warning(
                "SlackProducer: no filter (or filter disabled); messages "
                "would flood the queue. Not starting."
            )
            return
        try:
            self._setup()
        except SlackProducerError as exc:
            logger.warning("SlackProducer setup failed: %s", exc)
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ---- setup ----------------------------------------------------------

    def _setup(self) -> None:
        """Resolve channel names → IDs and identify the bot user."""
        if not self._user_id:
            try:
                resp = self._client.auth_test()
            except Exception as exc:
                raise SlackProducerError(f"auth.test failed: {exc}") from exc
            self._user_id = resp.get("user_id", "") or ""
            if not self._user_id:
                logger.warning("SlackProducer: auth.test returned no user_id.")

        wanted = {name.lstrip("#") for name in self._config.slack_channels}
        try:
            cursor: str | None = None
            seen: dict[str, str] = {}
            while True:
                resp = self._client.conversations_list(
                    limit=200,
                    types="public_channel,private_channel",
                    cursor=cursor,
                )
                for ch in resp.get("channels", []):
                    cid = ch.get("id")
                    cname = ch.get("name")
                    if cid and cname and cname in wanted:
                        seen[cid] = cname
                cursor = resp.get("response_metadata", {}).get("next_cursor", "")
                if not cursor:
                    break
        except Exception as exc:
            raise SlackProducerError(f"conversations.list failed: {exc}") from exc

        if not seen:
            raise SlackProducerError(
                f"None of the configured channels were found: {sorted(wanted)}"
            )
        self._channels = seen
        missing = wanted - set(seen.values())
        if missing:
            logger.warning(
                "SlackProducer: channels not found / not joined: %s", sorted(missing)
            )

    # ---- poll loop ------------------------------------------------------

    def _run(self) -> None:
        # Stagger a tiny bit so the producer isn't fighting for the network
        # the instant the orchestrator starts.
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
        for channel_id, channel_name in list(self._channels.items()):
            self._poll_channel(channel_id, channel_name)

    def _poll_channel(self, channel_id: str, channel_name: str) -> None:
        oldest = self._state.last_ts(channel_id)
        try:
            resp = self._client.conversations_history(
                channel=channel_id,
                oldest=oldest or None,
                limit=50,
            )
        except Exception as exc:
            logger.warning(
                "conversations.history failed for %s: %s", channel_name, exc
            )
            return

        # Slack returns messages newest-first. We want to process oldest-first
        # so last_ts only advances after we've fully handled earlier messages.
        messages = list(reversed(resp.get("messages", [])))
        if not messages:
            return

        for msg in messages:
            ts = msg.get("ts")
            if not ts:
                continue
            if oldest and ts <= oldest:
                continue
            try:
                self._handle_message(msg, channel_id, channel_name)
            except Exception:
                logger.exception("Failed to handle message in %s", channel_name)
            self._state.set_last_ts(channel_id, ts)

    def _handle_message(self, msg: dict, channel_id: str, channel_name: str) -> None:
        # Skip non-user messages, edits, joins, and our own posts.
        subtype = msg.get("subtype") or ""
        if subtype in ("channel_join", "channel_leave", "message_changed", "message_deleted"):
            return
        if msg.get("bot_id"):
            return
        sender_id = msg.get("user") or ""
        if not sender_id or sender_id == self._user_id:
            return

        text = msg.get("text") or ""
        if not text.strip():
            return

        sender_name = self._resolve_user_name(sender_id)
        try:
            verdict = self._filter.evaluate(
                text=text,
                sender_name=sender_name,
                channel_name=channel_name,
                is_dm=False,
                user_id=self._user_id,
            )
        except SlackFilterError as exc:
            logger.warning("Slack filter failed; skipping message: %s", exc)
            return

        if not verdict.get("relevant"):
            return

        headline = verdict.get("headline") or f"{sender_name}: {text.strip()[:60]}"
        urgency_label = verdict.get("urgency", "normal")
        urgency = _URGENCY_MAP.get(urgency_label, URGENCY_NORMAL)
        ts = msg.get("ts", "")
        thread_ts = msg.get("thread_ts") or ts

        task = Task(
            kind="slack_msg",
            topic=f"slack-{channel_name}",
            headline=headline,
            body=text,
            urgency=urgency,
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

    def _resolve_user_name(self, user_id: str) -> str:
        if not user_id:
            return "someone"
        cached = self._users.get(user_id)
        if cached:
            return cached
        try:
            resp = self._client.users_info(user=user_id)
            profile = resp.get("user", {}).get("profile", {}) or {}
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or resp.get("user", {}).get("name")
                or user_id
            )
        except Exception:
            name = user_id
        self._users[user_id] = name
        return name


class SlackProducerError(Exception):
    pass
