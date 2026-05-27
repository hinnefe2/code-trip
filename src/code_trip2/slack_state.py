"""Persistent Slack producer state.

v2 of the producer makes a single mention-search call per poll, so we
only need one cursor: the highest message timestamp we've already
surfaced. Stored at ``~/.code-trip/slack-state.json`` so the cursor
survives restarts and old messages don't resurface.

The file holds a flat ``{"last_search_ts": "1716000000.000200", ...}``
JSON object. Other keys are reserved for future producers that need
their own cursors.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


_LAST_SEARCH_KEY = "last_search_ts"
_CHANNEL_KEY_PREFIX = "channel:"


def default_state_path() -> Path:
    return Path.home() / ".code-trip" / "slack-state.json"


class SlackState:
    """JSON-backed cursor store for the Slack producer.

    Single-loop discipline (one asyncio task at a time touches this
    object) replaces the previous ``threading.Lock``.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_state_path()
        self._data: dict[str, str] = self._load()

    # ---- last_search_ts (high-water mark across all channels) -----------

    def last_search_ts(self) -> str | None:
        return self._data.get(_LAST_SEARCH_KEY)

    def set_last_search_ts(self, ts: str) -> None:
        """Advance the cursor. Refuses to go backwards.

        Slack timestamps are strings like ``"1716000000.000200"`` where
        the lexicographic order matches the numeric order, so we just
        string-compare.
        """
        if not ts:
            return
        prior = self._data.get(_LAST_SEARCH_KEY)
        if prior is not None and prior >= ts:
            return
        self._data[_LAST_SEARCH_KEY] = ts
        self._save()

    # ---- per-channel cursor (for watched channels) ----------------------

    def last_channel_ts(self, channel_id: str) -> str | None:
        return self._data.get(_CHANNEL_KEY_PREFIX + channel_id)

    def set_last_channel_ts(self, channel_id: str, ts: str) -> None:
        if not ts or not channel_id:
            return
        key = _CHANNEL_KEY_PREFIX + channel_id
        prior = self._data.get(key)
        if prior is not None and prior >= ts:
            return
        self._data[key] = ts
        self._save()

    def all(self) -> dict[str, str]:
        return dict(self._data)

    # ---- internals ------------------------------------------------------

    def _load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s; starting fresh.", self.path, exc_info=True)
            return {}
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Could not create %s", self.path.parent, exc_info=True)
            return
        try:
            fd, tmp = tempfile.mkstemp(
                prefix=".slack-state-", suffix=".json", dir=str(self.path.parent)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            logger.warning("Could not write %s", self.path, exc_info=True)
