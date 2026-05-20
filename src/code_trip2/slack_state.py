"""Persistent Slack producer state: last-seen message timestamp per channel.

Stored at ``~/.code-trip/slack-state.json``. Read on producer start,
written after each poll batch. Survives restarts so we don't resurface
messages the user has already seen.

The format is a flat ``{channel_id: ts_str}`` map. Slack timestamps are
strings of the form ``"1716321234.000200"`` (epoch seconds + 6-digit
ordinal); we treat them opaquely and pass them straight back to
``conversations.history`` as the ``oldest`` cursor.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


def default_state_path() -> Path:
    return Path.home() / ".code-trip" / "slack-state.json"


class SlackState:
    """Thread-safe ``channel_id → last_ts`` map with JSON persistence."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_state_path()
        self._lock = threading.Lock()
        self._data: dict[str, str] = self._load()

    def last_ts(self, channel_id: str) -> str | None:
        with self._lock:
            return self._data.get(channel_id)

    def set_last_ts(self, channel_id: str, ts: str) -> None:
        if not ts:
            return
        with self._lock:
            prior = self._data.get(channel_id)
            if prior is not None and prior >= ts:
                # Slack's lexicographic order on these strings is the same as
                # numeric order, so a smaller new ts is a regression — skip.
                return
            self._data[channel_id] = ts
        self._save()

    def all(self) -> dict[str, str]:
        with self._lock:
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
        # Atomic write so a crash mid-save doesn't leave a half-written file.
        try:
            fd, tmp = tempfile.mkstemp(
                prefix=".slack-state-", suffix=".json", dir=str(self.path.parent)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            logger.warning("Could not write %s", self.path, exc_info=True)
