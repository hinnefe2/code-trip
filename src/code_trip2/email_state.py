"""Persistent EmailProducer state.

Tracks the highest Unix timestamp we've already surfaced from the Gmail
inbox. Stored at ``~/.code-trip/email-state.json`` so the cursor
survives restarts and old messages don't resurface.

Mirror of :mod:`code_trip2.slack_state` — kept in a sibling file rather
than unified so each producer's persistence is independent and the
user can wipe one without touching the other.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


_LAST_MESSAGE_TS_KEY = "last_message_ts"


def default_state_path() -> Path:
    return Path.home() / ".code-trip" / "email-state.json"


class EmailState:
    """Thread-safe JSON-backed cursor store for the Email producer."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_state_path()
        self._lock = threading.Lock()
        self._data: dict[str, int] = self._load()

    def last_message_ts(self) -> int | None:
        with self._lock:
            v = self._data.get(_LAST_MESSAGE_TS_KEY)
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def set_last_message_ts(self, ts: int) -> None:
        """Advance the cursor. Refuses to go backwards."""
        if not ts or int(ts) <= 0:
            return
        new = int(ts)
        with self._lock:
            prior = self._data.get(_LAST_MESSAGE_TS_KEY)
            if prior is not None and int(prior) >= new:
                return
            self._data[_LAST_MESSAGE_TS_KEY] = new
        self._save()

    def all(self) -> dict[str, int]:
        with self._lock:
            return dict(self._data)

    def _load(self) -> dict[str, int]:
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
        out: dict[str, int] = {}
        for k, v in data.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Could not create %s", self.path.parent, exc_info=True)
            return
        try:
            fd, tmp = tempfile.mkstemp(
                prefix=".email-state-", suffix=".json", dir=str(self.path.parent)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            logger.warning("Could not write %s", self.path, exc_info=True)
