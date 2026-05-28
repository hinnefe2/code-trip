"""Persistent LinearProducer state.

Tracks the highest ``updatedAt`` ISO-8601 string we've already surfaced
from Linear. Stored at ``~/.code-trip/linear-state.json`` so the cursor
survives restarts and the incremental ``updatedAt: <iso>`` query keeps
working across sessions.

This is *not* the dedup mechanism — Linear is the source of truth. On
startup ``main.py`` drops any replayed ``linear_issue`` tasks and the
producer's first poll does a wide pull (no ``updatedAt`` floor). The
cursor just keeps mid-session incremental polls cheap.

Mirror of :mod:`code_trip2.email_state` — kept in a sibling file rather
than unified so each producer's persistence is independent and the
user can wipe one without touching the other.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


_LAST_UPDATED_AT_KEY = "last_updated_at"


def default_state_path() -> Path:
    return Path.home() / ".code-trip" / "linear-state.json"


class LinearState:
    """JSON-backed cursor store for the Linear producer.

    Single-loop discipline (one asyncio task at a time touches this
    object) replaces any threading lock.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or default_state_path()
        self._data: dict[str, str] = self._load()

    def last_updated_at(self) -> str | None:
        v = self._data.get(_LAST_UPDATED_AT_KEY)
        if not v:
            return None
        return str(v)

    def set_last_updated_at(self, ts: str) -> None:
        """Advance the cursor. Refuses to go backwards.

        ISO-8601 strings compare lexicographically as long as they share
        the same format and timezone — Linear's MCP returns them as
        UTC ``2026-05-28T17:00:53.328Z`` so the ordering is consistent.
        """
        if not ts:
            return
        new = str(ts)
        prior = self._data.get(_LAST_UPDATED_AT_KEY)
        if prior is not None and str(prior) >= new:
            return
        self._data[_LAST_UPDATED_AT_KEY] = new
        self._save()

    def all(self) -> dict[str, str]:
        return dict(self._data)

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
        return {str(k): str(v) for k, v in data.items() if v is not None}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning("Could not create %s", self.path.parent, exc_info=True)
            return
        try:
            fd, tmp = tempfile.mkstemp(
                prefix=".linear-state-", suffix=".json", dir=str(self.path.parent)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            logger.warning("Could not write %s", self.path, exc_info=True)
