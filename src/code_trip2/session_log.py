"""Per-session JSONL event log for offline analysis and prompt tuning."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)


def default_session_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.home() / ".code-trip" / "sessions" / f"{stamp}.jsonl"


class SessionLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: TextIO | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", buffering=1, encoding="utf-8")
        except OSError:
            logger.warning("Could not open session log %s", path, exc_info=True)

    def event(self, kind: str, **fields: Any) -> None:
        if self._fh is None:
            return
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "kind": kind,
            **fields,
        }
        try:
            self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except (OSError, TypeError):
            logger.warning("Failed to write session event %s", kind, exc_info=True)

    def close(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.close()
        finally:
            self._fh = None

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
