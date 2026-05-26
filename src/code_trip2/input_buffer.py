"""Thread-safe line-edit buffer for the in-TUI input box.

In local-STT mode the orchestrator puts stdin in cbreak (non-canonical,
no-echo) mode so the reader thread sees keystrokes — and pasted
transcripts — as the bytes arrive, instead of waiting for a newline.
That breaks normal terminal echo, so we render the accumulated input
ourselves in a TUI panel. Editing keys map to buffer ops:

- printable char    → append
- backspace / DEL   → remove last char
- Esc               → clear
- Enter (``\\r``/``\\n``) → submit (dispatched outside this class)

The buffer also tracks a ``last_change`` timestamp so the reader can
auto-submit on a quiet pause — used for Superwhisper-style pastes that
arrive without a trailing newline.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class InputBuffer:
    _chars: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _last_change: float = field(default_factory=time.time)

    def append(self, ch: str) -> None:
        with self._lock:
            self._chars.append(ch)
            self._last_change = time.time()

    def backspace(self) -> None:
        with self._lock:
            if self._chars:
                self._chars.pop()
                self._last_change = time.time()

    def clear(self) -> None:
        with self._lock:
            self._chars = []
            self._last_change = time.time()

    def get(self) -> str:
        with self._lock:
            return "".join(self._chars)

    def pop(self) -> str:
        """Return the current contents and clear the buffer atomically."""
        with self._lock:
            out = "".join(self._chars)
            self._chars = []
            self._last_change = time.time()
            return out

    def is_empty(self) -> bool:
        with self._lock:
            return not self._chars

    def quiet_seconds(self) -> float:
        """Seconds since the last buffer mutation."""
        with self._lock:
            return time.time() - self._last_change
