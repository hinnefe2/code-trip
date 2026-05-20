"""Top-level voice dispatch + the queue / focused-app mode flip.

The orchestrator runs in one of two **app modes**:

- ``focused`` — today's behavior. Voice routes via ``modes.handle_voice``
  (its globals, voice phrases, and focused-app-aware dispatch).
- ``queue`` — voice operates against the task queue. Globals are queue
  ops (next / skip / dismiss / snooze / what's in the queue). PTT input
  with an active task dispatches against that task's ``kind``.

The flip is triggered by the ACT solo tap (see ``chords.handle_tap``).
Each flip plays a distinct mode chime; the two chimes are tuned to
sound clearly different so the direction of the flip is audible
without a spoken label.

Auto-announce: when in queue mode with no active task, a background
consumer thread pulls the highest-scoring pending task and announces it.
The consumer wakes on any queue mutation and on a short timer (for
``ready_at`` tasks).
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TYPE_CHECKING

from code_trip2 import earcon, modes, remote, tasks as tasks_mod
from code_trip2.tasks import Task

if TYPE_CHECKING:
    from code_trip2.modes import Context

logger = logging.getLogger(__name__)


# --- mode flip -------------------------------------------------------------


MODE_QUEUE = "queue"
MODE_FOCUSED = "focused"


def flip_mode(ctx: "Context") -> None:
    """Toggle between queue and focused app-modes.

    Only an earcon is played — the two mode chimes are distinct enough
    that a spoken label would be redundant and noisy.
    """
    new = MODE_QUEUE if ctx.app_mode != MODE_QUEUE else MODE_FOCUSED
    ctx.app_mode = new
    logger.info("App mode → %s", new)
    try:
        earcon.mode_chime(new)
    except earcon.EarconError:
        pass
    ctx.log.event("app_mode", mode=new)


# --- top-level voice -------------------------------------------------------


def handle_voice(ctx: "Context", transcript: str) -> None:
    """Route a PTT transcript: queue mode first, else fall through to focused."""
    t = transcript.strip()
    if not t:
        return
    if ctx.app_mode == MODE_QUEUE:
        _handle_queue_voice(ctx, t)
    else:
        modes.handle_voice(ctx, t)


# --- queue-mode voice handlers --------------------------------------------


_SNOOZE_RE = re.compile(
    r"^snooze(?:\s+(?:for\s+)?(\d+)\s*(second|seconds|sec|minute|minutes|min|hour|hours|hr|hrs)?)?$"
)


def _handle_queue_voice(ctx: "Context", t: str) -> None:
    low = t.lower().strip(" .!?")

    # Globals first.
    if low in ("stop", "cancel", "stop talking", "shut up", "be quiet"):
        modes.stop_playback(ctx)
        return
    if (
        low == "what" or low.startswith("repeat")
        or "say that again" in low or "say it again" in low
    ):
        if not modes.replay_last(ctx):
            _speak(ctx, "Nothing to repeat.")
        return
    if low.startswith("status"):
        _speak(ctx, f"Queue mode. Window {ctx.active_window}.")
        return

    # Queue ops.
    if low in ("next", "what's next", "whats next"):
        _announce_next(ctx)
        return
    if low in ("skip", "later", "skip it"):
        _skip_current(ctx)
        return
    if low in ("dismiss", "drop it", "drop", "done", "done with this"):
        _drop_current(ctx)
        return
    m = _SNOOZE_RE.match(low)
    if m:
        _snooze_current(ctx, m.group(1), m.group(2))
        return
    if low in ("what's in the queue", "whats in the queue", "how many", "queue", "inbox"):
        _announce_count(ctx)
        return
    if low in ("go on", "continue", "tell me more", "expand"):
        _announce_body(ctx)
        return

    # Manual task add.
    m = re.match(r"^(?:add|remind me to|note)\s+(?:a\s+)?(?:task\s+)?(.+)$", low)
    if m:
        _add_manual(ctx, m.group(1))
        return

    # Anything else: dispatch against the active task by kind.
    if ctx.current_task is not None:
        _dispatch_task_response(ctx, ctx.current_task, t)
        return

    # No active task: fall through to focused-mode voice phrases (window
    # switching etc. is still useful from queue mode for setup).
    modes.handle_voice(ctx, t)


# --- queue ops -------------------------------------------------------------


def _announce_next(ctx: "Context") -> Task | None:
    """Pull the highest-scoring task and announce its headline."""
    if ctx.current_task is not None:
        # Treat 'next' with an active task as "skip this one".
        _skip_current(ctx)
    t = ctx.queue.pull(now=time.time(), recent=ctx.recent_topics)
    if t is None:
        _speak(ctx, "Queue is empty.")
        return None
    if ctx.queue_log is not None:
        ctx.queue_log.record("pull", t)
    ctx.current_task = t
    ctx.recent_topics.touch(t.topic)
    _announce_headline(ctx, t)
    return t


def _skip_current(ctx: "Context") -> None:
    t = ctx.current_task
    ctx.current_task = None
    if t is None:
        return
    ctx.queue.defer(t.id, 300.0)  # 5-minute soft-defer
    _speak(ctx, "Skipped.")


def _drop_current(ctx: "Context") -> None:
    t = ctx.current_task
    ctx.current_task = None
    if t is None:
        _speak(ctx, "Nothing active.")
        return
    ctx.queue.mark_done(t.id)
    _speak(ctx, "Done.")


def _snooze_current(ctx: "Context", amount: str | None, unit: str | None) -> None:
    t = ctx.current_task
    if t is None:
        _speak(ctx, "Nothing active.")
        return
    seconds = _parse_snooze_seconds(amount, unit)
    ctx.current_task = None
    ctx.queue.defer(t.id, seconds)
    _speak(ctx, f"Snoozed {int(seconds)} seconds.")


def _parse_snooze_seconds(amount: str | None, unit: str | None) -> float:
    if amount is None:
        return 600.0  # default 10 min
    try:
        n = int(amount)
    except (TypeError, ValueError):
        return 600.0
    u = (unit or "minute").lower()
    if u.startswith("sec"):
        return float(n)
    if u.startswith("min"):
        return float(n * 60)
    if u.startswith("hour") or u.startswith("hr"):
        return float(n * 3600)
    return float(n * 60)


def _announce_count(ctx: "Context") -> None:
    counts = ctx.queue.count_by_kind()
    if not counts:
        _speak(ctx, "Queue is empty.")
        return
    total = sum(counts.values())
    parts = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(counts.items()))
    _speak(ctx, f"{total} pending. {parts}.")


def _add_manual(ctx: "Context", body: str) -> None:
    t = Task(
        kind="note",
        topic="inbox",
        headline=body[:80],
        body=body,
    )
    ctx.queue.add(t)
    _speak(ctx, "Added.")


# --- announcement ---------------------------------------------------------


def _announce_headline(ctx: "Context", task: Task) -> None:
    """Speak the task headline. Body is held until user asks for it."""
    label = _kind_label(task)
    text = f"{label}: {task.headline}" if task.headline else label
    modes.speak_chunked(ctx, text)


def _announce_body(ctx: "Context") -> None:
    t = ctx.current_task
    if t is None or not t.body:
        _speak(ctx, "Nothing to expand.")
        return
    modes._speak_chunked(ctx, t.body)  # type: ignore[attr-defined]


def _kind_label(task: Task) -> str:
    if task.kind == "claude_reply":
        return f"Claude in {task.topic} replied"
    if task.kind == "slack_msg":
        return f"Slack in {task.topic}"
    if task.kind == "note":
        return "Note"
    if task.kind == "web":
        return "Web link"
    return task.kind


# --- per-kind response dispatch -------------------------------------------


def _dispatch_task_response(ctx: "Context", task: Task, transcript: str) -> None:
    """Route a free-form PTT transcript to whatever makes sense for this task."""
    if task.kind == "claude_reply":
        _respond_claude(ctx, task, transcript)
        return
    if task.kind == "slack_msg":
        _speak(ctx, "Slack reply is not yet implemented.")
        return
    _speak(ctx, "No response action for this task.")


def _respond_claude(ctx: "Context", task: Task, transcript: str) -> None:
    """Send the transcript to the source tmux window. Don't block — the
    ClaudeProducer will surface the next reply as a new task when ready."""
    win = task.source.get("window") or task.topic
    host, opts = ctx.ssh
    try:
        remote.send(host, opts, ctx.config.tmux_session, win, transcript)
    except remote.RemoteError as exc:
        _speak(ctx, f"Could not reach Claude: {exc}")
        return
    ctx.queue.mark_done(task.id)
    ctx.current_task = None
    ctx.log.event(
        "queue_turn", task_id=task.id, kind=task.kind, topic=task.topic, sent=transcript
    )
    _speak(ctx, "Sent.")


# --- macropad tap delegates (queue-mode YES/NO/ACT) -----------------------


def queue_yes_tap(ctx: "Context") -> None:
    """YES in queue mode: engage with current task or pull next."""
    if ctx.current_task is None:
        _announce_next(ctx)
        return
    _announce_body(ctx)


def queue_no_tap(ctx: "Context") -> None:
    """NO in queue mode: skip current task."""
    if ctx.current_task is None:
        _speak(ctx, "Nothing active.")
        return
    _skip_current(ctx)


# --- consumer / auto-announce thread --------------------------------------


class QueueConsumer:
    """Background thread that auto-announces when queue mode is idle.

    Wakes on every queue mutation and on a short poll interval (so
    ``ready_at`` snoozed tasks surface when their time arrives).
    """

    def __init__(self, ctx: "Context", *, poll_interval: float = 2.0) -> None:
        self._ctx = ctx
        self._poll = poll_interval
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def attach(self) -> None:
        """Subscribe to queue events so producer adds wake the consumer."""
        self._ctx.queue.add_listener(self._on_event)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _on_event(self, _kind: str, _task: Task) -> None:
        self._wake.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=self._poll)
            self._wake.clear()
            if self._stop.is_set():
                break
            self._maybe_announce()

    def _maybe_announce(self) -> None:
        ctx = self._ctx
        if ctx.app_mode != MODE_QUEUE:
            return
        if ctx.current_task is not None:
            return
        if modes.is_playback_active(ctx):
            return
        ranked = ctx.queue.ranked(now=time.time(), recent=ctx.recent_topics)
        if not ranked:
            return
        # Earcon to flag a new task before announcing.
        try:
            earcon.new_task()
        except earcon.EarconError:
            pass
        _announce_next(ctx)


# --- helpers ---------------------------------------------------------------


def _speak(ctx: "Context", text: str) -> None:
    if not text:
        return
    try:
        ctx.tts.speak(text)
    except Exception:
        logger.exception("TTS failed for: %s", text)
