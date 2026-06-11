"""Top-level voice dispatch + the queue / focused-app mode flip.

The orchestrator runs in one of two **app modes**:

- ``focused`` — today's behavior. Voice routes via ``modes.handle_focused_voice``
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

import asyncio
import json
import logging
import re
import time
from typing import TYPE_CHECKING

from code_trip2 import earcon, modes, remote
from code_trip2._async_utils import event_or_timeout
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


async def handle_voice(ctx: "Context", transcript: str) -> None:
    """Route a PTT transcript: queue mode first, else fall through to focused."""
    t = transcript.strip()
    if not t:
        return
    if ctx.app_mode == MODE_QUEUE:
        await _handle_queue_voice(ctx, t)
    else:
        await modes.handle_focused_voice(ctx, t)


async def handle_skill(ctx: "Context", transcript: str) -> None:
    """ACT+PTT path: hand the transcript + task context to Claude.

    The keypress (ACT held during PTT) flags the recording as a skill
    invocation. We ship the transcript, the active task's source data,
    and a brief preamble to ``claude --print``. Claude Code's own skill
    discovery (project ``.claude/skills/``) matches the user's request
    to a skill and runs it.

    Requires an active task — the whole point is to act on it. If
    nothing is active, we just say so.
    """
    t = transcript.strip()
    if not t:
        return
    task = ctx.current_task
    if task is None:
        await _speak(ctx, "Nothing active to act on.")
        return
    mcp = ctx.agent_mcp
    if mcp is None or not getattr(mcp, "enabled", False):
        await _speak(ctx, "Agent MCP is not configured.")
        return
    prompt = _build_skill_prompt(task, t)
    logger.info(
        "Skill mode: invoking agent for task %s with %d allowed tools",
        task.id, len(ctx.agent_allowed_tools),
    )
    ctx.thinking.start()
    try:
        try:
            summary = await mcp.run_agent(
                prompt=prompt,
                allowed_tools=ctx.agent_allowed_tools,
            )
        except Exception as exc:
            logger.exception("Skill invocation failed")
            await _speak(ctx, f"Skill failed: {exc}")
            return
    finally:
        ctx.thinking.stop()
    ctx.queue.mark_done(task.id)
    ctx.current_task = None
    ctx.log.event(
        "queue_turn",
        task_id=task.id,
        task_kind=task.kind,
        topic=task.topic,
        skill_transcript=t,
        skill_summary=summary,
    )
    await _speak(ctx, summary or "Done.")


def _build_skill_prompt(task: Task, transcript: str) -> str:
    """Build the prompt for an ACT+PTT skill invocation.

    Hands Claude the task context plus the user's brief instruction.
    Claude Code's skill discovery picks the matching skill from
    ``.claude/skills/`` based on the skill descriptions.
    """
    try:
        source_json = json.dumps(task.source, default=str)
    except (TypeError, ValueError):
        source_json = "{}"
    return (
        "You are completing a task inside a voice-driven inbox "
        "orchestrator. The user just spoke a brief instruction. Use "
        "the appropriate skill from `.claude/skills/` (and the MCP "
        "tools it references) to do what they asked. Don't ask for "
        "confirmation; act and report back in one sentence.\n"
        "\n"
        f"Task kind: {task.kind}\n"
        f"Task topic: {task.topic}\n"
        f"Task source: {source_json}\n"
        f"Task headline: {task.headline}\n"
        f"Task body:\n{task.body or '(empty)'}\n"
        "\n"
        f'User said: "{transcript}"'
    )


# --- queue-mode voice handlers --------------------------------------------


_SNOOZE_RE = re.compile(
    r"^snooze(?:\s+(?:for\s+)?(\d+)\s*(second|seconds|sec|minute|minutes|min|hour|hours|hr|hrs)?)?$"
)


async def _handle_queue_voice(ctx: "Context", t: str) -> None:
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
            await _speak(ctx, "Nothing to repeat.")
        return
    if low.startswith("status"):
        await _speak(ctx, f"Queue mode. Window {ctx.active_window}.")
        return

    # Queue ops.
    if low in ("next", "what's next", "whats next"):
        await _announce_next(ctx)
        return
    if low in ("skip", "later", "skip it"):
        await _skip_current(ctx)
        return
    if low in ("dismiss", "drop it", "drop", "done", "done with this"):
        await _drop_current(ctx)
        return
    m = _SNOOZE_RE.match(low)
    if m:
        await _snooze_current(ctx, m.group(1), m.group(2))
        return
    if low in ("what's in the queue", "whats in the queue", "how many", "queue", "inbox"):
        await _announce_count(ctx)
        return
    if low in ("go on", "continue", "tell me more", "expand"):
        await _announce_body(ctx)
        return

    # Manual task add.
    m = re.match(r"^(?:add|remind me to|note)\s+(?:a\s+)?(?:task\s+)?(.+)$", low)
    if m:
        await _add_manual(ctx, m.group(1))
        return

    # Anything else: dispatch against the active task by kind.
    if ctx.current_task is not None:
        await _dispatch_task_response(ctx, ctx.current_task, t)
        return

    # No active task: fall through to focused-mode voice phrases (window
    # switching etc. is still useful from queue mode for setup).
    await modes.handle_focused_voice(ctx, t)


# --- queue ops -------------------------------------------------------------


async def _announce_next(ctx: "Context") -> Task | None:
    """Set the cursor to the highest-scoring task and announce it.

    Voice "next" keeps its old semantic of "skip what I'm on, give me
    the next one": when there's already a cursor task, defer it first
    so it drops out of the ranking. The cursor itself is just a
    pointer — the task stays ``pending`` (visible in the queue list)
    until an engagement (PTT / skill / dismiss / snooze) actually
    removes it.
    """
    if ctx.current_task is not None:
        # Treat 'next' with a cursored task as "skip this one".
        await _skip_current(ctx)
    ranked = ctx.queue.ranked(now=time.time(), recent=ctx.recent_topics)
    if not ranked:
        await _speak(ctx, "Queue is empty.")
        return None
    t = ranked[0][0]
    if ctx.queue_log is not None:
        ctx.queue_log.record("pull", t)
    ctx.current_task = t
    ctx.recent_topics.touch(t.topic)
    _announce_headline(ctx, t)
    return t


async def _skip_current(ctx: "Context") -> None:
    t = ctx.current_task
    ctx.current_task = None
    if t is None:
        return
    # Cut any in-flight announcement (headline read, body expansion)
    # the moment the user skips — they've decided this isn't worth
    # finishing. Matches what ``dismiss_current_task`` already does.
    modes.stop_playback(ctx)
    ctx.queue.defer(t.id, 300.0)  # 5-minute soft-defer
    await _speak(ctx, "Skipped.")


async def _drop_current(ctx: "Context") -> None:
    t = ctx.current_task
    ctx.current_task = None
    if t is None:
        await _speak(ctx, "Nothing active.")
        return
    modes.stop_playback(ctx)
    ctx.queue.mark_done(t.id)
    await _speak(ctx, "Done.")


async def _snooze_current(ctx: "Context", amount: str | None, unit: str | None) -> None:
    t = ctx.current_task
    if t is None:
        await _speak(ctx, "Nothing active.")
        return
    seconds = _parse_snooze_seconds(amount, unit)
    ctx.current_task = None
    modes.stop_playback(ctx)
    ctx.queue.defer(t.id, seconds)
    await _speak(ctx, f"Snoozed {int(seconds)} seconds.")


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


async def _announce_count(ctx: "Context") -> None:
    counts = ctx.queue.count_by_kind()
    if not counts:
        await _speak(ctx, "Queue is empty.")
        return
    total = sum(counts.values())
    parts = ", ".join(f"{v} {k.replace('_', ' ')}" for k, v in sorted(counts.items()))
    await _speak(ctx, f"{total} pending. {parts}.")


async def _add_manual(ctx: "Context", body: str) -> None:
    t = Task(
        kind="note",
        topic="inbox",
        headline=body[:80],
        body=body,
    )
    ctx.queue.add(t)
    await _speak(ctx, "Added.")


# --- announcement ---------------------------------------------------------


def _announce_headline(ctx: "Context", task: Task) -> None:
    """Speak the task headline. Body is held until user asks for it.

    Sync because ``modes.speak_chunked`` is fire-and-forget — it
    schedules a playback task and returns.
    """
    label = _kind_label(task)
    text = f"{label}: {task.headline}" if task.headline else label
    modes.speak_chunked(ctx, text)


async def _announce_body(ctx: "Context") -> None:
    t = ctx.current_task
    if t is None or not t.body:
        await _speak(ctx, "Nothing to expand.")
        return
    modes.speak_chunked(ctx, t.body)


def _kind_label(task: Task) -> str:
    if task.kind == "claude_reply":
        return f"Claude in {task.topic} replied"
    if task.kind == "slack_msg":
        return f"Slack in {task.topic}"
    if task.kind == "email_msg":
        sender = task.source.get("sender_name") or task.source.get("sender_email") or ""
        return f"Email from {sender}" if sender else "Email"
    if task.kind == "linear_issue":
        identifier = task.source.get("identifier") or ""
        return f"Linear {identifier}" if identifier else "Linear issue"
    if task.kind == "meeting_followup":
        meeting = task.source.get("meeting") or task.topic or ""
        return f"Meeting follow-up from {meeting}" if meeting else "Meeting follow-up"
    if task.kind == "note":
        return "Note"
    if task.kind == "web":
        return "Web link"
    return task.kind


# --- per-kind response dispatch -------------------------------------------


async def _dispatch_task_response(ctx: "Context", task: Task, transcript: str) -> None:
    """Route a free-form PTT transcript to whatever makes sense for this task."""
    if task.kind == "claude_reply":
        await _respond_claude(ctx, task, transcript)
        return
    if task.kind == "slack_msg":
        await _respond_slack(ctx, task, transcript)
        return
    if task.kind == "email_msg":
        await _respond_email(ctx, task, transcript)
        return
    if task.kind == "linear_issue":
        await _respond_linear(ctx, task, transcript)
        return
    await _speak(ctx, "No response action for this task.")


async def _respond_claude(ctx: "Context", task: Task, transcript: str) -> None:
    """Send the transcript to the source tmux window. Don't block — the
    ClaudeProducer will surface the next reply as a new task when ready."""
    win = task.source.get("window") or task.topic
    host, opts = ctx.ssh
    try:
        await remote.send(host, opts, ctx.config.tmux_session, win, transcript)
    except remote.RemoteError as exc:
        await _speak(ctx, f"Could not reach Claude: {exc}")
        return
    ctx.queue.mark_done(task.id)
    ctx.current_task = None
    ctx.log.event(
        "queue_turn", task_id=task.id, task_kind=task.kind, topic=task.topic, sent=transcript
    )
    await _speak(ctx, "Sent.")


# "archive" / "archive it" / "archive this" / "archive the email" /
# "archive please" — tight enough that the word doesn't accidentally
# match if the user's actually trying to reply with a sentence
# containing "archive".
_EMAIL_ARCHIVE_RE = re.compile(
    r"^archive(\s+(it|this|please|the email))?[.!?]*$",
    re.IGNORECASE,
)


async def _respond_email(ctx: "Context", task: Task, transcript: str) -> None:
    """Voice response on an email task — archive or draft a reply.

    The claude.ai Gmail MCP has no send tool — only ``create_draft``.
    When the user says "archive" / "archive it", we call
    ``unlabel_thread`` (removing INBOX) instead of drafting. Any other
    transcript becomes the body of a draft reply; the user reviews and
    sends from Gmail. That's the safer default for voice anyway.
    """
    mcp = ctx.email_mcp
    if mcp is None:
        await _speak(ctx, "Gmail MCP is not configured.")
        return
    if _EMAIL_ARCHIVE_RE.match(transcript.strip()):
        await _archive_email(ctx, task)
        return
    to_addr = task.source.get("sender_email") or ""
    msg_id = task.source.get("message_id") or ""
    subject = task.source.get("subject") or ""
    if not to_addr:
        await _speak(ctx, "Missing sender email for this task.")
        return
    if not subject.lower().startswith("re:"):
        subject = f"Re: {subject}" if subject else "Re:"
    args: dict = {"to": [to_addr], "subject": subject, "body": transcript}
    if msg_id:
        args["replyToMessageId"] = msg_id
    try:
        await mcp.call_tool("create_draft", args)
    except Exception as exc:
        logger.exception("Email draft failed")
        await _speak(ctx, f"Could not draft email: {exc}")
        return
    ctx.queue.mark_done(task.id)
    ctx.current_task = None
    ctx.log.event(
        "queue_turn",
        task_id=task.id,
        task_kind=task.kind,
        topic=task.topic,
        sent=transcript,
    )
    await _speak(ctx, "Drafted.")


async def _archive_email(ctx: "Context", task: Task) -> None:
    """Archive the email by removing the INBOX label.

    Same Gmail MCP call the ``accept-invite`` skill uses. Marks the
    task done; on the next wide poll the producer won't re-surface it
    because the email is no longer ``in:inbox``.
    """
    mcp = ctx.email_mcp
    thread_id = (task.source or {}).get("thread_id") or ""
    if not thread_id:
        await _speak(ctx, "Missing thread id; can't archive.")
        return
    try:
        await mcp.call_tool(
            "unlabel_thread",
            # ``threadId`` (camelCase) is the Gmail MCP's required arg
            # name. Passing ``thread_id`` makes the MCP reject with a
            # misleading "Invalid label" error.
            {"threadId": thread_id, "labelIds": ["INBOX"]},
        )
    except Exception as exc:
        logger.exception("Email archive failed")
        await _speak(ctx, f"Could not archive: {exc}")
        return
    ctx.queue.mark_done(task.id)
    ctx.current_task = None
    ctx.log.event(
        "queue_turn",
        task_id=task.id,
        task_kind=task.kind,
        topic=task.topic,
        action="archive",
    )
    await _speak(ctx, "Archived.")


async def _respond_linear(ctx: "Context", task: Task, transcript: str) -> None:
    """Voice response on a Linear task: post the transcript as a comment.

    The claude.ai Linear MCP's ``save_comment`` tool creates a new
    top-level comment on the issue when given ``issueId`` and ``body``.
    On success the task is marked done; on the next wide poll the
    producer re-surfaces the issue only if it's still in an active
    state.
    """
    mcp = ctx.linear_mcp
    if mcp is None:
        await _speak(ctx, "Linear MCP is not configured.")
        return
    identifier = (task.source or {}).get("identifier") or ""
    if not identifier:
        await _speak(ctx, "Missing issue id for this task.")
        return
    try:
        await mcp.call_tool(
            "save_comment",
            {"issueId": identifier, "body": transcript},
        )
    except Exception as exc:
        logger.exception("Linear comment failed")
        await _speak(ctx, f"Could not post comment: {exc}")
        return
    ctx.queue.mark_done(task.id)
    ctx.current_task = None
    ctx.log.event(
        "queue_turn",
        task_id=task.id,
        task_kind=task.kind,
        topic=task.topic,
        sent=transcript,
    )
    await _speak(ctx, "Commented.")


async def _respond_slack(ctx: "Context", task: Task, transcript: str) -> None:
    """Reply in the Slack thread the task came from via the Slack MCP."""
    mcp = ctx.slack_mcp
    if mcp is None:
        await _speak(ctx, "Slack MCP is not configured.")
        return
    channel_id = task.source.get("channel_id")
    thread_ts = task.source.get("thread_ts") or task.source.get("ts")
    if not channel_id:
        await _speak(ctx, "Missing Slack channel for this task.")
        return
    try:
        await mcp.call_tool(
            "slack_send_message",
            {
                "channel_id": channel_id,
                "message": transcript,
                "thread_ts": thread_ts,
            },
        )
    except Exception as exc:
        logger.exception("Slack reply failed")
        await _speak(ctx, f"Could not send Slack reply: {exc}")
        return
    ctx.queue.mark_done(task.id)
    ctx.current_task = None
    ctx.log.event(
        "queue_turn",
        task_id=task.id,
        task_kind=task.kind,
        topic=task.topic,
        sent=transcript,
    )
    await _speak(ctx, "Sent.")


# --- macropad tap delegates (queue-mode YES/NO/ACT) -----------------------


async def queue_yes_tap(ctx: "Context") -> None:
    """YES in queue mode: submit whatever's in the Input widget.

    Acts as the Enter key for the TUI's Input — lets the user paste a
    PTT transcript, edit it, and submit on their schedule. If the
    Input is empty (or there is no TUI), this is a no-op: auto-
    advance takes care of the "I'm ready for the next task" case, and
    expanding the current task's body is a voice-only command ("go
    on" / "tell me more" / "expand").
    """
    submit = ctx.submit_input
    if submit is None:
        return
    try:
        submit()
    except Exception:
        logger.exception("queue_yes_tap: submit_input failed")


async def queue_no_tap(ctx: "Context") -> None:
    """NO in queue mode: skip current task."""
    if ctx.current_task is None:
        await _speak(ctx, "Nothing active.")
        return
    await _skip_current(ctx)


async def queue_navigate(ctx: "Context", *, direction: int) -> None:
    """Move the cursor through the ranked queue.

    ``direction`` is ``-1`` (up / previous) or ``+1`` (down / next).
    The cursor is just a pointer — no task state changes — so the
    queue list stays static and the cursor task remains visible in
    its rightful ranked position. Wraps at the boundaries: down on
    the bottom row jumps to the top, up on the top jumps to the
    bottom. Python's ``%`` makes the math symmetric for negatives.

    Doesn't touch ``recent_topics`` — arrow navigation is exploratory,
    so it shouldn't bias the scheduler the way an explicit "next" or
    a skill engagement does.
    """
    modes.stop_playback(ctx)
    ranked = ctx.queue.ranked(now=time.time(), recent=ctx.recent_topics)
    if not ranked:
        ctx.current_task = None
        await _speak(ctx, "Queue is empty.")
        return
    pending_ids = [t.id for t, _ in ranked]
    cur = ctx.current_task
    if cur is None:
        new_idx = 0 if direction > 0 else len(pending_ids) - 1
    else:
        try:
            cur_idx = pending_ids.index(cur.id)
        except ValueError:
            # Cursor task was removed from pending (e.g. just engaged
            # with). Treat as fresh — top for down, bottom for up.
            cur_idx = -1 if direction > 0 else len(pending_ids)
        new_idx = (cur_idx + direction) % len(pending_ids)
    ctx.current_task = ctx.queue.get(pending_ids[new_idx])
    _announce_headline(ctx, ctx.current_task)


async def dismiss_current_task(ctx: "Context") -> None:
    """ACT+NO chord in queue mode: mark current task done.

    Distinct from :func:`queue_no_tap` (which just defers): this is
    "permanently drop this task" — the user has decided they don't
    care. Stops any in-flight announcement first so we don't keep
    speaking about a task we just killed.
    """
    logger.info(
        "dismiss_current_task: current_task=%s",
        ctx.current_task.id if ctx.current_task else None,
    )
    if ctx.current_task is None:
        await _speak(ctx, "Nothing active.")
        return
    modes.stop_playback(ctx)
    await _drop_current(ctx)


# --- consumer / auto-announce thread --------------------------------------


class QueueConsumer:
    """Async task that auto-announces when queue mode is idle.

    Wakes on every queue mutation and on a short poll interval (so
    ``ready_at`` snoozed tasks surface when their time arrives).
    """

    def __init__(self, ctx: "Context", *, poll_interval: float = 2.0) -> None:
        self._ctx = ctx
        self._poll = poll_interval
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()

    def attach(self) -> None:
        """Subscribe to queue events so producer adds wake the consumer."""
        self._ctx.queue.add_listener(self._on_event)

    def request_stop(self) -> None:
        self._stop.set()
        # Also pop the wait so the loop exits promptly instead of
        # sitting through the rest of the poll interval.
        self._wake.set()

    def _on_event(self, _kind: str, _task: Task) -> None:
        # Sync listener fired from queue mutations; safe to call
        # asyncio.Event.set() because every mutation now originates on
        # the same loop thread.
        self._wake.set()

    async def run(self) -> None:
        while not self._stop.is_set():
            # We don't care whether the wake fired or the timeout elapsed —
            # either way we re-check state and maybe announce.
            await event_or_timeout(self._wake, self._poll)
            self._wake.clear()
            if self._stop.is_set():
                break
            await self._maybe_announce()

    async def _maybe_announce(self) -> None:
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
        await _announce_next(ctx)


# --- helpers ---------------------------------------------------------------


async def _speak(ctx: "Context", text: str) -> None:
    if not text:
        return
    try:
        await ctx.tts.speak(text)
    except Exception:
        logger.exception("TTS failed for: %s", text)
