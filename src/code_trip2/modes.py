"""Shared orchestrator state (:class:`Context`) + chunked audio playback.

:class:`Context` is the god-object every layer threads through —
config, TTS, the task queue, the MCP clients, and the playback state.

Long spoken responses (task headlines, bodies, the agent's skill
summaries) are split into ~1–3-sentence chunks and played sequentially
on a worker task. Macropad taps consult :func:`is_playback_active` to
decide whether to stop playback or fall through to their normal action.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from code_trip2 import earcon
from code_trip2.config import Config
from code_trip2.queue_log import QueueLog
from code_trip2.screener import AutohandleLogEntry
from code_trip2.session_log import SessionLogger
from code_trip2.summarizer import Summarizer
from code_trip2.tasks import RecentTopics, Task, TaskQueue
from code_trip2.tts_client import TTSClient, TTSClientError

logger = logging.getLogger(__name__)


# --- Context ---------------------------------------------------------------


@dataclass
class Context:
    config: Config
    tts: TTSClient
    log: SessionLogger
    thinking: earcon.Thinking
    # Display-only label for the TUI header / "status" voice command —
    # the tmux work window the orchestrator was launched against. Not a
    # live focus indicator any more; kept because the header and
    # ``_respond_claude`` still reference it.
    active_window: str = ""
    # Chunked playback state. Single-loop discipline: every mutation
    # happens on the event loop thread, no lock needed.
    playback_queue: list[str] = field(default_factory=list)
    last_response_chunks: list[str] = field(default_factory=list)
    _playback_task: "asyncio.Task | None" = field(
        default=None, init=False, repr=False
    )
    # Task-queue interaction surface.
    queue: TaskQueue = field(default_factory=TaskQueue)
    queue_log: QueueLog | None = None
    recent_topics: RecentTopics = field(default_factory=RecentTopics)
    current_task: Task | None = None
    # Pane-output → spoken-English summarizer. Owned here so the TUI
    # header can show its status; the ClaudeProducer holds its own.
    summarizer: Summarizer | None = None
    # Set when ``--tui`` is on. If the frontmost macOS app matches this
    # value, ``chords`` suppresses synthesized keystrokes so the
    # app-navigation chords don't scroll the alternate-screen buffer.
    tui_host_app: str | None = None
    # ClaudeMCPClient pointing at the claude.ai Slack MCP. Set by main.py
    # if the claude CLI is available. dispatch._respond_slack uses it to
    # thread-reply to active slack_msg tasks.
    slack_mcp: object | None = None
    # ClaudeMCPClient pointing at the claude.ai Gmail MCP. Used by
    # dispatch._respond_email to draft a reply to an email_msg task.
    email_mcp: object | None = None
    # ClaudeMCPClient pointing at the claude.ai Linear MCP. Used by
    # dispatch._respond_linear to post a comment on a linear_issue task.
    linear_mcp: object | None = None
    # ClaudeMCPClient used for free-form skill invocation (ACT+PTT).
    # No fixed server_id — claude.ai's skill discovery loads the
    # matching skill from ``.claude/skills/`` and uses whatever MCP
    # tools that skill needs.
    agent_mcp: object | None = None
    # Union of ``allowed-tools`` declared in every project skill's
    # frontmatter. Passed to ``run_agent`` so Claude can't reach for a
    # tool that isn't in any skill's declared set.
    agent_allowed_tools: tuple[str, ...] = ()
    # Bounded log of recent screener outcomes for TUI display. Only
    # "interesting" outcomes land here — handled / failed / dry-run-
    # nominated forwards — never a plain "no skill cared" pass-through.
    autohandle_log: "deque[AutohandleLogEntry]" = field(
        default_factory=lambda: deque(maxlen=20)
    )
    # Submits whatever's in the TUI Input widget, as if Enter were
    # pressed. Returns True if there was text to submit. None when no
    # TUI is attached (headless OpenAI-STT mode). Typed by shape
    # rather than by which class implements it — keeps the type-level
    # dependency one-way and avoids importing tui.py here.
    submit_input: Callable[[], bool] | None = None
    # Strong refs to in-flight background side effects (email archive,
    # Linear filing) spawned by dispatch._spawn_bg so the chord returns
    # immediately. Holding the refs keeps asyncio from GC'ing the
    # tasks mid-run; the done callback discards them.
    background_jobs: "set[asyncio.Task]" = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self.active_window:
            self.active_window = self.config.work_window

    @property
    def ssh(self) -> tuple[str, tuple[str, ...]]:
        return self.config.ssh_host, self.config.ssh_options


# --- chunked playback ------------------------------------------------------


def replay_last(ctx: Context) -> bool:
    """Re-queue the last response's chunks for playback. Returns True if
    anything was queued. The ``repeat`` / ``what`` voice command in
    ``dispatch`` calls this."""
    if not ctx.last_response_chunks:
        return False
    ctx.playback_queue = list(ctx.last_response_chunks)
    _start_playback_task(ctx)
    return True


_SENT_END = re.compile(r"(?<=[.!?])\s+")
_PARA_BREAK = re.compile(r"\n\s*\n")
_DEFAULT_CHUNK_CHARS = 200
_DEFAULT_CHUNK_SENTENCES = 3


def chunk_text(
    text: str,
    *,
    max_chars: int = _DEFAULT_CHUNK_CHARS,
    max_sentences: int = _DEFAULT_CHUNK_SENTENCES,
) -> list[str]:
    """Split text into ~1–3-sentence chunks, respecting paragraph breaks."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    for para in (p.strip() for p in _PARA_BREAK.split(text)):
        if not para:
            continue
        # Join terminal soft-wrapped lines within a paragraph.
        para = re.sub(r"\s*\n\s*", " ", para)
        sentences = [s.strip() for s in _SENT_END.split(para) if s.strip()]
        if not sentences:
            continue
        cur = ""
        cur_count = 0
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            if cur and (len(candidate) > max_chars or cur_count >= max_sentences):
                chunks.append(cur)
                cur = s
                cur_count = 1
            else:
                cur = candidate
                cur_count += 1
        if cur:
            chunks.append(cur)
    return chunks


def speak_chunked(ctx: Context, text: str) -> None:
    """Public entry point: chunk ``text`` and start playback. Replaces any
    pending playback. Sets ``last_response_chunks`` for repeat support.

    Sync because callers (dispatch handlers, the queue consumer) shouldn't
    wait for playback to finish — they want fire-and-forget. The actual
    speaking happens on an asyncio task spawned by ``_start_playback_task``.
    """
    chunks = chunk_text(text)
    if not chunks:
        return
    ctx.last_response_chunks = list(chunks)
    ctx.playback_queue = list(chunks)
    _start_playback_task(ctx)


def _start_playback_task(ctx: Context) -> None:
    if ctx._playback_task is not None and not ctx._playback_task.done():
        # An existing task will pick up the new queue items.
        return
    if not ctx.playback_queue:
        return
    ctx._playback_task = asyncio.create_task(_playback_loop(ctx), name="playback")


async def _playback_loop(ctx: Context) -> None:
    try:
        while ctx.playback_queue:
            chunk = ctx.playback_queue.pop(0)
            try:
                await ctx.tts.speak(chunk)
            except TTSClientError:
                logger.exception("TTS failed for chunk")
                break
        try:
            earcon.completion()
        except earcon.EarconError:
            pass
    finally:
        ctx._playback_task = None


def advance_playback(ctx: Context) -> None:
    """Skip the current chunk; the playback task will pick up the next."""
    ctx.tts.stop()


def stop_playback(ctx: Context) -> None:
    """Drop pending chunks and interrupt current playback.

    Sync so it can be called from the macropad's pynput listener thread
    (on_ptt_press); ``ctx.tts.stop()`` is also sync and thread-safe.
    """
    ctx.playback_queue.clear()
    ctx.tts.stop()


def is_playback_active(ctx: Context) -> bool:
    """True while audio is playing or chunks remain queued."""
    if ctx.tts.is_playing():
        return True
    return bool(ctx.playback_queue)


# --- pane-output cleanup ---------------------------------------------------
#
# Mechanical ANSI / box-drawing strip used as the summarizer's documented
# fallback for turning raw tmux pane capture into speakable text. The
# summarizer is the live path; this stays as a pure, tested utility the
# orchestrator can always fall back to.

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_BOX_RE = re.compile(r"[╭╮╰╯│─┌┐└┘├┤┬┴┼━┃┏┓┗┛┣┫┳┻╋]")
_ANCHOR_PREFIX_LEN = 30


def clean_output(raw: str, anchor: str | None = None) -> str:
    """Extract Claude's most recent message from captured pane text.

    With ``anchor``, find the **last** line containing the first
    ``_ANCHOR_PREFIX_LEN`` chars of the anchor (the user prompt) and
    return everything after that line — that's Claude's response,
    independent of pane height. Without an anchor (or when the anchor
    isn't found), fall back to the last 60 non-empty lines.
    """
    if not raw:
        return ""
    s = _ANSI_RE.sub("", raw)
    s = _BOX_RE.sub("", s)
    lines = [ln.rstrip() for ln in s.splitlines()]
    while lines and (not lines[-1].strip() or lines[-1].strip().startswith(">")):
        lines.pop()

    if anchor:
        body = _slice_after_anchor(lines, anchor)
    else:
        body = [ln for ln in lines if ln.strip()][-60:]

    cleaned = "\n".join(body).strip()
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


def _slice_after_anchor(lines: list[str], anchor: str) -> list[str]:
    needle = anchor.strip()[:_ANCHOR_PREFIX_LEN]
    if not needle:
        return [ln for ln in lines if ln.strip()][-60:]
    last_idx = -1
    for i, ln in enumerate(lines):
        if needle in ln:
            last_idx = i
    if last_idx < 0:
        logger.warning("Anchor %r not found in capture; falling back to tail.", needle)
        return [ln for ln in lines if ln.strip()][-60:]
    return [ln for ln in lines[last_idx + 1 :] if ln.strip()]
