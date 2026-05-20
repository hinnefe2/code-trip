"""Voice routing + chunked audio playback.

Routing layers (checked in order):

1. Global commands — status, repeat / what, stop / cancel.
2. Voice phrases — tmux window navigation; Linear ticket browsing.
3. App-focus dispatch — if the frontmost macOS app is in
   ``config.terminal_apps``, the transcript goes to the active tmux pane
   (talk-to-Claude). Otherwise it pastes into the focused app.

There is no semantic mode FSM. State that used to live on a "mode"
(active tmux window, ticket cursor) lives on :class:`Context` as plain
fields.

Long Claude responses are split into ~1–3-sentence chunks and played
sequentially on a worker thread. Macropad taps consult
:func:`is_playback_active` to decide whether to advance / stop playback
or fall through to the focused-app keystroke.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field

from code_trip2 import earcon, remote, window
from code_trip2.config import Config
from code_trip2.queue_log import QueueLog
from code_trip2.session_log import SessionLogger
from code_trip2.summarizer import Summarizer, SummarizerError
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
    # Active tmux window for talk-to-Claude turns (kept across turns).
    active_window: str = ""
    # Linear ticket cache + cursor.
    tickets: list[dict] = field(default_factory=list)
    ticket_index: int = 0
    # Last prompt sent to the active pane — anchor for finding Claude's
    # response in the captured pane text.
    last_sent_prompt: str = ""
    # Chunked playback state.
    playback_queue: list[str] = field(default_factory=list)
    last_response_chunks: list[str] = field(default_factory=list)
    _playback_lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )
    _playback_thread: "threading.Thread | None" = field(
        default=None, init=False, repr=False
    )
    # Task-queue interaction surface (queue mode).
    queue: TaskQueue = field(default_factory=TaskQueue)
    queue_log: QueueLog | None = None
    recent_topics: RecentTopics = field(default_factory=RecentTopics)
    current_task: Task | None = None
    # "queue" | "focused". See dispatch.py for the mode flip.
    app_mode: str = "focused"
    # Raw-pane-output → spoken-English summarizer; falls back to clean_output
    # when None or when summarize() raises.
    summarizer: Summarizer | None = None
    # Set when ``--tui`` is on. If the frontmost macOS app matches this
    # value, ``chords`` suppresses synthesized keystrokes so YES/NO/NAV
    # taps don't scroll the alternate-screen buffer.
    tui_host_app: str | None = None

    def __post_init__(self) -> None:
        if not self.active_window:
            self.active_window = self.config.work_window

    @property
    def ssh(self) -> tuple[str, tuple[str, ...]]:
        return self.config.ssh_host, self.config.ssh_options


# --- top-level dispatch ----------------------------------------------------


def handle_voice(ctx: Context, transcript: str) -> None:
    """Route a PTT transcript: globals → voice phrases → app focus."""
    t = transcript.strip()
    if not t:
        return

    if _try_global_commands(ctx, t):
        return
    if _try_voice_phrase(ctx, t):
        return
    _dispatch_by_focus(ctx, t)


def _dispatch_by_focus(ctx: Context, t: str) -> None:
    try:
        app = window.active_app()
    except window.WindowError as exc:
        logger.warning("active_app failed (%s); defaulting to WORK branch.", exc)
        _work_voice(ctx, t)
        return

    if app in ctx.config.terminal_apps:
        _work_voice(ctx, t)
    else:
        _dictate_voice(ctx, t)


# --- global commands -------------------------------------------------------


def replay_last(ctx: Context) -> bool:
    """Re-queue the last response's chunks for playback. Returns True if
    anything was queued. Used by both ``modes`` and ``dispatch`` repeat
    handlers."""
    if not ctx.last_response_chunks:
        return False
    with ctx._playback_lock:
        ctx.playback_queue = list(ctx.last_response_chunks)
    _start_playback_worker(ctx)
    return True


def _try_global_commands(ctx: Context, t: str) -> bool:
    low = t.lower().strip(" .!?")
    if (
        low == "what"
        or low.startswith("repeat")
        or "say that again" in low
        or "say it again" in low
    ):
        if not replay_last(ctx):
            _speak(ctx, "Nothing to repeat.")
        return True
    if low in ("stop", "cancel", "stop talking", "shut up", "be quiet"):
        stop_playback(ctx)
        return True
    if low.startswith("status"):
        try:
            app = window.active_app()
        except window.WindowError:
            app = "unknown"
        _speak(ctx, f"App {app}. Window {ctx.active_window}.")
        return True
    return False


# --- voice phrases ---------------------------------------------------------


def _try_voice_phrase(ctx: Context, t: str) -> bool:
    low = t.lower()

    if any(p in low for p in (
        "list tickets", "show tickets", "my tickets",
        "refresh tickets", "reload tickets",
    )):
        _linear_refresh(ctx)
        return True
    if ctx.tickets:
        bare = low.strip(" .!?")
        if bare in ("next", "next ticket"):
            _linear_step(ctx, +1)
            return True
        if bare in ("previous", "prev", "back", "previous ticket"):
            _linear_step(ctx, -1)
            return True
        m = re.match(r"(?:select|work on|start)\s+(?:ticket\s+)?(\d+|this)", low)
        if m:
            token = m.group(1)
            idx = ctx.ticket_index if token == "this" else int(token) - 1
            _linear_select(ctx, idx)
            return True

    if "list windows" in low or "what windows" in low:
        _announce_windows(ctx)
        return True
    m = re.match(r"(?:switch(?:\s+to)?|go\s+to)\s+(.+?)[\.\s]*$", low)
    if m:
        _select_window(ctx, m.group(1).strip())
        return True
    m = re.match(r"new\s+window\s+(.+?)[\.\s]*$", low)
    if m:
        _new_window(ctx, m.group(1).strip())
        return True

    return False


# --- DICTATE branch --------------------------------------------------------


def _dictate_voice(ctx: Context, t: str) -> None:
    try:
        window.paste_text(t)
    except window.WindowError as exc:
        _report_error(ctx, f"Could not paste: {exc}")
        return
    ctx.log.event("turn", branch="dictate", transcript=t)


# --- WORK branch -----------------------------------------------------------


def _work_voice(ctx: Context, t: str) -> None:
    host, opts = ctx.ssh
    win = ctx.active_window
    cfg = ctx.config

    # Drop any in-flight playback before sending a new turn.
    stop_playback(ctx)

    try:
        remote.send(host, opts, cfg.tmux_session, win, t)
    except remote.RemoteError as exc:
        _report_error(ctx, f"Could not reach Claude: {exc}")
        return
    ctx.last_sent_prompt = t

    ctx.thinking.start()
    raw: str | None = None
    try:
        try:
            remote.wait_done(host, opts, win, timeout=cfg.wait_timeout)
        except remote.WaitTimeout:
            _report_error(ctx, "Claude did not respond in time.")
            return
        except remote.RemoteError as exc:
            _report_error(ctx, f"Lost connection to Claude: {exc}")
            return
        try:
            raw = remote.capture(host, opts, cfg.tmux_session, win, lines=2000)
        except remote.RemoteError as exc:
            _report_error(ctx, f"Could not read Claude's response: {exc}")
            return
    finally:
        ctx.thinking.stop()

    spoken, summarized = _summarize_or_strip(ctx, raw or "", t)
    ctx.log.event(
        "turn",
        branch="work",
        user=t,
        remote_output=raw,
        spoken=spoken,
        summarized=summarized,
    )
    if not spoken:
        _speak(ctx, "No output.")
        return
    _speak_chunked(ctx, spoken)


def _summarize_or_strip(ctx: Context, raw: str, prompt: str) -> tuple[str, bool]:
    """Run the summarizer if available; otherwise fall back to clean_output.

    Returns ``(spoken_text, summarized)`` where ``summarized`` is True iff
    the LLM summarizer produced the text. The fallback also fires if the
    summarizer succeeded but returned an empty string.
    """
    if ctx.summarizer is not None and ctx.summarizer.enabled and raw.strip():
        try:
            text = ctx.summarizer.summarize(
                raw, context={"kind": "claude_reply", "user_prompt": prompt}
            )
        except SummarizerError as exc:
            logger.warning("Summarizer failed; falling back to clean_output: %s", exc)
        else:
            if text:
                return text, True
    return clean_output(raw, anchor=prompt), False


# --- output cleanup --------------------------------------------------------

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


# --- chunked playback ------------------------------------------------------

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


def _speak_chunked(ctx: Context, text: str) -> None:
    speak_chunked(ctx, text)


def speak_chunked(ctx: Context, text: str) -> None:
    """Public entry point: chunk ``text`` and start playback. Replaces any
    pending playback. Sets ``last_response_chunks`` for repeat support."""
    chunks = chunk_text(text)
    if not chunks:
        return
    with ctx._playback_lock:
        ctx.last_response_chunks = list(chunks)
        ctx.playback_queue = list(chunks)
    _start_playback_worker(ctx)


def _start_playback_worker(ctx: Context) -> None:
    with ctx._playback_lock:
        existing = ctx._playback_thread
        if existing is not None and existing.is_alive():
            # An existing worker will pick up the new queue items.
            return
        if not ctx.playback_queue:
            return
        thread = threading.Thread(target=_playback_loop, args=(ctx,), daemon=True)
        ctx._playback_thread = thread
    thread.start()


def _playback_loop(ctx: Context) -> None:
    try:
        while True:
            with ctx._playback_lock:
                if not ctx.playback_queue:
                    break
                chunk = ctx.playback_queue.pop(0)
            try:
                ctx.tts.speak(chunk)
            except TTSClientError:
                logger.exception("TTS failed for chunk")
                break
        try:
            earcon.completion()
        except earcon.EarconError:
            pass
    finally:
        with ctx._playback_lock:
            ctx._playback_thread = None


def advance_playback(ctx: Context) -> None:
    """Skip the current chunk; the worker will pick up the next one."""
    ctx.tts.stop()


def stop_playback(ctx: Context) -> None:
    """Drop pending chunks and interrupt current playback."""
    with ctx._playback_lock:
        ctx.playback_queue.clear()
    ctx.tts.stop()


def is_playback_active(ctx: Context) -> bool:
    """True while audio is playing or chunks remain queued."""
    if ctx.tts.is_playing():
        return True
    with ctx._playback_lock:
        return bool(ctx.playback_queue)


# --- window navigation -----------------------------------------------------


def _select_window(ctx: Context, name: str) -> None:
    host, opts = ctx.ssh
    try:
        remote.select_window(host, opts, ctx.config.tmux_session, name)
    except remote.RemoteError as exc:
        _report_error(ctx, f"Could not switch: {exc}")
        return
    ctx.active_window = name
    _speak(ctx, f"Switched to {name}.")


def _new_window(ctx: Context, name: str) -> None:
    host, opts = ctx.ssh
    try:
        remote.new_window(host, opts, ctx.config.tmux_session, name)
    except remote.RemoteError as exc:
        _report_error(ctx, f"Could not create window: {exc}")
        return
    ctx.active_window = name
    _speak(ctx, f"Created window {name}.")


def _announce_windows(ctx: Context) -> None:
    host, opts = ctx.ssh
    try:
        rows = remote.list_windows(host, opts, ctx.config.tmux_session)
    except remote.RemoteError as exc:
        _report_error(ctx, f"Could not list windows: {exc}")
        return
    if not rows:
        _speak(ctx, "No windows.")
        return
    names = ", ".join(name for _idx, name, _cwd in rows)
    _speak(ctx, f"{len(rows)} windows: {names}.")


# --- LINEAR ----------------------------------------------------------------

_LINEAR_REFRESH_PROMPT = (
    "List my assigned Linear tickets using the Linear MCP. "
    "Respond with ONLY a JSON array (no prose, no code fences). "
    'Each object: {"id","title","priority","assignee","branch"}. '
    'Use empty string for missing fields.'
)

_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)


def _linear_refresh(ctx: Context) -> None:
    host, opts = ctx.ssh
    cfg = ctx.config
    win = cfg.linear_window
    prompt = f"claude -p {json.dumps(_LINEAR_REFRESH_PROMPT)}"
    try:
        remote.send(host, opts, cfg.tmux_session, win, prompt)
        remote.wait_done(host, opts, win, timeout=cfg.wait_timeout)
        raw = remote.capture(host, opts, cfg.tmux_session, win, lines=400)
    except (remote.RemoteError, remote.WaitTimeout) as exc:
        _report_error(ctx, f"Could not refresh tickets: {exc}")
        return
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        _report_error(ctx, "No ticket JSON found.")
        return
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        _report_error(ctx, f"Invalid ticket JSON: {exc}")
        return
    tickets: list[dict] = [t for t in payload if isinstance(t, dict) and t.get("id")]
    ctx.tickets = tickets
    ctx.ticket_index = 0
    _speak(ctx, f"{len(tickets)} tickets.")
    _linear_announce(ctx)


def _linear_step(ctx: Context, delta: int) -> None:
    if not ctx.tickets:
        _speak(ctx, "No tickets. Say 'list tickets' to refresh.")
        return
    ctx.ticket_index = (ctx.ticket_index + delta) % len(ctx.tickets)
    _linear_announce(ctx)


def _linear_announce(ctx: Context) -> None:
    if not ctx.tickets:
        _speak(ctx, "No tickets.")
        return
    t = ctx.tickets[ctx.ticket_index]
    tid = str(t.get("id", "?"))
    title = str(t.get("title", ""))
    prio = str(t.get("priority", "")) or "no priority"
    assignee = str(t.get("assignee", "")) or "unassigned"
    _speak(
        ctx,
        f"{ctx.ticket_index + 1} of {len(ctx.tickets)}. {tid}. {title}. "
        f"Priority {prio}. Assigned to {assignee}.",
    )


def _linear_select(ctx: Context, idx: int) -> None:
    if not ctx.tickets or idx < 0 or idx >= len(ctx.tickets):
        _speak(ctx, "No such ticket.")
        return
    t = ctx.tickets[idx]
    tid = str(t.get("id", ""))
    branch = str(t.get("branch", "")) or tid.lower()
    ctx.ticket_index = idx
    host, opts = ctx.ssh
    try:
        remote.new_window(host, opts, ctx.config.tmux_session, branch)
        remote.send(host, opts, ctx.config.tmux_session, branch, "claude")
    except remote.RemoteError as exc:
        _report_error(ctx, f"Could not open window for {tid}: {exc}")
        return
    ctx.active_window = branch
    _speak(ctx, f"Opened {tid} in window {branch}.")


# --- helpers ---------------------------------------------------------------


def _speak(ctx: Context, text: str) -> None:
    if not text:
        return
    try:
        ctx.tts.speak(text)
    except TTSClientError:
        logger.exception("TTS failed for: %s", text)


def _report_error(ctx: Context, message: str) -> None:
    ctx.thinking.stop()
    try:
        earcon.error()
    except earcon.EarconError:
        pass
    _speak(ctx, message)
    ctx.log.event("error", message=message)
