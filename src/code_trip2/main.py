"""Entry point: wire everything and run the PTT loop."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import faulthandler
import logging
import os
import queue as queue_lib
import re
import select
import signal
import sys
import termios
import threading
import time
from pathlib import Path

from code_trip2 import earcon
from code_trip2.chords import handle_chord, handle_tap
from code_trip2.config import Config, load_config
from code_trip2.dispatch import QueueConsumer, handle_skill, handle_voice
from code_trip2.macropad import Macropad, resolve_key
from code_trip2.modes import Context, stop_playback
from code_trip2.producers import ProducerSupervisor
from code_trip2.email_state import EmailState
from code_trip2.input_buffer import InputBuffer
from code_trip2.producers.claude import ClaudeProducer
from code_trip2.producers.email import EmailProducer
from code_trip2.producers.linear import LinearProducer
from code_trip2.producers.manual import ManualProducer
from code_trip2.producers.slack import SlackProducer
from code_trip2.queue_log import QueueLog
from code_trip2.producers.claude_mcp import ClaudeMCPClient
from code_trip2.session_log import SessionLogger, default_session_path
from code_trip2.skills import load_skill_allowed_tools
from code_trip2.slack_state import SlackState
from code_trip2.stt_client import STTClient, STTClientError
from code_trip2.summarizer import Summarizer
from code_trip2.tasks import TaskQueue
from code_trip2.tts_client import TTSClient
from code_trip2.tui import Dashboard, detect_tui_host_app

logger = logging.getLogger(__name__)


def run(config: Config, *, tui: bool = False) -> None:
    asyncio.run(main_async(config, tui=tui))


async def main_async(config: Config, *, tui: bool = False) -> None:
    log = SessionLogger(default_session_path())
    log.event("session_start", config={
        "tmux_session": config.tmux_session,
        "work_window": config.work_window,
        "linear_window": config.linear_window,
        "stt_model": config.stt_model,
        "tts_model": config.tts_model,
        "tts_voice": config.tts_voice,
        "app_cycle": list(config.app_cycle),
        "terminal_apps": list(config.terminal_apps),
    })

    stt: STTClient | None = None
    if config.stt_provider == "openai":
        stt = STTClient(api_key=config.api_key, model=config.stt_model)
    else:
        logger.info("STT provider=%s; bypassing OpenAI STT.", config.stt_provider)
    tts = TTSClient(
        api_key=config.api_key,
        model=config.tts_model,
        voice=config.tts_voice,
        speed=config.tts_speed,
    )
    thinking = earcon.Thinking()
    summarizer = Summarizer(
        api_key=config.api_key,
        model=config.summarizer_model,
        max_chars=config.summarizer_max_chars,
    )
    if not summarizer.enabled:
        logger.info("Summarizer disabled (no API key); falling back to clean_output.")

    queue = TaskQueue()
    queue_log = QueueLog()
    queue_log.attach(queue)
    # Replay last 24h of queue events so deferred / snoozed work survives a
    # restart. Log records the replay-load so it shows up in offline analysis.
    replayed = queue_log.replay()
    if replayed:
        queue.load(replayed)
        logger.info("Replayed %d tasks from queue log", len(replayed))

    tui_host_app = detect_tui_host_app() if tui else None
    if tui and tui_host_app:
        logger.info("TUI host detected as %r; suppressing synthesized "
                    "keystrokes that would target it.", tui_host_app)

    slack_mcp = ClaudeMCPClient(server_id="claude_ai_Slack")
    if not slack_mcp.enabled:
        logger.info(
            "Slack MCP via claude CLI not available; Slack producer will "
            "stay idle. Install claude CLI to enable."
        )
    email_mcp = ClaudeMCPClient(server_id="claude_ai_Gmail")
    if not email_mcp.enabled:
        logger.info(
            "Gmail MCP via claude CLI not available; Email producer will "
            "stay idle. Install claude CLI to enable."
        )
    # Free-form skill invocation (ACT+PTT). server_id is unused — run_agent
    # doesn't restrict to a single tool.
    agent_mcp = ClaudeMCPClient(server_id="agent")
    # Union of allowed-tools declared by every skill in .claude/skills/.
    # Loaded once on startup; constrains what run_agent will let Claude
    # touch. Resolves relative to CWD because the user runs the
    # orchestrator from the project root.
    skills_dir = Path.cwd() / ".claude" / "skills"
    agent_allowed_tools = load_skill_allowed_tools(skills_dir)
    if agent_allowed_tools:
        logger.info(
            "Loaded %d allowed-tools across project skills from %s",
            len(agent_allowed_tools), skills_dir,
        )
    else:
        logger.info(
            "No skill allowed-tools found at %s; ACT+PTT will run Claude "
            "without --allowedTools restriction.", skills_dir,
        )

    # Live edit buffer for the TUI input panel. Only used in local-STT
    # mode (the stdin reader thread fills it as bytes arrive); the TUI
    # renders it whenever it's non-None.
    input_buffer = InputBuffer() if config.stt_provider != "openai" else None

    ctx = Context(
        config=config,
        tts=tts,
        log=log,
        thinking=thinking,
        queue=queue,
        queue_log=queue_log,
        summarizer=summarizer,
        tui_host_app=tui_host_app,
        slack_mcp=slack_mcp,
        email_mcp=email_mcp,
        agent_mcp=agent_mcp,
        agent_allowed_tools=agent_allowed_tools,
        input_buffer=input_buffer,
        app_mode=config.startup_mode if config.startup_mode in ("queue", "focused") else "focused",
    )

    # Producers run in their own threads; supervisor owns start/stop.
    supervisor = ProducerSupervisor()
    supervisor.add(ClaudeProducer(config=config, queue=queue, summarizer=summarizer))
    supervisor.add(SlackProducer(config=config, queue=queue, mcp=slack_mcp, state=SlackState()))
    supervisor.add(EmailProducer(config=config, queue=queue, mcp=email_mcp, state=EmailState()))
    supervisor.add(LinearProducer(config=config, queue=queue))
    supervisor.add(ManualProducer())

    consumer = QueueConsumer(ctx)
    consumer.attach()

    # Two shutdown events bridged together. The asyncio.Event is what
    # main_async awaits; the threading.Event is what the still-threaded
    # producers, consumer, and stdin paste reader poll. Both fire
    # together so neither side has to know about the other. The
    # threading bridge goes away in Phase 8 once everything is async.
    shutdown = threading.Event()
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _trigger_shutdown() -> None:
        shutdown.set()
        shutdown_event.set()

    def _process_audio(path: Path, skill_mode: bool) -> None:
        if stt is None:
            logger.warning("on_audio fired in non-openai STT mode; ignoring %s", path)
            return
        try:
            transcript = stt.transcribe(path)
        except STTClientError as exc:
            logger.exception("STT failed")
            try:
                earcon.error()
                tts.speak(f"Transcription failed: {exc}")
            except Exception:
                pass
            return
        logger.info("Transcribed (skill_mode=%s): %s", skill_mode, transcript)
        if skill_mode:
            handle_skill(ctx, transcript)
        else:
            handle_voice(ctx, transcript)

    def on_audio(path: Path, *, skill_mode: bool = False) -> None:
        # Run STT + dispatch off the macropad listener thread so taps stay
        # responsive while a turn is in flight (wait_done can take 5–15s).
        threading.Thread(
            target=_process_audio, args=(path, skill_mode), daemon=True
        ).start()

    # --- local-STT (Superwhisper / clipboard-paste) integration ----------
    #
    # In local STT mode the macropad presses ``ptt_forward_key`` while PTT
    # is held; the external STT tool (Superwhisper) records, transcribes,
    # populates the clipboard, and pastes into whatever app is focused.
    # The orchestrator never captures the audio.
    #
    # To make voice work in queue mode we just read the paste from
    # stdin: when the user keeps the TUI's host terminal focused, the
    # Cmd+V from Superwhisper lands as stdin bytes for our process. We
    # accumulate the bytes in a worker thread, emit when there's a
    # quiet pause (the paste doesn't end with a newline), and dispatch
    # via ``handle_voice`` / ``handle_skill`` — matching the next paste
    # to the most recent PTT release's skill-mode flag via a small
    # FIFO that the macropad pushes to on PTT release.
    #
    # Focused mode is intentionally hands-off: Superwhisper's paste
    # *is* the action there (typing into Slack, kitty, etc.), so the
    # orchestrator drops any stdin lines that arrive while not in
    # queue mode and discards stale skill-mode flags.

    ptt_release_skill_q: queue_lib.Queue[bool] = queue_lib.Queue()
    _BRACKETED_PASTE_RE = (b"\x1b[200~", b"\x1b[201~")
    # ANSI CSI escape sequence: ESC [ <params> <intermediates> <final>.
    # Covers things like the forwarded Delete key (\x1b[3~) and arrow
    # keys (\x1b[A, …) — we strip them before they pollute the input
    # buffer with junk that the user can't see was typed.
    _ANSI_CSI_RE = re.compile(rb"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]")

    def on_ptt_release(skill_mode: bool) -> None:
        # Only push when in queue mode — otherwise Superwhisper pasted
        # into some other app and we won't see a stdin transcript to
        # match the flag against.
        logger.info(
            "on_ptt_release(skill_mode=%s) — app_mode=%s — %s",
            skill_mode, ctx.app_mode,
            "queued for stdin" if ctx.app_mode == "queue" else "dropped (not in queue mode)",
        )
        if ctx.app_mode == "queue":
            # Wipe any junk that accumulated during the PTT hold (e.g.
            # the forwarded delete key delivered as \x1b[3~ before we
            # got to strip it, or stray keystrokes from before the PTT
            # cycle). The upcoming paste should land in a clean buffer.
            if input_buffer is not None:
                input_buffer.clear()
            ptt_release_skill_q.put(skill_mode)

    # Auto-submit window after the last byte arrives. Long enough that
    # a burst-paste (Superwhisper) finishes cleanly, short enough that
    # the user doesn't feel a delay before the skill kicks off.
    _STDIN_QUIET_S = 0.4

    def _stdin_paste_loop() -> None:
        """Read raw stdin chars into ``ctx.input_buffer``.

        Stdin is in cbreak mode (set up before this thread starts), so
        ``os.read`` returns bytes as soon as they're available — no
        line buffering. We append printable chars to the buffer,
        handle backspace / Esc / Enter as editing keys, and auto-submit
        on a quiet pause when there's a recent PTT release (the
        common voice flow: Superwhisper pastes a burst, then idle).
        Without a PTT release pending we wait for Enter — that's the
        manual-typing / edit-the-paste flow.
        """
        if input_buffer is None:
            return
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, OSError):
            logger.info("stdin has no fileno; paste reader disabled.")
            return
        try:
            stdin_isatty = sys.stdin.isatty()
        except Exception:
            stdin_isatty = False
        logger.info("stdin paste reader live (fd=%d, isatty=%s)", fd, stdin_isatty)
        while not shutdown.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
            except (OSError, ValueError):
                return
            if ready:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    return
                if not chunk:
                    logger.info("stdin EOF; paste reader exiting.")
                    return
                logger.info("stdin got %d bytes: %r", len(chunk), chunk[:120])
                _ingest_stdin_chunk(chunk)
            # Auto-submit on quiet pause when a PTT release is pending
            # (voice-driven flow). Manual typing without a recent PTT
            # waits for Enter so the user can edit.
            if (
                ctx.app_mode == "queue"
                and not input_buffer.is_empty()
                and input_buffer.quiet_seconds() > _STDIN_QUIET_S
                and not ptt_release_skill_q.empty()
            ):
                _submit_input_buffer()

    def _ingest_stdin_chunk(chunk: bytes) -> None:
        if input_buffer is None:
            return
        # Strip ANSI CSI sequences first so the forwarded Delete key
        # (\x1b[3~) and friends don't leak into the buffer as junk.
        text = _ANSI_CSI_RE.sub(b"", chunk)
        for marker in _BRACKETED_PASTE_RE:
            text = text.replace(marker, b"")
        decoded = text.decode("utf-8", errors="replace")
        for ch in decoded:
            if ch in ("\r", "\n"):
                _submit_input_buffer()
            elif ch in ("\x7f", "\x08"):  # DEL / Backspace
                input_buffer.backspace()
            elif ch == "\x1b":  # bare Esc (strip-markers already removed CSI prefixes)
                input_buffer.clear()
            elif ch == "\x03":
                # Ctrl-C in cbreak mode still raises SIGINT via ISIG.
                # Nothing to do here.
                pass
            elif ch == " " or ch.isprintable():
                input_buffer.append(ch)
            # Other control chars are dropped.

    def _submit_input_buffer() -> None:
        if input_buffer is None:
            return
        transcript = input_buffer.pop().strip()
        if not transcript:
            return
        if ctx.app_mode != "queue":
            # Focused mode — discard, and drain any stale flag too.
            try:
                ptt_release_skill_q.get_nowait()
            except queue_lib.Empty:
                pass
            return
        try:
            skill_mode = ptt_release_skill_q.get_nowait()
        except queue_lib.Empty:
            skill_mode = False
        logger.info("submit transcript (skill_mode=%s): %s", skill_mode, transcript)
        if skill_mode:
            handle_skill(ctx, transcript)
        else:
            handle_voice(ctx, transcript)

    def on_ptt_press() -> None:
        # PTT-press while Claude is speaking should stop playback so the
        # mic input isn't talking over TTS.
        stop_playback(ctx)

    # Dispatch off the pynput listener thread — anything slow (TTS,
    # ssh capture) here would freeze the keyboard.
    def on_chord(name: str) -> None:
        threading.Thread(target=handle_chord, args=(ctx, name), daemon=True).start()

    def on_tap(name: str) -> None:
        threading.Thread(target=handle_tap, args=(ctx, name), daemon=True).start()

    ptt_forward_key = (
        resolve_key(config.stt_local_hotkey) if config.stt_provider == "local" else None
    )
    macropad = Macropad(
        keymap={
            "ptt": resolve_key(config.ptt_key),
            "act": resolve_key(config.act_key),
            "yes": resolve_key(config.yes_key),
            "no": resolve_key(config.no_key),
            "nav": resolve_key(config.nav_key),
        },
        on_audio=on_audio,
        on_chord=on_chord,
        on_tap=on_tap,
        on_ptt_press=on_ptt_press,
        on_ptt_release=on_ptt_release,
        ptt_forward_key=ptt_forward_key,
        sample_rate=config.sample_rate,
        device=config.audio_device,
    )

    def _on_signal(signum: int) -> None:
        logger.info("Received signal %d; shutting down.", signum)
        _trigger_shutdown()

    # add_signal_handler runs the callback on the loop, not in signal
    # context, so it can safely set asyncio.Event. Only works on the
    # main thread (asyncio.run runs there). Unix-only; macOS supported.
    loop.add_signal_handler(signal.SIGINT, _on_signal, signal.SIGINT)
    loop.add_signal_handler(signal.SIGTERM, _on_signal, signal.SIGTERM)

    ptt_desc = (
        f"{config.ptt_key} (forwards to {config.stt_local_hotkey})"
        if ptt_forward_key is not None
        else f"{config.ptt_key} (OpenAI STT)"
    )
    logger.info(
        "Starting code-trip. PTT=%s NAV=%s (hold NAV + key for chords). Ctrl-C to quit.",
        ptt_desc,
        config.nav_key,
    )
    # In local STT mode, the stdin reader needs to see chars as they
    # arrive (Superwhisper pastes don't end with a newline). Flip stdin
    # to cbreak: disable canonical buffering + echo, keep ISIG so
    # Ctrl-C still raises SIGINT.
    stdin_termios_saved: list | None = None
    if config.stt_provider != "openai":
        try:
            stdin_fd = sys.stdin.fileno()
            if sys.stdin.isatty():
                stdin_termios_saved = termios.tcgetattr(stdin_fd)
                new_attr = termios.tcgetattr(stdin_fd)
                new_attr[3] &= ~(termios.ICANON | termios.ECHO)
                termios.tcsetattr(stdin_fd, termios.TCSANOW, new_attr)
                logger.info("stdin in cbreak mode (no canonical buffering, no echo)")
                # atexit so even a hard crash that bypasses our finally
                # block still leaves the user with a usable terminal.
                _saved = stdin_termios_saved
                _fd = stdin_fd

                def _restore() -> None:
                    try:
                        termios.tcsetattr(_fd, termios.TCSANOW, _saved)
                    except Exception:
                        pass

                atexit.register(_restore)
        except (AttributeError, OSError, termios.error):
            logger.warning("could not put stdin in cbreak mode", exc_info=True)
            stdin_termios_saved = None

    macropad.start()
    supervisor.start_all()
    consumer.start()
    if config.stt_provider != "openai":
        # Reader runs only when the external STT tool is in charge of
        # capture+transcribe (e.g. Superwhisper). In openai mode the
        # orchestrator captures audio itself and dispatches via
        # ``on_audio``, so there's nothing pasted into stdin.
        threading.Thread(target=_stdin_paste_loop, daemon=True).start()
        logger.info("Started stdin paste reader for local STT mode.")
    dashboard = Dashboard(ctx, supervisor=supervisor) if tui else None
    if dashboard is not None:
        dashboard.start()
    try:
        await shutdown_event.wait()
    finally:
        if dashboard is not None:
            dashboard.stop()
        consumer.stop()
        supervisor.stop_all()
        macropad.stop()
        thinking.stop()
        log.close()
        if stdin_termios_saved is not None:
            # Restore the user's terminal modes so they don't end up
            # in a no-echo shell after we exit.
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSANOW, stdin_termios_saved
                )
            except (OSError, termios.error):
                logger.warning("could not restore stdin termios", exc_info=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="code-trip")
    parser.add_argument("--config", type=Path, required=True, help="Path to TOML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Show a live status dashboard. Suppresses Python logging output.",
    )
    args = parser.parse_args(argv)

    if args.tui:
        # The live display owns the terminal; route logs to a file so they
        # don't clobber the dashboard. tail -f the file in another pane to
        # debug.
        log_dir = Path.home() / ".code-trip" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "orchestrator.log"
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
            handlers=[logging.FileHandler(log_path)],
            force=True,
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        )

    # Native crashes (segfaults, aborts from C extensions like PortAudio
    # / pynput Quartz / Rich's terminal handling) don't print a Python
    # traceback by default. faulthandler installs signal handlers that
    # dump a Python-level stack trace to a dedicated file before the
    # process exits, so "Python crashed with no message" leaves at least
    # one breadcrumb to chase.
    fault_log_dir = Path.home() / ".code-trip" / "logs"
    fault_log_dir.mkdir(parents=True, exist_ok=True)
    fault_log = fault_log_dir / "faulthandler.log"
    try:
        # ``open`` deliberately unclosed — faulthandler needs the fd alive
        # for the whole process lifetime.
        faulthandler.enable(file=open(fault_log, "a"), all_threads=True)
    except Exception:
        pass

    config = load_config(args.config)
    run(config, tui=args.tui)
    return 0


if __name__ == "__main__":
    sys.exit(main())
