"""Entry point: wire everything and run the PTT loop."""

from __future__ import annotations

import argparse
import logging
import os
import queue as queue_lib
import select
import signal
import sys
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

    shutdown = threading.Event()

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

    def on_ptt_release(skill_mode: bool) -> None:
        # Only push when in queue mode — otherwise Superwhisper pasted
        # into some other app and we won't see a stdin transcript to
        # match the flag against.
        if ctx.app_mode == "queue":
            ptt_release_skill_q.put(skill_mode)

    def _stdin_paste_loop() -> None:
        """Continuously read stdin in chunks; emit transcripts on quiet pauses.

        Superwhisper's paste arrives as a burst of bytes without a
        trailing newline, so ``readline()`` would block forever. We
        ``select()`` on stdin, accumulate whatever bytes arrive, and
        treat a 250 ms quiet stretch after the first byte as "paste
        done". Bracketed-paste escape markers (``\\x1b[200~`` /
        ``\\x1b[201~``) are stripped — some terminals emit them in
        alt-screen mode, some don't.
        """
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, OSError):
            logger.info("stdin has no fileno; paste reader disabled.")
            return
        buffer = b""
        last_byte_at: float | None = None
        while not shutdown.is_set():
            try:
                ready, _, _ = select.select([fd], [], [], 0.1)
            except (OSError, ValueError):
                return  # stdin closed (e.g. on shutdown)
            if ready:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    return
                if not chunk:
                    return  # EOF
                buffer += chunk
                last_byte_at = time.time()
                continue
            if buffer and last_byte_at is not None and (time.time() - last_byte_at) > 0.25:
                text = buffer
                for marker in _BRACKETED_PASTE_RE:
                    text = text.replace(marker, b"")
                transcript = text.decode("utf-8", errors="replace").strip()
                buffer = b""
                last_byte_at = None
                if transcript:
                    _dispatch_stdin_transcript(transcript)

    def _dispatch_stdin_transcript(transcript: str) -> None:
        if ctx.app_mode != "queue":
            # Drain any stale flag and stay out of the focused-mode
            # paste path entirely.
            try:
                ptt_release_skill_q.get_nowait()
            except queue_lib.Empty:
                pass
            return
        try:
            skill_mode = ptt_release_skill_q.get_nowait()
        except queue_lib.Empty:
            skill_mode = False
        logger.info("stdin transcript (skill_mode=%s): %s", skill_mode, transcript)
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

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("Received signal %d; shutting down.", signum)
        shutdown.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

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
        # Poll instead of an indefinite wait so signal handlers (SIGINT
        # in particular) get to run promptly. Python's signal delivery
        # doesn't always interrupt threading.Event().wait() with no
        # timeout on the main thread.
        while not shutdown.is_set():
            shutdown.wait(timeout=1.0)
    finally:
        if dashboard is not None:
            dashboard.stop()
        consumer.stop()
        supervisor.stop_all()
        macropad.stop()
        thinking.stop()
        log.close()


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

    config = load_config(args.config)
    run(config, tui=args.tui)
    return 0


if __name__ == "__main__":
    sys.exit(main())
