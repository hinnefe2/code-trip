"""Entry point: wire everything and run the PTT loop."""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import logging
import signal
import sys
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
from code_trip2.tui import CodeTripApp, detect_tui_host_app

logger = logging.getLogger(__name__)


def run(config: Config, *, tui: bool = False) -> None:
    asyncio.run(main_async(config, tui=tui))


async def main_async(config: Config, *, tui: bool = False) -> None:
    if config.stt_provider != "openai" and not tui:
        raise SystemExit(
            "Local STT mode (stt_provider != 'openai') requires --tui — "
            "the Textual Input widget is the only path that accepts "
            "pasted transcripts now that the stdin reader has been removed."
        )

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

    supervisor = ProducerSupervisor()
    supervisor.add(ClaudeProducer(config=config, queue=queue, summarizer=summarizer))
    supervisor.add(SlackProducer(config=config, queue=queue, mcp=slack_mcp, state=SlackState()))
    supervisor.add(EmailProducer(config=config, queue=queue, mcp=email_mcp, state=EmailState()))
    supervisor.add(LinearProducer(config=config, queue=queue))
    supervisor.add(ManualProducer())

    consumer = QueueConsumer(ctx)
    consumer.attach()

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    local_stt = config.stt_provider != "openai"
    app: CodeTripApp | None = None
    if tui:
        app = CodeTripApp(ctx, supervisor=supervisor, local_stt=local_stt)

    # Bridge from threaded callbacks (macropad's pynput listener) onto the
    # asyncio loop. The done callback surfaces task exceptions to the log
    # instead of asyncio silently warning on GC.
    def _from_thread(coro_fn, *args, **kwargs) -> None:
        def _schedule() -> None:
            task = loop.create_task(coro_fn(*args, **kwargs))
            task.add_done_callback(_log_task_exception)
        loop.call_soon_threadsafe(_schedule)

    def _log_task_exception(task: "asyncio.Task") -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.exception(
                "Background task %r failed", task.get_name(), exc_info=exc
            )

    async def _process_audio(path: Path, skill_mode: bool) -> None:
        if stt is None:
            logger.warning("on_audio fired in non-openai STT mode; ignoring %s", path)
            return
        try:
            transcript = await stt.transcribe(path)
        except STTClientError as exc:
            logger.exception("STT failed")
            try:
                earcon.error()
                await tts.speak(f"Transcription failed: {exc}")
            except Exception:
                pass
            return
        logger.info("Transcribed (skill_mode=%s): %s", skill_mode, transcript)
        if skill_mode:
            await handle_skill(ctx, transcript)
        else:
            await handle_voice(ctx, transcript)

    def on_audio(path: Path, *, skill_mode: bool = False) -> None:
        _from_thread(_process_audio, path, skill_mode)

    def on_ptt_press() -> None:
        # Sync because it runs on the pynput thread; tts.stop is sync and
        # thread-safe (threading.Event under the hood).
        stop_playback(ctx)

    def on_ptt_release(skill_mode: bool) -> None:
        # Local-STT only: Superwhisper is about to paste the transcript
        # into our terminal. The Textual app's Input widget catches the
        # paste; PttReleased clears the field and arms auto-submit.
        logger.info(
            "on_ptt_release(skill_mode=%s) — app=%s",
            skill_mode, "running" if app is not None else "none",
        )
        if app is not None:
            app.post_ptt_release_from_thread(skill_mode)

    def on_chord(name: str) -> None:
        _from_thread(handle_chord, ctx, name)

    def on_tap(name: str) -> None:
        _from_thread(handle_tap, ctx, name)

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
        shutdown_event.set()
        if app is not None:
            try:
                app.exit()
            except Exception:
                logger.exception("app.exit() during signal failed")

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

    macropad.start()
    supervisor.start_all()
    consumer_task = asyncio.create_task(consumer.run(), name="consumer")
    try:
        if app is not None:
            # Textual owns the foreground until exit. The signal handler
            # calls app.exit() so SIGINT/SIGTERM unwinds cleanly.
            await app.run_async()
        else:
            await shutdown_event.wait()
    finally:
        consumer.request_stop()
        try:
            await asyncio.wait_for(consumer_task, timeout=2.0)
        except asyncio.TimeoutError:
            consumer_task.cancel()
            try:
                await consumer_task
            except (asyncio.CancelledError, Exception):
                pass
        await supervisor.stop_all()
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
