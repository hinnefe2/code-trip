"""Entry point: wire everything and run the PTT loop."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import faulthandler
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Callable

from code_trip2 import audio_out, earcon
from code_trip2.chords import handle_chord, handle_tap
from code_trip2.config import Config, load_config
from code_trip2.dispatch import QueueConsumer, handle_skill, handle_voice
from code_trip2.linear_state import LinearState
from code_trip2.macropad import Macropad, resolve_key
from code_trip2.modes import Context, stop_playback
from code_trip2.poll_health import PollHealth
from code_trip2.producers import ProducerSupervisor
from code_trip2.email_state import EmailState
from code_trip2.producers.email import EmailProducer
from code_trip2.producers.linear import LinearProducer
from code_trip2.producers.manual import ManualProducer
from code_trip2.producers.reconcile import ReconcileProducer
from code_trip2.producers.slack import SlackProducer
from code_trip2.producers.windows import WindowProducer
from code_trip2.queue_log import QueueLog
from code_trip2.producers.claude_mcp import ClaudeMCPClient
from code_trip2.producers.mcp_batch import BatchedMCPClient, MCPBatcher
from code_trip2.screener import (
    AutohandleLogEntry,
    ScreeningOutcome,
    candidates_for,
    run_screener_loop,
)
from code_trip2.session_log import SessionLogger, default_session_path
from code_trip2.skills import load_skill_manifests
from code_trip2.slack_state import SlackState
from code_trip2.stt_client import STTClient, STTClientError
from code_trip2.summarizer import Summarizer
from code_trip2.tasks import STATE_SCREENING, Task, TaskQueue
from code_trip2.tts_client import SilentTTSClient, TTSClient
from code_trip2.tui import CodeTripApp, detect_tui_host_app

logger = logging.getLogger(__name__)


def run(config: Config, *, tui: bool = False, silent: bool = False) -> None:
    asyncio.run(main_async(config, tui=tui, silent=silent))


async def main_async(config: Config, *, tui: bool = False, silent: bool = False) -> None:
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
    tts: TTSClient | SilentTTSClient
    if silent or config.audio_silent:
        tts = SilentTTSClient()
        earcon.set_silent(True)
        source = "--silent" if silent else "[audio] silent"
        logger.info("Silent mode (%s): TTS + earcons disabled.", source)
    else:
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
    # Drop producer-sourced tasks from the replay where the producer
    # itself is the source of truth. EmailProducer's first poll does a
    # wide pull from the current inbox; LinearProducer's first poll
    # does a wide pull from Linear. Anything the user already
    # archived / closed won't come back; anything still active
    # resurfaces as a fresh task. No need to remember dismissed-
    # without-archive state across restarts.
    _source_of_truth_kinds = {"email_msg", "linear_issue"}
    dropped_sot = sum(1 for t in replayed if t.kind in _source_of_truth_kinds)
    replayed = [t for t in replayed if t.kind not in _source_of_truth_kinds]
    if replayed:
        queue.load(replayed)
        logger.info("Replayed %d tasks from queue log", len(replayed))
    if dropped_sot:
        logger.info(
            "Dropped %d replayed producer-sourced tasks; first poll will refresh",
            dropped_sot,
        )

    tui_host_app = detect_tui_host_app() if tui else None
    if tui and tui_host_app:
        logger.info("TUI host detected as %r; suppressing synthesized "
                    "keystrokes that would target it.", tui_host_app)

    # All single-tool MCP calls funnel through one coalescing batcher:
    # producers poll on wall-clock-aligned ticks, so their calls land
    # inside the same collection window and run in ONE claude session
    # instead of one each — headless claude calls bill outside plan
    # limits, so sessions are the cost unit that matters.
    mcp_batcher = MCPBatcher(window_s=config.mcp_batch_window)
    slack_mcp = BatchedMCPClient(server_id="claude_ai_Slack", batcher=mcp_batcher)
    if not slack_mcp.enabled:
        logger.info(
            "Slack MCP via claude CLI not available; Slack producer will "
            "stay idle. Install claude CLI to enable."
        )
    email_mcp = BatchedMCPClient(server_id="claude_ai_Gmail", batcher=mcp_batcher)
    if not email_mcp.enabled:
        logger.info(
            "Gmail MCP via claude CLI not available; Email producer will "
            "stay idle. Install claude CLI to enable."
        )
    linear_mcp = BatchedMCPClient(server_id="claude_ai_Linear", batcher=mcp_batcher)
    if not linear_mcp.enabled:
        logger.info(
            "Linear MCP via claude CLI not available; Linear producer will "
            "stay idle. Install claude CLI to enable."
        )
    # Free-form skill invocation (ACT+PTT). ``server_id`` is unused —
    # run_agent doesn't restrict to a single tool.
    agent_mcp = ClaudeMCPClient(server_id="agent")
    # Screener classifier (skill nomination) and executor (skill running)
    # clients. Both configurable; a weak model nominates the wrong skill
    # and narrates side effects it never performed, so both default to a
    # stronger model.
    classifier_mcp = ClaudeMCPClient(
        server_id="agent", model=config.autohandle_classifier_model,
    )
    executor_mcp = ClaudeMCPClient(
        server_id="agent", model=config.autohandle_executor_model,
    )
    # Skill manifests for both the ACT+PTT path (allowed-tools union)
    # and the screener (per-skill metadata: description, auto-handle
    # flags, kinds). Loaded once on startup. Resolves relative to CWD
    # because the user runs the orchestrator from the project root.
    skills_dir = Path.cwd() / ".claude" / "skills"
    skill_manifests = load_skill_manifests(skills_dir)
    agent_allowed_tools: tuple[str, ...] = tuple(sorted({
        tool for m in skill_manifests for tool in m.allowed_tools
    }))
    if agent_allowed_tools:
        logger.info(
            "Loaded %d skill manifests, %d unique allowed-tools from %s",
            len(skill_manifests), len(agent_allowed_tools), skills_dir,
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
        linear_mcp=linear_mcp,
        agent_mcp=agent_mcp,
        agent_allowed_tools=agent_allowed_tools,
    )

    # Task screener. Every producer routes new tasks through ``submit``
    # below: screenable tasks land in the queue in the ``screening``
    # state (resident, so re-sightings collapse into them) and their ids
    # go onto the screener's work queue; everything else lands straight
    # as ``pending``. The screener coroutine is only scheduled when
    # autohandle is actually doing something — otherwise every task
    # takes the pending path and the worker isn't needed.
    autohandle_active = (
        config.autohandle_enabled
        and config.autohandle_kinds
        and any(m.auto_handle for m in skill_manifests)
        and agent_mcp.enabled
    )
    # Work items are ``(task_id, mode)``: "intake" for a task parked in
    # the screening state, "reconsider" for an already-visible task the
    # producer wants re-judged (e.g. SlackProducer when a tracked thread
    # gets new activity). One queue — the screener is serial anyway.
    screener_work: "asyncio.Queue[tuple[str, str]]" = asyncio.Queue()
    screener_stop = asyncio.Event()

    def _on_screener_outcome(outcome: ScreeningOutcome) -> None:
        # Append to the TUI log first — the session log and queue log
        # are useful for offline analysis but the user is staring at
        # the TUI right now, and we don't want a logging hiccup to
        # eat the visible entry.
        if outcome.action != "forward" or outcome.dry_run_nominated:
            ctx.autohandle_log.append(
                AutohandleLogEntry(ts=time.time(), outcome=outcome)
            )
        # ``task_kind``, not ``kind`` — SessionLogger.event() takes
        # ``kind`` as its positional event-name parameter, so a kwarg
        # of the same name collides at call time.
        log.event(
            "task_screened",
            task_id=outcome.task.id,
            task_kind=outcome.task.kind,
            topic=outcome.task.topic,
            action=outcome.action,
            skill=outcome.skill,
            summary=outcome.summary,
            error=outcome.error,
            dry_run_nominated=outcome.dry_run_nominated,
        )

    # The single pipeline entry point for producer tasks. Screenable
    # kinds park in the ``screening`` state; the rest go straight to
    # ``pending``. Either way the task is resident in ``queue`` from
    # the first moment, so identity checks (``upsert``) see it — no
    # window where a task exists but is invisible, no matter how long
    # ``claude --print`` takes. The screenability check is sync and
    # pure (kind gate + manifest match), so unscreenable tasks skip the
    # screener entirely instead of queueing behind a slow executor run.
    allowed_kinds = frozenset(config.autohandle_kinds)

    def submit(task: Task, *, if_terminal: str = "new") -> Task:
        screenable = (
            autohandle_active
            and task.kind in allowed_kinds
            and bool(candidates_for(task, skill_manifests))
        )
        if screenable:
            task.state = STATE_SCREENING
        resident = queue.upsert(task, if_terminal=if_terminal)
        # Only a task that actually entered needs screening; an upsert
        # that collapsed into an existing task (screening or already
        # user-visible) has nothing new to judge.
        if screenable and resident is task:
            screener_work.put_nowait((task.id, "intake"))
        return resident

    # Reconsider is gated by the same kinds list. SlackProducer only
    # fires it when slack_msg is in autohandle_kinds.
    submit_to_reconsider: Callable[[Task], None] | None = None
    if autohandle_active and "slack_msg" in config.autohandle_kinds:
        submit_to_reconsider = (
            lambda t: screener_work.put_nowait((t.id, "reconsider"))
        )
    if autohandle_active:
        logger.info(
            "Autohandle ON: kinds=%s dry_run=%s (%d eligible skills)",
            list(config.autohandle_kinds),
            config.autohandle_dry_run,
            sum(1 for m in skill_manifests if m.auto_handle),
        )
    else:
        logger.info(
            "Autohandle OFF (enabled=%s, kinds=%s, agent_mcp.enabled=%s)",
            config.autohandle_enabled,
            list(config.autohandle_kinds),
            agent_mcp.enabled,
        )

    # Consecutive-failure surfacing for the MCP pollers: a dead
    # pipeline becomes a queue task instead of a silent WARNING streak
    # (a 10-day Gmail outage went unnoticed before this existed).
    # Threshold 0 disables.
    def _health(producer: str) -> PollHealth | None:
        threshold = int(config.poll_failure_alert_threshold)
        if threshold <= 0:
            return None
        return PollHealth(producer=producer, queue=queue, threshold=threshold)

    supervisor = ProducerSupervisor()
    supervisor.add(WindowProducer(
        config=config, queue=queue, submit=submit, mirrors=ctx.window_mirrors,
    ))
    supervisor.add(SlackProducer(
        config=config, queue=queue, mcp=slack_mcp, state=SlackState(),
        submit=submit, reconsider=submit_to_reconsider,
        health=_health("slack"),
    ))
    email_producer = EmailProducer(
        config=config, queue=queue, mcp=email_mcp,
        state=EmailState(), submit=submit, health=_health("email"),
    )
    supervisor.add(email_producer)
    supervisor.add(LinearProducer(
        config=config, queue=queue, mcp=linear_mcp,
        state=LinearState(), submit=submit, health=_health("linear"),
    ))
    supervisor.add(ManualProducer())

    # Reconcile sweep: walks the standing queue on a timer and retires
    # tasks handled outside the orchestrator. LinearProducer already
    # retires closed tickets on its own poll, so the gap this closes is
    # email — an archived/read/replied mail whose task would otherwise
    # linger. Only wire the email reconciler when the Gmail MCP is live.
    reconcilers = []
    if email_mcp is not None and getattr(email_mcp, "enabled", False):
        reconcilers.append(email_producer.reconcile_inbox)
    if reconcilers:
        supervisor.add(ReconcileProducer(config=config, reconcilers=reconcilers))

    consumer = QueueConsumer(ctx)
    consumer.attach()

    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    local_stt = config.stt_provider != "openai"
    app: CodeTripApp | None = None
    if tui:
        app = CodeTripApp(ctx, supervisor=supervisor, local_stt=local_stt)
        # Wire just the callable dispatch needs — submit-the-Input. The
        # macropad YES tap in queue mode calls this via Context. Passing
        # only the bound method avoids exposing the whole app surface
        # to dispatch and keeps the import direction one-way.
        ctx.submit_input = app.submit_input

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

    async def _verify_side_effect(task: Task, verify_type: str) -> bool | None:
        """Ground-truth check that a handled skill's side effect happened.

        ``left-inbox``: the skill was supposed to archive the email, so
        confirm the thread no longer carries Gmail's ``INBOX`` label.
        Returns None (inconclusive → trust the skill) for an unknown check
        type, a task with no thread id, or a failed/empty Gmail read — we
        only override a "handled" verdict on positive evidence it didn't
        happen, never on an inability to check.
        """
        if verify_type != "left-inbox":
            return None
        thread_id = (task.source or {}).get("thread_id")
        if not thread_id or not email_mcp.enabled:
            return None
        try:
            result = await email_mcp.call_tool(
                "get_thread",
                {"threadId": thread_id, "messageFormat": "MINIMAL"},
            )
        except Exception:
            logger.warning(
                "verify left-inbox: get_thread failed for %s", thread_id,
                exc_info=True,
            )
            return None
        messages = (result or {}).get("messages") or []
        if not messages:
            return None
        still_in_inbox = any(
            "INBOX" in (m.get("labelIds") or []) for m in messages
        )
        return not still_in_inbox

    screener_task: asyncio.Task | None = None
    if autohandle_active:
        screener_task = asyncio.create_task(
            run_screener_loop(
                work=screener_work,
                queue=queue,
                manifests=skill_manifests,
                mcp=executor_mcp,
                classifier_mcp=classifier_mcp,
                verify_side_effect=_verify_side_effect,
                on_outcome=_on_screener_outcome,
                allowed_kinds=allowed_kinds,
                dry_run=config.autohandle_dry_run,
                stop=screener_stop,
                # Follow-up spawns land terminal-suppressed: a
                # respawned follow-up whose original was already filed
                # or dismissed stays gone (terminal durable records
                # are replayed for exactly this).
                submit_follow_up=lambda t: submit(t, if_terminal="skip"),
            ),
            name="screener",
        )
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
        if screener_task is not None:
            screener_stop.set()
            try:
                await asyncio.wait_for(screener_task, timeout=2.0)
            except asyncio.TimeoutError:
                screener_task.cancel()
                try:
                    await screener_task
                except (asyncio.CancelledError, Exception):
                    pass
        await supervisor.stop_all()
        macropad.stop()
        thinking.stop()
        # Release the shared CoreAudio stream so the OS doesn't keep
        # the device pinned at our sample rate — leaving it open is
        # what makes the *next* app's first sound get clipped.
        audio_out.shutdown()
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
    parser.add_argument(
        "--silent",
        action="store_true",
        help=(
            "Disable all audio output (TTS speech + earcon beeps). "
            "Same as [audio] silent = true in the config."
        ),
    )
    parser.add_argument(
        "--after-hours",
        action="store_true",
        help=(
            "Ignore the [poll] active-hours window and poll around the "
            "clock. Use for late-night work that would otherwise wait "
            "until the window reopens."
        ),
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
    if args.after_hours:
        config = dataclasses.replace(config, poll_ignore_active_hours=True)
    run(config, tui=args.tui, silent=args.silent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
