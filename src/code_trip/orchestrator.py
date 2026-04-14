"""Orchestrator: wires all Stage 1a components into the voice loop."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from pynput import keyboard

from code_trip.audio_recorder import AudioRecorder
from code_trip.browse import BrowseController, register_browse_handlers
from code_trip.config import Config
from code_trip.earcon import ThinkingEarcon, play_completion, play_error
from code_trip.mode_fsm import KEY_BEHAVIORS, Gesture, Key, Mode, ModeFSM
from code_trip.push_to_talk import PushToTalk
from code_trip.remote_tmux import RemoteTmux, RemoteTmuxError, WaitTimeout
from code_trip.review import ReviewController, register_review_handlers
from code_trip.ship import ShipController, register_ship_handlers
from code_trip.stt_client import STTClient, STTClientError
from code_trip.summarizer import Summarizer, SummarizerError, Verbosity
from code_trip.tts_client import TTSClient, TTSClientError
from code_trip.work import WorkController, register_work_handlers

logger = logging.getLogger(__name__)


class OrchestratorError(Exception):
    """Raised for orchestrator wiring problems."""


@dataclass
class OrchestratorDeps:
    tmux: RemoteTmux
    recorder: AudioRecorder
    ptt: PushToTalk
    stt: STTClient
    summarizer: Summarizer
    tts: TTSClient
    thinking: ThinkingEarcon


def _build_deps(config: Config, handler) -> OrchestratorDeps:
    try:
        hotkey = getattr(keyboard.Key, config.audio.hotkey)
    except AttributeError as exc:
        raise OrchestratorError(
            f"Unknown hotkey '{config.audio.hotkey}'"
        ) from exc

    try:
        verbosity = Verbosity[config.openai.verbosity.upper()]
    except KeyError as exc:
        raise OrchestratorError(
            f"Unknown verbosity '{config.openai.verbosity}'"
        ) from exc

    tmux = RemoteTmux(host=config.ssh.host, ssh_options=config.ssh.options)
    recorder = AudioRecorder(
        sample_rate=config.audio.sample_rate, device=config.audio.device
    )
    ptt = PushToTalk(recorder=recorder, hotkey=hotkey, on_recording_complete=handler)
    stt = STTClient(api_key=config.openai.api_key, model=config.openai.stt_model)
    summarizer = Summarizer(
        model=config.openai.summarizer_model,
        api_key=config.openai.api_key,
        verbosity=verbosity,
    )
    tts = TTSClient(
        api_key=config.openai.api_key,
        model=config.openai.tts_model,
        voice=config.openai.tts_voice,
        speed=config.openai.tts_speed,
    )
    return OrchestratorDeps(
        tmux=tmux,
        recorder=recorder,
        ptt=ptt,
        stt=stt,
        summarizer=summarizer,
        tts=tts,
        thinking=ThinkingEarcon(),
    )


class Orchestrator:
    def __init__(
        self, config: Config, deps: OrchestratorDeps | None = None
    ) -> None:
        self._config = config
        self._deps = deps if deps is not None else _build_deps(config, self._handle_recording)
        # When caller supplied deps, they own the PTT wiring — do nothing here.
        if deps is None:
            # _build_deps already wired the callback; nothing more to do.
            pass
        self._shutdown = threading.Event()
        self._fsm = ModeFSM(tts=self._deps.tts)
        self._browse = BrowseController(
            tmux=self._deps.tmux,
            tts=self._deps.tts,
            session=config.tmux.session,
            browse_window=config.tmux.browse_window,
            wait_timeout=config.claude.wait_timeout,
        )
        register_browse_handlers(self._fsm, self._browse)
        self._work = WorkController(
            tmux=self._deps.tmux,
            tts=self._deps.tts,
            summarizer=self._deps.summarizer,
            thinking=self._deps.thinking,
            session=config.tmux.session,
            window=config.tmux.window,
            wait_timeout=config.claude.wait_timeout,
        )
        register_work_handlers(self._fsm, self._work)
        self._review = ReviewController(
            tmux=self._deps.tmux,
            tts=self._deps.tts,
            summarizer=self._deps.summarizer,
            thinking=self._deps.thinking,
            session=config.tmux.session,
            window=config.tmux.window,
            wait_timeout=config.claude.wait_timeout,
        )
        register_review_handlers(self._fsm, self._review)
        self._ship = ShipController(
            tmux=self._deps.tmux,
            tts=self._deps.tts,
            thinking=self._deps.thinking,
            session=config.tmux.session,
            window=config.tmux.window,
            wait_timeout=config.claude.wait_timeout,
        )
        register_ship_handlers(self._fsm, self._ship)
        self._work.on_escalate = lambda _fsm: self._review.enter()
        self._review.on_ship = lambda _fsm: self._ship.enter()
        self._install_browse_to_work_handoff()
        # TODO: swap PushToTalk for KeypadListener to dispatch NAV/ACT/OK/NO
        # gestures into self._fsm.handle_key. Until then, gestures aren't wired.

    def start(self) -> None:
        logger.info("Starting orchestrator")
        self._deps.ptt.start()
        try:
            self._shutdown.wait()
        finally:
            self._deps.ptt.stop()

    def stop(self) -> None:
        self._shutdown.set()

    # --- core loop --------------------------------------------------------

    def _install_browse_to_work_handoff(self) -> None:
        """Chain WORK:PLAN entry after BROWSE's ACT-short ticket selection."""
        key = (Mode.BROWSE, Key.ACT, Gesture.SHORT)
        original = KEY_BEHAVIORS[key]

        def act_short(fsm: ModeFSM) -> None:
            original(fsm)
            if fsm.current is Mode.WORK and self._browse.selected is not None:
                ticket = self._browse.selected
                self._work.enter(ticket.id, ticket.title)

        KEY_BEHAVIORS[key] = act_short

    def _handle_recording(self, audio_path: Path) -> None:
        cfg = self._config
        user_request: str | None = None
        try:
            user_request = self._deps.stt.transcribe(audio_path)
            logger.info("Transcribed: %s", user_request)
        except STTClientError as exc:
            logger.exception("STT failed")
            self._report_error(f"Transcription failed: {exc}")
            return

        if self._fsm.current is Mode.BROWSE:
            self._browse.handle_voice(user_request)
            return
        if self._fsm.current is Mode.WORK:
            self._work.handle_voice(user_request)
            return
        if self._fsm.current is Mode.REVIEW:
            self._review.handle_voice(user_request)
            return
        if self._fsm.current is Mode.SHIP:
            self._ship.handle_voice(user_request)
            return

        try:
            self._deps.tmux.send_keys(
                cfg.tmux.session, cfg.tmux.window, user_request
            )
        except RemoteTmuxError as exc:
            logger.exception("Failed to send keys")
            self._report_error(f"Could not reach Claude: {exc}")
            return

        self._deps.thinking.start()
        try:
            try:
                self._deps.tmux.wait_for_claude(
                    cfg.tmux.session,
                    cfg.tmux.window,
                    timeout=cfg.claude.wait_timeout,
                )
            except WaitTimeout:
                self._report_error("Claude did not respond in time.")
                return
            except RemoteTmuxError as exc:
                self._report_error(f"Lost connection to Claude: {exc}")
                return

            try:
                raw_output = self._deps.tmux.capture_pane(
                    cfg.tmux.session, cfg.tmux.window
                )
            except RemoteTmuxError as exc:
                self._report_error(f"Could not read Claude's response: {exc}")
                return
        finally:
            self._deps.thinking.stop()

        try:
            summary = self._deps.summarizer.summarize(
                raw_output, user_request=user_request
            )
        except SummarizerError as exc:
            logger.exception("Summarizer failed")
            self._report_error(f"Summarization failed: {exc}")
            return

        try:
            self._deps.tts.speak(summary)
        except TTSClientError as exc:
            logger.exception("TTS failed")
            self._report_error(f"Speech failed: {exc}")
            return

        try:
            play_completion()
        except Exception:
            logger.exception("Completion earcon failed")

    def _report_error(self, message: str) -> None:
        self._deps.thinking.stop()
        try:
            play_error()
        except Exception:
            logger.exception("Error earcon failed")
        try:
            self._deps.tts.speak(message)
        except Exception:
            logger.exception("Failed to speak error message: %s", message)
