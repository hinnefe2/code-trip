"""SHIP mode: create a draft PR via remote Claude and read back the URL."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from code_trip.earcon import ThinkingEarcon, play_completion, play_error
from code_trip.mode_fsm import (
    KEY_BEHAVIORS,
    Gesture,
    Key,
    Mode,
    ModeFSM,
)
from code_trip.remote_tmux import RemoteTmux, RemoteTmuxError, WaitTimeout
from code_trip.tts_client import TTSClient, TTSClientError

logger = logging.getLogger(__name__)


PUSH_PROMPT = (
    "Create a draft PR for the current branch using "
    "'gh pr create --draft --fill'. Respond with ONLY the PR URL on a "
    "single line."
)

_URL_RE = re.compile(r"https?://\S+")


class ShipError(Exception):
    """Raised when SHIP cannot drive the remote session."""


@dataclass
class ShipController:
    tmux: RemoteTmux
    tts: TTSClient
    thinking: ThinkingEarcon
    session: str
    window: str
    wait_timeout: float = 300.0
    pr_url: str | None = None

    def enter(self) -> None:
        self.pr_url = None
        self.tts.speak("Ready to ship. Press okay to create the draft PR.")

    def push_pr(self, fsm: ModeFSM) -> None:
        try:
            self.tmux.send_keys(self.session, self.window, PUSH_PROMPT)
        except RemoteTmuxError as exc:
            self._report_error(f"Could not reach Claude: {exc}")
            return

        self.thinking.start()
        try:
            try:
                self.tmux.wait_for_claude(
                    self.session, self.window, timeout=self.wait_timeout
                )
            except WaitTimeout:
                self._report_error("Claude did not respond in time.")
                return
            except RemoteTmuxError as exc:
                self._report_error(f"Lost connection to Claude: {exc}")
                return
            try:
                raw = self.tmux.capture_pane(
                    self.session, self.window, lines=200
                )
            except RemoteTmuxError as exc:
                self._report_error(f"Could not read Claude's response: {exc}")
                return
        finally:
            self.thinking.stop()

        match = _URL_RE.search(raw)
        if not match:
            self._report_error("Could not find a PR URL in the response.")
            return
        url = match.group(0).rstrip(".,)\"'")
        self.pr_url = url

        try:
            self.tts.speak(f"Pull request created. {_spoken_url(url)}")
        except TTSClientError as exc:
            self._report_error(f"Speech failed: {exc}")
            return

        try:
            play_completion()
        except Exception:
            logger.exception("Completion earcon failed")

        fsm.transition_to(Mode.IDLE)

    def handle_voice(self, transcript: str) -> None:
        text = transcript.strip()
        if not text:
            return
        try:
            self.tmux.send_keys(self.session, self.window, text)
        except RemoteTmuxError as exc:
            self._report_error(f"Could not reach Claude: {exc}")
            return

        self.thinking.start()
        try:
            try:
                self.tmux.wait_for_claude(
                    self.session, self.window, timeout=self.wait_timeout
                )
            except WaitTimeout:
                self._report_error("Claude did not respond in time.")
                return
            except RemoteTmuxError as exc:
                self._report_error(f"Lost connection to Claude: {exc}")
                return
        finally:
            self.thinking.stop()

        try:
            self.tts.speak("Updated.")
        except TTSClientError as exc:
            self._report_error(f"Speech failed: {exc}")
            return

        try:
            play_completion()
        except Exception:
            logger.exception("Completion earcon failed")

    def back_to_review(self, fsm: ModeFSM) -> None:
        fsm.transition_to(Mode.REVIEW)

    def _report_error(self, message: str) -> None:
        self.thinking.stop()
        try:
            play_error()
        except Exception:
            logger.exception("Error earcon failed")
        try:
            self.tts.speak(message)
        except Exception:
            logger.exception("Failed to speak error: %s", message)


def _spoken_url(url: str) -> str:
    # Replace "://" and "/" so TTS reads a URL more naturally.
    return url.replace("://", " ").replace("/", " slash ")


def register_ship_handlers(fsm: ModeFSM, ctl: ShipController) -> None:
    """Install SHIP-mode handlers into the global KEY_BEHAVIORS table."""

    del fsm

    def ok_short(f: ModeFSM) -> None:
        ctl.push_pr(f)

    def no_short(f: ModeFSM) -> None:
        ctl.back_to_review(f)

    handlers: dict[tuple[Mode, Key, Gesture], Callable[[ModeFSM], None]] = {
        (Mode.SHIP, Key.OK, Gesture.SHORT): ok_short,
        (Mode.SHIP, Key.NO, Gesture.SHORT): no_short,
    }
    KEY_BEHAVIORS.update(handlers)
