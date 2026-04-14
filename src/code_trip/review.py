"""REVIEW mode: step through post-implementation findings.

Follows the ``WorkController`` pattern (see ``work.py``). Review findings
come from a remote Claude turn that returns a JSON array of short strings.
PTT voice feedback is forwarded as a normal turn.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from code_trip.earcon import ThinkingEarcon, play_completion, play_error
from code_trip.mode_fsm import (
    KEY_BEHAVIORS,
    Gesture,
    Key,
    Mode,
    ModeFSM,
)
from code_trip.remote_tmux import RemoteTmux, RemoteTmuxError, WaitTimeout
from code_trip.summarizer import Summarizer, SummarizerError
from code_trip.tts_client import TTSClient, TTSClientError

logger = logging.getLogger(__name__)


REVIEW_PROMPT = (
    "Review the recent changes. Respond with ONLY a JSON array of short "
    "finding strings (no prose, no code fences). Empty array if nothing "
    "to flag."
)

RERUN_PROMPT = "Re-run the review tools and list any new findings."

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class ReviewError(Exception):
    """Raised when REVIEW cannot drive the remote session."""


@dataclass
class ReviewController:
    tmux: RemoteTmux
    tts: TTSClient
    summarizer: Summarizer
    thinking: ThinkingEarcon
    session: str
    window: str
    wait_timeout: float = 300.0

    findings: list[str] = field(default_factory=list)
    cursor: int = 0
    on_ship: Optional[Callable[[ModeFSM], None]] = None

    # --- lifecycle --------------------------------------------------------

    def enter(self) -> None:
        """Fetch findings from remote Claude and announce the count."""
        self.findings = []
        self.cursor = 0
        try:
            self.tmux.send_keys(self.session, self.window, REVIEW_PROMPT)
            self.thinking.start()
            try:
                self.tmux.wait_for_claude(
                    self.session, self.window, timeout=self.wait_timeout
                )
                raw = self.tmux.capture_pane(
                    self.session, self.window, lines=400
                )
            finally:
                self.thinking.stop()
        except (RemoteTmuxError, WaitTimeout) as exc:
            self._report_error(f"Could not fetch review: {exc}")
            return

        items = _parse_findings(raw)
        if items is None:
            self._report_error("Claude did not return a findings array.")
            return
        self.findings = items
        if not items:
            self.tts.speak("No findings. Press okay to ship.")
            return
        self.tts.speak(
            f"{len(items)} finding{'s' if len(items) != 1 else ''}. "
            "Press next to step through."
        )

    # --- navigation -------------------------------------------------------

    def next(self) -> None:
        if not self.findings:
            self.tts.speak("No findings.")
            return
        self.cursor = (self.cursor + 1) % len(self.findings)
        self._announce()

    def prev(self) -> None:
        if not self.findings:
            self.tts.speak("No findings.")
            return
        self.cursor = (self.cursor - 1) % len(self.findings)
        self._announce()

    def _announce(self) -> None:
        self.tts.speak(
            f"Finding {self.cursor + 1}. {self.findings[self.cursor]}"
        )

    # --- actions ----------------------------------------------------------

    def rerun(self) -> None:
        self._run_turn(RERUN_PROMPT)

    def handle_voice(self, transcript: str) -> None:
        text = transcript.strip()
        if not text:
            return
        self._run_turn(text)

    def approve_all(self, fsm: ModeFSM) -> None:
        fsm.transition_to(Mode.SHIP)
        if self.on_ship is not None:
            self.on_ship(fsm)

    def back_to_work(self, fsm: ModeFSM) -> None:
        fsm.transition_to(Mode.WORK)

    # --- turn loop --------------------------------------------------------
    # Duplicates WorkController._run_turn on purpose until all modes land.

    def _run_turn(self, text: str) -> None:
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
            try:
                raw = self.tmux.capture_pane(self.session, self.window)
            except RemoteTmuxError as exc:
                self._report_error(f"Could not read Claude's response: {exc}")
                return
        finally:
            self.thinking.stop()

        try:
            summary = self.summarizer.summarize(raw, user_request=text)
        except SummarizerError as exc:
            self._report_error(f"Summarization failed: {exc}")
            return

        try:
            self.tts.speak(summary)
        except TTSClientError as exc:
            self._report_error(f"Speech failed: {exc}")
            return

        try:
            play_completion()
        except Exception:
            logger.exception("Completion earcon failed")

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


def _parse_findings(raw: str) -> list[str] | None:
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    return [str(item) for item in payload if isinstance(item, (str, int, float))]


# --- KEY_BEHAVIORS registration ------------------------------------------


def register_review_handlers(fsm: ModeFSM, ctl: ReviewController) -> None:
    """Install REVIEW-mode handlers into the global KEY_BEHAVIORS table."""

    del fsm

    def nav_short(_f: ModeFSM) -> None:
        ctl.next()

    def nav_long(_f: ModeFSM) -> None:
        ctl.prev()

    def act_short(_f: ModeFSM) -> None:
        ctl.rerun()

    def ok_short(f: ModeFSM) -> None:
        ctl.approve_all(f)

    def no_short(f: ModeFSM) -> None:
        ctl.back_to_work(f)

    handlers: dict[tuple[Mode, Key, Gesture], Callable[[ModeFSM], None]] = {
        (Mode.REVIEW, Key.NAV, Gesture.SHORT): nav_short,
        (Mode.REVIEW, Key.NAV, Gesture.LONG): nav_long,
        (Mode.REVIEW, Key.ACT, Gesture.SHORT): act_short,
        (Mode.REVIEW, Key.OK, Gesture.SHORT): ok_short,
        (Mode.REVIEW, Key.NO, Gesture.SHORT): no_short,
    }
    KEY_BEHAVIORS.update(handlers)
