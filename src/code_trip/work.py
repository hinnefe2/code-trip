"""WORK mode: plan → execute → review loop over a remote Claude session.

Mirrors the ``BrowseController`` pattern (see ``browse.py``): the controller
owns ticket/plan/history state, and ``register_work_handlers`` installs the
WORK-mode entries into ``mode_fsm.KEY_BEHAVIORS``. The orchestrator drives
voice (PTT) into ``handle_voice()``.
"""

from __future__ import annotations

import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from code_trip.earcon import ThinkingEarcon, play_completion, play_error
from code_trip.mode_fsm import (
    KEY_BEHAVIORS,
    Gesture,
    Key,
    Mode,
    ModeFSM,
    WorkSubMode,
)
from code_trip.remote_tmux import RemoteTmux, RemoteTmuxError, WaitTimeout
from code_trip.summarizer import Summarizer, SummarizerError
from code_trip.tts_client import TTSClient, TTSClientError

logger = logging.getLogger(__name__)


HISTORY_LIMIT = 10

PLAN_PROMPT = (
    "Produce a short numbered plan for ticket {ticket_id}: {ticket_title}. "
    "Respond with ONLY a JSON array of short step strings (no prose, no code fences)."
)

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


class WorkError(Exception):
    """Raised when WORK cannot drive the remote session."""


@dataclass
class WorkController:
    tmux: RemoteTmux
    tts: TTSClient
    summarizer: Summarizer
    thinking: ThinkingEarcon
    session: str
    window: str
    wait_timeout: float = 300.0

    plan_items: list[str] = field(default_factory=list)
    plan_cursor: int = 0
    history: deque[str] = field(
        default_factory=lambda: deque(maxlen=HISTORY_LIMIT)
    )
    history_cursor: int = 0
    ticket_id: str = ""
    ticket_title: str = ""
    on_escalate: Optional[Callable[["ModeFSM"], None]] = None

    # --- lifecycle --------------------------------------------------------

    def enter(self, ticket_id: str, ticket_title: str) -> None:
        """Called on transition into WORK:PLAN to fetch an initial plan."""
        self.ticket_id = ticket_id
        self.ticket_title = ticket_title
        self.plan_items = []
        self.plan_cursor = 0
        self.history.clear()
        self.history_cursor = 0

        prompt = PLAN_PROMPT.format(
            ticket_id=ticket_id, ticket_title=ticket_title
        )
        try:
            self.tmux.send_keys(self.session, self.window, prompt)
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
            self._report_error(f"Could not fetch plan: {exc}")
            return

        items = _parse_plan(raw)
        if not items:
            self._report_error("Claude did not return a plan.")
            return
        self.plan_items = items
        self.tts.speak(f"{len(items)} step plan. Press next to step through.")

    # --- PLAN handlers ----------------------------------------------------

    def plan_next(self) -> None:
        if not self.plan_items:
            self.tts.speak("No plan yet.")
            return
        self.plan_cursor = (self.plan_cursor + 1) % len(self.plan_items)
        self._announce_plan_item()

    def plan_prev(self) -> None:
        if not self.plan_items:
            self.tts.speak("No plan yet.")
            return
        self.plan_cursor = (self.plan_cursor - 1) % len(self.plan_items)
        self._announce_plan_item()

    def _announce_plan_item(self) -> None:
        item = self.plan_items[self.plan_cursor]
        self.tts.speak(f"Step {self.plan_cursor + 1}. {item}")

    def approve_plan(self, fsm: ModeFSM) -> None:
        fsm.transition_work_sub(WorkSubMode.EXECUTING)
        self._run_turn("Proceed with the plan.")

    def reject_plan(self) -> None:
        self.tts.speak("Rejected. Press talk to give feedback.")

    # --- EXECUTING handlers -----------------------------------------------

    def keep_going(self) -> None:
        self._run_turn("Keep going.")

    def stop_and_explain(self) -> None:
        self._run_turn("Stop and explain what you just did.")

    def escalate_to_review(self, fsm: ModeFSM) -> None:
        fsm.transition_to(Mode.REVIEW)
        if self.on_escalate is not None:
            self.on_escalate(fsm)

    # --- NAV replay -------------------------------------------------------

    def replay_next(self) -> None:
        if not self.history:
            self.tts.speak("No messages to replay.")
            return
        self.history_cursor = (self.history_cursor + 1) % len(self.history)
        self.tts.speak(self.history[self.history_cursor])

    def replay_prev(self) -> None:
        if not self.history:
            self.tts.speak("No messages to replay.")
            return
        self.history_cursor = (self.history_cursor - 1) % len(self.history)
        self.tts.speak(self.history[self.history_cursor])

    # --- PTT voice --------------------------------------------------------

    def handle_voice(self, transcript: str) -> None:
        """Forward transcribed text to Claude and run a turn."""
        text = transcript.strip()
        if not text:
            return
        self._run_turn(text)

    # --- turn loop --------------------------------------------------------
    # Duplicates Orchestrator._handle_recording; that copy will be retired
    # once every mode owns its own controller.

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

        self.history.append(summary)
        self.history_cursor = len(self.history) - 1
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


def _parse_plan(raw: str) -> list[str]:
    match = _JSON_ARRAY_RE.search(raw)
    if not match:
        return []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if isinstance(item, (str, int, float))]


# --- KEY_BEHAVIORS registration ------------------------------------------


def register_work_handlers(fsm: ModeFSM, ctl: WorkController) -> None:
    """Install WORK-mode handlers into the global KEY_BEHAVIORS table."""

    del fsm  # handlers receive the fsm at dispatch time

    def plan_nav_short(_f: ModeFSM) -> None:
        if _f.work_sub is WorkSubMode.PLAN:
            ctl.plan_next()
        else:
            ctl.replay_next()

    def plan_nav_long(_f: ModeFSM) -> None:
        if _f.work_sub is WorkSubMode.PLAN:
            ctl.plan_prev()
        else:
            ctl.replay_prev()

    def ok_short(f: ModeFSM) -> None:
        if f.work_sub is WorkSubMode.PLAN:
            ctl.approve_plan(f)
        elif f.work_sub is WorkSubMode.EXECUTING:
            ctl.keep_going()

    def no_short(f: ModeFSM) -> None:
        if f.work_sub is WorkSubMode.PLAN:
            ctl.reject_plan()
        elif f.work_sub is WorkSubMode.EXECUTING:
            ctl.stop_and_explain()

    def act_short(f: ModeFSM) -> None:
        if f.work_sub is WorkSubMode.EXECUTING:
            ctl.escalate_to_review(f)

    handlers: dict[tuple[Mode, Key, Gesture], Callable[[ModeFSM], None]] = {
        (Mode.WORK, Key.NAV, Gesture.SHORT): plan_nav_short,
        (Mode.WORK, Key.NAV, Gesture.LONG): plan_nav_long,
        (Mode.WORK, Key.OK, Gesture.SHORT): ok_short,
        (Mode.WORK, Key.NO, Gesture.SHORT): no_short,
        (Mode.WORK, Key.ACT, Gesture.SHORT): act_short,
    }
    KEY_BEHAVIORS.update(handlers)
