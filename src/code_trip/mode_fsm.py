"""Mode state machine for the voice-driven orchestrator.

Tracks which interaction mode the user is in (IDLE, BROWSE, WORK, REVIEW,
SHIP) and, within WORK, a sub-state (PLAN, EXECUTING, REVIEWING). Each valid
transition plays a distinct earcon chime and announces the new mode via TTS.

Per-mode key behavior (PTT/NAV/ACT/OK/NO) is captured as stubs in this
module; real handlers land in SHOA-113/115/116. Design intent per key and
mode follows docs/voice-driven-claude-design.md §"Modal System":

    IDLE    PTT: voice command  NAV: cycle worktrees  ACT: enter BROWSE
    BROWSE  PTT: filter/search  NAV: next/prev ticket ACT: select -> WORK
            OK:  read full desc NO:  exit to IDLE
    WORK    PTT: talk to Claude NAV: cycle messages   ACT: run review
            OK:  approve action NO:  reject/retry
    REVIEW  PTT: feedback       NAV: step findings    ACT: rerun tools
            OK:  approve -> SHIP NO: back to WORK
    SHIP    PTT: edit PR text   OK:  push draft PR    NO: back to REVIEW
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Callable

from code_trip.earcon import play_mode_chime
from code_trip.tts_client import TTSClient

logger = logging.getLogger(__name__)


class Mode(enum.Enum):
    IDLE = "idle"
    BROWSE = "browse"
    WORK = "work"
    REVIEW = "review"
    SHIP = "ship"


class WorkSubMode(enum.Enum):
    PLAN = "plan"
    EXECUTING = "executing"
    REVIEWING = "reviewing"


class Key(enum.Enum):
    PTT = "ptt"
    NAV = "nav"
    ACT = "act"
    OK = "ok"
    NO = "no"


class InvalidTransition(Exception):
    """Raised when a caller attempts a transition not in the allowed table."""


TRANSITIONS: dict[Mode, frozenset[Mode]] = {
    Mode.IDLE: frozenset({Mode.BROWSE}),
    Mode.BROWSE: frozenset({Mode.WORK, Mode.IDLE}),
    Mode.WORK: frozenset({Mode.REVIEW}),
    Mode.REVIEW: frozenset({Mode.WORK, Mode.SHIP}),
    Mode.SHIP: frozenset({Mode.REVIEW, Mode.IDLE}),
}

WORK_SUB_TRANSITIONS: dict[WorkSubMode, frozenset[WorkSubMode]] = {
    WorkSubMode.PLAN: frozenset({WorkSubMode.EXECUTING}),
    WorkSubMode.EXECUTING: frozenset({WorkSubMode.REVIEWING}),
    WorkSubMode.REVIEWING: frozenset({WorkSubMode.PLAN, WorkSubMode.EXECUTING}),
}

MODE_ANNOUNCEMENTS: dict[Mode, str] = {
    Mode.IDLE: "Idle mode.",
    Mode.BROWSE: "Browse mode.",
    Mode.WORK: "Work mode.",
    Mode.REVIEW: "Review mode.",
    Mode.SHIP: "Ship mode.",
}


# --- key-behavior stubs ---------------------------------------------------


def _stub(mode: Mode, key: Key) -> Callable[["ModeFSM"], None]:
    def handler(_fsm: "ModeFSM") -> None:
        logger.info("%s.%s pressed (stub)", mode.name, key.name)

    handler.__name__ = f"_stub_{mode.name}_{key.name}"
    return handler


KEY_BEHAVIORS: dict[tuple[Mode, Key], Callable[["ModeFSM"], None]] = {
    (mode, key): _stub(mode, key)
    for mode in Mode
    for key in Key
}


# --- FSM ------------------------------------------------------------------


@dataclass
class ModeFSM:
    """Tracks current mode + WORK sub-state and drives earcon/TTS on entry."""

    tts: TTSClient
    current: Mode = Mode.IDLE
    work_sub: WorkSubMode | None = field(default=None)

    def transition_to(self, target: Mode) -> None:
        allowed = TRANSITIONS[self.current]
        if target not in allowed:
            raise InvalidTransition(
                f"Cannot transition from {self.current.name} to {target.name}"
            )

        self.current = target
        self.work_sub = WorkSubMode.PLAN if target is Mode.WORK else None
        self._announce(target)

    def transition_work_sub(self, target: WorkSubMode) -> None:
        if self.current is not Mode.WORK or self.work_sub is None:
            raise InvalidTransition(
                "WORK sub-state transitions require current mode WORK"
            )
        allowed = WORK_SUB_TRANSITIONS[self.work_sub]
        if target not in allowed:
            raise InvalidTransition(
                f"Cannot transition WORK sub-state from "
                f"{self.work_sub.name} to {target.name}"
            )
        self.work_sub = target
        # Sub-state chime reuses the WORK chime so the mode stays audibly consistent.
        play_mode_chime(Mode.WORK.name)

    def handle_key(self, key: Key) -> None:
        handler = KEY_BEHAVIORS.get((self.current, key))
        if handler is None:
            return
        handler(self)

    # --- internals --------------------------------------------------------

    def _announce(self, mode: Mode) -> None:
        play_mode_chime(mode.name)
        self.tts.speak(MODE_ANNOUNCEMENTS[mode])
