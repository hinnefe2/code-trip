"""BROWSE mode: navigate Linear tickets via TTS.

Ticket data is fetched by driving a dedicated tmux pane with
``claude -p "list my tickets"`` (Linear MCP is configured inside that
Claude Code instance). The response is asked to be JSON so the local side
stays dumb — no free-text parsing beyond extracting the array.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from code_trip.mode_fsm import KEY_BEHAVIORS, Gesture, Key, Mode, ModeFSM
from code_trip.remote_tmux import RemoteTmux, RemoteTmuxError, WaitTimeout
from code_trip.tts_client import TTSClient

logger = logging.getLogger(__name__)


REFRESH_PROMPT = (
    "List my assigned Linear tickets using the Linear MCP. "
    "Respond with ONLY a JSON array (no prose, no code fences). "
    'Each object: {"id","title","priority","assignee","description","branch"}. '
    'Use empty string for missing fields.'
)

_JSON_ARRAY_RE = re.compile(r"\[\s*\{.*\}\s*\]", re.DOTALL)


class BrowseError(Exception):
    """Raised when BROWSE cannot fetch or parse tickets."""


@dataclass(frozen=True)
class Ticket:
    id: str
    title: str
    priority: str = ""
    assignee: str = ""
    description: str = ""
    branch: str = ""


@dataclass
class BrowseController:
    tmux: RemoteTmux
    tts: TTSClient
    session: str
    browse_window: str
    wait_timeout: float = 120.0
    tickets: list[Ticket] = field(default_factory=list)
    index: int = 0
    selected: Ticket | None = None
    _filtered: list[Ticket] | None = None

    # --- view helpers -----------------------------------------------------

    def _view(self) -> list[Ticket]:
        return self._filtered if self._filtered is not None else self.tickets

    def current(self) -> Ticket | None:
        view = self._view()
        if not view:
            return None
        return view[self.index % len(view)]

    # --- lifecycle --------------------------------------------------------

    def refresh(self) -> None:
        """Fetch tickets via the BROWSE tmux pane and populate the cache."""
        try:
            self.tmux.send_keys(
                self.session, self.browse_window, f'claude -p {json.dumps(REFRESH_PROMPT)}'
            )
            self.tmux.wait_for_claude(
                self.session, self.browse_window, timeout=self.wait_timeout
            )
            raw = self.tmux.capture_pane(
                self.session, self.browse_window, lines=400
            )
        except (RemoteTmuxError, WaitTimeout) as exc:
            raise BrowseError(f"Failed to fetch tickets: {exc}") from exc

        match = _JSON_ARRAY_RE.search(raw)
        if not match:
            raise BrowseError("No JSON array found in Claude response")
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise BrowseError(f"Invalid JSON in Claude response: {exc}") from exc
        if not isinstance(payload, list):
            raise BrowseError("Expected JSON array of tickets")

        tickets: list[Ticket] = []
        for item in payload:
            if not isinstance(item, dict) or "id" not in item or "title" not in item:
                continue
            tickets.append(
                Ticket(
                    id=str(item.get("id", "")),
                    title=str(item.get("title", "")),
                    priority=str(item.get("priority", "")),
                    assignee=str(item.get("assignee", "")),
                    description=str(item.get("description", "")),
                    branch=str(item.get("branch", "")),
                )
            )
        self.tickets = tickets
        self.index = 0
        self._filtered = None
        logger.info("Loaded %d tickets", len(tickets))

    # --- navigation -------------------------------------------------------

    def next(self) -> None:
        view = self._view()
        if not view:
            return
        self.index = (self.index + 1) % len(view)

    def prev(self) -> None:
        view = self._view()
        if not view:
            return
        self.index = (self.index - 1) % len(view)

    def announce_current(self) -> None:
        ticket = self.current()
        if ticket is None:
            self.tts.speak("No tickets.")
            return
        priority = ticket.priority or "no priority"
        assignee = ticket.assignee or "unassigned"
        self.tts.speak(
            f"{ticket.id}. {ticket.title}. Priority {priority}. Assigned to {assignee}."
        )

    def read_description(self) -> None:
        ticket = self.current()
        if ticket is None:
            self.tts.speak("No tickets.")
            return
        if not ticket.description:
            self.tts.speak(f"{ticket.id} has no description.")
            return
        self.tts.speak(ticket.description)

    # --- filter -----------------------------------------------------------

    def filter(self, query: str) -> None:
        q = query.strip().lower()
        if not q:
            self.clear_filter()
            return
        matches = [
            t
            for t in self.tickets
            if q in t.title.lower()
            or q in t.description.lower()
            or q in t.priority.lower()
            or q in t.assignee.lower()
            or q in t.id.lower()
        ]
        self._filtered = matches
        self.index = 0
        if matches:
            self.tts.speak(f"{len(matches)} tickets match.")
        else:
            self.tts.speak("No tickets match.")

    def clear_filter(self) -> None:
        self._filtered = None
        self.index = 0

    def handle_voice(self, transcript: str) -> None:
        """Entry point for PTT-in-BROWSE. Routes transcript to filter()."""
        self.filter(transcript)

    # --- selection --------------------------------------------------------

    def select_index(self, n: int) -> Ticket | None:
        """Select the *n*-th ticket (1-based) from the current view."""
        view = self._view()
        if not view:
            self.tts.speak("No tickets.")
            return None
        if n < 1 or n > len(view):
            self.tts.speak(f"No ticket {n}.")
            return None
        self.index = n - 1
        return self.select_current()

    def select_current(self) -> Ticket | None:
        ticket = self.current()
        if ticket is None:
            self.tts.speak("No ticket selected.")
            return None
        self.selected = ticket
        self.tts.speak(f"Selected {ticket.id}.")
        return ticket


# --- KEY_BEHAVIORS registration ------------------------------------------


def register_browse_handlers(fsm: ModeFSM, ctl: BrowseController) -> None:
    """Install BROWSE-mode handlers into the global KEY_BEHAVIORS table.

    Handlers close over *ctl* so they can manipulate the cache/cursor. The
    *fsm* reference is unused here but kept in the signature to mirror how
    WORK/REVIEW controllers will likely register.
    """

    del fsm  # reserved for symmetry with future controllers

    def nav_short(_f: ModeFSM) -> None:
        ctl.next()
        ctl.announce_current()

    def nav_long(_f: ModeFSM) -> None:
        ctl.prev()
        ctl.announce_current()

    def ok_short(_f: ModeFSM) -> None:
        ctl.read_description()

    def act_short(f: ModeFSM) -> None:
        ticket = ctl.select_current()
        if ticket is None:
            return
        # TODO(SHOA-114): create worktree + tmux window from ticket.branch
        # before transitioning, and surface errors if that fails.
        f.transition_to(Mode.WORK)

    def no_short(f: ModeFSM) -> None:
        ctl.clear_filter()
        f.transition_to(Mode.IDLE)

    handlers: dict[tuple[Mode, Key, Gesture], Callable[[ModeFSM], None]] = {
        (Mode.BROWSE, Key.NAV, Gesture.SHORT): nav_short,
        (Mode.BROWSE, Key.NAV, Gesture.LONG): nav_long,
        (Mode.BROWSE, Key.OK, Gesture.SHORT): ok_short,
        (Mode.BROWSE, Key.ACT, Gesture.SHORT): act_short,
        (Mode.BROWSE, Key.NO, Gesture.SHORT): no_short,
    }
    KEY_BEHAVIORS.update(handlers)
