"""Unit tests for BROWSE mode (SHOA-113)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from code_trip import browse as browse_module
from code_trip import mode_fsm as mode_fsm_module
from code_trip.browse import (
    BrowseController,
    BrowseError,
    Ticket,
    register_browse_handlers,
)
from code_trip.mode_fsm import Gesture, Key, Mode, ModeFSM


SAMPLE_TICKETS = [
    {
        "id": "SHOA-113",
        "title": "Linear ticket browsing",
        "priority": "Medium",
        "assignee": "Henry",
        "description": "Implement BROWSE mode.",
        "branch": "henry/shoa-113",
    },
    {
        "id": "SHOA-114",
        "title": "Worktree creation bug",
        "priority": "High",
        "assignee": "Henry",
        "description": "Fix worktree race.",
        "branch": "henry/shoa-114",
    },
]


def _make_ctl(*, tmux=None, tts=None, tickets=None):
    tmux = tmux or MagicMock()
    tts = tts or MagicMock()
    ctl = BrowseController(
        tmux=tmux,
        tts=tts,
        session="sess",
        browse_window="browse",
    )
    if tickets is not None:
        ctl.tickets = list(tickets)
    return ctl


@pytest.fixture
def ctl():
    return _make_ctl()


@pytest.fixture
def populated():
    return _make_ctl(
        tickets=[
            Ticket(
                id=t["id"],
                title=t["title"],
                priority=t["priority"],
                assignee=t["assignee"],
                description=t["description"],
                branch=t["branch"],
            )
            for t in SAMPLE_TICKETS
        ]
    )


# --- refresh --------------------------------------------------------------


def test_refresh_parses_json(ctl):
    ctl.tmux.capture_pane.return_value = json.dumps(SAMPLE_TICKETS)
    ctl.refresh()
    assert len(ctl.tickets) == 2
    assert ctl.tickets[0].id == "SHOA-113"
    assert ctl.tickets[1].priority == "High"
    ctl.tmux.send_keys.assert_called_once()
    ctl.tmux.wait_for_claude.assert_called_once()


def test_refresh_tolerates_surrounding_text(ctl):
    noise = "user@host $ claude -p '...'\n" + json.dumps(SAMPLE_TICKETS) + "\nuser@host $ "
    ctl.tmux.capture_pane.return_value = noise
    ctl.refresh()
    assert len(ctl.tickets) == 2


def test_refresh_raises_on_no_json(ctl):
    ctl.tmux.capture_pane.return_value = "some prose with no array"
    with pytest.raises(BrowseError):
        ctl.refresh()


def test_refresh_raises_on_invalid_json(ctl):
    ctl.tmux.capture_pane.return_value = "[{broken json"
    with pytest.raises(BrowseError):
        ctl.refresh()


def test_refresh_resets_filter_and_index(populated):
    populated.index = 1
    populated._filtered = populated.tickets[:1]
    populated.tmux.capture_pane.return_value = json.dumps(SAMPLE_TICKETS)
    populated.refresh()
    assert populated.index == 0
    assert populated._filtered is None


# --- navigation -----------------------------------------------------------


def test_next_prev_wraps(populated):
    assert populated.current().id == "SHOA-113"
    populated.next()
    assert populated.current().id == "SHOA-114"
    populated.next()
    assert populated.current().id == "SHOA-113"
    populated.prev()
    assert populated.current().id == "SHOA-114"


def test_nav_on_empty_is_noop(ctl):
    ctl.next()
    ctl.prev()
    assert ctl.current() is None


def test_announce_current_speaks_metadata(populated):
    populated.announce_current()
    spoken = populated.tts.speak.call_args.args[0]
    assert "SHOA-113" in spoken
    assert "Medium" in spoken
    assert "Henry" in spoken
    assert "Linear ticket browsing" in spoken


def test_announce_empty(ctl):
    ctl.announce_current()
    ctl.tts.speak.assert_called_once_with("No tickets.")


def test_read_description(populated):
    populated.read_description()
    populated.tts.speak.assert_called_once_with("Implement BROWSE mode.")


def test_read_description_empty(ctl):
    ctl.tickets = [Ticket(id="X-1", title="t", description="")]
    ctl.read_description()
    assert "no description" in ctl.tts.speak.call_args.args[0].lower()


# --- filter ---------------------------------------------------------------


def test_filter_substring_case_insensitive(populated):
    populated.filter("BUG")
    assert populated._filtered is not None
    assert len(populated._filtered) == 1
    assert populated._filtered[0].id == "SHOA-114"


def test_filter_matches_priority(populated):
    populated.filter("high")
    assert len(populated._filtered) == 1
    assert populated._filtered[0].id == "SHOA-114"


def test_filter_no_matches_speaks(populated):
    populated.filter("nonsense")
    assert populated._filtered == []
    populated.tts.speak.assert_called_with("No tickets match.")


def test_clear_filter_restores(populated):
    populated.filter("bug")
    populated.clear_filter()
    assert populated._filtered is None
    assert populated.current().id == "SHOA-113"


def test_handle_voice_delegates_to_filter(populated):
    populated.handle_voice("bug")
    assert len(populated._filtered) == 1


# --- selection ------------------------------------------------------------


def test_select_current_stores_and_returns(populated):
    ticket = populated.select_current()
    assert ticket.id == "SHOA-113"
    assert populated.selected is ticket


def test_select_current_empty(ctl):
    assert ctl.select_current() is None
    assert ctl.selected is None


# --- handler registration -------------------------------------------------


@pytest.fixture
def fsm_ctl(monkeypatch, populated):
    # Snapshot the five BROWSE keys we mutate so tests don't leak.
    keys = [
        (Mode.BROWSE, Key.NAV, Gesture.SHORT),
        (Mode.BROWSE, Key.NAV, Gesture.LONG),
        (Mode.BROWSE, Key.OK, Gesture.SHORT),
        (Mode.BROWSE, Key.ACT, Gesture.SHORT),
        (Mode.BROWSE, Key.NO, Gesture.SHORT),
    ]
    originals = {k: mode_fsm_module.KEY_BEHAVIORS[k] for k in keys}

    monkeypatch.setattr(mode_fsm_module, "play_mode_chime", MagicMock())
    fsm = ModeFSM(tts=populated.tts, current=Mode.BROWSE)
    register_browse_handlers(fsm, populated)
    yield fsm, populated

    for k, v in originals.items():
        mode_fsm_module.KEY_BEHAVIORS[k] = v


def test_nav_short_advances_and_announces(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.NAV, Gesture.SHORT)
    assert ctl.current().id == "SHOA-114"
    assert ctl.tts.speak.called


def test_nav_long_goes_back(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.NAV, Gesture.LONG)
    assert ctl.current().id == "SHOA-114"  # wrap


def test_ok_short_reads_description(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.OK, Gesture.SHORT)
    assert ctl.tts.speak.call_args.args[0] == "Implement BROWSE mode."


def test_act_short_selects_and_transitions_to_work(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.ACT, Gesture.SHORT)
    assert ctl.selected.id == "SHOA-113"
    assert fsm.current is Mode.WORK


def test_no_short_transitions_to_idle_and_clears_filter(fsm_ctl):
    fsm, ctl = fsm_ctl
    ctl.filter("bug")
    fsm.handle_key(Key.NO, Gesture.SHORT)
    assert fsm.current is Mode.IDLE
    assert ctl._filtered is None


def test_register_does_not_touch_other_modes():
    # Sanity: IDLE handlers should remain stubs after registration.
    tts = MagicMock()
    ctl = _make_ctl(tts=tts)
    fsm = ModeFSM(tts=tts)
    register_browse_handlers(fsm, ctl)
    handler = mode_fsm_module.KEY_BEHAVIORS[(Mode.IDLE, Key.NAV, Gesture.SHORT)]
    assert handler.__name__.startswith("_stub_")
