"""Unit tests for the mode state machine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from code_trip import mode_fsm as mode_fsm_module
from code_trip.mode_fsm import (
    TRANSITIONS,
    Gesture,
    InvalidTransition,
    Key,
    Mode,
    ModeFSM,
    WorkSubMode,
)


@pytest.fixture
def tts():
    return MagicMock()


@pytest.fixture
def chime(monkeypatch):
    m = MagicMock()
    monkeypatch.setattr(mode_fsm_module, "play_mode_chime", m)
    return m


@pytest.fixture
def fsm(tts):
    return ModeFSM(tts=tts)


def _set_state(fsm: ModeFSM, mode: Mode) -> None:
    fsm.current = mode
    fsm.work_sub = WorkSubMode.PLAN if mode is Mode.WORK else None


# --- valid transitions ----------------------------------------------------


@pytest.mark.parametrize(
    "source,target",
    [(src, tgt) for src, tgts in TRANSITIONS.items() for tgt in tgts],
)
def test_valid_transition(source, target, fsm, tts, chime):
    _set_state(fsm, source)
    tts.reset_mock()
    chime.reset_mock()

    fsm.transition_to(target)

    assert fsm.current is target
    chime.assert_called_once_with(target.name)
    tts.speak.assert_called_once()
    assert target.name.lower() in tts.speak.call_args.args[0].lower()


@pytest.mark.parametrize(
    "source,target",
    [
        (src, tgt)
        for src in Mode
        for tgt in Mode
        if tgt not in TRANSITIONS[src]
    ],
)
def test_invalid_transition(source, target, fsm, tts, chime):
    _set_state(fsm, source)
    sub_before = fsm.work_sub

    with pytest.raises(InvalidTransition):
        fsm.transition_to(target)

    assert fsm.current is source
    assert fsm.work_sub is sub_before
    chime.assert_not_called()
    tts.speak.assert_not_called()


# --- WORK sub-state -------------------------------------------------------


def test_entering_work_sets_plan(fsm, chime):
    _set_state(fsm, Mode.BROWSE)
    fsm.transition_to(Mode.WORK)
    assert fsm.work_sub is WorkSubMode.PLAN


def test_exiting_work_clears_sub(fsm, chime):
    _set_state(fsm, Mode.WORK)
    fsm.transition_to(Mode.REVIEW)
    assert fsm.work_sub is None


def test_work_sub_transition_plays_chime(fsm, chime):
    _set_state(fsm, Mode.WORK)
    fsm.transition_work_sub(WorkSubMode.EXECUTING)
    assert fsm.work_sub is WorkSubMode.EXECUTING
    chime.assert_called_once_with(Mode.WORK.name)


def test_work_sub_invalid_jump_rejected(fsm, chime):
    _set_state(fsm, Mode.WORK)
    fsm.transition_work_sub(WorkSubMode.EXECUTING)
    with pytest.raises(InvalidTransition):
        # EXECUTING -> EXECUTING is not in the table.
        fsm.transition_work_sub(WorkSubMode.EXECUTING)
    assert fsm.work_sub is WorkSubMode.EXECUTING


def test_work_sub_requires_work_mode(fsm):
    _set_state(fsm, Mode.IDLE)
    with pytest.raises(InvalidTransition):
        fsm.transition_work_sub(WorkSubMode.EXECUTING)


def test_work_sub_cycle_back_to_plan(fsm, chime):
    _set_state(fsm, Mode.WORK)
    fsm.transition_work_sub(WorkSubMode.EXECUTING)
    fsm.transition_work_sub(WorkSubMode.PLAN)
    assert fsm.work_sub is WorkSubMode.PLAN


# --- key dispatch ---------------------------------------------------------


def test_handle_key_dispatches_to_registered_stub(fsm, monkeypatch):
    called = []
    monkeypatch.setitem(
        mode_fsm_module.KEY_BEHAVIORS,
        (Mode.IDLE, Key.ACT, Gesture.SHORT),
        lambda f: called.append(f),
    )
    fsm.handle_key(Key.ACT, Gesture.SHORT)
    assert called == [fsm]


def test_handle_key_routes_by_gesture(fsm, monkeypatch):
    short_calls: list = []
    long_calls: list = []
    monkeypatch.setitem(
        mode_fsm_module.KEY_BEHAVIORS,
        (Mode.IDLE, Key.NAV, Gesture.SHORT),
        lambda f: short_calls.append(f),
    )
    monkeypatch.setitem(
        mode_fsm_module.KEY_BEHAVIORS,
        (Mode.IDLE, Key.NAV, Gesture.LONG),
        lambda f: long_calls.append(f),
    )
    fsm.handle_key(Key.NAV, Gesture.LONG)
    assert long_calls == [fsm]
    assert short_calls == []


def test_handle_key_unmapped_is_noop(fsm):
    key = (Mode.IDLE, Key.NO, Gesture.HOLD)
    original = mode_fsm_module.KEY_BEHAVIORS.pop(key)
    try:
        fsm.handle_key(Key.NO, Gesture.HOLD)  # should not raise
    finally:
        mode_fsm_module.KEY_BEHAVIORS[key] = original


def test_every_mode_key_gesture_triple_has_stub():
    for mode in Mode:
        for key in Key:
            for gesture in Gesture:
                assert (mode, key, gesture) in mode_fsm_module.KEY_BEHAVIORS
