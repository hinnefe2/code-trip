"""Unit tests for the mode state machine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from code_trip import mode_fsm as mode_fsm_module
from code_trip.mode_fsm import (
    TRANSITIONS,
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


def test_work_sub_invalid_jump_rejected(fsm):
    _set_state(fsm, Mode.WORK)
    with pytest.raises(InvalidTransition):
        fsm.transition_work_sub(WorkSubMode.REVIEWING)
    assert fsm.work_sub is WorkSubMode.PLAN


def test_work_sub_requires_work_mode(fsm):
    _set_state(fsm, Mode.IDLE)
    with pytest.raises(InvalidTransition):
        fsm.transition_work_sub(WorkSubMode.EXECUTING)


def test_work_sub_cycle_back_to_plan(fsm, chime):
    _set_state(fsm, Mode.WORK)
    fsm.transition_work_sub(WorkSubMode.EXECUTING)
    fsm.transition_work_sub(WorkSubMode.REVIEWING)
    fsm.transition_work_sub(WorkSubMode.PLAN)
    assert fsm.work_sub is WorkSubMode.PLAN


# --- key dispatch ---------------------------------------------------------


def test_handle_key_dispatches_to_registered_stub(fsm, monkeypatch):
    called = []
    monkeypatch.setitem(
        mode_fsm_module.KEY_BEHAVIORS,
        (Mode.IDLE, Key.ACT),
        lambda f: called.append(f),
    )
    fsm.handle_key(Key.ACT)
    assert called == [fsm]


def test_handle_key_unmapped_is_noop(fsm, monkeypatch):
    monkeypatch.setitem(
        mode_fsm_module.KEY_BEHAVIORS, (Mode.IDLE, Key.NO), None
    )
    # Pop the entry so .get() returns None.
    mode_fsm_module.KEY_BEHAVIORS.pop((Mode.IDLE, Key.NO))
    try:
        fsm.handle_key(Key.NO)  # should not raise
    finally:
        mode_fsm_module.KEY_BEHAVIORS[(Mode.IDLE, Key.NO)] = (
            mode_fsm_module._stub(Mode.IDLE, Key.NO)
        )


def test_every_mode_key_pair_has_stub():
    for mode in Mode:
        for key in Key:
            assert (mode, key) in mode_fsm_module.KEY_BEHAVIORS
