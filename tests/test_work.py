"""Unit tests for WORK mode (SHOA-115)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from code_trip import mode_fsm as mode_fsm_module
from code_trip.mode_fsm import Gesture, Key, Mode, ModeFSM, WorkSubMode
from code_trip.remote_tmux import RemoteTmuxError, WaitTimeout
from code_trip.summarizer import SummarizerError
from code_trip.work import (
    HISTORY_LIMIT,
    WorkController,
    register_work_handlers,
)


def _make_ctl():
    return WorkController(
        tmux=MagicMock(),
        tts=MagicMock(),
        summarizer=MagicMock(),
        thinking=MagicMock(),
        session="sess",
        window="work",
        wait_timeout=30.0,
    )


@pytest.fixture
def ctl():
    c = _make_ctl()
    c.summarizer.summarize.return_value = "summary text"
    c.tmux.capture_pane.return_value = "raw claude output"
    return c


# --- enter / plan fetch ---------------------------------------------------


def test_enter_fetches_and_parses_plan(ctl):
    ctl.tmux.capture_pane.return_value = json.dumps(
        ["write tests", "implement controller", "wire orchestrator"]
    )
    ctl.enter("SHOA-115", "WORK mode")
    assert ctl.plan_items == [
        "write tests",
        "implement controller",
        "wire orchestrator",
    ]
    assert ctl.plan_cursor == 0
    ctl.tmux.send_keys.assert_called_once()
    ctl.tmux.wait_for_claude.assert_called_once()
    ctl.thinking.start.assert_called_once()
    ctl.thinking.stop.assert_called()


def test_enter_resets_state(ctl):
    ctl.plan_items = ["old"]
    ctl.plan_cursor = 3
    ctl.history.append("old summary")
    ctl.history_cursor = 7
    ctl.tmux.capture_pane.return_value = json.dumps(["new"])

    ctl.enter("X-1", "t")

    assert ctl.plan_items == ["new"]
    assert ctl.plan_cursor == 0
    assert len(ctl.history) == 0
    assert ctl.history_cursor == 0


@patch("code_trip.work.play_error")
def test_enter_wait_timeout_reports_error(play_err, ctl):
    ctl.tmux.wait_for_claude.side_effect = WaitTimeout("slow")
    ctl.enter("X-1", "t")
    assert ctl.plan_items == []
    play_err.assert_called_once()


@patch("code_trip.work.play_error")
def test_enter_no_json_reports_error(play_err, ctl):
    ctl.tmux.capture_pane.return_value = "no array here"
    ctl.enter("X-1", "t")
    assert ctl.plan_items == []
    play_err.assert_called_once()


# --- plan navigation ------------------------------------------------------


def test_plan_next_prev_wraps(ctl):
    ctl.plan_items = ["a", "b", "c"]
    ctl.plan_next()
    assert ctl.plan_cursor == 1
    ctl.plan_next()
    ctl.plan_next()
    assert ctl.plan_cursor == 0
    ctl.plan_prev()
    assert ctl.plan_cursor == 2


def test_plan_nav_empty_speaks(ctl):
    ctl.plan_next()
    assert "no plan" in ctl.tts.speak.call_args.args[0].lower()


# --- turn loop (_run_turn via handle_voice) -------------------------------


@patch("code_trip.work.play_completion")
def test_handle_voice_runs_turn(play_comp, ctl):
    ctl.handle_voice("do the thing")
    ctl.tmux.send_keys.assert_called_once_with("sess", "work", "do the thing")
    ctl.summarizer.summarize.assert_called_once_with(
        "raw claude output", user_request="do the thing"
    )
    ctl.tts.speak.assert_called_once_with("summary text")
    assert list(ctl.history) == ["summary text"]
    assert ctl.history_cursor == 0
    play_comp.assert_called_once()


def test_handle_voice_empty_is_noop(ctl):
    ctl.handle_voice("   ")
    ctl.tmux.send_keys.assert_not_called()


@patch("code_trip.work.play_completion")
def test_history_caps_at_limit(play_comp, ctl):
    for i in range(HISTORY_LIMIT + 3):
        ctl.summarizer.summarize.return_value = f"summary {i}"
        ctl.handle_voice(f"msg {i}")
    assert len(ctl.history) == HISTORY_LIMIT
    assert ctl.history[0] == f"summary {3}"
    assert ctl.history[-1] == f"summary {HISTORY_LIMIT + 2}"


@patch("code_trip.work.play_error")
def test_run_turn_send_keys_failure(play_err, ctl):
    ctl.tmux.send_keys.side_effect = RemoteTmuxError("ssh down")
    ctl.handle_voice("hi")
    ctl.thinking.start.assert_not_called()
    play_err.assert_called_once()


@patch("code_trip.work.play_error")
def test_run_turn_summarizer_failure(play_err, ctl):
    ctl.summarizer.summarize.side_effect = SummarizerError("bad")
    ctl.handle_voice("hi")
    ctl.thinking.stop.assert_called()
    play_err.assert_called_once()


# --- replay ---------------------------------------------------------------


def test_replay_next_prev_cycles(ctl):
    ctl.history.extend(["first", "second", "third"])
    ctl.history_cursor = 0
    ctl.replay_next()
    assert ctl.tts.speak.call_args.args[0] == "second"
    ctl.replay_next()
    assert ctl.tts.speak.call_args.args[0] == "third"
    ctl.replay_prev()
    assert ctl.tts.speak.call_args.args[0] == "second"


def test_replay_empty_speaks(ctl):
    ctl.replay_next()
    assert "no messages" in ctl.tts.speak.call_args.args[0].lower()


# --- handler registration / dispatch -------------------------------------


@pytest.fixture
def fsm_ctl(monkeypatch, ctl):
    keys = [
        (Mode.WORK, Key.NAV, Gesture.SHORT),
        (Mode.WORK, Key.NAV, Gesture.LONG),
        (Mode.WORK, Key.OK, Gesture.SHORT),
        (Mode.WORK, Key.NO, Gesture.SHORT),
        (Mode.WORK, Key.ACT, Gesture.SHORT),
    ]
    originals = {k: mode_fsm_module.KEY_BEHAVIORS[k] for k in keys}

    monkeypatch.setattr(mode_fsm_module, "play_mode_chime", MagicMock())
    fsm = ModeFSM(
        tts=ctl.tts, current=Mode.WORK, work_sub=WorkSubMode.PLAN
    )
    register_work_handlers(fsm, ctl)
    yield fsm, ctl

    for k, v in originals.items():
        mode_fsm_module.KEY_BEHAVIORS[k] = v


def test_nav_short_in_plan_steps_plan(fsm_ctl):
    fsm, ctl = fsm_ctl
    ctl.plan_items = ["a", "b"]
    fsm.handle_key(Key.NAV, Gesture.SHORT)
    assert ctl.plan_cursor == 1


def test_nav_short_in_executing_replays_history(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.work_sub = WorkSubMode.EXECUTING
    ctl.history.extend(["one", "two"])
    ctl.history_cursor = 0
    fsm.handle_key(Key.NAV, Gesture.SHORT)
    assert ctl.tts.speak.call_args.args[0] == "two"


@patch("code_trip.work.play_completion")
def test_ok_in_plan_transitions_to_executing(_play, fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.OK, Gesture.SHORT)
    assert fsm.work_sub is WorkSubMode.EXECUTING
    ctl.tmux.send_keys.assert_called_with("sess", "work", "Proceed with the plan.")


@patch("code_trip.work.play_completion")
def test_ok_in_executing_keeps_going(_play, fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.work_sub = WorkSubMode.EXECUTING
    fsm.handle_key(Key.OK, Gesture.SHORT)
    ctl.tmux.send_keys.assert_called_with("sess", "work", "Keep going.")


@patch("code_trip.work.play_completion")
def test_no_in_executing_stops_and_explains(_play, fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.work_sub = WorkSubMode.EXECUTING
    fsm.handle_key(Key.NO, Gesture.SHORT)
    ctl.tmux.send_keys.assert_called_with(
        "sess", "work", "Stop and explain what you just did."
    )


def test_act_in_executing_escalates_to_review(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.work_sub = WorkSubMode.EXECUTING
    calls = []
    ctl.on_escalate = lambda f: calls.append(f)
    fsm.handle_key(Key.ACT, Gesture.SHORT)
    assert fsm.current is Mode.REVIEW
    assert calls == [fsm]


def test_act_in_executing_without_callback(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.work_sub = WorkSubMode.EXECUTING
    fsm.handle_key(Key.ACT, Gesture.SHORT)
    assert fsm.current is Mode.REVIEW


def test_no_in_plan_prompts_for_feedback(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.NO, Gesture.SHORT)
    assert "feedback" in ctl.tts.speak.call_args.args[0].lower()
