"""Unit tests for REVIEW mode (SHOA-116)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from code_trip import mode_fsm as mode_fsm_module
from code_trip.mode_fsm import Gesture, Key, Mode, ModeFSM
from code_trip.remote_tmux import RemoteTmuxError, WaitTimeout
from code_trip.summarizer import SummarizerError
from code_trip.review import ReviewController, register_review_handlers


def _make_ctl():
    return ReviewController(
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


# --- enter / findings fetch ----------------------------------------------


def test_enter_fetches_and_parses_findings(ctl):
    ctl.tmux.capture_pane.return_value = json.dumps(
        ["missing docstring", "unused import"]
    )
    ctl.enter()
    assert ctl.findings == ["missing docstring", "unused import"]
    assert ctl.cursor == 0
    ctl.tmux.send_keys.assert_called_once()
    ctl.thinking.start.assert_called_once()
    ctl.thinking.stop.assert_called()


def test_enter_empty_array_announces(ctl):
    ctl.tmux.capture_pane.return_value = "[]"
    ctl.enter()
    assert ctl.findings == []
    msg = ctl.tts.speak.call_args.args[0].lower()
    assert "no findings" in msg or "ship" in msg


@patch("code_trip.review.play_error")
def test_enter_wait_timeout_reports_error(play_err, ctl):
    ctl.tmux.wait_for_claude.side_effect = WaitTimeout("slow")
    ctl.enter()
    assert ctl.findings == []
    play_err.assert_called_once()


@patch("code_trip.review.play_error")
def test_enter_no_json_reports_error(play_err, ctl):
    ctl.tmux.capture_pane.return_value = "no array here"
    ctl.enter()
    assert ctl.findings == []
    play_err.assert_called_once()


def test_enter_resets_state(ctl):
    ctl.findings = ["old"]
    ctl.cursor = 5
    ctl.tmux.capture_pane.return_value = json.dumps(["fresh"])
    ctl.enter()
    assert ctl.findings == ["fresh"]
    assert ctl.cursor == 0


# --- navigation -----------------------------------------------------------


def test_next_prev_wraps(ctl):
    ctl.findings = ["a", "b", "c"]
    ctl.next()
    assert ctl.cursor == 1
    ctl.next()
    ctl.next()
    assert ctl.cursor == 0
    ctl.prev()
    assert ctl.cursor == 2


def test_nav_empty_speaks(ctl):
    ctl.next()
    assert "no findings" in ctl.tts.speak.call_args.args[0].lower()


# --- turn loop ------------------------------------------------------------


@patch("code_trip.review.play_completion")
def test_handle_voice_runs_turn(play_comp, ctl):
    ctl.handle_voice("fix that")
    ctl.tmux.send_keys.assert_called_once_with("sess", "work", "fix that")
    ctl.summarizer.summarize.assert_called_once_with(
        "raw claude output", user_request="fix that"
    )
    ctl.tts.speak.assert_called_once_with("summary text")
    play_comp.assert_called_once()


def test_handle_voice_empty_is_noop(ctl):
    ctl.handle_voice("   ")
    ctl.tmux.send_keys.assert_not_called()


@patch("code_trip.review.play_completion")
def test_rerun_runs_turn(play_comp, ctl):
    ctl.rerun()
    args = ctl.tmux.send_keys.call_args.args
    assert args[0] == "sess" and args[1] == "work"
    assert "review" in args[2].lower() or "re-run" in args[2].lower()


@patch("code_trip.review.play_error")
def test_run_turn_send_keys_failure(play_err, ctl):
    ctl.tmux.send_keys.side_effect = RemoteTmuxError("ssh down")
    ctl.handle_voice("hi")
    play_err.assert_called_once()


@patch("code_trip.review.play_error")
def test_run_turn_summarizer_failure(play_err, ctl):
    ctl.summarizer.summarize.side_effect = SummarizerError("bad")
    ctl.handle_voice("hi")
    play_err.assert_called_once()


# --- transitions ----------------------------------------------------------


@pytest.fixture
def fsm_ctl(monkeypatch, ctl):
    keys = [
        (Mode.REVIEW, Key.NAV, Gesture.SHORT),
        (Mode.REVIEW, Key.NAV, Gesture.LONG),
        (Mode.REVIEW, Key.ACT, Gesture.SHORT),
        (Mode.REVIEW, Key.OK, Gesture.SHORT),
        (Mode.REVIEW, Key.NO, Gesture.SHORT),
    ]
    originals = {k: mode_fsm_module.KEY_BEHAVIORS[k] for k in keys}

    monkeypatch.setattr(mode_fsm_module, "play_mode_chime", MagicMock())
    fsm = ModeFSM(tts=ctl.tts, current=Mode.REVIEW)
    register_review_handlers(fsm, ctl)
    yield fsm, ctl

    for k, v in originals.items():
        mode_fsm_module.KEY_BEHAVIORS[k] = v


def test_ok_transitions_to_ship_and_calls_hook(fsm_ctl):
    fsm, ctl = fsm_ctl
    calls = []
    ctl.on_ship = lambda f: calls.append(f)
    fsm.handle_key(Key.OK, Gesture.SHORT)
    assert fsm.current is Mode.SHIP
    assert calls == [fsm]


def test_no_transitions_back_to_work(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.NO, Gesture.SHORT)
    assert fsm.current is Mode.WORK


def test_nav_short_steps(fsm_ctl):
    fsm, ctl = fsm_ctl
    ctl.findings = ["a", "b"]
    fsm.handle_key(Key.NAV, Gesture.SHORT)
    assert ctl.cursor == 1


def test_nav_long_steps_back(fsm_ctl):
    fsm, ctl = fsm_ctl
    ctl.findings = ["a", "b", "c"]
    fsm.handle_key(Key.NAV, Gesture.LONG)
    assert ctl.cursor == 2


@patch("code_trip.review.play_completion")
def test_act_short_reruns(_play, fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.ACT, Gesture.SHORT)
    ctl.tmux.send_keys.assert_called_once()
