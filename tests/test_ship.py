"""Unit tests for SHIP mode (SHOA-116)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from code_trip import mode_fsm as mode_fsm_module
from code_trip.mode_fsm import Gesture, Key, Mode, ModeFSM
from code_trip.remote_tmux import RemoteTmuxError, WaitTimeout
from code_trip.ship import ShipController, register_ship_handlers


def _make_ctl():
    return ShipController(
        tmux=MagicMock(),
        tts=MagicMock(),
        thinking=MagicMock(),
        session="sess",
        window="work",
        wait_timeout=30.0,
    )


@pytest.fixture
def ctl():
    c = _make_ctl()
    c.tmux.capture_pane.return_value = (
        "Some chatter\nhttps://github.com/acme/repo/pull/42\n"
    )
    return c


def test_enter_announces(ctl):
    ctl.enter()
    assert ctl.pr_url is None
    msg = ctl.tts.speak.call_args.args[0].lower()
    assert "ship" in msg or "pr" in msg or "okay" in msg


@patch("code_trip.ship.play_completion")
def test_push_pr_happy_path(play_comp, ctl):
    fsm = MagicMock()
    ctl.push_pr(fsm)

    ctl.tmux.send_keys.assert_called_once()
    sent = ctl.tmux.send_keys.call_args.args[2]
    assert "gh pr create" in sent
    assert "draft" in sent

    assert ctl.pr_url == "https://github.com/acme/repo/pull/42"
    ctl.tts.speak.assert_called_once()
    play_comp.assert_called_once()
    fsm.transition_to.assert_called_once_with(Mode.IDLE)


@patch("code_trip.ship.play_error")
def test_push_pr_no_url_in_response(play_err, ctl):
    ctl.tmux.capture_pane.return_value = "no url in this output"
    fsm = MagicMock()
    ctl.push_pr(fsm)

    assert ctl.pr_url is None
    play_err.assert_called_once()
    fsm.transition_to.assert_not_called()


@patch("code_trip.ship.play_error")
def test_push_pr_send_keys_failure(play_err, ctl):
    ctl.tmux.send_keys.side_effect = RemoteTmuxError("ssh down")
    fsm = MagicMock()
    ctl.push_pr(fsm)
    play_err.assert_called_once()
    fsm.transition_to.assert_not_called()


@patch("code_trip.ship.play_error")
def test_push_pr_wait_timeout(play_err, ctl):
    ctl.tmux.wait_for_claude.side_effect = WaitTimeout("slow")
    fsm = MagicMock()
    ctl.push_pr(fsm)
    play_err.assert_called_once()
    fsm.transition_to.assert_not_called()


def test_back_to_review_transitions(ctl):
    fsm = MagicMock()
    ctl.back_to_review(fsm)
    fsm.transition_to.assert_called_once_with(Mode.REVIEW)


@patch("code_trip.ship.play_completion")
def test_handle_voice_forwards_text(play_comp, ctl):
    ctl.handle_voice("make the title snappier")
    ctl.tmux.send_keys.assert_called_once_with(
        "sess", "work", "make the title snappier"
    )
    play_comp.assert_called_once()


def test_handle_voice_empty_is_noop(ctl):
    ctl.handle_voice("   ")
    ctl.tmux.send_keys.assert_not_called()


# --- handler registration -------------------------------------------------


@pytest.fixture
def fsm_ctl(monkeypatch, ctl):
    keys = [
        (Mode.SHIP, Key.OK, Gesture.SHORT),
        (Mode.SHIP, Key.NO, Gesture.SHORT),
    ]
    originals = {k: mode_fsm_module.KEY_BEHAVIORS[k] for k in keys}

    monkeypatch.setattr(mode_fsm_module, "play_mode_chime", MagicMock())
    fsm = ModeFSM(tts=ctl.tts, current=Mode.SHIP)
    register_ship_handlers(fsm, ctl)
    yield fsm, ctl

    for k, v in originals.items():
        mode_fsm_module.KEY_BEHAVIORS[k] = v


@patch("code_trip.ship.play_completion")
def test_ok_pushes_pr(_play, fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.OK, Gesture.SHORT)
    assert fsm.current is Mode.IDLE
    assert ctl.pr_url == "https://github.com/acme/repo/pull/42"


def test_no_goes_back_to_review(fsm_ctl):
    fsm, ctl = fsm_ctl
    fsm.handle_key(Key.NO, Gesture.SHORT)
    assert fsm.current is Mode.REVIEW
