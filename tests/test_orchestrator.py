"""Unit tests for Orchestrator with all components mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from code_trip.config import (
    AudioConfig,
    ClaudeConfig,
    Config,
    OpenAIConfig,
    SSHConfig,
    TmuxConfig,
)
from code_trip.orchestrator import Orchestrator, OrchestratorDeps
from code_trip.remote_tmux import RemoteTmuxError, WaitTimeout
from code_trip.stt_client import STTClientError
from code_trip.summarizer import SummarizerError
from code_trip.tts_client import TTSClientError


@pytest.fixture
def config():
    return Config(
        ssh=SSHConfig(host="h"),
        tmux=TmuxConfig(session="s", window="w"),
        audio=AudioConfig(),
        openai=OpenAIConfig(api_key="sk-test"),
        claude=ClaudeConfig(wait_timeout=30.0),
    )


@pytest.fixture
def deps():
    d = OrchestratorDeps(
        tmux=MagicMock(),
        recorder=MagicMock(),
        ptt=MagicMock(),
        stt=MagicMock(),
        summarizer=MagicMock(),
        tts=MagicMock(),
        thinking=MagicMock(),
        intent_classifier=MagicMock(),
    )
    d.stt.transcribe.return_value = "list files"
    d.tmux.capture_pane.return_value = "ls output"
    d.summarizer.summarize.return_value = "Here are the files."
    d.intent_classifier.classify.return_value = None
    return d


@pytest.fixture
def orch(config, deps):
    return Orchestrator(config, deps=deps)


def _audio():
    return Path("/tmp/code-trip-audio/rec.wav")


@patch("code_trip.orchestrator.play_completion")
@patch("code_trip.orchestrator.play_error")
def test_happy_path(play_err, play_comp, orch, deps):
    orch._handle_recording(_audio())

    deps.stt.transcribe.assert_called_once_with(_audio())
    deps.tmux.send_keys.assert_called_once_with("s", "w", "list files")
    deps.thinking.start.assert_called_once()
    deps.tmux.wait_for_claude.assert_called_once_with("s", "w", timeout=30.0)
    deps.thinking.stop.assert_called()
    deps.tmux.capture_pane.assert_called_once_with("s", "w")
    deps.summarizer.summarize.assert_called_once_with(
        "ls output", user_request="list files"
    )
    deps.tts.speak.assert_called_once_with("Here are the files.")
    play_comp.assert_called_once()
    play_err.assert_not_called()


@patch("code_trip.orchestrator.play_completion")
@patch("code_trip.orchestrator.play_error")
def test_thinking_starts_before_wait_and_stops_after(play_err, play_comp, orch, deps):
    order = []
    deps.thinking.start.side_effect = lambda: order.append("think_start")
    deps.thinking.stop.side_effect = lambda: order.append("think_stop")
    deps.tmux.wait_for_claude.side_effect = lambda *a, **kw: order.append("wait")
    deps.tmux.capture_pane.side_effect = lambda *a, **kw: (
        order.append("capture"),
        "ls output",
    )[1]

    orch._handle_recording(_audio())

    assert order.index("think_start") < order.index("wait")
    assert order.index("capture") < order.index("think_stop")


@patch("code_trip.orchestrator.play_completion")
@patch("code_trip.orchestrator.play_error")
def test_stt_failure_reports_error(play_err, play_comp, orch, deps):
    deps.stt.transcribe.side_effect = STTClientError("nope")

    orch._handle_recording(_audio())

    deps.tmux.send_keys.assert_not_called()
    play_err.assert_called_once()
    deps.tts.speak.assert_called_once()
    assert "Transcription failed" in deps.tts.speak.call_args.args[0]
    play_comp.assert_not_called()


@patch("code_trip.orchestrator.play_completion")
@patch("code_trip.orchestrator.play_error")
def test_send_keys_failure_reports_error(play_err, play_comp, orch, deps):
    deps.tmux.send_keys.side_effect = RemoteTmuxError("ssh down")

    orch._handle_recording(_audio())

    deps.thinking.start.assert_not_called()
    play_err.assert_called_once()
    assert "Could not reach Claude" in deps.tts.speak.call_args.args[0]


@patch("code_trip.orchestrator.play_completion")
@patch("code_trip.orchestrator.play_error")
def test_wait_timeout_stops_thinking_and_reports(play_err, play_comp, orch, deps):
    deps.tmux.wait_for_claude.side_effect = WaitTimeout("too slow")

    orch._handle_recording(_audio())

    deps.thinking.start.assert_called_once()
    deps.thinking.stop.assert_called()
    deps.tmux.capture_pane.assert_not_called()
    play_err.assert_called_once()
    assert "did not respond" in deps.tts.speak.call_args.args[0]
    play_comp.assert_not_called()


@patch("code_trip.orchestrator.play_completion")
@patch("code_trip.orchestrator.play_error")
def test_summarizer_failure_reports_error(play_err, play_comp, orch, deps):
    deps.summarizer.summarize.side_effect = SummarizerError("bad")

    orch._handle_recording(_audio())

    deps.thinking.stop.assert_called()
    play_err.assert_called_once()
    assert "Summarization failed" in deps.tts.speak.call_args.args[0]


@patch("code_trip.orchestrator.play_completion")
@patch("code_trip.orchestrator.play_error")
def test_tts_failure_reports_error(play_err, play_comp, orch, deps):
    # First speak call (main summary) fails; error reporter then tries again.
    deps.tts.speak.side_effect = [TTSClientError("api down"), None]

    orch._handle_recording(_audio())

    assert deps.tts.speak.call_count == 2
    play_err.assert_called_once()
    play_comp.assert_not_called()


def test_stop_sets_shutdown(orch):
    orch.stop()
    assert orch._shutdown.is_set()


# --- mode-aware PTT dispatch ---------------------------------------------


from code_trip.browse import Ticket
from code_trip.mode_fsm import Gesture, Key, Mode, WorkSubMode


def test_ptt_in_browse_routes_to_browse_handle_voice(orch, deps):
    orch._fsm.current = Mode.BROWSE
    orch._browse.handle_voice = MagicMock()
    deps.stt.transcribe.return_value = "bug"

    orch._handle_recording(_audio())

    orch._browse.handle_voice.assert_called_once_with("bug")
    deps.tmux.send_keys.assert_not_called()


def test_ptt_in_work_routes_to_work_handle_voice(orch, deps):
    orch._fsm.current = Mode.WORK
    orch._fsm.work_sub = WorkSubMode.EXECUTING
    orch._work.handle_voice = MagicMock()
    deps.stt.transcribe.return_value = "refactor this"

    orch._handle_recording(_audio())

    orch._work.handle_voice.assert_called_once_with("refactor this")


def test_ptt_in_review_routes_to_review_handle_voice(orch, deps):
    orch._fsm.current = Mode.REVIEW
    orch._review.handle_voice = MagicMock()
    deps.stt.transcribe.return_value = "ignore it"

    orch._handle_recording(_audio())

    orch._review.handle_voice.assert_called_once_with("ignore it")
    deps.tmux.send_keys.assert_not_called()


def test_ptt_in_ship_routes_to_ship_handle_voice(orch, deps):
    orch._fsm.current = Mode.SHIP
    orch._ship.handle_voice = MagicMock()
    deps.stt.transcribe.return_value = "make title shorter"

    orch._handle_recording(_audio())

    orch._ship.handle_voice.assert_called_once_with("make title shorter")
    deps.tmux.send_keys.assert_not_called()


def test_work_escalate_enters_review(orch, monkeypatch):
    from code_trip import mode_fsm as fsm_mod

    monkeypatch.setattr(fsm_mod, "play_mode_chime", MagicMock())
    orch._fsm.current = Mode.WORK
    orch._fsm.work_sub = WorkSubMode.EXECUTING
    orch._review.enter = MagicMock()

    orch._fsm.handle_key(Key.ACT, Gesture.SHORT)

    assert orch._fsm.current is Mode.REVIEW
    orch._review.enter.assert_called_once()


def test_review_approve_enters_ship(orch, monkeypatch):
    from code_trip import mode_fsm as fsm_mod

    monkeypatch.setattr(fsm_mod, "play_mode_chime", MagicMock())
    orch._fsm.current = Mode.REVIEW
    orch._ship.enter = MagicMock()

    orch._fsm.handle_key(Key.OK, Gesture.SHORT)

    assert orch._fsm.current is Mode.SHIP
    orch._ship.enter.assert_called_once()


# --- intent routing -------------------------------------------------------


from code_trip.intent_classifier import (
    Intent,
    IntentClassifierError,
    IntentResult,
)
from code_trip.summarizer import Verbosity


def test_intent_list_tickets_from_idle_refreshes_browse(orch, deps, monkeypatch):
    from code_trip import mode_fsm as fsm_mod

    monkeypatch.setattr(fsm_mod, "play_mode_chime", MagicMock())
    deps.intent_classifier.classify.return_value = IntentResult(Intent.LIST_TICKETS)
    orch._browse.refresh = MagicMock()
    orch._browse.announce_current = MagicMock()

    orch._handle_recording(_audio())

    assert orch._fsm.current is Mode.BROWSE
    orch._browse.refresh.assert_called_once()
    orch._browse.announce_current.assert_called_once()
    deps.tmux.send_keys.assert_not_called()


def test_intent_list_tickets_from_work_is_noop_with_explanation(orch, deps):
    orch._fsm.current = Mode.WORK
    deps.intent_classifier.classify.return_value = IntentResult(Intent.LIST_TICKETS)
    orch._browse.refresh = MagicMock()
    orch._work.handle_voice = MagicMock()

    orch._handle_recording(_audio())

    assert orch._fsm.current is Mode.WORK
    orch._browse.refresh.assert_not_called()
    orch._work.handle_voice.assert_not_called()
    spoken = deps.tts.speak.call_args.args[0]
    assert "can't" in spoken.lower()
    assert "work" in spoken.lower()


def test_intent_ship_it_from_review_enters_ship(orch, deps, monkeypatch):
    from code_trip import mode_fsm as fsm_mod

    monkeypatch.setattr(fsm_mod, "play_mode_chime", MagicMock())
    orch._fsm.current = Mode.REVIEW
    deps.intent_classifier.classify.return_value = IntentResult(Intent.SHIP_IT)
    orch._ship.enter = MagicMock()

    orch._handle_recording(_audio())

    assert orch._fsm.current is Mode.SHIP
    orch._ship.enter.assert_called_once()


def test_intent_ship_it_from_idle_is_noop(orch, deps):
    deps.intent_classifier.classify.return_value = IntentResult(Intent.SHIP_IT)
    orch._ship.enter = MagicMock()

    orch._handle_recording(_audio())

    assert orch._fsm.current is Mode.IDLE
    orch._ship.enter.assert_not_called()
    deps.tmux.send_keys.assert_not_called()


def test_intent_status_from_idle(orch, deps):
    deps.intent_classifier.classify.return_value = IntentResult(Intent.STATUS)

    orch._handle_recording(_audio())

    spoken = deps.tts.speak.call_args.args[0]
    assert "idle" in spoken.lower()
    assert "no active ticket" in spoken.lower()


def test_intent_status_with_active_ticket_in_work(orch, deps):
    orch._fsm.current = Mode.WORK
    orch._fsm.work_sub = WorkSubMode.PLAN
    orch._browse.selected = Ticket(id="SHOA-117", title="Intent routing")
    deps.intent_classifier.classify.return_value = IntentResult(Intent.STATUS)

    orch._handle_recording(_audio())

    spoken = deps.tts.speak.call_args.args[0]
    assert "SHOA-117" in spoken
    assert "Intent routing" in spoken
    assert "plan" in spoken.lower()


def test_intent_set_verbosity_updates_summarizer(orch, deps):
    deps.intent_classifier.classify.return_value = IntentResult(
        Intent.SET_VERBOSITY, arg="brief"
    )

    orch._handle_recording(_audio())

    assert deps.summarizer.verbosity == Verbosity.BRIEF
    deps.tts.speak.assert_called_once()
    assert "brief" in deps.tts.speak.call_args.args[0].lower()


def test_intent_switch_ticket_in_browse_enters_work(orch, deps, monkeypatch):
    from code_trip import mode_fsm as fsm_mod

    monkeypatch.setattr(fsm_mod, "play_mode_chime", MagicMock())
    orch._fsm.current = Mode.BROWSE
    orch._browse.tickets = [
        Ticket(id="SHOA-1", title="One"),
        Ticket(id="SHOA-2", title="Two"),
    ]
    deps.intent_classifier.classify.return_value = IntentResult(
        Intent.SWITCH_TICKET, arg=2
    )
    orch._work.enter = MagicMock()

    orch._handle_recording(_audio())

    assert orch._fsm.current is Mode.WORK
    orch._work.enter.assert_called_once_with("SHOA-2", "Two")


def test_intent_classifier_error_falls_through_to_mode(orch, deps):
    orch._fsm.current = Mode.BROWSE
    deps.intent_classifier.classify.side_effect = IntentClassifierError("boom")
    orch._browse.handle_voice = MagicMock()
    deps.stt.transcribe.return_value = "find python"

    orch._handle_recording(_audio())

    orch._browse.handle_voice.assert_called_once_with("find python")


def test_unknown_intent_falls_through_to_mode(orch, deps):
    orch._fsm.current = Mode.WORK
    deps.intent_classifier.classify.return_value = None
    orch._work.handle_voice = MagicMock()
    deps.stt.transcribe.return_value = "do the thing"

    orch._handle_recording(_audio())

    orch._work.handle_voice.assert_called_once_with("do the thing")


def test_browse_to_work_handoff_invokes_work_enter(orch, monkeypatch):
    from code_trip import mode_fsm as fsm_mod

    monkeypatch.setattr(fsm_mod, "play_mode_chime", MagicMock())
    orch._fsm.current = Mode.BROWSE
    orch._browse.tickets = [Ticket(id="SHOA-115", title="WORK mode")]
    orch._work.enter = MagicMock()

    orch._fsm.handle_key(Key.ACT, Gesture.SHORT)

    assert orch._fsm.current is Mode.WORK
    orch._work.enter.assert_called_once_with("SHOA-115", "WORK mode")
