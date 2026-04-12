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
    )
    d.stt.transcribe.return_value = "list files"
    d.tmux.capture_pane.return_value = "ls output"
    d.summarizer.summarize.return_value = "Here are the files."
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
