"""Unit tests for code_trip2.modes — chunking, anchor capture, dispatcher."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from code_trip2 import chords, modes, window
from code_trip2.earcon import Thinking
from code_trip2.session_log import SessionLogger
from conftest import make_mock_tts


# --- chunk_text ------------------------------------------------------------


def test_chunk_text_empty():
    assert modes.chunk_text("") == []
    assert modes.chunk_text("   \n\n   ") == []


def test_chunk_text_single_sentence():
    assert modes.chunk_text("Tests passed.") == ["Tests passed."]


def test_chunk_text_groups_short_sentences():
    out = modes.chunk_text("First. Second. Third.")
    assert out == ["First. Second. Third."]


def test_chunk_text_splits_on_max_sentences():
    out = modes.chunk_text("A. B. C. D.", max_sentences=2)
    assert out == ["A. B.", "C. D."]


def test_chunk_text_splits_on_max_chars():
    long = "x" * 80 + ". " + "y" * 80 + ". " + "z" * 80 + "."
    out = modes.chunk_text(long, max_chars=100)
    assert len(out) == 3


def test_chunk_text_paragraph_break_is_hard_boundary():
    text = "Para one is short.\n\nPara two is also short."
    out = modes.chunk_text(text)
    assert out == ["Para one is short.", "Para two is also short."]


def test_chunk_text_joins_terminal_wrapped_lines():
    text = "This is a long sentence that\nwraps at the terminal edge."
    out = modes.chunk_text(text)
    assert out == ["This is a long sentence that wraps at the terminal edge."]


# --- clean_output (anchor capture) ----------------------------------------


def test_clean_output_empty():
    assert modes.clean_output("") == ""


def test_clean_output_anchor_takes_text_after_last_match():
    raw = (
        "earlier output\n"
        "implement the auth fix\n"  # echo of user msg
        "Sure, I'll start by reading the auth module.\n"
        "Done. The fix is in place.\n"
        ">\n"
    )
    out = modes.clean_output(raw, anchor="implement the auth fix")
    assert "Sure, I'll start" in out
    assert "Done. The fix is in place." in out
    assert "earlier output" not in out


def test_clean_output_anchor_uses_last_occurrence():
    # Claude often echoes the user message after a status block, so prefer the
    # *last* occurrence — that's where the response begins.
    raw = (
        "implement the auth fix\n"  # input area redraw
        "(some status block)\n"
        "implement the auth fix\n"  # echo in conversation flow
        "Here is the response.\n"
    )
    out = modes.clean_output(raw, anchor="implement the auth fix")
    assert out == "Here is the response."


def test_clean_output_anchor_not_found_falls_back_to_tail():
    raw = "line 1\nline 2\nline 3\n"
    out = modes.clean_output(raw, anchor="never appears here")
    assert "line 1" in out and "line 3" in out


def test_clean_output_strips_ansi_and_box():
    raw = (
        "\x1b[1mecho\x1b[0m\n"
        "anchor text\n"
        "│ box │\n"
        "actual response\n"
    )
    out = modes.clean_output(raw, anchor="anchor text")
    assert "actual response" in out
    assert "\x1b" not in out
    assert "│" not in out


def test_clean_output_drops_trailing_prompt_block():
    raw = "anchor\nresponse line\n>\n>\n"
    out = modes.clean_output(raw, anchor="anchor")
    assert out == "response line"


# --- Context init ----------------------------------------------------------


def _real_ctx(*, terminal_apps=("kitty",)) -> modes.Context:
    config = SimpleNamespace(
        ssh_host="",
        ssh_options=(),
        tmux_session="s",
        work_window="work",
        linear_window="linear",
        terminal_apps=terminal_apps,
        wait_timeout=1.0,
    )
    tts = make_mock_tts()
    log = create_autospec(SessionLogger, instance=True)
    thinking = create_autospec(Thinking, instance=True)
    return modes.Context(config=config, tts=tts, log=log, thinking=thinking)


def test_context_active_window_default_from_config():
    ctx = _real_ctx()
    assert ctx.active_window == "work"


# --- playback state --------------------------------------------------------


def test_is_playback_active_false_initially():
    ctx = _real_ctx()
    assert modes.is_playback_active(ctx) is False


def test_is_playback_active_true_when_queue_nonempty():
    ctx = _real_ctx()
    ctx.playback_queue = ["chunk one"]
    assert modes.is_playback_active(ctx) is True


def test_is_playback_active_true_when_tts_playing():
    ctx = _real_ctx()
    ctx.tts.is_playing.return_value = True
    assert modes.is_playback_active(ctx) is True


def test_stop_playback_clears_queue_and_stops_tts():
    ctx = _real_ctx()
    ctx.playback_queue = ["a", "b", "c"]
    modes.stop_playback(ctx)
    assert ctx.playback_queue == []
    ctx.tts.stop.assert_called_once()


def test_advance_playback_stops_tts_keeps_queue():
    ctx = _real_ctx()
    ctx.playback_queue = ["a", "b"]
    modes.advance_playback(ctx)
    ctx.tts.stop.assert_called_once()
    assert ctx.playback_queue == ["a", "b"]


# --- chord tap routing -----------------------------------------------------


@pytest.mark.asyncio
async def test_nav_tap_is_noop(monkeypatch):
    """NAV solo tap does nothing — it's only meaningful held as a chord
    modifier. Playback and the queue are left untouched."""
    from code_trip2 import dispatch
    ctx = _real_ctx()
    ctx.playback_queue = ["a", "b"]
    sent: list = []
    monkeypatch.setattr(window, "send_keystroke", AsyncMock(side_effect=lambda s: sent.append(s)))
    yes = AsyncMock()
    no = AsyncMock()
    monkeypatch.setattr(dispatch, "queue_yes_tap", yes)
    monkeypatch.setattr(dispatch, "queue_no_tap", no)

    await chords.handle_tap(ctx, "nav")

    assert sent == []
    ctx.tts.stop.assert_not_called()
    yes.assert_not_called()
    no.assert_not_called()
    assert ctx.playback_queue == ["a", "b"]


@pytest.mark.asyncio
async def test_act_tap_stops_playback_when_active(monkeypatch):
    """ACT solo tap interrupts TTS playback."""
    ctx = _real_ctx()
    ctx.playback_queue = ["a", "b"]
    sent: list = []
    monkeypatch.setattr(window, "send_keystroke", AsyncMock(side_effect=lambda s: sent.append(s)))

    await chords.handle_tap(ctx, "act")

    assert sent == []
    ctx.tts.stop.assert_called_once()
    assert ctx.playback_queue == []  # stop_playback clears the queue


@pytest.mark.asyncio
async def test_act_tap_idle_is_noop(monkeypatch):
    """ACT solo with no playback does nothing (no focused-app action)."""
    ctx = _real_ctx()
    sent: list = []
    monkeypatch.setattr(window, "send_keystroke", AsyncMock(side_effect=lambda s: sent.append(s)))

    await chords.handle_tap(ctx, "act")

    assert sent == []
    ctx.tts.stop.assert_not_called()


@pytest.mark.asyncio
async def test_no_tap_skips_task(monkeypatch):
    """NO tap always skips the current task."""
    from code_trip2 import dispatch
    ctx = _real_ctx()
    ctx.playback_queue = ["a"]
    skipped: list = []

    async def fake_no_tap(c):
        skipped.append(c)

    monkeypatch.setattr(dispatch, "queue_no_tap", fake_no_tap)

    await chords.handle_tap(ctx, "no")

    assert skipped == [ctx]


@pytest.mark.asyncio
async def test_yes_tap_submits_queue_input(monkeypatch):
    """YES tap always submits the TUI input via queue_yes_tap."""
    from code_trip2 import dispatch
    ctx = _real_ctx()
    submitted: list = []

    async def fake_yes_tap(c):
        submitted.append(c)

    monkeypatch.setattr(dispatch, "queue_yes_tap", fake_yes_tap)

    await chords.handle_tap(ctx, "yes")

    assert submitted == [ctx]


# --- chunked playback worker (integration with mocked tts) ----------------


@pytest.mark.asyncio
async def test_speak_chunked_drains_queue_via_worker(monkeypatch):
    ctx = _real_ctx()
    spoken: list[str] = []

    async def fake_speak(text: str) -> None:
        spoken.append(text)

    ctx.tts.speak = AsyncMock(side_effect=fake_speak)
    # Avoid actually calling sounddevice via earcon.completion in the worker.
    monkeypatch.setattr(modes.earcon, "completion", lambda: None)

    modes.speak_chunked(ctx, "Sentence one. Sentence two.\n\nNew para here.")
    # Wait for the playback task to finish.
    if ctx._playback_task is not None:
        await asyncio.wait_for(ctx._playback_task, timeout=2.0)

    assert spoken == ["Sentence one. Sentence two.", "New para here."]
    assert ctx.last_response_chunks == ["Sentence one. Sentence two.", "New para here."]
    assert ctx.playback_queue == []


@pytest.mark.asyncio
async def test_stop_during_playback_drops_remaining(monkeypatch):
    ctx = _real_ctx()
    started = asyncio.Event()
    proceed = asyncio.Event()
    spoken: list[str] = []

    async def slow_speak(text: str) -> None:
        spoken.append(text)
        started.set()
        try:
            await asyncio.wait_for(proceed.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

    ctx.tts.speak = AsyncMock(side_effect=slow_speak)
    monkeypatch.setattr(modes.earcon, "completion", lambda: None)

    modes.speak_chunked(ctx, "Chunk one. " + ("x. " * 80))
    await asyncio.wait_for(started.wait(), timeout=2.0)
    modes.stop_playback(ctx)
    proceed.set()
    if ctx._playback_task is not None:
        await asyncio.wait_for(ctx._playback_task, timeout=2.0)

    # Only the first chunk got spoken because we stopped before the rest.
    assert len(spoken) == 1
    assert ctx.playback_queue == []
