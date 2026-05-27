"""Unit tests for code_trip2.modes — chunking, anchor capture, dispatcher."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from code_trip2 import chords, modes, window
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
    log = MagicMock()
    thinking = MagicMock()
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


# --- _try_global_commands --------------------------------------------------


@pytest.mark.asyncio
async def test_global_repeat_with_no_history_speaks_message():
    ctx = _real_ctx()
    handled = await modes._try_global_commands(ctx, "repeat")
    assert handled is True
    ctx.tts.speak.assert_called_once()
    assert "nothing" in ctx.tts.speak.call_args.args[0].lower()


@pytest.mark.asyncio
async def test_global_repeat_with_history_replays(monkeypatch):
    ctx = _real_ctx()
    ctx.last_response_chunks = ["chunk one.", "chunk two."]
    started: list[str] = []
    monkeypatch.setattr(modes, "_start_playback_task", lambda c: started.append("ok"))
    handled = await modes._try_global_commands(ctx, "Repeat that.")
    assert handled is True
    assert ctx.playback_queue == ["chunk one.", "chunk two."]
    assert started == ["ok"]


@pytest.mark.asyncio
async def test_global_stop_clears_playback():
    ctx = _real_ctx()
    ctx.playback_queue = ["a", "b"]
    handled = await modes._try_global_commands(ctx, "stop")
    assert handled is True
    assert ctx.playback_queue == []
    ctx.tts.stop.assert_called_once()


@pytest.mark.asyncio
async def test_global_status_speaks_app_and_window(monkeypatch):
    ctx = _real_ctx()
    monkeypatch.setattr(window, "active_app", AsyncMock(return_value="kitty"))
    handled = await modes._try_global_commands(ctx, "status")
    assert handled is True
    msg = ctx.tts.speak.call_args.args[0]
    assert "kitty" in msg.lower()
    assert "work" in msg.lower()


# --- _try_voice_phrase -----------------------------------------------------


@pytest.mark.asyncio
async def test_voice_phrase_next_without_tickets_falls_through():
    ctx = _real_ctx()
    handled = await modes._try_voice_phrase(ctx, "next")
    assert handled is False


@pytest.mark.asyncio
async def test_voice_phrase_next_with_tickets_steps(monkeypatch):
    ctx = _real_ctx()
    ctx.tickets = [{"id": "T-1", "title": "one"}, {"id": "T-2", "title": "two"}]
    handled = await modes._try_voice_phrase(ctx, "next")
    assert handled is True
    assert ctx.ticket_index == 1


@pytest.mark.asyncio
async def test_voice_phrase_select_n_with_tickets(monkeypatch):
    ctx = _real_ctx()
    ctx.tickets = [
        {"id": "T-1", "title": "one", "branch": "t-1"},
        {"id": "T-2", "title": "two", "branch": "t-2"},
    ]
    monkeypatch.setattr(modes.remote, "new_window", AsyncMock())
    monkeypatch.setattr(modes.remote, "send", AsyncMock())
    handled = await modes._try_voice_phrase(ctx, "select 2")
    assert handled is True
    assert ctx.active_window == "t-2"
    assert ctx.ticket_index == 1


@pytest.mark.asyncio
async def test_voice_phrase_list_windows(monkeypatch):
    ctx = _real_ctx()
    monkeypatch.setattr(
        modes.remote,
        "list_windows",
        AsyncMock(return_value=[(0, "main", "/"), (1, "linear", "/x")]),
    )
    handled = await modes._try_voice_phrase(ctx, "list windows")
    assert handled is True
    msg = ctx.tts.speak.call_args.args[0]
    assert "main" in msg and "linear" in msg


@pytest.mark.asyncio
async def test_voice_phrase_switch_to(monkeypatch):
    ctx = _real_ctx()
    sel = AsyncMock()
    monkeypatch.setattr(modes.remote, "select_window", sel)
    handled = await modes._try_voice_phrase(ctx, "switch to ticket-1")
    assert handled is True
    sel.assert_called_once()
    assert ctx.active_window == "ticket-1"


# --- handle_voice dispatcher ----------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_terminal_app_routes_to_work(monkeypatch):
    ctx = _real_ctx(terminal_apps=("kitty", "iTerm2"))
    work_calls: list[str] = []
    monkeypatch.setattr(window, "active_app", AsyncMock(return_value="kitty"))

    async def fake_work(c, t):
        work_calls.append(t)

    monkeypatch.setattr(modes, "_work_voice", fake_work)
    await modes.handle_focused_voice(ctx, "do the thing")
    assert work_calls == ["do the thing"]


@pytest.mark.asyncio
async def test_dispatch_non_terminal_app_routes_to_dictate(monkeypatch):
    ctx = _real_ctx()
    dictate_calls: list[str] = []
    monkeypatch.setattr(window, "active_app", AsyncMock(return_value="Google Chrome"))

    async def fake_dictate(c, t):
        dictate_calls.append(t)

    monkeypatch.setattr(modes, "_dictate_voice", fake_dictate)
    await modes.handle_focused_voice(ctx, "type this in the browser")
    assert dictate_calls == ["type this in the browser"]


@pytest.mark.asyncio
async def test_dispatch_falls_back_to_work_on_active_app_error(monkeypatch):
    ctx = _real_ctx()
    work_calls: list[str] = []

    async def boom():
        raise window.WindowError("osascript broken")

    monkeypatch.setattr(window, "active_app", boom)

    async def fake_work(c, t):
        work_calls.append(t)

    monkeypatch.setattr(modes, "_work_voice", fake_work)
    await modes.handle_focused_voice(ctx, "hi")
    assert work_calls == ["hi"]


@pytest.mark.asyncio
async def test_dispatch_voice_phrase_wins_over_focus(monkeypatch):
    ctx = _real_ctx()
    ctx.tickets = [{"id": "T-1", "title": "one"}]
    monkeypatch.setattr(window, "active_app", AsyncMock(return_value="Google Chrome"))
    work_calls: list[str] = []

    async def fake_work(c, t):
        work_calls.append(t)

    monkeypatch.setattr(modes, "_work_voice", fake_work)
    await modes.handle_focused_voice(ctx, "next")
    # ticket step ran, NOT _work_voice
    assert work_calls == []
    assert ctx.ticket_index == 0  # only one ticket; modulo wraps to 0


@pytest.mark.asyncio
async def test_empty_transcript_is_noop(monkeypatch):
    ctx = _real_ctx()
    monkeypatch.setattr(window, "active_app", AsyncMock(return_value="kitty"))
    await modes.handle_focused_voice(ctx, "   ")
    ctx.tts.speak.assert_not_called()


# --- chord tap routing during playback ------------------------------------


@pytest.mark.asyncio
async def test_nav_tap_flips_mode_even_during_playback(monkeypatch):
    """NAV solo tap flips app-mode regardless of playback state. The
    earlier 'advance chunk' behavior on NAV is gone — chunks auto-
    advance via the playback worker."""
    from code_trip2 import dispatch
    ctx = _real_ctx()
    ctx.playback_queue = ["a", "b"]
    ctx.app_mode = "focused"
    sent: list = []
    monkeypatch.setattr(window, "send_keystroke", AsyncMock(side_effect=lambda s: sent.append(s)))
    flipped: list = []
    monkeypatch.setattr(dispatch, "flip_mode", lambda c: flipped.append(c))

    await chords.handle_tap(ctx, "nav")

    assert flipped == [ctx]
    assert sent == []
    ctx.tts.stop.assert_not_called()
    assert ctx.playback_queue == ["a", "b"]


@pytest.mark.asyncio
async def test_act_tap_stops_playback_when_active(monkeypatch):
    """ACT solo tap interrupts TTS playback (was: NO tap)."""
    ctx = _real_ctx()
    ctx.playback_queue = ["a", "b"]
    sent: list = []
    monkeypatch.setattr(window, "send_keystroke", AsyncMock(side_effect=lambda s: sent.append(s)))

    await chords.handle_tap(ctx, "act")

    assert sent == []
    ctx.tts.stop.assert_called_once()
    assert ctx.playback_queue == []  # stop_playback clears the queue


@pytest.mark.asyncio
async def test_no_tap_skips_task_during_playback_in_queue_mode(monkeypatch):
    """NO no longer stops playback as a side effect — in queue mode it
    just skips the current task (which on its own ends the announce-
    ment via the worker draining)."""
    from code_trip2 import dispatch
    ctx = _real_ctx()
    ctx.app_mode = "queue"
    ctx.playback_queue = ["a"]
    skipped: list = []

    async def fake_no_tap(c):
        skipped.append(c)

    monkeypatch.setattr(dispatch, "queue_no_tap", fake_no_tap)

    await chords.handle_tap(ctx, "no")

    assert skipped == [ctx]


@pytest.mark.asyncio
async def test_yes_tap_unchanged_during_playback(monkeypatch):
    ctx = _real_ctx()
    ctx.app_mode = "focused"  # YES in focused mode = Enter keystroke
    ctx.playback_queue = ["a"]
    sent: list = []
    monkeypatch.setattr(window, "send_keystroke", AsyncMock(side_effect=lambda s: sent.append(s)))

    await chords.handle_tap(ctx, "yes")

    assert sent == [chords._TAP_YES]


@pytest.mark.asyncio
async def test_no_tap_when_idle_in_focused_mode_sends_escape(monkeypatch):
    ctx = _real_ctx()
    ctx.app_mode = "focused"  # queue empty, tts not playing
    sent: list = []
    monkeypatch.setattr(window, "send_keystroke", AsyncMock(side_effect=lambda s: sent.append(s)))
    monkeypatch.setattr(window, "active_app", AsyncMock(return_value="TextEdit"))

    await chords.handle_tap(ctx, "no")

    assert sent == [chords._TAP_NO]


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
