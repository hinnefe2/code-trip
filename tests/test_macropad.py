"""Unit tests for Macropad key dispatch and NAV-chord handling."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pynput import keyboard

from code_trip2 import chords, window
from code_trip2 import macropad as macropad_module
from code_trip2.macropad import Macropad
from code_trip2.window import Chord, KeyStroke


KEYMAP = {
    "ptt": keyboard.Key.f13,
    "act": keyboard.Key.f14,
    "yes": keyboard.Key.f15,
    "no": keyboard.Key.f16,
    "nav": keyboard.Key.f17,
}


@dataclass
class Recorder:
    chords: list[str] = field(default_factory=list)
    taps: list[str] = field(default_factory=list)
    audio: list[Path] = field(default_factory=list)

    def on_chord(self, name: str) -> None:
        self.chords.append(name)

    def on_tap(self, name: str) -> None:
        self.taps.append(name)

    def on_audio(self, path: Path) -> None:
        self.audio.append(path)


def _make(monkeypatch: pytest.MonkeyPatch) -> tuple[Macropad, Recorder, MagicMock, MagicMock]:
    rec = Recorder()
    pad = Macropad(
        keymap=KEYMAP,
        on_audio=rec.on_audio,
        on_chord=rec.on_chord,
        on_tap=rec.on_tap,
    )
    start_stub = MagicMock()
    finish_stub = MagicMock()
    monkeypatch.setattr(pad, "_start_recording", start_stub)
    monkeypatch.setattr(pad, "_finish_recording", finish_stub)
    return pad, rec, start_stub, finish_stub


# --- Macropad dispatch -----------------------------------------------------


def test_ptt_alone_starts_recording(monkeypatch):
    pad, rec, start, finish = _make(monkeypatch)
    pad._on_press(keyboard.Key.f13)
    start.assert_called_once()
    assert rec.chords == []
    pad._on_release(keyboard.Key.f13)
    finish.assert_called_once()


def test_nav_plus_ptt_fires_chord_no_recording(monkeypatch):
    pad, rec, start, _finish = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)  # NAV down
    pad._on_press(keyboard.Key.f13)  # PTT while NAV held
    assert rec.chords == ["nav+ptt"]
    start.assert_not_called()
    assert rec.audio == []
    pad._on_release(keyboard.Key.f13)
    pad._on_release(keyboard.Key.f17)
    assert rec.audio == []


def test_nav_plus_yes_fires_chord(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)  # NAV
    pad._on_press(keyboard.Key.f15)  # YES
    assert rec.chords == ["nav+yes"]
    pad._on_release(keyboard.Key.f15)
    pad._on_release(keyboard.Key.f17)


def test_nav_plus_no_fires_chord(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)
    pad._on_press(keyboard.Key.f16)
    assert rec.chords == ["nav+no"]


def test_nav_plus_act_fires_chord(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)
    pad._on_press(keyboard.Key.f14)
    assert rec.chords == ["nav+act"]


def test_released_nav_no_longer_modifies(monkeypatch):
    pad, rec, start, _finish = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)
    pad._on_release(keyboard.Key.f17)
    pad._on_press(keyboard.Key.f13)
    assert rec.chords == []
    start.assert_called_once()


def test_yes_without_nav_fires_tap(monkeypatch):
    pad, rec, start, _ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f15)
    assert rec.chords == []
    assert rec.taps == ["yes"]
    start.assert_not_called()


def test_no_without_nav_fires_tap(monkeypatch):
    pad, rec, start, _ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f16)
    assert rec.taps == ["no"]
    start.assert_not_called()


def test_act_press_alone_does_not_fire_tap_yet(monkeypatch):
    pad, rec, start, _ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f14)
    assert rec.chords == []
    assert rec.taps == []
    start.assert_not_called()


def test_act_solo_press_release_fires_act_tap(monkeypatch):
    pad, rec, start, _ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f14)
    pad._on_release(keyboard.Key.f14)
    assert rec.chords == []
    assert rec.taps == ["act"]


def test_nav_plus_act_chord_does_not_also_fire_act_tap(monkeypatch):
    """Regression: pressing ACT while NAV is held used to reset
    _act_chorded to False, so releasing ACT fired a stray solo tap and
    flipped app-mode in addition to the nav+act chord."""
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)  # NAV down
    pad._on_press(keyboard.Key.f14)  # ACT under NAV → nav+act chord
    pad._on_release(keyboard.Key.f14)
    pad._on_release(keyboard.Key.f17)
    assert rec.chords == ["nav+act"]
    assert rec.taps == []  # no stray "act" tap


def test_act_plus_no_chord_does_not_fire_act_tap(monkeypatch):
    """Same regression class for the ACT-as-modifier case: act+no should
    not fire a separate act tap on release."""
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f14)  # ACT down
    pad._on_press(keyboard.Key.f16)  # NO under ACT → act+no chord
    pad._on_release(keyboard.Key.f16)
    pad._on_release(keyboard.Key.f14)
    assert rec.chords == ["act+no"]
    assert rec.taps == []


def test_nav_modifier_suppresses_tap(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)  # NAV
    pad._on_press(keyboard.Key.f15)  # YES under NAV
    assert rec.chords == ["nav+yes"]
    assert rec.taps == []


def test_nav_tap_alone_fires_tap_on_release(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)
    assert rec.taps == []  # nothing yet — tap fires on release
    pad._on_release(keyboard.Key.f17)
    assert rec.taps == ["nav"]
    assert rec.chords == []


def test_nav_chord_suppresses_nav_tap(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)  # NAV
    pad._on_press(keyboard.Key.f15)  # YES under NAV → nav+yes
    pad._on_release(keyboard.Key.f15)
    pad._on_release(keyboard.Key.f17)
    assert rec.chords == ["nav+yes"]
    assert rec.taps == []  # NAV was used as a modifier, no nav tap


def test_nav_tap_across_multiple_presses(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    # First NAV press: used as modifier — no tap
    pad._on_press(keyboard.Key.f17)
    pad._on_press(keyboard.Key.f15)
    pad._on_release(keyboard.Key.f15)
    pad._on_release(keyboard.Key.f17)
    # Second NAV press: solo — tap should fire
    pad._on_press(keyboard.Key.f17)
    pad._on_release(keyboard.Key.f17)
    assert rec.taps == ["nav"]


def test_act_plus_no_fires_chord(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f14)  # ACT
    pad._on_press(keyboard.Key.f16)  # NO under ACT
    assert rec.chords == ["act+no"]
    assert rec.taps == []


def test_act_plus_yes_is_noop(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f14)  # ACT
    pad._on_press(keyboard.Key.f15)  # YES under ACT (no mapping)
    assert rec.chords == []
    assert rec.taps == []


def test_nav_beats_act_when_both_held(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)  # NAV
    pad._on_press(keyboard.Key.f14)  # ACT under NAV → fires nav+act
    pad._on_press(keyboard.Key.f16)  # NO under NAV+ACT → should be nav+no, not act+no
    assert rec.chords == ["nav+act", "nav+no"]


# --- forward-key (local STT) mode -----------------------------------------


def _make_forward(monkeypatch: pytest.MonkeyPatch):
    rec = Recorder()
    controller = MagicMock()
    pad = Macropad(
        keymap=KEYMAP,
        on_audio=rec.on_audio,
        on_chord=rec.on_chord,
        on_tap=rec.on_tap,
        ptt_forward_key=keyboard.Key.home,
    )
    monkeypatch.setattr(pad, "_get_controller", lambda: controller)
    start_stub = MagicMock()
    finish_stub = MagicMock()
    monkeypatch.setattr(pad, "_start_recording", start_stub)
    monkeypatch.setattr(pad, "_finish_recording", finish_stub)
    return pad, rec, controller, start_stub, finish_stub


def test_forward_mode_presses_and_releases_key(monkeypatch):
    pad, rec, controller, start, finish = _make_forward(monkeypatch)

    pad._on_press(keyboard.Key.f13)
    controller.press.assert_called_once_with(keyboard.Key.home)
    start.assert_not_called()
    assert pad._forwarding is True

    pad._on_release(keyboard.Key.f13)
    controller.release.assert_called_once_with(keyboard.Key.home)
    finish.assert_not_called()
    assert pad._forwarding is False


def test_forward_mode_no_audio_callback(monkeypatch):
    pad, rec, *_ = _make_forward(monkeypatch)
    pad._on_press(keyboard.Key.f13)
    pad._on_release(keyboard.Key.f13)
    assert rec.audio == []


def test_nav_plus_ptt_in_forward_mode_fires_chord_no_press(monkeypatch):
    pad, rec, controller, *_ = _make_forward(monkeypatch)
    pad._on_press(keyboard.Key.f17)  # NAV
    pad._on_press(keyboard.Key.f13)  # PTT while NAV held
    assert rec.chords == ["nav+ptt"]
    controller.press.assert_not_called()
    pad._on_release(keyboard.Key.f13)
    controller.release.assert_not_called()


def test_stop_releases_forwarded_key(monkeypatch):
    pad, rec, controller, *_ = _make_forward(monkeypatch)
    pad._on_press(keyboard.Key.f13)
    assert pad._forwarding is True
    pad.stop()
    controller.release.assert_called_once_with(keyboard.Key.home)
    assert pad._forwarding is False


def test_key_repeat_ignored(monkeypatch):
    pad, rec, *_ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f17)
    pad._on_press(keyboard.Key.f15)
    pad._on_press(keyboard.Key.f15)  # repeat while still held
    assert rec.chords == ["nav+yes"]


def test_unmapped_key_ignored(monkeypatch):
    pad, rec, start, _ = _make(monkeypatch)
    pad._on_press(keyboard.Key.space)
    assert rec.chords == []
    start.assert_not_called()


# --- darwin_intercept (macropad-key suppression) --------------------------


def test_darwin_intercept_suppresses_macropad_key(monkeypatch):
    pad, _rec, *_ = _make(monkeypatch)
    f17_vk = keyboard.Key.f17.value.vk
    monkeypatch.setattr(macropad_module, "CGEventGetIntegerValueField", lambda e, f: f17_vk)
    monkeypatch.setattr(macropad_module, "kCGKeyboardEventKeycode", 0)
    assert pad._darwin_intercept(0, object()) is None


def test_darwin_intercept_passes_through_other_keys(monkeypatch):
    pad, _rec, *_ = _make(monkeypatch)
    space_vk = keyboard.Key.space.value.vk
    sentinel = object()
    monkeypatch.setattr(macropad_module, "CGEventGetIntegerValueField", lambda e, f: space_vk)
    monkeypatch.setattr(macropad_module, "kCGKeyboardEventKeycode", 0)
    assert pad._darwin_intercept(0, sentinel) is sentinel


def test_suppress_vks_covers_all_macropad_keys():
    pad = Macropad(keymap=KEYMAP, on_audio=lambda p: None, on_chord=lambda n: None)
    expected = {k.value.vk for k in KEYMAP.values()}
    assert pad._suppress_vks == expected


# --- Chord handler ---------------------------------------------------------


def _ctx(
    app_cycle=("kitty", "Google Chrome", "Slack"),
    *,
    playing=False,
    queue=None,
    terminal_apps=("kitty",),
    ssh_host="",
    ssh_options=(),
    tmux_session="s",
    active_window="work",
):
    tts = MagicMock()
    tts.is_playing.return_value = playing
    config = SimpleNamespace(
        app_cycle=app_cycle,
        terminal_apps=terminal_apps,
        tmux_session=tmux_session,
    )
    return SimpleNamespace(
        tts=tts,
        config=config,
        playback_queue=list(queue or []),
        _playback_lock=threading.Lock(),
        ssh=(ssh_host, ssh_options),
        active_window=active_window,
        app_mode="focused",
    )


def test_chord_yes_sends_chrome_next_tab(monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(window, "active_app", lambda: "Google Chrome")
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_chord(ctx, "nav+yes")
    assert sent == [chords._CHROME_NEXT_TAB]


def test_chord_no_sends_kitty_prev_window(monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(window, "active_app", lambda: "kitty")
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_chord(ctx, "nav+no")
    assert sent == [chords._KITTY_PREV]


def test_chord_yes_on_slack_sends_next_unread(monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(window, "active_app", lambda: "Slack")
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_chord(ctx, "nav+yes")
    assert sent == [chords._SLACK_NEXT_UNREAD]


def test_chord_no_on_slack_sends_history_back(monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(window, "active_app", lambda: "Slack")
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_chord(ctx, "nav+no")
    assert sent == [chords._SLACK_BACK_HISTORY]


def test_chord_unknown_app_speaks_error(monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(window, "active_app", lambda: "TextEdit")
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))
    monkeypatch.setattr(chords.earcon, "error", lambda: None)

    chords.handle_chord(ctx, "nav+yes")
    assert sent == []
    ctx.tts.speak.assert_called_once()
    assert "TextEdit" in ctx.tts.speak.call_args.args[0]


def test_chord_act_cycles_apps(monkeypatch):
    ctx = _ctx()
    activated: list[str] = []
    monkeypatch.setattr(window, "active_app", lambda: "kitty")
    monkeypatch.setattr(window, "activate_app", lambda n: activated.append(n))

    chords.handle_chord(ctx, "nav+act")
    assert activated == ["Google Chrome"]


def test_chord_act_wraps_around(monkeypatch):
    ctx = _ctx()
    activated: list[str] = []
    monkeypatch.setattr(window, "active_app", lambda: "Slack")
    monkeypatch.setattr(window, "activate_app", lambda n: activated.append(n))

    chords.handle_chord(ctx, "nav+act")
    assert activated == ["kitty"]


def test_chord_act_unknown_app_goes_to_first(monkeypatch):
    ctx = _ctx()
    activated: list[str] = []
    monkeypatch.setattr(window, "active_app", lambda: "TextEdit")
    monkeypatch.setattr(window, "activate_app", lambda n: activated.append(n))

    chords.handle_chord(ctx, "nav+act")
    assert activated == ["kitty"]


def test_chord_ptt_speaks_active_app(monkeypatch):
    ctx = _ctx()
    monkeypatch.setattr(window, "active_app", lambda: "Google Chrome")

    chords.handle_chord(ctx, "nav+ptt")
    ctx.tts.speak.assert_called_once_with("Google Chrome")


def test_tap_yes_sends_enter(monkeypatch):
    ctx = _ctx()
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_tap(ctx, "yes")
    assert sent == [chords._TAP_YES]
    assert sent[0].chords[0].key == keyboard.Key.enter


def test_tap_no_sends_escape(monkeypatch):
    ctx = _ctx()
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_tap(ctx, "no")
    assert sent == [chords._TAP_NO]
    assert sent[0].chords[0].key == keyboard.Key.esc


# --- NAV solo tap: app-mode flip ----------------------------------------


def test_tap_nav_flips_mode(monkeypatch):
    """NAV solo tap toggles the app-mode (queue ↔ focused) regardless
    of focused app or playback state."""
    from code_trip2 import dispatch
    ctx = _ctx()
    ctx.app_mode = "focused"
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))
    flipped: list = []
    monkeypatch.setattr(dispatch, "flip_mode", lambda c: flipped.append(c))

    chords.handle_tap(ctx, "nav")

    assert flipped == [ctx]
    assert sent == []  # no synthesized keystroke


def test_tap_nav_during_playback_still_flips_mode(monkeypatch):
    """The 'advance playback' meaning that used to live on NAV-during-
    playback is gone — chunks auto-advance now, NAV is just mode flip."""
    from code_trip2 import dispatch
    ctx = _ctx(queue=["chunk a", "chunk b"])
    flipped: list = []
    monkeypatch.setattr(dispatch, "flip_mode", lambda c: flipped.append(c))

    chords.handle_tap(ctx, "nav")

    assert flipped == [ctx]
    ctx.tts.stop.assert_not_called()


# --- ACT solo tap: stop audio / per-app handler -------------------------


def test_tap_act_during_playback_stops_audio(monkeypatch):
    """In any mode, ACT-tap while TTS is speaking interrupts playback."""
    ctx = _ctx(queue=["chunk a"])  # playback_queue non-empty → is_playing
    monkeypatch.setattr(window, "active_app", lambda: "kitty")

    chords.handle_tap(ctx, "act")

    ctx.tts.stop.assert_called_once()


def test_tap_act_in_queue_mode_no_playback_is_noop(monkeypatch):
    """User is away from the screen, no audio playing — ACT does nothing
    rather than firing focused-app behavior."""
    ctx = _ctx()
    ctx.app_mode = "queue"
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))
    opened: list[list[str]] = []
    monkeypatch.setattr(chords.subprocess, "run", lambda cmd, **kw: opened.append(cmd))

    chords.handle_tap(ctx, "act")

    assert sent == []
    assert opened == []
    ctx.tts.stop.assert_not_called()


def test_tap_act_in_focused_mode_chrome_opens_new_tab(monkeypatch):
    ctx = _ctx()
    ctx.app_mode = "focused"
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "active_app", lambda: "Google Chrome")
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_tap(ctx, "act")

    assert sent == [chords._CHROME_NEW_TAB]
    assert sent[0].chords[0].key == "t"
    assert sent[0].chords[0].modifiers == (keyboard.Key.cmd,)


def test_tap_act_in_focused_mode_terminal_opens_last_pane_url(monkeypatch):
    ctx = _ctx()
    ctx.app_mode = "focused"
    monkeypatch.setattr(window, "active_app", lambda: "kitty")
    pane = (
        "Earlier output https://old.example.com/x referenced.\n"
        "Final URL is https://github.com/owner/repo/pull/42 here.\n"
        ">\n"
    )
    monkeypatch.setattr(chords.remote, "capture", lambda *a, **kw: pane)
    opened: list[list[str]] = []
    monkeypatch.setattr(
        chords.subprocess, "run",
        lambda cmd, **kw: opened.append(cmd) or MagicMock(),
    )
    monkeypatch.setattr(chords.earcon, "completion", lambda: None)

    chords.handle_tap(ctx, "act")

    assert opened == [["open", "-a", "Google Chrome", "https://github.com/owner/repo/pull/42"]]


def test_tap_act_in_focused_mode_terminal_strips_trailing_punctuation(monkeypatch):
    ctx = _ctx()
    ctx.app_mode = "focused"
    monkeypatch.setattr(window, "active_app", lambda: "kitty")
    monkeypatch.setattr(
        chords.remote, "capture", lambda *a, **kw: "see https://example.com/path."
    )
    opened: list[list[str]] = []
    monkeypatch.setattr(
        chords.subprocess, "run",
        lambda cmd, **kw: opened.append(cmd) or MagicMock(),
    )
    monkeypatch.setattr(chords.earcon, "completion", lambda: None)

    chords.handle_tap(ctx, "act")

    assert opened[0][-1] == "https://example.com/path"


def test_tap_act_in_focused_mode_terminal_no_url_speaks_error(monkeypatch):
    ctx = _ctx()
    ctx.app_mode = "focused"
    monkeypatch.setattr(window, "active_app", lambda: "kitty")
    monkeypatch.setattr(chords.remote, "capture", lambda *a, **kw: "no urls here")
    monkeypatch.setattr(chords.earcon, "error", lambda: None)

    chords.handle_tap(ctx, "act")

    ctx.tts.speak.assert_called_once()
    assert "url" in ctx.tts.speak.call_args.args[0].lower()


def test_tap_act_in_focused_mode_in_other_app_is_silent(monkeypatch):
    """Frontmost app isn't terminal or Chrome and there's no playback;
    ACT is a silent no-op (no Down arrow fallback any more)."""
    ctx = _ctx()
    ctx.app_mode = "focused"
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "active_app", lambda: "TextEdit")
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_tap(ctx, "act")

    assert sent == []


def test_chord_act_no_in_focused_mode_sends_ctrl_u(monkeypatch):
    ctx = _ctx()
    ctx.app_mode = "focused"
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_chord(ctx, "act+no")
    assert sent == [chords._ACT_NO_CLEAR_LINE]
    stroke = sent[0].chords[0]
    assert stroke.key == "u"
    assert stroke.modifiers == (keyboard.Key.ctrl,)


def test_chord_act_no_in_queue_mode_dismisses_task(monkeypatch):
    """ACT+NO in queue mode dismisses the current task (mark done) —
    the 'permanent' counterpart to NO-tap's 5-min defer."""
    from code_trip2 import dispatch
    ctx = _ctx()
    ctx.app_mode = "queue"
    dismissed: list = []
    monkeypatch.setattr(dispatch, "dismiss_current_task", lambda c: dismissed.append(c))
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_chord(ctx, "act+no")

    assert dismissed == [ctx]
    assert sent == []  # no Ctrl+U leaking into the focused app


def test_tap_unknown_is_noop(monkeypatch):
    ctx = _ctx()
    sent: list[KeyStroke] = []
    monkeypatch.setattr(window, "send_keystroke", lambda s: sent.append(s))

    chords.handle_tap(ctx, "bogus")
    assert sent == []


def test_chord_active_app_error_reported(monkeypatch):
    ctx = _ctx()

    def raise_(*_a):
        raise window.WindowError("boom")

    monkeypatch.setattr(window, "active_app", raise_)
    monkeypatch.setattr(chords.earcon, "error", lambda: None)

    chords.handle_chord(ctx, "nav+ptt")
    ctx.tts.speak.assert_called_once()
    assert "boom" in ctx.tts.speak.call_args.args[0]
