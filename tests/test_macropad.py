"""Unit tests for Macropad key dispatch and NAV-chord handling."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pynput import keyboard

from code_trip2 import chords, window
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
    audio: list[Path] = field(default_factory=list)

    def on_chord(self, name: str) -> None:
        self.chords.append(name)

    def on_audio(self, path: Path) -> None:
        self.audio.append(path)


def _make(monkeypatch: pytest.MonkeyPatch) -> tuple[Macropad, Recorder, MagicMock]:
    rec = Recorder()
    pad = Macropad(keymap=KEYMAP, on_audio=rec.on_audio, on_chord=rec.on_chord)
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


def test_yes_without_nav_is_noop(monkeypatch):
    pad, rec, start, _ = _make(monkeypatch)
    pad._on_press(keyboard.Key.f15)
    assert rec.chords == []
    start.assert_not_called()


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


# --- Chord handler ---------------------------------------------------------


def _ctx(app_cycle=("kitty", "Google Chrome", "Slack")):
    tts = MagicMock()
    config = SimpleNamespace(app_cycle=app_cycle)
    return SimpleNamespace(tts=tts, config=config)


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


def test_chord_active_app_error_reported(monkeypatch):
    ctx = _ctx()

    def raise_(*_a):
        raise window.WindowError("boom")

    monkeypatch.setattr(window, "active_app", raise_)
    monkeypatch.setattr(chords.earcon, "error", lambda: None)

    chords.handle_chord(ctx, "nav+ptt")
    ctx.tts.speak.assert_called_once()
    assert "boom" in ctx.tts.speak.call_args.args[0]
