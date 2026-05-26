"""Tests for the TUI input buffer."""

from __future__ import annotations

import time

from code_trip2.input_buffer import InputBuffer


def test_append_accumulates():
    b = InputBuffer()
    for ch in "hello":
        b.append(ch)
    assert b.get() == "hello"
    assert not b.is_empty()


def test_backspace_removes_last_char():
    b = InputBuffer()
    for ch in "abc":
        b.append(ch)
    b.backspace()
    assert b.get() == "ab"


def test_backspace_on_empty_is_noop():
    b = InputBuffer()
    b.backspace()
    assert b.get() == ""
    assert b.is_empty()


def test_clear_empties_buffer():
    b = InputBuffer()
    for ch in "abc":
        b.append(ch)
    b.clear()
    assert b.get() == ""


def test_pop_returns_and_clears():
    b = InputBuffer()
    for ch in "done":
        b.append(ch)
    assert b.pop() == "done"
    assert b.get() == ""


def test_quiet_seconds_grows_when_idle():
    b = InputBuffer()
    b.append("x")
    initial = b.quiet_seconds()
    time.sleep(0.02)
    assert b.quiet_seconds() > initial


def test_quiet_seconds_resets_on_append():
    b = InputBuffer()
    b.append("x")
    time.sleep(0.02)
    before = b.quiet_seconds()
    b.append("y")
    after = b.quiet_seconds()
    assert after < before
