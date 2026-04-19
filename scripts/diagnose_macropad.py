#!/usr/bin/env python3
"""Macropad diagnostic: confirm F13-F17 keys are firing and classify gestures.

Listens for all five macropad keys and prints each press/release event with
timing and gesture classification matching the thresholds used in KeypadListener.

Usage:
    uv run python scripts/diagnose_macropad.py

Press Ctrl+C to exit and see a per-key event summary.
"""

from __future__ import annotations

import signal
import sys
import termios
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from pynput import keyboard


def _suppress_echo() -> list | None:
    """Disable terminal echo so raw F-key escape codes don't appear in output."""
    try:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        return old
    except termios.error:
        return None  # not a TTY (e.g. piped input), skip


def _restore_echo(old: list) -> None:
    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)
    except termios.error:
        pass

# Gesture thresholds (must match KeypadConfig in code_trip/keypad.py)
LONG_THRESHOLD = 0.5   # seconds; >= this → LONG
HOLD_THRESHOLD = 1.5   # seconds; >= this → HOLD

KEY_LABELS: dict[keyboard.Key, str] = {
    keyboard.Key.f13: "F13 (PTT)",
    keyboard.Key.f14: "F14 (NAV)",
    keyboard.Key.f15: "F15 (ACT)",
    keyboard.Key.f16: "F16 (OK) ",
    keyboard.Key.f17: "F17 (NO) ",
}

MACROPAD_KEYS = set(KEY_LABELS)


@dataclass
class _KeyState:
    press_time: float
    hold_timer: threading.Timer | None = None
    hold_fired: bool = False


def _gesture(duration: float) -> str:
    if duration >= HOLD_THRESHOLD:
        return "HOLD"
    if duration >= LONG_THRESHOLD:
        return "LONG"
    return "SHORT"


def main() -> None:
    old_term = _suppress_echo()
    states: dict[keyboard.Key, _KeyState] = {}
    counts: dict[str, int] = defaultdict(int)
    done = threading.Event()

    def on_press(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key not in MACROPAD_KEYS:
            return
        if key in states:
            return  # ignore key-repeat
        label = KEY_LABELS[key]
        ts = time.strftime("%H:%M:%S")

        def fire_hold() -> None:
            state = states.get(key)
            if state is None or state.hold_fired:
                return
            state.hold_fired = True
            counts[f"{label.split()[0]} HOLD"] += 1
            print(f"  [{ts}] {label}  ↓ … HOLD")

        timer = threading.Timer(HOLD_THRESHOLD, fire_hold)
        timer.daemon = True
        states[key] = _KeyState(press_time=time.monotonic(), hold_timer=timer)
        timer.start()
        print(f"  [{ts}] {label}  ↓ press")

    def on_release(key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key not in MACROPAD_KEYS:
            return
        state = states.pop(key, None)
        if state is None:
            return
        if state.hold_timer is not None:
            state.hold_timer.cancel()

        label = KEY_LABELS[key]
        ts = time.strftime("%H:%M:%S")
        duration = time.monotonic() - state.press_time

        if state.hold_fired:
            print(f"  [{ts}] {label}  ↑ release  ({duration:.2f}s)")
            return

        gesture = _gesture(duration)
        counts[f"{label.split()[0]} {gesture}"] += 1
        print(f"  [{ts}] {label}  ↑ {gesture:<5}  ({duration:.2f}s)")

    def shutdown(*_: object) -> None:
        done.set()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("Macropad diagnostic — press F13-F17, Ctrl+C to quit\n")
    print(f"  Keys monitored: {', '.join(KEY_LABELS.values())}")
    print(f"  Gesture thresholds: SHORT <{LONG_THRESHOLD}s | LONG {LONG_THRESHOLD}-{HOLD_THRESHOLD}s | HOLD >{HOLD_THRESHOLD}s\n")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    done.wait()
    listener.stop()

    if old_term is not None:
        _restore_echo(old_term)

    print("\n--- Summary ---")
    if counts:
        for event, n in sorted(counts.items()):
            print(f"  {event}: {n}")
    else:
        print("  No macropad key events detected.")
        print("  Check that the macropad is connected and flashed with the correct firmware.")


if __name__ == "__main__":
    main()
