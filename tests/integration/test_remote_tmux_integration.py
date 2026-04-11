"""Integration tests for RemoteTmux against a real tmux session.

Requires environment variables:
    CODETRIP_TEST_HOST: SSH host alias (e.g., "remote")
    CODETRIP_TEST_SESSION: tmux session name on the remote host

Run with:
    CODETRIP_TEST_HOST=myhost CODETRIP_TEST_SESSION=test pytest tests/integration/ -v
"""

from __future__ import annotations

import os
import time

import pytest

from code_trip.remote_tmux import RemoteTmux

HOST = os.environ.get("CODETRIP_TEST_HOST", "")
SESSION = os.environ.get("CODETRIP_TEST_SESSION", "")

pytestmark = pytest.mark.skipif(
    not HOST or not SESSION,
    reason="CODETRIP_TEST_HOST and CODETRIP_TEST_SESSION must be set",
)

WINDOW_NAME = "code-trip-integration-test"


@pytest.fixture
def tmux():
    return RemoteTmux(host=HOST)


@pytest.fixture
def test_window(tmux):
    """Create a test window and clean it up after the test."""
    tmux.new_window(SESSION, WINDOW_NAME)
    time.sleep(0.5)
    yield WINDOW_NAME
    tmux.send_keys(SESSION, WINDOW_NAME, "exit")


def test_send_and_capture(tmux, test_window):
    tmux.send_keys(SESSION, test_window, "echo CODE_TRIP_TEST_MARKER")
    time.sleep(1)
    output = tmux.capture_pane(SESSION, test_window)
    assert "CODE_TRIP_TEST_MARKER" in output


def test_list_windows_includes_test_window(tmux, test_window):
    windows = tmux.list_windows(SESSION)
    names = [w.name for w in windows]
    assert WINDOW_NAME in names


# --- completion detection ---


def test_signal_file_lifecycle(tmux, test_window):
    """Create, detect, and clear a signal file via SSH."""
    # Initially no signal file
    assert tmux.check_signal_file(test_window) is False

    # Simulate the Stop hook by touching the signal file
    from code_trip.remote_tmux import SIGNAL_FILE_PREFIX

    signal_path = f"{SIGNAL_FILE_PREFIX}{test_window}"
    tmux._run_ssh(f"touch {signal_path}")

    assert tmux.check_signal_file(test_window) is True

    # Clear it
    tmux.clear_signal_file(test_window)
    assert tmux.check_signal_file(test_window) is False


def test_wait_for_claude_with_simulated_hook(tmux, test_window):
    """wait_for_claude returns when a signal file is created externally."""
    import threading

    from code_trip.remote_tmux import SIGNAL_FILE_PREFIX

    signal_path = f"{SIGNAL_FILE_PREFIX}{test_window}"

    def create_signal_after_delay():
        time.sleep(2)
        tmux._run_ssh(f"touch {signal_path}")

    thread = threading.Thread(target=create_signal_after_delay)
    thread.start()

    # Should return within a few seconds (signal created after 2s)
    tmux.wait_for_claude(SESSION, test_window, timeout=10, poll_interval=0.5)
    thread.join()

    # Signal file should have been cleaned up
    assert tmux.check_signal_file(test_window) is False


def test_is_claude_ready_on_shell_prompt(tmux, test_window):
    """Fallback detection on a regular shell (should not look like Claude prompt)."""
    time.sleep(0.5)
    # A normal shell prompt (e.g., $) should not be detected as Claude's > prompt
    ready = tmux.is_claude_ready(SESSION, test_window)
    assert ready is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
