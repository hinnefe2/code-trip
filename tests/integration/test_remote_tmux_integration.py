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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
