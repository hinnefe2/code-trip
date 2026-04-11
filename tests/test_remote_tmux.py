"""Unit tests for RemoteTmux with mocked subprocess calls."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from code_trip.remote_tmux import RemoteTmux, RemoteTmuxError, TmuxWindow


@pytest.fixture
def tmux():
    return RemoteTmux(host="remote")


@pytest.fixture
def mock_run():
    with patch("code_trip.remote_tmux.subprocess.run") as m:
        m.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        yield m


# --- send_keys ---


def test_send_keys_basic(tmux, mock_run):
    tmux.send_keys("mysession", "0", "echo hello")
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ssh"
    assert cmd[-2] == "remote"
    remote_cmd = cmd[-1]
    assert "tmux send-keys" in remote_cmd
    assert "mysession:0" in remote_cmd
    assert remote_cmd.endswith(" Enter")


def test_send_keys_no_enter(tmux, mock_run):
    tmux.send_keys("mysession", "0", "partial", enter=False)
    remote_cmd = mock_run.call_args[0][0][-1]
    assert not remote_cmd.endswith(" Enter")


def test_send_keys_special_characters(tmux, mock_run):
    tmux.send_keys("mysession", "0", "echo 'hello world' && echo \"done\"")
    remote_cmd = mock_run.call_args[0][0][-1]
    # The text should be shell-quoted on the remote side
    assert "send-keys" in remote_cmd
    # Should not raise


# --- capture_pane ---


def test_capture_pane_default_lines(tmux, mock_run):
    mock_run.return_value.stdout = "line1\nline2\nline3\n"
    result = tmux.capture_pane("mysession", "0")
    remote_cmd = mock_run.call_args[0][0][-1]
    assert "-S -50" in remote_cmd
    assert result == "line1\nline2\nline3\n"


def test_capture_pane_custom_lines(tmux, mock_run):
    mock_run.return_value.stdout = "output\n"
    tmux.capture_pane("mysession", "0", lines=100)
    remote_cmd = mock_run.call_args[0][0][-1]
    assert "-S -100" in remote_cmd


# --- list_windows ---


def test_list_windows_parses_output(tmux, mock_run):
    mock_run.return_value.stdout = (
        "0\tmain\t/home/user/project\n"
        "1\tticket-42\t/home/user/worktrees/ticket-42\n"
        "2\tticket-99\t/home/user/worktrees/ticket-99\n"
    )
    windows = tmux.list_windows("mysession")
    assert len(windows) == 3
    assert windows[0] == TmuxWindow(index=0, name="main", working_dir="/home/user/project")
    assert windows[1] == TmuxWindow(index=1, name="ticket-42", working_dir="/home/user/worktrees/ticket-42")
    assert windows[2] == TmuxWindow(index=2, name="ticket-99", working_dir="/home/user/worktrees/ticket-99")


def test_list_windows_empty(tmux, mock_run):
    mock_run.return_value.stdout = ""
    windows = tmux.list_windows("mysession")
    assert windows == []


# --- new_window ---


def test_new_window_with_working_dir(tmux, mock_run):
    tmux.new_window("mysession", "ticket-42", working_dir="/home/user/worktrees/ticket-42")
    remote_cmd = mock_run.call_args[0][0][-1]
    assert "new-window" in remote_cmd
    assert "-n" in remote_cmd
    assert "ticket-42" in remote_cmd
    assert "-c" in remote_cmd
    assert "/home/user/worktrees/ticket-42" in remote_cmd


def test_new_window_without_working_dir(tmux, mock_run):
    tmux.new_window("mysession", "ticket-42")
    remote_cmd = mock_run.call_args[0][0][-1]
    assert "new-window" in remote_cmd
    assert "-c" not in remote_cmd


# --- error handling ---


def test_ssh_failure_raises_error(tmux, mock_run):
    mock_run.side_effect = subprocess.CalledProcessError(
        returncode=1, cmd=["ssh"], stderr="Connection refused"
    )
    with pytest.raises(RemoteTmuxError) as exc_info:
        tmux.capture_pane("mysession", "0")
    assert exc_info.value.returncode == 1
    assert "Connection refused" in exc_info.value.stderr


def test_ssh_timeout_raises_error(tmux, mock_run):
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ssh"], timeout=30)
    with pytest.raises(RemoteTmuxError, match="timed out"):
        tmux.capture_pane("mysession", "0")


# --- ssh_options ---


def test_ssh_options_passed_through(mock_run):
    tmux = RemoteTmux(host="remote", ssh_options=("-o", "ConnectTimeout=5"))
    tmux.send_keys("mysession", "0", "hello")
    cmd = mock_run.call_args[0][0]
    assert "-o" in cmd
    assert "ConnectTimeout=5" in cmd
