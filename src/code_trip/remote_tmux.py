"""SSH + tmux remote interface.

Wraps SSH and tmux commands to communicate with a remote tmux session.
This is the foundation for the code-trip orchestrator.

SSH ControlMaster Setup
-----------------------
For best performance, configure SSH connection multiplexing in ~/.ssh/config::

    Host <your-remote>
        ControlMaster auto
        ControlPath ~/.ssh/sockets/%r@%h-%p
        ControlPersist 600

Make sure the socket directory exists::

    mkdir -p ~/.ssh/sockets
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field


class RemoteTmuxError(Exception):
    """Raised when an SSH or tmux command fails."""

    def __init__(
        self, message: str, returncode: int | None = None, stderr: str = ""
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True)
class TmuxWindow:
    """A tmux window with its index, name, and working directory."""

    index: int
    name: str
    working_dir: str


@dataclass
class RemoteTmux:
    """Interface to a remote tmux session over SSH.

    Args:
        host: SSH host alias (must be configured in ~/.ssh/config).
        ssh_options: Extra SSH flags, e.g. ("-o", "ConnectTimeout=5").
    """

    host: str
    ssh_options: tuple[str, ...] = field(default_factory=tuple)

    def _run_ssh(
        self, command: str, *, capture_output: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Execute a command on the remote host via SSH."""
        cmd = ["ssh", *self.ssh_options, self.host, command]
        try:
            return subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                check=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as e:
            raise RemoteTmuxError(
                f"SSH command failed: {command}",
                returncode=e.returncode,
                stderr=e.stderr or "",
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RemoteTmuxError(f"SSH command timed out: {command}") from e

    def send_keys(
        self, session: str, window: str, text: str, *, enter: bool = True
    ) -> None:
        """Send keystrokes to a tmux pane.

        Args:
            session: tmux session name.
            window: window name or index.
            text: text to send.
            enter: whether to append Enter keystroke (default True).
        """
        target = shlex.quote(f"{session}:{window}")
        tmux_cmd = f"tmux send-keys -t {target} {shlex.quote(text)}"
        if enter:
            tmux_cmd += " Enter"
        self._run_ssh(tmux_cmd, capture_output=False)

    def capture_pane(
        self, session: str, window: str, *, lines: int = 50
    ) -> str:
        """Capture visible pane output.

        Args:
            session: tmux session name.
            window: window name or index.
            lines: number of lines to capture from bottom (default 50).

        Returns:
            The captured pane text.
        """
        target = shlex.quote(f"{session}:{window}")
        tmux_cmd = f"tmux capture-pane -t {target} -p -S -{lines}"
        result = self._run_ssh(tmux_cmd)
        return result.stdout

    def list_windows(self, session: str) -> list[TmuxWindow]:
        """List all windows in a tmux session.

        Returns:
            List of TmuxWindow with index, name, and working_dir.
        """
        fmt = "#{window_index}\t#{window_name}\t#{pane_current_path}"
        tmux_cmd = (
            f"tmux list-windows -t {shlex.quote(session)} -F {shlex.quote(fmt)}"
        )
        result = self._run_ssh(tmux_cmd)
        windows = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                windows.append(
                    TmuxWindow(
                        index=int(parts[0]),
                        name=parts[1],
                        working_dir=parts[2],
                    )
                )
        return windows

    def new_window(
        self, session: str, name: str, *, working_dir: str | None = None
    ) -> None:
        """Create a new window in a tmux session.

        Args:
            session: tmux session name.
            name: name for the new window.
            working_dir: initial working directory (optional).
        """
        tmux_cmd = (
            f"tmux new-window -t {shlex.quote(session)} -n {shlex.quote(name)}"
        )
        if working_dir:
            tmux_cmd += f" -c {shlex.quote(working_dir)}"
        self._run_ssh(tmux_cmd, capture_output=False)
