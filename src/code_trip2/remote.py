"""SSH + tmux as free functions.

Single done-detection strategy: poll for the Stop-hook signal file at
/tmp/claude-done-<window>. ControlMaster multiplexing in ~/.ssh/config
keeps the per-call overhead low.
"""

from __future__ import annotations

import shlex
import subprocess
import time


SIGNAL_PREFIX = "/tmp/claude-done-"


class RemoteError(Exception):
    pass


class WaitTimeout(RemoteError):
    pass


def _ssh(host: str, opts: tuple[str, ...], cmd: str, *, capture: bool = True) -> str:
    full = ["ssh", *opts, host, cmd]
    try:
        r = subprocess.run(full, capture_output=capture, text=True, check=True, timeout=30)
    except subprocess.CalledProcessError as e:
        raise RemoteError(f"SSH failed ({e.returncode}): {cmd}\n{e.stderr}") from e
    except subprocess.TimeoutExpired as e:
        raise RemoteError(f"SSH timed out: {cmd}") from e
    return r.stdout if capture else ""


def send(host: str, opts: tuple[str, ...], session: str, window: str, text: str, *, enter: bool = True) -> None:
    target = shlex.quote(f"{session}:{window}")
    cmd = f"tmux send-keys -t {target} {shlex.quote(text)}"
    if enter:
        cmd += " Enter"
    _ssh(host, opts, cmd, capture=False)


def capture(host: str, opts: tuple[str, ...], session: str, window: str, *, lines: int = 100) -> str:
    target = shlex.quote(f"{session}:{window}")
    return _ssh(host, opts, f"tmux capture-pane -t {target} -p -S -{lines}")


def list_windows(host: str, opts: tuple[str, ...], session: str) -> list[tuple[int, str, str]]:
    fmt = "#{window_index}\t#{window_name}\t#{pane_current_path}"
    out = _ssh(host, opts, f"tmux list-windows -t {shlex.quote(session)} -F {shlex.quote(fmt)}")
    rows: list[tuple[int, str, str]] = []
    for line in out.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            rows.append((int(parts[0]), parts[1], parts[2]))
    return rows


def new_window(host: str, opts: tuple[str, ...], session: str, name: str, *, cwd: str | None = None) -> None:
    cmd = f"tmux new-window -t {shlex.quote(session)} -n {shlex.quote(name)}"
    if cwd:
        cmd += f" -c {shlex.quote(cwd)}"
    _ssh(host, opts, cmd, capture=False)


def select_window(host: str, opts: tuple[str, ...], session: str, window: str) -> None:
    target = shlex.quote(f"{session}:{window}")
    _ssh(host, opts, f"tmux select-window -t {target}", capture=False)


def _signal_path(window: str) -> str:
    return f"{SIGNAL_PREFIX}{window}"


def clear_signal(host: str, opts: tuple[str, ...], window: str) -> None:
    path = shlex.quote(_signal_path(window))
    try:
        _ssh(host, opts, f"rm -f {path}", capture=False)
    except RemoteError:
        pass


def wait_done(host: str, opts: tuple[str, ...], window: str, *, timeout: float = 300.0, poll: float = 1.0) -> None:
    """Poll for the Stop-hook signal file. Raises WaitTimeout on timeout."""
    clear_signal(host, opts, window)
    path = shlex.quote(_signal_path(window))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            _ssh(host, opts, f"test -f {path}", capture=False)
            clear_signal(host, opts, window)
            return
        except RemoteError:
            pass
        time.sleep(poll)
    raise WaitTimeout(f"Claude did not finish within {timeout}s in {window!r}")
