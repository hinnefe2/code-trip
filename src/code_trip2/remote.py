"""SSH + tmux as free functions.

ControlMaster multiplexing in ~/.ssh/config keeps the per-call
overhead low. Claude session state (running vs waiting) is detected
from ``capture`` output by :mod:`code_trip2.claude_screen` — there is
no hook or signal-file machinery on the remote.
"""

from __future__ import annotations

import asyncio
import shlex


class RemoteError(Exception):
    pass


async def _ssh(
    host: str,
    opts: tuple[str, ...],
    cmd: str,
    *,
    capture: bool = True,
    timeout: float = 30.0,
) -> str:
    """Run one SSH command via asyncio.subprocess. Raises RemoteError on failure."""
    argv = ["ssh", *opts, host, cmd]
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE if capture else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise RemoteError(f"SSH spawn failed: {cmd}: {exc}") from exc
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        try:
            await proc.wait()
        except Exception:
            pass
        raise RemoteError(f"SSH timed out: {cmd}") from exc
    if proc.returncode != 0:
        stderr = (stderr_b or b"").decode(errors="replace")
        raise RemoteError(f"SSH failed ({proc.returncode}): {cmd}\n{stderr}")
    if capture:
        return (stdout_b or b"").decode(errors="replace")
    return ""


async def send(
    host: str,
    opts: tuple[str, ...],
    session: str,
    window: str,
    text: str,
    *,
    enter: bool = True,
) -> None:
    target = shlex.quote(f"{session}:{window}")
    cmd = f"tmux send-keys -t {target} {shlex.quote(text)}"
    if enter:
        cmd += " Enter"
    await _ssh(host, opts, cmd, capture=False)


async def capture(
    host: str,
    opts: tuple[str, ...],
    session: str,
    window: str,
    *,
    lines: int = 100,
    ansi: bool = False,
) -> str:
    """Capture the pane text. ``ansi=True`` keeps SGR escapes (-e) so the
    TUI mirror can re-render colors via ``Text.from_ansi``."""
    target = shlex.quote(f"{session}:{window}")
    flags = "-p -e" if ansi else "-p"
    return await _ssh(host, opts, f"tmux capture-pane -t {target} {flags} -S -{lines}")


async def list_windows(
    host: str, opts: tuple[str, ...], session: str,
) -> list[tuple[int, str, str]]:
    fmt = "#{window_index}\t#{window_name}\t#{pane_current_path}"
    out = await _ssh(
        host, opts, f"tmux list-windows -t {shlex.quote(session)} -F {shlex.quote(fmt)}"
    )
    rows: list[tuple[int, str, str]] = []
    for line in out.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            rows.append((int(parts[0]), parts[1], parts[2]))
    return rows


async def new_window(
    host: str, opts: tuple[str, ...], session: str, name: str, *, cwd: str | None = None,
) -> None:
    cmd = f"tmux new-window -t {shlex.quote(session)} -n {shlex.quote(name)}"
    if cwd:
        cmd += f" -c {shlex.quote(cwd)}"
    await _ssh(host, opts, cmd, capture=False)


async def select_window(
    host: str, opts: tuple[str, ...], session: str, window: str,
) -> None:
    target = shlex.quote(f"{session}:{window}")
    await _ssh(host, opts, f"tmux select-window -t {target}", capture=False)


