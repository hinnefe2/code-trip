"""SSH + tmux as free functions.

Single done-detection strategy: poll for the Stop-hook signal file at
/tmp/claude-done-<window>. ControlMaster multiplexing in ~/.ssh/config
keeps the per-call overhead low.
"""

from __future__ import annotations

import asyncio
import shlex


SIGNAL_PREFIX = "/tmp/claude-done-"


class RemoteError(Exception):
    pass


class WaitTimeout(RemoteError):
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
) -> str:
    target = shlex.quote(f"{session}:{window}")
    return await _ssh(host, opts, f"tmux capture-pane -t {target} -p -S -{lines}")


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


def _signal_path(window: str) -> str:
    return f"{SIGNAL_PREFIX}{window}"


async def clear_signal(host: str, opts: tuple[str, ...], window: str) -> None:
    path = shlex.quote(_signal_path(window))
    try:
        await _ssh(host, opts, f"rm -f {path}", capture=False)
    except RemoteError:
        pass


async def wait_done(
    host: str,
    opts: tuple[str, ...],
    window: str,
    *,
    timeout: float = 300.0,
    poll: float = 1.0,
) -> None:
    """Poll for the Stop-hook signal file. Raises WaitTimeout on timeout."""
    await clear_signal(host, opts, window)
    path = shlex.quote(_signal_path(window))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            await _ssh(host, opts, f"test -f {path}", capture=False)
            await clear_signal(host, opts, window)
            return
        except RemoteError:
            pass
        await asyncio.sleep(poll)
    raise WaitTimeout(f"Claude did not finish within {timeout}s in {window!r}")
