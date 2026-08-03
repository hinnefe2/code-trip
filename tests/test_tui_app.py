"""Pilot tests for CodeTripApp — the mirror scroll viewport.

The pure panel-builders are covered in test_tui.py; these drive the
mounted app headlessly to pin the VerticalScroll + anchor() behavior
that makes the remote-window mirror act like a terminal: stuck to the
newest output until the user scrolls up, re-stuck when they return to
the bottom or move to another task.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from code_trip2 import modes
from code_trip2.tasks import Task
from code_trip2.tui import CodeTripApp


def _make_ctx():
    cfg = SimpleNamespace(
        ssh_host="",
        ssh_options=(),
        tmux_session="main",
        work_window="work",
        linear_window="linear",
        terminal_apps=("kitty",),
        autohandle_enabled=False,
        autohandle_kinds=(),
    )
    ctx = modes.Context(
        config=cfg,  # type: ignore[arg-type]
        tts=MagicMock(),
        log=MagicMock(),
        thinking=MagicMock(),
    )
    ctx.current_task = Task(
        kind="remote_window",
        topic="AI-42",
        headline="waiting for input",
        source={"window": "AI-42", "claude_state": "waiting_input"},
    )
    ctx.window_mirrors["AI-42"] = "\n".join(f"line {i}" for i in range(120))
    return ctx


@pytest.mark.asyncio
async def test_mirror_scroll_lifecycle():
    ctx = _make_ctx()
    app = CodeTripApp(ctx, None, local_stt=False)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        scroll = app.query_one("#task_scroll")

        # Mounted anchored: a 120-line mirror starts scrolled to bottom.
        assert scroll.scroll_y > 0
        y_bottom = scroll.scroll_y

        # PageUp scrolls back and releases the anchor.
        await pilot.press("pageup")
        await pilot.pause()
        assert scroll.scroll_y < y_bottom
        assert scroll._anchor_released

        # Paging back down (even past the bottom) re-arms the anchor —
        # the no-movement pagedown-at-bottom case is re-armed by the
        # refresh tick, not the position watcher.
        for _ in range(6):
            await pilot.press("pagedown")
        await pilot.pause(0.7)
        assert scroll.scroll_y >= scroll.max_scroll_y
        assert not scroll._anchor_released

        # While anchored, appended mirror content drags the view down.
        ctx.window_mirrors["AI-42"] += "\n" + "\n".join(
            f"new {i}" for i in range(30)
        )
        await pilot.pause(0.7)
        assert scroll.scroll_y > y_bottom

        # Moving to a non-mirror task disarms the anchor and shows the
        # top of the panel — bottom-sticking is mirror-only semantics.
        ctx.current_task = Task(
            kind="note",
            topic="t",
            headline="other",
            body="\n".join(f"body {i}" for i in range(100)),
        )
        await pilot.pause(0.7)
        assert not scroll._anchored
        assert scroll.scroll_y == 0

        # Coming back to a mirror re-anchors at the (new) bottom.
        ctx.current_task = Task(
            kind="remote_window",
            topic="AI-42",
            headline="waiting for input",
            source={"window": "AI-42", "claude_state": "waiting_input"},
        )
        await pilot.pause(0.7)
        assert scroll._anchored and not scroll._anchor_released
        assert scroll.scroll_y >= scroll.max_scroll_y > 0
