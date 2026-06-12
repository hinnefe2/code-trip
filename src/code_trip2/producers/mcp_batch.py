"""Coalesce concurrent MCP tool calls into one ``claude --print`` session.

Every ``claude --print`` invocation pays a fixed overhead — subprocess
start, the full MCP tool catalog loaded into context, and (since the
2026-06-15 billing change) headless-credit burn — regardless of how many
tools it calls. The producers poll on wall-clock-aligned ticks (see
``_async_utils.next_tick_delay``), so their poll calls arrive within
milliseconds of each other; collecting requests for a short window and
issuing a single session with ``--allowedTools`` spanning all of them
cuts the per-tick session count from three-plus to one.

Correctness is structural, not prompt-trust:

- Results are read from the stream-json ``tool_use`` / ``tool_result``
  blocks — the same protocol-level data the single-call path parses —
  never from the model's prose. The model cannot corrupt a result by
  paraphrasing it because we never look at what it says, only at what
  the tools returned.
- Each request is matched to its ``tool_use`` block by tool name plus
  **exact argument equality**, joined to its ``tool_result`` through
  the protocol's own ``tool_use_id``. A call the model skipped, or
  whose arguments it mutated, simply has no structured match.
- Unmatched requests fall back to their own single-tool invocation
  (the pre-batching behavior), so a flaky batch degrades to the old
  cost profile rather than to missing data.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from code_trip2.producers.claude_mcp import (
    ClaudeMCPClient,
    ClaudeMCPError,
    _run_subprocess,
    parse_result_text,
)

logger = logging.getLogger(__name__)


@dataclass
class _Pending:
    tool_id: str
    args: dict[str, Any]
    future: "asyncio.Future[dict[str, Any]]"


@dataclass
class _CompletedCall:
    """One tool invocation recovered from the stream-json output."""

    name: str
    input: dict[str, Any]
    result_text: str
    consumed: bool = False


def _collect_tool_calls(stdout: str) -> list[_CompletedCall]:
    """Join ``tool_use`` and ``tool_result`` blocks by tool_use_id.

    Walks the stream-json events, indexing assistant ``tool_use``
    blocks (id, name, input) and user ``tool_result`` blocks
    (tool_use_id, content text). Returns the completed pairs in
    tool_use order.
    """
    uses: dict[str, tuple[str, dict[str, Any]]] = {}  # id -> (name, input)
    order: list[str] = []
    results: dict[str, str] = {}  # tool_use_id -> result text
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = event.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                use_id = block.get("id")
                name = block.get("name")
                inp = block.get("input")
                if isinstance(use_id, str) and isinstance(name, str):
                    if use_id not in uses:
                        order.append(use_id)
                    uses[use_id] = (name, inp if isinstance(inp, dict) else {})
            elif block.get("type") == "tool_result":
                use_id = block.get("tool_use_id")
                text = _result_block_text(block)
                if isinstance(use_id, str) and text is not None:
                    results[use_id] = text
    out: list[_CompletedCall] = []
    for use_id in order:
        if use_id in results:
            name, inp = uses[use_id]
            out.append(_CompletedCall(name=name, input=inp, result_text=results[use_id]))
    return out


def _result_block_text(block: dict) -> str | None:
    inner = block.get("content")
    if isinstance(inner, str):
        return inner
    if isinstance(inner, list):
        parts = []
        for b in inner:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif isinstance(b, str):
                parts.append(b)
        if parts:
            return "\n".join(parts)
    return None


@dataclass
class MCPBatcher:
    """Shared coalescing executor behind every ``BatchedMCPClient``.

    The first ``call_tool`` after an idle period opens a collection
    window of ``window_s`` seconds; every request arriving inside it
    joins the same ``claude --print`` session. A request that arrives
    alone (no other call within the window — e.g. an interactive
    voice-driven draft) just runs as a single-tool session after the
    window, identical to the pre-batching behavior plus the window's
    latency.
    """

    binary: str = "claude"
    model: str = "haiku"
    window_s: float = 2.0
    # Timeout / budget scale with batch size: each extra tool call is
    # another MCP roundtrip inside the same session.
    base_timeout: float = 60.0
    per_call_timeout: float = 20.0
    base_budget_usd: float = 0.05
    per_call_budget_usd: float = 0.04

    _client: ClaudeMCPClient = field(init=False, repr=False)
    _pending: list[_Pending] = field(default_factory=list, init=False, repr=False)
    _flush_task: "asyncio.Task | None" = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Reused for availability detection and single-call fallback;
        # server_id is irrelevant because callers pass fully-qualified
        # tool ids (``mcp__<server>__<tool>``).
        self._client = ClaudeMCPClient(binary=self.binary, model=self.model)

    @property
    def enabled(self) -> bool:
        return self._client.enabled

    async def call_tool(self, tool_id: str, args: dict[str, Any]) -> dict[str, Any]:
        """Enqueue one tool call; resolves when its batch completes."""
        if not self.enabled:
            raise ClaudeMCPError("claude CLI not available")
        fut: "asyncio.Future[dict[str, Any]]" = asyncio.get_running_loop().create_future()
        self._pending.append(_Pending(tool_id=tool_id, args=args, future=fut))
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(
                self._flush_after_window(), name="mcp-batch-flush",
            )
        return await fut

    async def _flush_after_window(self) -> None:
        await asyncio.sleep(self.window_s)
        batch, self._pending = self._pending, []
        if not batch:
            return
        try:
            await self._execute(batch)
        except Exception as exc:  # backstop — futures must always resolve
            logger.exception("MCP batch execution crashed")
            for p in batch:
                if not p.future.done():
                    p.future.set_exception(
                        ClaudeMCPError(f"batch execution crashed: {exc}")
                    )

    async def _execute(self, batch: list[_Pending]) -> None:
        if len(batch) == 1:
            await self._execute_single(batch[0])
            return
        logger.info(
            "MCP batch: %d calls in one session (%s)",
            len(batch), ", ".join(p.tool_id for p in batch),
        )
        tool_ids = sorted({p.tool_id for p in batch})
        n = len(batch)
        cmd = [
            self.binary,
            "--print",
            "--output-format", "stream-json",
            "--verbose",
            "--model", self.model,
            "--no-session-persistence",
            "--disable-slash-commands",
            "--permission-mode", "bypassPermissions",
            "--max-budget-usd",
            str(self.base_budget_usd + self.per_call_budget_usd * n),
            "--allowedTools", *tool_ids,
        ]
        prompt = self._format_batch_prompt(batch)
        timeout = self.base_timeout + self.per_call_timeout * n
        try:
            stdout, _stderr, _rc = await _run_subprocess(
                cmd, input_=prompt, timeout=timeout, what="claude (batch)",
            )
            completed = _collect_tool_calls(stdout)
        except ClaudeMCPError as exc:
            logger.warning(
                "MCP batch of %d failed (%s); falling back to single calls",
                n, exc,
            )
            completed = []
        # Deterministic matching: name + exact argument equality, in
        # request order. ``consumed`` prevents two identical requests
        # from sharing one result.
        misses: list[_Pending] = []
        for p in batch:
            short_name = p.tool_id.rsplit("__", 1)[-1]
            match = next(
                (
                    c for c in completed
                    if not c.consumed
                    and c.name in (p.tool_id, short_name)
                    and c.input == p.args
                ),
                None,
            )
            if match is None:
                misses.append(p)
                continue
            match.consumed = True
            try:
                p.future.set_result(parse_result_text(match.result_text))
            except Exception as exc:
                p.future.set_exception(
                    ClaudeMCPError(f"could not parse batched result: {exc}")
                )
        if misses:
            logger.warning(
                "MCP batch: %d/%d calls unmatched; retrying individually (%s)",
                len(misses), n, ", ".join(p.tool_id for p in misses),
            )
            await asyncio.gather(*(self._execute_single(p) for p in misses))

    async def _execute_single(self, p: _Pending) -> None:
        """Single-tool path — delegates to the proven ClaudeMCPClient flow."""
        try:
            result = await self._client.call_tool_id(p.tool_id, p.args)
        except Exception as exc:
            if not p.future.done():
                p.future.set_exception(exc)
            return
        if not p.future.done():
            p.future.set_result(result)

    def _format_batch_prompt(self, batch: list[_Pending]) -> str:
        calls = "\n\n".join(
            f"{i}. Tool `{p.tool_id}`:\n```json\n{json.dumps(p.args, indent=2)}\n```"
            for i, p in enumerate(batch, start=1)
        )
        return (
            f"Call ALL {len(batch)} of the following MCP tools, each exactly "
            "once, with the EXACT arguments given — do not modify, merge, or "
            "skip any call. Make the calls in the order listed.\n\n"
            f"{calls}\n\n"
            "After every tool has returned, your job is complete — do not "
            "summarize, comment, or interpret the results."
        )


@dataclass
class BatchedMCPClient:
    """Drop-in for :class:`ClaudeMCPClient`'s ``call_tool`` surface.

    Binds a server id and funnels calls through the shared
    :class:`MCPBatcher` so concurrent calls from different producers
    (Slack, Gmail, Linear) coalesce into one claude session.
    """

    server_id: str
    batcher: MCPBatcher

    @property
    def enabled(self) -> bool:
        return self.batcher.enabled

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        return await self.batcher.call_tool(
            f"mcp__{self.server_id}__{tool_name}", args,
        )
