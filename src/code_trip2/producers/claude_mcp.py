"""Invoke MCP tools by shelling out to the ``claude`` CLI.

We use Claude as a thin auth proxy. claude.ai already holds OAuth tokens
for its hosted MCP servers (Slack, Linear, Gmail, etc.); rather than
register a Slack app and manage tokens ourselves, we let Claude make the
tool call on our behalf and pluck the structured result out of its
stream-json output.

Trade-offs versus a direct API call:

- **Auth**: free — piggy-backs on whatever the user already authorized
  via claude.ai.
- **Latency**: ~3–5 s per call (subprocess start + LLM inference + MCP
  round-trip). Fine for 30-second poll intervals.
- **Cost**: ~$0.001 per call with Haiku, ``--bare``-style minimal
  context, and tight ``--allowedTools`` constraints. Capped per call
  via ``--max-budget-usd``.
- **Robustness**: the LLM is in the loop, but we constrain it to one
  tool with explicit args, and we parse the *tool_result* block out of
  the stream — not the assistant's prose. So the LLM can ramble all it
  wants; we ignore everything except the tool's structured response.

The prompt template demands exactly one tool call with the supplied
args and tells the model not to summarize. With ``--allowedTools``
limited to a single tool and ``--max-budget-usd`` set low, the worst
case is the model refusing to call — which surfaces as a
``ClaudeMCPError`` we report cleanly.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ClaudeMCPError(Exception):
    pass


@dataclass
class ClaudeMCPClient:
    """Calls one MCP tool per invocation via ``claude --print``.

    ``server_id`` is the underscore form of the server name (e.g.
    ``claude_ai_Slack`` for the server listed as ``claude.ai Slack``
    in ``claude mcp list``). Tool IDs follow the ``mcp__<server>__<tool>``
    convention exposed to Claude as tool names.
    """

    binary: str = "claude"
    model: str = "haiku"
    server_id: str = "claude_ai_Slack"
    timeout: float = 60.0
    # Per-call budget cap. Real-world cost is ~$0.02–0.04 per call because
    # claude --print loads the full catalog of available MCP tools into
    # context, not just the allowed one — most of that is cache reads but
    # they add up. The cap exists to keep a runaway invocation from
    # spiraling (e.g., if the model retries), not as a normal-path knob.
    max_budget_usd: float = 0.05

    _available: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._available = shutil.which(self.binary) is not None
        if not self._available:
            logger.info(
                "ClaudeMCPClient: %r not found on PATH; client disabled.",
                self.binary,
            )

    @property
    def enabled(self) -> bool:
        return self._available

    def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Invoke ``server.<tool_name>(**args)`` and return the parsed result.

        Returns a dict (the tool's JSON response, parsed). Raises
        :class:`ClaudeMCPError` on subprocess failure, timeout, missing
        tool_result, or unparseable output.
        """
        if not self._available:
            raise ClaudeMCPError("claude CLI not available")

        tool_id = f"mcp__{self.server_id}__{tool_name}"
        prompt = self._format_prompt(tool_name, args)
        # The prompt goes on stdin, not as a positional arg, because
        # ``--allowedTools`` is variadic (``<tools...>``) and will silently
        # swallow a trailing prompt as a second tool name. claude --print
        # reads from stdin when no positional prompt is given.
        cmd = [
            self.binary,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",  # stream-json requires verbose
            "--model",
            self.model,
            "--no-session-persistence",
            "--disable-slash-commands",
            "--permission-mode",
            "bypassPermissions",
            "--max-budget-usd",
            str(self.max_budget_usd),
            "--allowedTools",
            tool_id,
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ClaudeMCPError(f"claude timed out after {self.timeout}s") from exc

        # Try to parse the tool_result regardless of returncode. Claude can
        # exit nonzero *after* the tool ran successfully (e.g. budget hit
        # post-call, the model adding extra commentary, etc.) — we'd rather
        # surface the good result than throw it away.
        try:
            return self._parse_stream_for_tool_result(proc.stdout, tool_id)
        except ClaudeMCPError as parse_exc:
            if proc.returncode != 0:
                raise ClaudeMCPError(
                    f"claude exited {proc.returncode}: "
                    f"{proc.stderr[:500].strip() or '(no stderr)'}"
                ) from parse_exc
            raise

    # ---- internals ------------------------------------------------------

    def _format_prompt(self, tool_name: str, args: dict[str, Any]) -> str:
        return (
            f"Call the {tool_name} MCP tool with these EXACT arguments:\n\n"
            f"```json\n{json.dumps(args, indent=2)}\n```\n\n"
            "Make exactly one tool call. Do not modify the arguments. Do not "
            "call any other tool. After the tool returns, your job is complete "
            "— do not summarize, comment, or interpret the result."
        )

    def _parse_stream_for_tool_result(self, stdout: str, tool_id: str) -> dict[str, Any]:
        """Walk the stream-json output and extract the tool_result block.

        ``stream-json`` emits one JSON object per line. The shapes we care
        about all live in ``event["message"]["content"]`` as lists of
        ``{"type": "tool_use" | "tool_result" | "text", ...}`` blocks. We
        scan for the latest ``tool_result`` block and parse its text
        payload (which the Slack MCP returns as a JSON string).
        """
        last_result_text: str | None = None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = self._extract_tool_result_text(event)
            if text is not None:
                last_result_text = text

        if last_result_text is None:
            raise ClaudeMCPError(
                f"No tool_result block found in claude output for {tool_id}. "
                f"Stdout head: {stdout[:300]!r}"
            )

        return self._parse_result_text(last_result_text)

    def _extract_tool_result_text(self, event: dict) -> str | None:
        """Pull text out of a tool_result content block, if present."""
        msg = event.get("message")
        if not isinstance(msg, dict):
            return None
        content = msg.get("content")
        if not isinstance(content, list):
            return None
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_result":
                continue
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

    def _parse_result_text(self, text: str) -> dict[str, Any]:
        """Best-effort parse of the tool result string into a dict."""
        text = text.strip()
        if not text:
            return {}
        # Direct JSON object/array.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        # Fish a JSON object out of stray prose.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
        # Last resort: return as opaque text.
        return {"_raw": text}
