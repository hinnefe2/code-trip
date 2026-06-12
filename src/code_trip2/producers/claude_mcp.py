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

import asyncio
import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ClaudeMCPError(Exception):
    pass


async def _run_subprocess(
    cmd: list[str], *, input_: str, timeout: float, what: str,
) -> tuple[str, str, int]:
    """Run ``cmd`` async with text I/O. Returns (stdout, stderr, returncode).

    Raises :class:`ClaudeMCPError` on timeout (kills the process first).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise ClaudeMCPError(f"{what} spawn failed: {exc}") from exc
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(input_.encode("utf-8")), timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        proc.kill()
        try:
            await proc.wait()
        except Exception:
            pass
        raise ClaudeMCPError(f"{what} timed out after {timeout}s") from exc
    stdout = (stdout_b or b"").decode("utf-8", errors="replace")
    stderr = (stderr_b or b"").decode("utf-8", errors="replace")
    return stdout, stderr, proc.returncode or 0


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

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Invoke ``server.<tool_name>(**args)`` and return the parsed result.

        Returns a dict (the tool's JSON response, parsed). Raises
        :class:`ClaudeMCPError` on subprocess failure, timeout, missing
        tool_result, or unparseable output.
        """
        return await self.call_tool_id(f"mcp__{self.server_id}__{tool_name}", args)

    async def call_tool_id(self, tool_id: str, args: dict[str, Any]) -> dict[str, Any]:
        """Same as :meth:`call_tool` but takes a fully-qualified tool id.

        Used by the batch fallback path, where the server prefix comes
        from the request rather than this client's ``server_id``.
        """
        if not self._available:
            raise ClaudeMCPError("claude CLI not available")

        tool_name = tool_id.rsplit("__", 1)[-1]
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
        stdout, stderr, returncode = await _run_subprocess(
            cmd, input_=prompt, timeout=self.timeout, what="claude",
        )

        # Try to parse the tool_result regardless of returncode. Claude can
        # exit nonzero *after* the tool ran successfully (e.g. budget hit
        # post-call, the model adding extra commentary, etc.) — we'd rather
        # surface the good result than throw it away.
        try:
            return self._parse_stream_for_tool_result(stdout, tool_id)
        except ClaudeMCPError as parse_exc:
            if returncode != 0:
                raise ClaudeMCPError(
                    f"claude exited {returncode}: "
                    f"{stderr[:500].strip() or '(no stderr)'}"
                ) from parse_exc
            raise

    async def run_agent(
        self,
        *,
        prompt: str,
        allowed_tools: list[str] | tuple[str, ...] = (),
        timeout: float | None = None,
        max_budget_usd: float | None = None,
        transcript: bool = False,
    ) -> str:
        """Free-form Claude invocation: no single-tool constraint.

        Used for skill execution where Claude orchestrates multiple MCP
        tools toward a goal described in the prompt. Claude Code's own
        skill discovery (``.claude/skills/``) handles matching the
        user's request to a skill; this method just hands the prompt
        over and reads back the final assistant text.

        ``allowed_tools`` is the union of ``allowed-tools`` declared by
        every available skill — passed through to ``--allowedTools`` so
        Claude can't reach for any tool a skill doesn't already say it
        needs. Pass an empty list/tuple to leave the flag off (Claude
        can use any configured MCP tool); skill files should declare
        their tools instead.

        ``--permission-mode bypassPermissions`` skips interactive
        prompts. Budget cap defaults higher than :meth:`call_tool`
        because skill flows tend to make several tool calls.

        ``transcript=True`` returns all assistant text blocks joined by
        newlines — the screener needs this to recover structured output
        (``FOLLOWUP_TASK:`` lines) that the model emits *before* its
        final tool call, which would otherwise be discarded. ACT+PTT
        leaves it False so the spoken summary stays clean.
        """
        if not self._available:
            raise ClaudeMCPError("claude CLI not available")

        budget = max_budget_usd if max_budget_usd is not None else max(self.max_budget_usd, 0.20)
        # Skill flows chain several claude.ai MCP roundtrips (skill
        # discovery, then each tool call proxies through the remote
        # server) — a healthy file-meeting-followup run measured ~90s
        # end-to-end, so the old 120s floor left no headroom and killed
        # a working run mid-flight. Worse than slow: a kill can land
        # after the side effect (issue created, email archived) but
        # before the summary, leaving the task active for a re-trigger.
        # 300s is a ceiling, not a wait — healthy runs are unaffected.
        deadline = timeout if timeout is not None else max(self.timeout, 300.0)
        cmd = [
            self.binary,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model,
            "--no-session-persistence",
            "--disable-slash-commands",
            "--permission-mode",
            "bypassPermissions",
            "--max-budget-usd",
            str(budget),
        ]
        if allowed_tools:
            # ``--allowedTools`` is variadic; must come last so the variadic
            # capture doesn't swallow a subsequent flag. Prompt goes on
            # stdin (same reason as :meth:`call_tool`).
            cmd += ["--allowedTools", *allowed_tools]
        stdout, stderr, returncode = await _run_subprocess(
            cmd, input_=prompt, timeout=deadline, what="claude",
        )
        if returncode != 0:
            raise ClaudeMCPError(
                f"claude exited {returncode}: "
                f"{stderr[:500].strip() or '(no stderr)'}"
            )
        return self._extract_assistant_text(stdout, transcript=transcript)

    @staticmethod
    def _extract_assistant_text(stdout: str, *, transcript: bool = False) -> str:
        """Pull assistant ``text`` content from a stream-json stdout.

        Default: return only the *final* assistant text block — the
        natural-language summary the user hears spoken back. Skill
        runs interleave ``tool_use`` and ``tool_result`` blocks with
        assistant ``text`` blocks, so picking the last text-typed
        block gives the user-facing summary without intermediate
        reasoning.

        ``transcript=True``: return every assistant text block joined
        by newlines. Used by the screener so structured emissions
        like ``FOLLOWUP_TASK: {...}`` lines — which the model often
        prints in a separate block *before* its final tool call —
        survive into ``parse_follow_up_tasks``.
        """
        blocks: list[str] = []
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
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    txt = (block.get("text") or "").strip()
                    if txt:
                        blocks.append(txt)
        if not blocks:
            return ""
        if transcript:
            return "\n".join(blocks)
        return blocks[-1]

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
        return parse_result_text(text)

    def _recover_from_dumped_file(self, text: str) -> dict[str, Any] | None:
        return _recover_from_dumped_file(text)


def parse_result_text(text: str) -> dict[str, Any]:
    """Best-effort parse of the tool result string into a dict."""
    text = text.strip()
    if not text:
        return {}
    # Large tool results get diverted by claude --print to a file
    # on disk; the tool_result content is then just a notice with
    # the file path. Read the file ourselves so producers don't
    # silently see "no results" when the real payload is huge.
    recovered = _recover_from_dumped_file(text)
    if recovered is not None:
        return recovered
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


# Sentinel claude --print emits when a tool result exceeds its
# per-tool token cap. The file is a JSON array of content blocks
# (``[{"type": "text", "text": "<actual payload>"}]``); the
# ``text`` field carries the real tool result as a JSON string.
_DUMPED_PATH_RE = re.compile(r"Output has been saved to (\S+\.txt)")


def _recover_from_dumped_file(text: str) -> dict[str, Any] | None:
    """Read a dumped tool-result file when claude --print diverted it.

    Returns the parsed payload, or ``None`` when the text doesn't
    carry a dump-path sentinel (caller should fall back to its
    normal parse path). Errors reading or parsing the file also
    return ``None`` so the caller's default ``{"_raw": ...}``
    still wins as a last resort.
    """
    m = _DUMPED_PATH_RE.search(text)
    if not m:
        return None
    path = m.group(1)
    logger.info("claude --print diverted tool result to %s; reading", path)
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as exc:
        logger.warning("Could not read diverted tool-result file %s: %s", path, exc)
        return None
    # The file is the standard MCP content-block list. Pull the
    # text out of each block and join — usually there's just one.
    try:
        blocks = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Diverted file %s is not JSON; treating as opaque", path)
        return {"_raw": raw}
    parts: list[str] = []
    if isinstance(blocks, list):
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text", "")))
            elif isinstance(b, str):
                parts.append(b)
    elif isinstance(blocks, dict):
        return blocks
    joined = "\n".join(p for p in parts if p)
    if not joined:
        return None
    # Recurse on the recovered payload, but strip our sentinel-
    # detection out of the recursion so a degenerate file that
    # itself contains the sentinel can't loop forever.
    try:
        parsed = json.loads(joined)
    except json.JSONDecodeError:
        return {"_raw": joined}
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    return {"_raw": joined}
