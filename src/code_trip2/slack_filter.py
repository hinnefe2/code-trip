"""LLM relevance filter for incoming Slack messages.

Per-message decision: should this message become a task in the queue?
Returns a verdict (``relevant``, ``urgency``, ``headline``) so the
producer can both gate task creation and pre-populate the audio
announcement headline in one call.

Designed to be cheap and tunable:

- Uses ``gpt-4o-mini`` by default (~$0.0001 per call at typical sizes).
- The user can append context to the system prompt via
  ``config.slack_filter_extra`` — e.g. project names, teammates' handles,
  channels they actually care about.
- Every decision is logged at INFO so false-negatives are reviewable in
  ``~/.code-trip/logs/orchestrator.log``.

Falls **closed** on any error (no task created) and **open** on
ambiguity — better to surface a maybe-irrelevant ping than to silently
drop a real ask.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT_BASE = """You are filtering Slack messages for a software engineer who
wants to be interrupted by audio notifications only when truly relevant.

Decide whether the incoming message warrants their attention right now.
Respond with STRICT JSON only — no prose, no code fences:

{
  "relevant": true | false,
  "urgency": "interrupt" | "normal" | "background",
  "headline": "<=60 char spoken summary, sender + gist"
}

Default to "relevant": true for anything that addresses the user directly
or implicitly asks for their input. Default to "relevant": false for
broadcast chatter, bot pings, automated alerts unrelated to them, and
discussion they can scroll-read later.

Urgency:
- "interrupt"  → production alert mentioning their systems, urgent ask, time-sensitive blocker
- "normal"     → typical mention or ask
- "background" → relevant but not pressing; can wait

Headline must be one short spoken-English line, NO markdown, NO emojis.
Lead with the sender name."""


class SlackFilterError(Exception):
    pass


class SlackFilter:
    """Wraps an OpenAI chat completion that returns a JSON verdict."""

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "gpt-4o-mini",
        extra_prompt: str = "",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._extra = extra_prompt.strip()
        self._client = None
        if api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai package not installed; Slack filter disabled.")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def evaluate(
        self,
        *,
        text: str,
        sender_name: str,
        channel_name: str,
        is_dm: bool,
        user_id: str | None,
    ) -> dict:
        """Return verdict dict. Raises :class:`SlackFilterError` on failure."""
        if self._client is None:
            raise SlackFilterError("Slack filter not configured (no API key).")
        if not text or not text.strip():
            return {"relevant": False, "urgency": "background", "headline": ""}

        system = _SYSTEM_PROMPT_BASE
        if self._extra:
            system = system + "\n\nExtra context from the user:\n" + self._extra

        user_msg = self._format_user_msg(text, sender_name, channel_name, is_dm, user_id)

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            raise SlackFilterError(str(exc)) from exc

        raw = (resp.choices[0].message.content or "").strip()
        verdict = self._parse(raw)
        logger.info(
            "Slack filter %s/%s sender=%s relevant=%s urgency=%s headline=%r",
            channel_name,
            "dm" if is_dm else "ch",
            sender_name,
            verdict.get("relevant"),
            verdict.get("urgency"),
            verdict.get("headline"),
        )
        return verdict

    # ---- internals ------------------------------------------------------

    def _format_user_msg(
        self,
        text: str,
        sender_name: str,
        channel_name: str,
        is_dm: bool,
        user_id: str | None,
    ) -> str:
        parts: list[str] = []
        if user_id:
            parts.append(f"The user's Slack ID is {user_id}.")
        parts.append(f"Channel: {channel_name} ({'DM' if is_dm else 'public/private'})")
        parts.append(f"Sender: {sender_name}")
        parts.append("Message:")
        parts.append(text.strip())
        return "\n".join(parts)

    def _parse(self, raw: str) -> dict:
        """Parse the LLM's JSON output, tolerating mild stray prose."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Fish JSON out of stray prose / fences.
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise SlackFilterError(f"No JSON in filter response: {raw!r}")
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError as exc:
                raise SlackFilterError(f"Bad JSON in filter response: {exc}") from exc

        if not isinstance(data, dict):
            raise SlackFilterError(f"Filter returned non-object: {data!r}")

        relevant = bool(data.get("relevant", False))
        urgency = str(data.get("urgency") or "normal")
        if urgency not in ("interrupt", "normal", "background"):
            urgency = "normal"
        headline = str(data.get("headline") or "").strip()[:80]
        return {"relevant": relevant, "urgency": urgency, "headline": headline}
