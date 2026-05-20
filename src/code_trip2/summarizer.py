"""Summarizer: raw Claude pane output → spoken English.

A thin wrapper around an OpenAI chat completion. Replaces the
placeholder ``modes.clean_output`` (mechanical ANSI/box-drawing strip)
for any code path that wants audio-shaped text.

The system prompt mirrors the one drafted in
``docs/voice-driven-claude-design.md`` — keep it in code so the
summarizer can ship without separate template files.

Callers should treat ``SummarizerError`` as non-fatal and fall back to
``modes.clean_output``: the orchestrator should always be able to speak
*something* when Claude finishes a turn.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """You are converting developer tool output into spoken audio for a user
wearing earbuds who cannot see a screen. You will receive raw output from a
coding assistant.

Produce a concise spoken summary following these rules:
- NEVER include code, diffs, file contents, or raw terminal output
- NEVER use markdown formatting, bullet characters, or special symbols
- Summarize in natural spoken English, as if briefing a colleague
- For file paths, say just the filename unless the directory matters
- For errors, state the error type, the affected file, and the fix in plain English
- Keep it concise — aim for 15 to 30 seconds of speech
- Use ordinal markers for lists: "First... Second... Third..."
- Describe WHAT changed and WHY, not the literal code
- If the user needs to make a decision, state the options clearly
"""


class SummarizerError(Exception):
    pass


class Summarizer:
    """Audio-shaped summary of raw Claude pane output.

    Disabled if no API key is configured; callers should check
    :attr:`enabled` before calling. ``summarize`` raises on API errors so
    the caller can fall back without us hiding the failure.
    """

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str = "gpt-4o-mini",
        max_chars: int = 600,
        max_input_chars: int = 6000,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_chars = max_chars
        self._max_input_chars = max_input_chars
        self._client = None
        if api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai package not installed; summarizer disabled.")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def summarize(self, raw: str, *, context: dict | None = None) -> str:
        """Return an audio-shaped summary of ``raw``. Raises on API error."""
        if self._client is None:
            raise SummarizerError("Summarizer not configured (no OPENAI API key).")
        if not raw or not raw.strip():
            return ""
        user_msg = self._format_user_msg(raw, context)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=400,
            )
        except Exception as exc:  # openai SDK raises many exception types
            raise SummarizerError(str(exc)) from exc
        text = (resp.choices[0].message.content or "").strip()
        if len(text) > self._max_chars:
            text = text[: self._max_chars].rsplit(" ", 1)[0] + "…"
        return text

    def _format_user_msg(self, raw: str, context: dict | None) -> str:
        parts: list[str] = []
        if context:
            kind = context.get("kind")
            if kind:
                parts.append(f"Source: {kind}")
            user_prompt = context.get("user_prompt")
            if user_prompt:
                parts.append(f'User asked: "{user_prompt}"')
        # Cap input size: long captures bloat the prompt with little extra
        # signal because the useful response is usually at the tail.
        if len(raw) > self._max_input_chars:
            raw = "[... truncated ...]\n" + raw[-self._max_input_chars :]
        parts.append("Raw output:")
        parts.append(raw)
        return "\n".join(parts)
