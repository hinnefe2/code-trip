"""Summarizer LLM pipeline for converting Claude output to spoken English.

Passes raw terminal output from Claude Code through a small, fast LLM
(gpt-4o-mini by default) to produce concise spoken-English summaries
suitable for TTS playback.
"""

from __future__ import annotations

import enum
import os
from dataclasses import dataclass, field

import openai
from openai import APIConnectionError, APIError, AuthenticationError

# --- Module-level defaults ------------------------------------------------

DEFAULT_MODEL: str = "gpt-4o-mini"
ENV_API_KEY: str = "OPENAI_API_KEY"

SUMMARIZER_PROMPT: str = (
    "You are converting developer tool output into spoken audio for a user "
    "wearing earbuds who cannot see a screen. You will receive the raw output "
    "from a coding assistant.\n"
    "\n"
    "Produce a concise spoken summary following these rules:\n"
    "- NEVER include code, diffs, file contents, or raw terminal output\n"
    "- NEVER use markdown formatting, bullet characters, or special symbols\n"
    "- Summarize in natural spoken English, as if briefing a colleague\n"
    "- For file paths, say just the filename unless the directory matters\n"
    "- For errors, state the error type, the affected file, and the fix in "
    "plain English\n"
    "- Keep it concise — aim for 15-30 seconds of speech\n"
    "- Use ordinal markers for lists: \"First... Second... Third...\"\n"
    "- Describe WHAT changed and WHY, not the literal code\n"
    "- If the user needs to make a decision, state the options clearly"
)

VERBOSITY_INSTRUCTIONS: dict[str, str] = {
    "brief": "Keep your response to 1-2 sentences maximum.",
    "detailed": "Provide a full summary. Aim for 15-30 seconds of speech.",
    "verbose": (
        "Provide a thorough summary. Include file names, line counts, "
        "and test names where relevant."
    ),
}


# --- Enums ----------------------------------------------------------------


class Verbosity(str, enum.Enum):
    """Controls how much detail the summarizer produces."""

    BRIEF = "brief"
    DETAILED = "detailed"
    VERBOSE = "verbose"


# --- Exceptions -----------------------------------------------------------


class SummarizerError(Exception):
    """Raised when a summarization operation fails."""


# --- Summarizer -----------------------------------------------------------


@dataclass
class Summarizer:
    """Converts raw Claude Code output into spoken-English summaries.

    Args:
        model: OpenAI chat model name (default ``"gpt-4o-mini"``).
        api_key: OpenAI API key.  If ``None``, reads from the
            ``OPENAI_API_KEY`` environment variable.
        verbosity: Level of detail in summaries (default ``DETAILED``).
    """

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    verbosity: Verbosity = Verbosity.DETAILED

    _client: openai.OpenAI = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        resolved_key = self.api_key or os.environ.get(ENV_API_KEY)
        if not resolved_key:
            raise SummarizerError(
                f"No API key: set {ENV_API_KEY} or pass api_key"
            )
        self._client = openai.OpenAI(api_key=resolved_key)

    # --- public API -------------------------------------------------------

    def summarize(
        self,
        raw_output: str,
        mode: str | None = None,
        user_request: str | None = None,
    ) -> str:
        """Summarize raw Claude Code output as spoken English.

        Args:
            raw_output: Raw terminal output captured from Claude Code.
            mode: Current orchestrator mode (e.g. ``"WORK"``), prepended
                as context for the summarizer.
            user_request: What the user originally asked, prepended as
                context for the summarizer.

        Returns:
            A natural-language summary suitable for TTS.

        Raises:
            SummarizerError: if the API call fails or the response is empty.
        """
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(raw_output, mode, user_request)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
        except AuthenticationError as exc:
            raise SummarizerError(
                f"Authentication failed: {exc}"
            ) from exc
        except APIConnectionError as exc:
            raise SummarizerError(f"Network error: {exc}") from exc
        except APIError as exc:
            raise SummarizerError(f"API error: {exc}") from exc

        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise SummarizerError("Empty summary returned")
        return text

    # --- internals --------------------------------------------------------

    def _build_system_prompt(self) -> str:
        instruction = VERBOSITY_INSTRUCTIONS[self.verbosity.value]
        return f"{SUMMARIZER_PROMPT}\n\n{instruction}"

    @staticmethod
    def _build_user_message(
        raw_output: str,
        mode: str | None,
        user_request: str | None,
    ) -> str:
        parts: list[str] = []
        if mode is not None:
            parts.append(f"[Mode: {mode}]")
        if user_request is not None:
            parts.append(f'[User asked: "{user_request}"]')
        parts.append(raw_output)
        return " ".join(parts) if len(parts) > 1 else raw_output
