"""Speech-to-text via the OpenAI Whisper API.

Wraps the OpenAI ``/v1/audio/transcriptions`` endpoint behind a simple
``transcribe(audio_path) -> str`` interface.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import openai
from openai import APIConnectionError, APIError, AsyncOpenAI, AuthenticationError

# --- Module-level defaults ------------------------------------------------

DEFAULT_MODEL: str = "whisper-1"
ENV_API_KEY: str = "OPENAI_API_KEY"


# --- Exceptions -----------------------------------------------------------


class STTClientError(Exception):
    """Raised when a speech-to-text operation fails."""


# --- STTClient ------------------------------------------------------------


@dataclass
class STTClient:
    """Transcribes audio files via the OpenAI Whisper API.

    Args:
        api_key: OpenAI API key.  If ``None``, reads from the
            ``OPENAI_API_KEY`` environment variable.
        model: Whisper model name (default ``"whisper-1"``).
        language: Optional language hint (ISO-639-1), e.g. ``"en"``.
    """

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    language: str | None = None

    _client: AsyncOpenAI = field(default=None, init=False, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        resolved_key = self.api_key or os.environ.get(ENV_API_KEY)
        if not resolved_key:
            raise STTClientError(
                f"No API key: set {ENV_API_KEY} or pass api_key"
            )
        self._client = AsyncOpenAI(api_key=resolved_key)

    # --- public API -------------------------------------------------------

    async def transcribe(self, audio_path: Path | str) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to a WAV (or other supported) audio file.

        Returns:
            The transcription text.

        Raises:
            STTClientError: if the file is missing, the API call fails,
                or the transcription is empty.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise STTClientError(f"Audio file not found: {audio_path}")

        kwargs: dict = dict(model=self.model)
        if self.language is not None:
            kwargs["language"] = self.language

        try:
            with open(audio_path, "rb") as f:
                kwargs["file"] = f
                response = await self._client.audio.transcriptions.create(**kwargs)
        except AuthenticationError as exc:
            raise STTClientError(
                f"Authentication failed: {exc}"
            ) from exc
        except APIConnectionError as exc:
            raise STTClientError(
                f"Network error: {exc}"
            ) from exc
        except APIError as exc:
            raise STTClientError(f"API error: {exc}") from exc

        text = response.text.strip()
        if not text:
            raise STTClientError("Empty transcription returned")
        return text
