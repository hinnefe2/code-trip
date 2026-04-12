"""Text-to-speech via the OpenAI TTS API.

Wraps the OpenAI ``/v1/audio/speech`` endpoint behind a simple
``speak(text) -> None`` interface.  Synthesized audio is requested as WAV
and played through ``aplay`` via a subprocess, which also gives us a
cheap interruption mechanism: ``stop()`` kills the player process.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field

import openai
from openai import APIConnectionError, APIError, AuthenticationError

# --- Module-level defaults ------------------------------------------------

DEFAULT_MODEL: str = "gpt-4o-mini-tts"
DEFAULT_VOICE: str = "nova"
DEFAULT_SPEED: float = 1.15
DEFAULT_FORMAT: str = "wav"
ENV_API_KEY: str = "OPENAI_API_KEY"

# Player reads audio from stdin.  aplay auto-detects WAV format.
PLAYER_CMD: list[str] = ["aplay", "-q", "-"]


# --- Exceptions -----------------------------------------------------------


class TTSClientError(Exception):
    """Raised when a text-to-speech operation fails."""


# --- TTSClient ------------------------------------------------------------


@dataclass
class TTSClient:
    """Synthesizes speech from text via the OpenAI TTS API.

    Args:
        api_key: OpenAI API key.  If ``None``, reads from the
            ``OPENAI_API_KEY`` environment variable.
        model: TTS model name (default ``"gpt-4o-mini-tts"``).
        voice: Voice preset (default ``"nova"``).
        speed: Playback speed multiplier (default ``1.15``).
    """

    api_key: str | None = None
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    speed: float = DEFAULT_SPEED

    _client: openai.OpenAI = field(default=None, init=False, repr=False)  # type: ignore[assignment]
    _playback: subprocess.Popen | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        resolved_key = self.api_key or os.environ.get(ENV_API_KEY)
        if not resolved_key:
            raise TTSClientError(
                f"No API key: set {ENV_API_KEY} or pass api_key"
            )
        self._client = openai.OpenAI(api_key=resolved_key)

    # --- public API -------------------------------------------------------

    def speak(self, text: str) -> None:
        """Synthesize ``text`` and play it through the system audio.

        Blocks until playback finishes.  Call :meth:`stop` from another
        thread to interrupt mid-sentence.

        Raises:
            TTSClientError: if the text is empty, the API call fails,
                or the player cannot be started.
        """
        text = text.strip()
        if not text:
            raise TTSClientError("Empty text cannot be synthesized")

        try:
            response = self._client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                speed=self.speed,
                response_format=DEFAULT_FORMAT,
            )
        except AuthenticationError as exc:
            raise TTSClientError(f"Authentication failed: {exc}") from exc
        except APIConnectionError as exc:
            raise TTSClientError(f"Network error: {exc}") from exc
        except APIError as exc:
            raise TTSClientError(f"API error: {exc}") from exc

        audio: bytes = response.content
        if not audio:
            raise TTSClientError("Empty audio returned")

        # Stop anything still playing before starting new audio.
        self.stop()

        try:
            self._playback = subprocess.Popen(
                PLAYER_CMD,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise TTSClientError(
                f"Audio player not found: {PLAYER_CMD[0]}"
            ) from exc

        try:
            assert self._playback.stdin is not None
            self._playback.stdin.write(audio)
            self._playback.stdin.close()
            self._playback.wait()
        except BrokenPipeError:
            # Player exited early (e.g. stop() was called) — not an error.
            pass
        finally:
            self._playback = None

    def stop(self) -> None:
        """Interrupt any in-progress playback.  No-op if nothing is playing."""
        proc = self._playback
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self._playback = None
