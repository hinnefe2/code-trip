"""Text-to-speech via the OpenAI TTS API.

Wraps the OpenAI ``/v1/audio/speech`` endpoint behind a simple
``speak(text) -> None`` interface.  Audio is requested as WAV, decoded
via the stdlib ``wave`` module, and played through ``sounddevice``
(cross-platform: Linux/macOS/Windows).  :meth:`stop` interrupts
playback mid-sentence.
"""

from __future__ import annotations

import io
import os
import wave
from dataclasses import dataclass, field

import numpy as np
import openai
from openai import APIConnectionError, APIError, AuthenticationError

try:
    import sounddevice as sd
except OSError:
    sd = None  # type: ignore[assignment]  # PortAudio not installed; tests mock this

# --- Module-level defaults ------------------------------------------------

DEFAULT_MODEL: str = "gpt-4o-mini-tts"
DEFAULT_VOICE: str = "nova"
DEFAULT_SPEED: float = 1.15
DEFAULT_FORMAT: str = "wav"
ENV_API_KEY: str = "OPENAI_API_KEY"

# Pop/click suppression. The OpenAI WAV doesn't always start or end at
# zero amplitude, and the audio device often goes idle between
# utterances (especially on Bluetooth earbuds), so the first sample of
# playback hits an asleep DAC and we hear a click. A short fade + a
# silence cushion lets the device wake up before the speech itself
# plays.
_FADE_MS: int = 12
_SILENCE_PAD_MS: int = 40


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
    _playing: bool = field(default=False, init=False, repr=False)

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
            TTSClientError: if the text is empty or the API call fails.
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

        samples, sample_rate = _decode_wav(audio)

        # Stop anything still playing before starting new audio.
        sd.stop()
        self._playing = True
        try:
            sd.play(samples, sample_rate)
            sd.wait()
        finally:
            self._playing = False

    def stop(self) -> None:
        """Interrupt any in-progress playback.  No-op if nothing is playing."""
        sd.stop()

    def is_playing(self) -> bool:
        """True while ``speak()`` is blocked on playback."""
        return self._playing


# --- helpers --------------------------------------------------------------


def _decode_wav(audio: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes into a numpy int16 array and sample rate.

    Applies a small fade-in/out and silence pad to suppress the DAC-wakeup
    click that's especially noticeable for short utterances ("Queue is
    empty.") and on Bluetooth audio paths.
    """
    with wave.open(io.BytesIO(audio), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    samples = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    samples = _shape_samples(samples, sample_rate)
    return samples, sample_rate


def _shape_samples(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply fade-in/out and prepend/append silence so playback doesn't pop."""
    if samples.size == 0:
        return samples
    n_fade = int(sample_rate * _FADE_MS / 1000)
    n_pad = int(sample_rate * _SILENCE_PAD_MS / 1000)
    n = len(samples)
    if 2 * n_fade < n:
        env = np.ones(n, dtype=np.float32)
        env[:n_fade] = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
        env[-n_fade:] = np.linspace(1.0, 0.0, n_fade, dtype=np.float32)
        if samples.ndim == 1:
            faded = (samples.astype(np.float32) * env).astype(np.int16)
        else:
            faded = (samples.astype(np.float32) * env[:, np.newaxis]).astype(np.int16)
    else:
        faded = samples
    if n_pad > 0:
        if samples.ndim == 1:
            silence = np.zeros(n_pad, dtype=np.int16)
        else:
            silence = np.zeros((n_pad, samples.shape[1]), dtype=np.int16)
        return np.concatenate([silence, faded, silence])
    return faded
